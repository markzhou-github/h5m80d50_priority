"""Daily, memory-bounded runner for the existing h5m80d50 signal package.

Place beside generate_signals.py and generate_signals_range.py. Requires their
unchanged signal functions, config.json and models. Input: one parquet file.
"""
from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from generate_signals import (
    add_base_selections, add_predictions, add_signal_models, add_watchlist,
    diagnostic_columns, load_config, load_models, normalize_keys, signal_columns,
)
from generate_signals_range import write_one_date


def date_arg(value: str) -> str:
    value = str(value).replace("-", "")
    try:
        pd.to_datetime(value, format="%Y%m%d", errors="raise")
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("Use a valid YYYYMMDD date") from exc
    if len(value) != 8 or not value.isdigit():
        raise argparse.ArgumentTypeError("Use YYYYMMDD")
    return value


def normalize_dates(values: pd.Series) -> pd.Series:
    dates = values.astype(str).str.replace("-", "", regex=False)
    if not dates.str.fullmatch(r"\d{8}").all():
        raise ValueError("trade_date must contain non-null YYYYMMDD integers/strings or YYYY-MM-DD strings")
    return dates


def inspect_dates(path: Path, start: str, end: str) -> tuple[list[str], bool]:
    """Read only the date column in bounded batches; never load the wide panel."""
    found = set()
    ordered = True
    previous = None
    with pq.ParquetFile(path, pre_buffer=False) as pf:
        for batch in pf.iter_batches(columns=["trade_date"], batch_size=65536, use_threads=False):
            values = normalize_dates(batch.column(0).to_pandas())
            if values.empty:
                continue
            ordered = ordered and values.is_monotonic_increasing and (previous is None or previous <= values.iloc[0])
            previous = values.iloc[-1]
            found.update(values[values.between(start, end)].unique().tolist())
    return sorted(found), bool(ordered)


def config_dependencies(node) -> set[str]:
    found = set()
    if isinstance(node, dict):
        if isinstance(node.get("base"), str):
            found.add(node["base"])
        for value in node.values():
            found.update(config_dependencies(value))
    elif isinstance(node, list):
        if len(node) == 3 and isinstance(node[0], str) and isinstance(node[1], str) and node[1] in {"<=", ">="}:
            found.add(node[0])
        else:
            for value in node:
                found.update(config_dependencies(value))
    return found


def selected_columns(names: list[str], bundles, cfg: dict, allow_missing: bool) -> list[str]:
    available = set(names)
    required_keys = {"trade_date", "ts_code", "turnover_prank_1500"}
    if not required_keys <= available:
        raise ValueError(f"Missing required input columns: {sorted(required_keys - available)}")
    model_features = set()
    for bundle in bundles:
        missing = set(bundle.features) - available
        if missing and not allow_missing:
            raise ValueError(f"{bundle.name}: missing {len(missing)} features, first: {sorted(missing)[:20]}")
        model_features.update(bundle.features)
    # Query existing output selectors using only an empty schema, not actual data.
    empty = pd.DataFrame(columns=names)
    wanted = (required_keys | model_features | config_dependencies(cfg)
              | set(signal_columns(empty)) | set(diagnostic_columns(empty)))
    wanted.update(c for c in names if c.startswith(("pred_", "rank_")))
    return [c for c in names if c in wanted]


def iter_sorted_days(path: Path, columns: list[str], start: str, end: str,
                     batch_rows: int, max_day_rows: int):
    """Single wide scan. Complete dates may span read batches and row groups."""
    active = None
    pieces = []
    count = 0
    with pq.ParquetFile(path, pre_buffer=False) as pf:
        for batch in pf.iter_batches(columns=columns, batch_size=batch_rows, use_threads=False):
            pdf = batch.to_pandas()
            pdf["trade_date"] = normalize_dates(pdf["trade_date"])
            values = pdf["trade_date"].to_numpy()
            boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                date = values[left]
                if active is not None and date != active:
                    ready = pd.concat(pieces, ignore_index=True)
                    pieces = []
                    count = 0
                    yield active, ready
                    del ready
                    active = None
                if date < start:
                    continue
                if date > end:
                    return
                active = date
                count += right - left
                if count > max_day_rows:
                    raise ValueError(f"{date}: more than {max_day_rows} rows; inspect duplicates or increase --max-day-rows")
                # Copy only the fragment so a retained day cannot pin old batches.
                pieces.append(pdf.iloc[left:right].copy())
            del pdf, batch
        if pieces:
            yield active, pd.concat(pieces, ignore_index=True)


def iter_filtered_days(path: Path, columns: list[str], dates: list[str], max_day_rows: int):
    """Unsorted fallback: one full date at a time; potentially many disk scans."""
    source = pl.scan_parquet(path).select(columns)
    dtype = source.collect_schema()["trade_date"]
    for date in dates:
        if dtype.is_integer():
            predicate = pl.col("trade_date") == int(date)
        else:
            # Direct string predicates allow row-group statistics pruning.
            dashed = f"{date[:4]}-{date[4:6]}-{date[6:]}"
            predicate = pl.col("trade_date").is_in([date, dashed])
        frame = source.filter(predicate).limit(max_day_rows + 1).collect(engine="streaming")
        if frame.height > max_day_rows:
            raise ValueError(f"{date}: more than {max_day_rows} rows; inspect duplicates or increase --max-day-rows")
        yield date, frame.to_pandas()
        del frame


def process_day(df: pd.DataFrame, bundles, cfg: dict, allow_missing: bool,
                watchlist_top_n: int, watchlist_vote_min: int) -> pd.DataFrame:
    df = normalize_keys(df)
    if df["trade_date"].nunique() != 1:
        raise ValueError("Prediction batch must contain one complete date")
    if df.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("Duplicate (trade_date, ts_code) input keys; refusing ambiguous ranking/join expansion")
    before = set(df.columns)
    df = add_predictions(df, bundles, allow_missing)
    # Subsequent signal functions copy their input. Discard model-only features
    # after inference, while retaining all rule, diagnostic and output columns.
    needed = (set(signal_columns(df)) | set(diagnostic_columns(df))
              | config_dependencies(cfg) | (set(df.columns) - before))
    needed.update(c for c in df.columns if c.startswith(("pred_", "rank_")))
    df = df[[c for c in df.columns if c in needed]].copy()
    df = add_base_selections(df, cfg, [b.name for b in bundles])
    df = add_signal_models(df, cfg)
    return add_watchlist(df, watchlist_top_n, watchlist_vote_min)


def rss_message() -> str:
    try:
        import psutil
        return f"rss={psutil.Process().memory_info().rss / 2**30:.2f}GiB"
    except ImportError:
        return "rss=unavailable (optional: install psutil)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--start-date", type=date_arg, required=True)
    parser.add_argument("--end-date", type=date_arg, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "signals_memory_safe")
    parser.add_argument("--allow-missing-features", action="store_true")
    parser.add_argument("--save-ranked", action="store_true")
    parser.add_argument("--diagnostic-top-n", type=int, default=30)
    parser.add_argument("--watchlist-top-n", type=int, default=3)
    parser.add_argument("--watchlist-vote-min", type=int, default=3)
    parser.add_argument("--read-batch-rows", type=int, default=2048)
    parser.add_argument("--max-day-rows", type=int, default=20000)
    parser.add_argument("--read-mode", choices=["auto", "sorted", "filtered"], default="auto")
    args = parser.parse_args()
    if args.input.suffix.lower() != ".parquet":
        parser.error("This bounded runner accepts a single parquet file")
    if args.start_date > args.end_date:
        parser.error("start-date must be <= end-date")
    if min(args.read_batch_rows, args.max_day_rows, args.diagnostic_top_n,
           args.watchlist_top_n, args.watchlist_vote_min) < 1:
        parser.error("Batch sizes and top/vote counts must be positive")

    print("[inspect] reading date column only", flush=True)
    dates, ordered = inspect_dates(args.input, args.start_date, args.end_date)
    if not dates:
        raise ValueError("No input dates in requested range")
    if args.read_mode == "sorted" and not ordered:
        raise ValueError("Input is not date-sorted; use --read-mode auto or filtered")
    mode = "sorted" if ordered and args.read_mode != "filtered" else "filtered"
    cfg = load_config()
    bundles = load_models(cfg)
    if not bundles:
        raise ValueError("No models configured")
    with pq.ParquetFile(args.input, pre_buffer=False) as pf:
        names = pf.schema_arrow.names
    columns = selected_columns(names, bundles, cfg, args.allow_missing_features)
    print(f"[plan] dates={len(dates)} models={len(bundles)} columns={len(columns)}/{len(names)} "
          f"reader={mode} batch_rows={args.read_batch_rows} {rss_message()}", flush=True)
    if mode == "filtered":
        print("[read] unsorted/filtered mode may scan the file repeatedly; each prediction still uses one complete day", flush=True)
    days = (iter_sorted_days(args.input, columns, args.start_date, args.end_date,
                             args.read_batch_rows, args.max_day_rows) if mode == "sorted"
            else iter_filtered_days(args.input, columns, dates, args.max_day_rows))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"range_generation_summary_{dates[0]}_{dates[-1]}.csv"
    records = []
    for i, (date, day) in enumerate(days, 1):
        t0 = time.perf_counter()
        result = process_day(day, bundles, cfg, args.allow_missing_features,
                             args.watchlist_top_n, args.watchlist_vote_min)
        record = write_one_date(result, date, args.out_dir, args.save_ranked, args.diagnostic_top_n)
        records.append(record)
        # Small summary checkpoint; completed daily files survive a later failure.
        temporary = summary_path.with_suffix(".csv.partial")
        pd.DataFrame(records).to_csv(temporary, index=False, encoding="utf_8_sig")
        temporary.replace(summary_path)
        del result, day
        gc.collect()
        print(f"[{i}/{len(dates)}] {date} rows={record['rows']} trade={record['trade_rows']} "
              f"watchlist={record['watchlist_rows']} seconds={time.perf_counter()-t0:.2f} {rss_message()}", flush=True)
    if [r["trade_date"] for r in records] != dates:
        raise RuntimeError("Written dates differ from preflight; check whether input changed during the run")
    print(f"[SAVE] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
