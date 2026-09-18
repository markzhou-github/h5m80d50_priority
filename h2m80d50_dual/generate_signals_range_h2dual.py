from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from generate_signals2 import (
    PACKAGE_DIR, add_day_regime, add_family_predictions, add_seed_cv,
    add_signal_tags, load_all_models, load_config, normalize_keys, write_outputs,
)


def read_chunks(path: Path, columns: list[str] | None, batch_rows: int):
    if path.suffix.lower() == ".parquet":
        with pq.ParquetFile(path, pre_buffer=False) as pf:
            for batch in pf.iter_batches(columns=columns, batch_size=batch_rows, use_threads=False):
                yield batch.to_pandas()
    elif path.suffix.lower() in {".csv", ".txt"}:
        with pd.read_csv(path, dtype={"trade_date": str, "ts_code": str},
                         usecols=columns, chunksize=batch_rows) as chunks:
            yield from chunks
    else:
        raise ValueError(f"Unsupported input file: {path}")


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
    dates = values.astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    if not dates.str.fullmatch(r"\d{8}").all():
        raise ValueError("trade_date must contain non-null YYYYMMDD integers/strings or YYYY-MM-DD strings")
    return dates


def inspect_dates(path: Path, start: str, end: str) -> tuple[list[str], bool]:
    """Read only the date column in bounded batches; never load the wide panel."""
    found = set()
    ordered = True
    previous = None
    for chunk in read_chunks(path, ["trade_date"], 65536):
        values = normalize_dates(chunk["trade_date"])
        if values.empty:
            continue
        ordered = ordered and values.is_monotonic_increasing and (previous is None or previous <= values.iloc[0])
        previous = values.iloc[-1]
        found.update(values[values.between(start, end)].unique().tolist())
    return sorted(found), bool(ordered)


def iter_sorted_days(path: Path, columns: list[str] | None, start: str, end: str,
                     batch_rows: int, max_day_rows: int):
    """Single wide scan. Complete dates may span read batches and row groups."""
    active = None
    pieces = []
    count = 0
    for pdf in read_chunks(path, columns, batch_rows):
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
        del pdf
    if pieces:
        yield active, pd.concat(pieces, ignore_index=True)


def iter_filtered_days(path: Path, columns: list[str] | None, dates: list[str], max_day_rows: int):
    """Unsorted fallback: one full date at a time; potentially many disk scans."""
    if path.suffix.lower() in {".csv", ".txt"}:
        for date in dates:
            parts = []
            count = 0
            for chunk in read_chunks(path, columns, 2048):
                normalized = normalize_dates(chunk["trade_date"])
                part = chunk.loc[normalized.eq(date)].copy()
                if part.empty:
                    continue
                part["trade_date"] = date
                count += len(part)
                if count > max_day_rows:
                    raise ValueError(f"{date}: exceeds --max-day-rows")
                parts.append(part)
            if parts:
                yield date, pd.concat(parts, ignore_index=True)
        return
    source = pl.scan_parquet(path).with_columns(
        pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", "").str.slice(0, 8))
    for date in dates:
        frame = source.filter(pl.col("trade_date") == date).limit(max_day_rows + 1).collect(engine="streaming")
        if frame.height > max_day_rows:
            raise ValueError(f"{date}: exceeds --max-day-rows")
        yield date, frame.to_pandas()
        del frame


def build_range_summary(scored: pd.DataFrame, candidate_top_n: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, part in scored.groupby("trade_date", sort=True):
        rows.append(
            {
                "trade_date": date,
                "row_count": len(part),
                "signal_count": int(part["signal_layer"].ne("none").sum()),
                "candidate_count": int((part["avg_rank"] <= candidate_top_n).sum()),
                "overlap_core_count": int((part["signal_layer"] == "overlap_core").sum()),
                "strong_only_count": int((part["signal_layer"] == "strong_only").sum()),
                "strict_only_count": int((part["signal_layer"] == "strict_only").sum()),
                "layer2_signal_count": int(part["layer2_signal"].sum()),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate h2m80d50 dual-model signals for a date range."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Prepared feature panel parquet/csv.",
    )
    parser.add_argument("--start-date", required=True, type=date_arg, help="First date, YYYYMMDD.")
    parser.add_argument("--end-date", required=True, type=date_arg, help="Last date, YYYYMMDD.")
    parser.add_argument("--out-dir", type=Path, default=PACKAGE_DIR / "signals")
    parser.add_argument("--allow-missing-features", action="store_true")
    parser.add_argument(
        "--save-ranked",
        action="store_true",
        help="Also save the full ranked universe for every date.",
    )
    parser.add_argument(
        "--candidate-top-n",
        type=int,
        default=20,
        help="Save top-N base candidates for empty-signal diagnostics.",
    )
    parser.add_argument("--skip-seed", action="store_true")
    parser.add_argument("--skip-day-regime", action="store_true")
    parser.add_argument("--read-batch-rows", type=int, default=2048)
    parser.add_argument("--max-day-rows", type=int, default=20000)
    parser.add_argument("--read-mode", choices=["auto", "sorted", "filtered"], default="auto")
    return parser.parse_args()


def rss_message() -> str:
    try:
        import psutil
        return f"rss={psutil.Process().memory_info().rss / 2**30:.2f}GiB"
    except ImportError:
        return "rss=unavailable (optional: install psutil)"


def score_day(frame, family, seeds, day_bundle, cfg, allow_missing):
    frame = normalize_keys(frame)
    if frame["trade_date"].nunique() != 1:
        raise ValueError("Expected one complete trading day")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("Duplicate trade_date/ts_code rows in requested range")
    scored = add_family_predictions(frame, family, allow_missing)
    scored = add_seed_cv(scored, seeds, allow_missing)
    scored = add_day_regime(scored, day_bundle, allow_missing)
    return add_signal_tags(scored, cfg)


def main() -> None:
    args = parse_args()
    if args.candidate_top_n < 0:
        raise ValueError("candidate_top_n must be >= 0")
    if args.start_date > args.end_date:
        raise ValueError("start-date must be <= end-date")
    if min(args.read_batch_rows, args.max_day_rows) < 1:
        raise ValueError("Read batch and maximum daily rows must be positive")
    print("[inspect] reading trade_date only", flush=True)
    dates, ordered = inspect_dates(args.input, args.start_date, args.end_date)
    if not dates:
        raise ValueError("No rows in requested range")
    if args.read_mode == "sorted" and not ordered:
        raise ValueError("Input is not date-sorted; use --read-mode auto or filtered")
    mode = "sorted" if ordered and args.read_mode != "filtered" else "filtered"
    cfg = load_config()
    family, seed_bundles, day_bundle = load_all_models(
        cfg, include_seed=not args.skip_seed, include_day=not args.skip_day_regime)
    if not family:
        raise ValueError("No family models configured")
    print(f"[models] family={len(family)} seed={len(seed_bundles)} "
          f"day={day_bundle.name if day_bundle else 'none'}", flush=True)
    print(f"[input] dates={len(dates)} date_min={dates[0]} date_max={dates[-1]} "
          f"reader={mode} batch_rows={args.read_batch_rows}", flush=True)
    if mode == "filtered":
        print("[read] one complete day per scan; unsorted input may be slower", flush=True)
    # Keep ALL original columns: write_outputs exports the complete input schema.
    days = (iter_sorted_days(args.input, None, args.start_date, args.end_date,
                             args.read_batch_rows, args.max_day_rows)
            if mode == "sorted" else iter_filtered_days(args.input, None, dates, args.max_day_rows))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"range_generation_summary_{dates[0]}_{dates[-1]}.csv"
    summaries = []
    for i, (date, frame) in enumerate(days, 1):
        t0 = time.perf_counter()
        scored = score_day(frame, family, seed_bundles, day_bundle, cfg, args.allow_missing_features)
        write_outputs(scored, args.out_dir, args.save_ranked, args.candidate_top_n)
        summaries.append(build_range_summary(scored, args.candidate_top_n))
        temporary = summary_path.with_suffix(".csv.partial")
        pd.concat(summaries, ignore_index=True).to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(summary_path)
        rows = len(scored)
        del scored, frame
        gc.collect()
        print(f"[{i}/{len(dates)}] {date} rows={rows} seconds={time.perf_counter()-t0:.2f} "
              f"{rss_message()}", flush=True)
    if pd.concat(summaries)["trade_date"].tolist() != dates:
        raise RuntimeError("Output dates differ from preflight; input may have changed")
    print(f"[SAVE] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
