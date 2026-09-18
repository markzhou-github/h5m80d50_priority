#!/usr/bin/env python3
"""Generate frozen H5 competing-direction Top3 signals for a feature-date range."""

from __future__ import annotations

import argparse
import json
import gc
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import lightgbm as lgb

from generate_signals import HERE, EPS, lines, platt


def normalize_date(value: object) -> str:
    """Normalize YYYYMMDD / YYYY-MM-DD-like values to YYYYMMDD."""
    return str(value).strip().replace("-", "")[:8]


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
    source = pl.scan_parquet(path).select(columns).with_columns(
        pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", "").str.slice(0, 8))
    for date in dates:
        frame = source.filter(pl.col("trade_date") == date).limit(max_day_rows + 1).collect(engine="streaming")
        if frame.height > max_day_rows:
            raise ValueError(f"{date}: exceeds --max-day-rows")
        yield date, frame.to_pandas()
        del frame


def rss_message() -> str:
    try:
        import psutil
        return f"rss={psutil.Process().memory_info().rss / 2**30:.2f}GiB"
    except ImportError:
        return "rss=unavailable (optional: install psutil)"


def configure_model_cache(max_models: int = 64) -> None:
    global _load_model
    if max_models < 0:
        raise ValueError("model-cache-size must be >= 0")
    previous = globals().get("_load_model")
    if previous is not None:
        previous.cache_clear()

    @lru_cache(maxsize=max_models)
    def load_model(path: str):
        return lgb.Booster(model_file=path)

    _load_model = load_model


configure_model_cache()


def predict(frame, model_root, features, seeds):
    # Match the original float32 conversion, schema check, iteration choice and
    # averaging order. Cache only model objects, never input frames/predictions.
    x = frame[features].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    values = []
    for seed in seeds:
        model = _load_model(str((model_root / f"seed{seed}" / "model.txt").resolve()))
        if model.feature_name() != features:
            raise ValueError(f"Feature schema mismatch: {model_root.name}/seed{seed}")
        values.append(model.predict(x, num_iteration=model.current_iteration()))
    return np.column_stack(values).mean(axis=1)


def build_ranked_range(
    frame: pd.DataFrame,
    cfg: dict,
    calibration: dict,
    feature_lists: dict[str, list[str]],
) -> pd.DataFrame:
    """Run model predictions once, then apply ranking independently by date."""
    ranked = frame[["trade_date", "ts_code"]].copy()

    # ------------------------------------------------------------------
    # Stage 1 families
    # ------------------------------------------------------------------
    for family, spec in cfg["stage1_families"].items():
        print(f"[predict] stage1 family={family}", flush=True)
        ranked[f"{family}_pred"] = predict(
            frame,
            HERE / "models/stage1" / family,
            feature_lists[f"stage1_{family}"],
            cfg["stage1_seeds"],
        )

        ranked[f"{family}_rank"] = (
            ranked.groupby("trade_date", sort=False)[f"{family}_pred"]
            .rank(method="first", ascending=False)
            .astype(int)
        )

        ranked[f"{family}_selected"] = (
            ranked[f"{family}_rank"] <= int(spec["top_k"])
        )

    selected_columns = [
        f"{family}_selected" for family in cfg["stage1_families"]
    ]

    ranked["family_count"] = ranked[selected_columns].sum(axis=1).astype(int)
    ranked["stage1_candidate"] = ranked["family_count"] >= 1
    ranked["stage1_ensemble_pred"] = ranked[
        [f"{family}_pred" for family in cfg["stage1_families"]]
    ].mean(axis=1)

    # ------------------------------------------------------------------
    # Competing-direction heads
    # ------------------------------------------------------------------
    for head in ["up", "down", "intensity", "direction"]:
        print(f"[predict] competing head={head}", flush=True)
        ranked[f"p_{head}"] = predict(
            frame,
            HERE / "models/competing" / head,
            feature_lists[f"competing_{head}"],
            cfg["competing_seeds"],
        )

    ranked["p_up_calibrated"] = platt(
        ranked["p_up"].to_numpy(), calibration["up"]
    )
    ranked["p_down_calibrated"] = platt(
        ranked["p_down"].to_numpy(), calibration["down"]
    )

    ranked["direction_ratio"] = ranked["p_up_calibrated"] / (
        ranked["p_up_calibrated"]
        + ranked["p_down_calibrated"]
        + EPS
    )

    ranked["conditional_score"] = (
        ranked["p_intensity"] * ranked["p_direction"]
    )

    # conditional_rank in generate_signals.py is assigned only among stage1
    # candidates. It therefore must restart separately for each date.
    ranked["conditional_rank"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")

    candidate_mask = ranked["stage1_candidate"]
    candidate_rows = ranked.loc[candidate_mask].copy()
    candidate_rows = candidate_rows.sort_values(
        [
            "trade_date",
            "conditional_score",
            "stage1_ensemble_pred",
            "ts_code",
        ],
        ascending=[True, False, False, True],
        kind="stable",
    )

    candidate_rows["conditional_rank"] = (
        candidate_rows.groupby("trade_date", sort=False)
        .cumcount()
        .add(1)
        .astype(int)
    )

    ranked.loc[candidate_rows.index, "conditional_rank"] = (
        candidate_rows["conditional_rank"].astype("Int64")
    )

    policy = cfg["policy"]
    threshold = float(policy["direction_ratio_threshold"])
    top_k = int(policy["top_k"])

    ranked["direction_gate_pass"] = ranked["direction_ratio"] >= threshold
    ranked["production_signal"] = (
        ranked["stage1_candidate"]
        & ranked["conditional_rank"].le(top_k).fillna(False)
        & ranked["direction_gate_pass"]
    )

    ranked["signal_tag"] = np.where(
        ranked["production_signal"],
        "H5_COMPETING_DIRECTION_TOP3",
        "WATCHLIST",
    )

    return ranked


def write_one_date(
    ranked_range: pd.DataFrame,
    date: str,
    out_dir: Path,
    save_ranked: bool,
    cfg: dict,
) -> dict:
    """Write the same per-date artifacts as generate_signals.py."""
    ranked = ranked_range.loc[ranked_range["trade_date"].eq(date)].copy()

    candidates = ranked.loc[ranked["stage1_candidate"]].copy()
    candidates = candidates.sort_values(
        ["conditional_score", "stage1_ensemble_pred", "ts_code"],
        ascending=[False, False, True],
        kind="stable",
    )

    # conditional_rank has already been computed date-by-date, but sort order is
    # kept identical to generate_signals.py for output readability.
    signals = candidates.loc[candidates["production_signal"]].copy()

    signal_path = out_dir / f"signals_{date}.csv"
    watchlist_path = out_dir / f"watchlist_{date}.csv"

    signals.to_csv(signal_path, index=False)
    signals.to_csv(out_dir / "signals_latest.csv", index=False)

    candidates.to_csv(watchlist_path, index=False)
    candidates.to_csv(out_dir / "watchlist_latest.csv", index=False)

    if save_ranked:
        ranked.to_parquet(out_dir / f"ranked_{date}.parquet", index=False)

    summary = {
        "trade_date": date,
        "universe_rows": int(len(ranked)),
        "stage1_candidates": int(len(candidates)),
        "signals": int(len(signals)),
        "policy": cfg["policy"],
    }

    (out_dir / f"summary_{date}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


def write_range_summary(summaries: list[dict], summary_path: Path) -> None:
    temporary = summary_path.with_suffix(".csv.partial")
    pd.DataFrame(
        [
            {
                **{k: v for k, v in summary.items() if k != "policy"},
                "policy": json.dumps(summary["policy"], ensure_ascii=False),
            }
            for summary in summaries
        ]
    ).to_csv(temporary, index=False, encoding="utf_8_sig")

    temporary.replace(summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate frozen H5 competing-direction Top3 signals for an "
            "inclusive feature-date range."
        )
    )

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--start-date", required=True, help="First feature date (YYYYMMDD).")
    parser.add_argument("--end-date", required=True, help="Last feature date (YYYYMMDD).")
    parser.add_argument("--out-dir", type=Path, default=HERE / "signals")
    parser.add_argument("--save-ranked", action="store_true")
    parser.add_argument("--read-batch-rows", type=int, default=2048)
    parser.add_argument("--max-day-rows", type=int, default=20000)
    parser.add_argument("--read-mode", choices=["auto", "sorted", "filtered"], default="auto")
    parser.add_argument("--model-cache-size", type=int, default=64,
                        help="Maximum cached model objects; 0 disables caching. Default fits the 35 configured models.")
    args = parser.parse_args()
    if args.input.suffix.lower() != ".parquet":
        parser.error("Input must be a parquet file")
    if min(args.read_batch_rows, args.max_day_rows) < 1 or args.model_cache_size < 0:
        parser.error("Read sizes must be positive and model-cache-size must be nonnegative")
    configure_model_cache(args.model_cache_size)

    start_date = date_arg(args.start_date)
    end_date = date_arg(args.end_date)
    if start_date > end_date:
        raise ValueError(
            f"start_date {start_date} is after end_date {end_date}"
        )

    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    calibration = json.loads(
        (HERE / "calibration.json").read_text(encoding="utf-8")
    )

    feature_lists: dict[str, list[str]] = {}

    for family in cfg["stage1_families"]:
        feature_lists[f"stage1_{family}"] = lines(
            HERE / "models/stage1" / family / "features.txt"
        )

    for head in ["up", "down", "intensity", "direction"]:
        feature_lists[f"competing_{head}"] = lines(
            HERE / "models/competing" / head / "features.txt"
        )

    required_features = sorted(set().union(*feature_lists.values()))

    print(
        f"[load] {args.input} {start_date} ~ {end_date}",
        flush=True,
    )

    required = list(dict.fromkeys(["trade_date", "ts_code", *required_features]))
    schema = pl.read_parquet_schema(args.input)
    missing = sorted(set(required) - set(schema.names()))
    if missing:
        raise KeyError(f"Input lacks {len(missing)} required columns: {missing[:30]}")
    print("[inspect] reading date column only", flush=True)
    dates, ordered = inspect_dates(args.input, start_date, end_date)
    if not dates:
        raise ValueError("No rows in requested range")
    if args.read_mode == "sorted" and not ordered:
        raise ValueError("Input is not date-sorted; use --read-mode auto or filtered")
    mode = "sorted" if ordered and args.read_mode != "filtered" else "filtered"
    print(f"[plan] dates={len(dates)} features={len(required_features)} reader={mode} "
          f"model_cache_limit={args.model_cache_size}", flush=True)
    if mode == "filtered":
        print("[read] daily filtered scans may be slower for unsorted input", flush=True)
    days = (iter_sorted_days(args.input, required, start_date, end_date,
                             args.read_batch_rows, args.max_day_rows)
            if mode == "sorted" else iter_filtered_days(args.input, required, dates, args.max_day_rows))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for index, (date, frame) in enumerate(days, start=1):
        t0 = time.perf_counter()
        frame["trade_date"] = normalize_dates(frame["trade_date"])
        frame["ts_code"] = frame["ts_code"].astype(str)
        if frame.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError(f"Duplicate trade_date/ts_code rows on {date}")
        ranked_range = build_ranked_range(frame, cfg, calibration, feature_lists)
        summary = write_one_date(ranked_range, date, args.out_dir, args.save_ranked, cfg)
        summaries.append(summary)
        checkpoint = args.out_dir / f"range_generation_summary_{dates[0]}_{dates[-1]}.csv"
        write_range_summary(summaries, checkpoint)
        del frame, ranked_range
        gc.collect()
        print(f"[{index}/{len(dates)}] {date} universe={summary['universe_rows']} "
              f"candidates={summary['stage1_candidates']} signals={summary['signals']} "
              f"seconds={time.perf_counter()-t0:.2f} {rss_message()}", flush=True)
    if [summary["trade_date"] for summary in summaries] != dates:
        raise RuntimeError("Written dates differ from preflight; input may have changed")

    summary_path = (
        args.out_dir
        / f"range_generation_summary_{dates[0]}_{dates[-1]}.csv"
    )

    write_range_summary(summaries, summary_path)

    print(f"[SAVE] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
