from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl

from generate_signals2 import (
    PACKAGE_DIR,
    add_day_regime,
    add_family_predictions,
    add_seed_cv,
    add_signal_tags,
    load_all_models,
    load_config,
    normalize_keys,
    write_outputs,
)


def normalize_date(value: object) -> str:
    return str(value).replace("-", "")[:8]


def read_input_range(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """Read rows whose normalized trade_date lies in the inclusive range."""
    start = normalize_date(start_date)
    end = normalize_date(end_date)
    if start > end:
        raise ValueError(f"start_date {start} is after end_date {end}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = (
            pl.scan_parquet(str(path))
            .with_columns(
                pl.col("trade_date")
                .cast(pl.Utf8)
                .str.replace_all("-", "")
                .str.slice(0, 8)
                .alias("trade_date"),
                pl.col("ts_code").cast(pl.Utf8),
            )
            .filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))
            .collect()
            .to_pandas()
        )
    elif suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path, dtype={"trade_date": str, "ts_code": str})
        frame = normalize_keys(frame)
        frame = frame[frame["trade_date"].between(start, end)].copy()
    else:
        raise ValueError(f"Unsupported input file type: {path}")

    if frame.empty:
        raise ValueError(f"No rows for date range {start} ~ {end} in {path}")

    frame = normalize_keys(frame)
    duplicate_mask = frame[["trade_date", "ts_code"]].duplicated(keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, ["trade_date", "ts_code"]].head(10)
        raise ValueError(
            "Duplicate trade_date/ts_code rows in requested range:\n"
            f"{examples.to_string(index=False)}"
        )
    return frame


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
    parser.add_argument("--start-date", required=True, help="First date, YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="Last date, YYYYMMDD.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_top_n < 0:
        raise ValueError("candidate_top_n must be >= 0")

    cfg = load_config()
    family, seed_bundles, day_bundle = load_all_models(
        cfg,
        include_seed=not args.skip_seed,
        include_day=not args.skip_day_regime,
    )
    print(
        f"[models] family={len(family)} seed={len(seed_bundles)} "
        f"day={day_bundle.name if day_bundle else 'none'}"
    )

    print(f"[load] {args.input} {args.start_date} ~ {args.end_date}")
    frame = read_input_range(args.input, args.start_date, args.end_date)
    dates = sorted(frame["trade_date"].astype(str).unique().tolist())
    print(
        f"[input] rows={len(frame)} dates={len(dates)} "
        f"date_min={dates[0]} date_max={dates[-1]}"
    )

    scored = add_family_predictions(frame, family, args.allow_missing_features)
    scored = add_seed_cv(scored, seed_bundles, args.allow_missing_features)
    scored = add_day_regime(scored, day_bundle, args.allow_missing_features)
    scored = add_signal_tags(scored, cfg)
    write_outputs(scored, args.out_dir, args.save_ranked, args.candidate_top_n)

    summary = build_range_summary(scored, args.candidate_top_n)
    summary_path = (
        args.out_dir
        / f"range_generation_summary_{dates[0]}_{dates[-1]}.csv"
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()
