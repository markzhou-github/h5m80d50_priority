#!/usr/bin/env python3
"""Generate frozen H5 competing-direction Top3 signals for a feature-date range."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from generate_signals import HERE, EPS, lines, platt, predict


def normalize_date(value: object) -> str:
    """Normalize YYYYMMDD / YYYY-MM-DD-like values to YYYYMMDD."""
    return str(value).strip().replace("-", "")[:8]


def read_input_range(
    path: Path,
    start_date: str,
    end_date: str,
    required_features: list[str],
) -> pd.DataFrame:
    """Read only required columns and rows in the inclusive date range."""
    start = normalize_date(start_date)
    end = normalize_date(end_date)

    if start > end:
        raise ValueError(f"start_date {start} is after end_date {end}")

    schema = pl.read_parquet_schema(path)
    required = ["trade_date", "ts_code", *required_features]
    missing = sorted(set(required) - set(schema.names()))
    if missing:
        raise KeyError(
            f"Input lacks {len(missing)} required columns: {missing[:30]}"
        )

    frame = (
        pl.scan_parquet(path)
        .select(required)
        .with_columns(
            pl.col("trade_date")
            .cast(pl.Utf8)
            .str.replace_all("-", "")
            .str.slice(0, 8)
            .alias("trade_date"),
            pl.col("ts_code").cast(pl.Utf8),
        )
        .filter(
            (pl.col("trade_date") >= start)
            & (pl.col("trade_date") <= end)
        )
        .collect()
        .to_pandas()
    )

    if frame.empty:
        raise ValueError(f"No rows for date range {start} ~ {end} in {path}")

    duplicate_mask = frame[["trade_date", "ts_code"]].duplicated(keep=False)
    if duplicate_mask.any():
        examples = frame.loc[
            duplicate_mask,
            ["trade_date", "ts_code"],
        ].head(10)
        raise ValueError(
            "Duplicate trade_date/ts_code rows in requested range:\n"
            f"{examples.to_string(index=False)}"
        )

    return frame


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
    args = parser.parse_args()

    start_date = normalize_date(args.start_date)
    end_date = normalize_date(args.end_date)
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

    frame = read_input_range(
        path=args.input,
        start_date=start_date,
        end_date=end_date,
        required_features=required_features,
    )

    dates = sorted(frame["trade_date"].astype(str).unique().tolist())

    print(
        f"[rows] {len(frame)} dates={len(dates)} "
        f"date_min={dates[0]} date_max={dates[-1]}",
        flush=True,
    )

    ranked_range = build_ranked_range(
        frame=frame,
        cfg=cfg,
        calibration=calibration,
        feature_lists=feature_lists,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []

    for index, date in enumerate(dates, start=1):
        summary = write_one_date(
            ranked_range=ranked_range,
            date=date,
            out_dir=args.out_dir,
            save_ranked=args.save_ranked,
            cfg=cfg,
        )

        summaries.append(summary)

        print(
            f"[{index}/{len(dates)}] {date} "
            f"universe={summary['universe_rows']} "
            f"candidates={summary['stage1_candidates']} "
            f"signals={summary['signals']}",
            flush=True,
        )

    summary_path = (
        args.out_dir
        / f"range_generation_summary_{dates[0]}_{dates[-1]}.csv"
    )

    pd.DataFrame(
        [
            {
                **{k: v for k, v in summary.items() if k != "policy"},
                "policy": json.dumps(summary["policy"], ensure_ascii=False),
            }
            for summary in summaries
        ]
    ).to_csv(summary_path, index=False, encoding="utf_8_sig")

    print(f"[SAVE] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
