from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from generate_signals import HERE, predict_family, read_lines


def normalize_date(value: object) -> str:
    return str(value).replace("-", "")[:8]


def read_input_range(
    path: Path,
    start_date: str,
    end_date: str,
    columns: list[str],
) -> pd.DataFrame:
    """Read only required columns and rows in the inclusive date range."""
    start = normalize_date(start_date)
    end = normalize_date(end_date)
    if start > end:
        raise ValueError(f"start_date {start} is after end_date {end}")

    schema = pl.read_parquet_schema(path)
    required = ["trade_date", "ts_code", *columns]
    missing = sorted(set(required) - set(schema.names()))
    if missing:
        raise KeyError(f"Input lacks {len(missing)} required columns: {missing[:20]}")

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
        .filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))
        .collect()
        .to_pandas()
    )

    if frame.empty:
        raise ValueError(f"No rows for date range {start} ~ {end} in {path}")
    if frame[["trade_date", "ts_code"]].duplicated().any():
        examples = frame.loc[
            frame[["trade_date", "ts_code"]].duplicated(keep=False),
            ["trade_date", "ts_code"],
        ].head(10)
        raise ValueError(
            "Duplicate trade_date/ts_code rows in requested range:\n"
            f"{examples.to_string(index=False)}"
        )
    return frame


def build_ranked(
    frame: pd.DataFrame,
    cfg: dict[str, object],
    family_features: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Predict the range and apply every rank/gate independently by date."""
    families = cfg["families"]
    policy = cfg["signal_policy"]
    gate_source = str(policy["gate_source_feature"])

    ranked = frame[["trade_date", "ts_code"]].copy()
    for family in families:
        print(f"[predict] family={family}")
        pred = predict_family(frame, family, cfg["seeds"])
        ranked = pd.concat([ranked, pred], axis=1)
        ranked[f"{family}_rank"] = (
            ranked.groupby("trade_date", sort=False)[f"{family}_pred"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        ranked[f"{family}_selected"] = (
            ranked[f"{family}_rank"] <= int(families[family]["top_k"])
        )

    selected_cols = [f"{name}_selected" for name in families]
    ranked["family_count"] = ranked[selected_cols].sum(axis=1).astype(int)
    ranked["ensemble_pred"] = ranked[
        [f"{name}_pred" for name in families]
    ].mean(axis=1)
    ranked["ensemble_seed_std"] = ranked[
        [f"{name}_seed_std" for name in families]
    ].mean(axis=1)
    ranked["legacy_signal_tag"] = np.select(
        [ranked.family_count == 3, ranked.family_count == 2, ranked.family_count == 1],
        ["CONSENSUS_3", "CONSENSUS_2", "FAMILY_ONLY"],
        default="",
    )
    ranked["family_tags"] = ranked.apply(
        lambda row: "|".join(
            name for name in families if row[f"{name}_selected"]
        ),
        axis=1,
    )

    valid_gate_values = frame[gate_source].replace([np.inf, -np.inf], np.nan)
    ranked[gate_source] = valid_gate_values
    gate_rank = valid_gate_values.groupby(frame["trade_date"], sort=False).rank(
        method="average",
        ascending=True,
    )
    valid_count_by_date = valid_gate_values.notna().groupby(
        frame["trade_date"], sort=False
    ).transform("sum")
    ranked[str(policy["gate_feature"])] = gate_rank / valid_count_by_date.replace(0, np.nan)

    threshold = float(policy["gate_threshold"])
    ranked["production_gate_pass"] = (
        ranked[str(policy["gate_feature"])].le(threshold).fillna(False)
    )
    ranked["production_signal"] = (
        ranked[f"{policy['family']}_rank"].le(int(policy["top_k"]))
        & ranked["production_gate_pass"]
    )
    ranked["ensemble_signal"] = ranked["family_count"].ge(1)
    ranked["output_signal"] = ranked["production_signal"] | ranked["ensemble_signal"]
    ranked["signal_tag"] = ranked.apply(
        lambda row: "|".join(
            tag
            for tag in [
                str(policy["production_tag"]) if row["production_signal"] else "",
                str(row["legacy_signal_tag"]) if row["ensemble_signal"] else "",
            ]
            if tag
        ),
        axis=1,
    )
    ranked["gate_threshold"] = threshold
    return ranked, policy


def write_one_date(
    ranked_range: pd.DataFrame,
    date: str,
    out_dir: Path,
    save_ranked: bool,
    policy: dict[str, object],
    gate_source: str,
) -> dict[str, object]:
    """Write the same artifacts produced by generate_signals.py for one date."""
    ranked = ranked_range[ranked_range["trade_date"].eq(date)].copy()
    ranked = ranked.sort_values(
        [
            "production_signal",
            f"{policy['family']}_rank",
            "family_count",
            "ensemble_pred",
            "ts_code",
        ],
        ascending=[False, True, False, False, True],
    )
    signals = ranked[ranked.output_signal].copy()
    production_signals = ranked[ranked.production_signal].copy()

    signal_path = out_dir / f"signals_{date}.csv"
    production_path = out_dir / f"production_signals_{date}.csv"
    signals.to_csv(signal_path, index=False)
    signals.to_csv(out_dir / "signals_latest.csv", index=False)
    production_signals.to_csv(production_path, index=False)
    production_signals.to_csv(out_dir / "production_signals_latest.csv", index=False)
    if save_ranked:
        ranked.to_parquet(out_dir / f"ranked_{date}.parquet", index=False)

    summary = {
        "trade_date": date,
        "universe_rows": len(ranked),
        "signal_rows": len(signals),
        "ensemble_signal_rows": int(ranked.ensemble_signal.sum()),
        "production_policy": str(policy["production_tag"]),
        "f220_top1_rows": int((ranked.F220_rank == 1).sum()),
        "gate_pass_rows": int(ranked.production_gate_pass.sum()),
        "production_signal_rows": int(ranked.production_signal.sum()),
        "consensus_3_rows": int((ranked.legacy_signal_tag == "CONSENSUS_3").sum()),
        "consensus_2_rows": int((ranked.legacy_signal_tag == "CONSENSUS_2").sum()),
        "family_only_rows": int((ranked.legacy_signal_tag == "FAMILY_ONLY").sum()),
        "gate_source_nonnull_rows": int(ranked[gate_source].notna().sum()),
        "gate_threshold": float(policy["gate_threshold"]),
    }
    (out_dir / f"summary_{date}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate transparent F150/C185/F220 ensemble signals for a date range."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Feature parquet, usually processed/train_v5b/train_v5b.parquet",
    )
    parser.add_argument("--start-date", required=True, help="First feature date (YYYYMMDD).")
    parser.add_argument("--end-date", required=True, help="Last feature date (YYYYMMDD).")
    parser.add_argument("--out-dir", type=Path, default=HERE / "signals")
    parser.add_argument(
        "--save-ranked",
        action="store_true",
        help="Save all stocks with scores and ranks for every processed date.",
    )
    args = parser.parse_args()

    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    family_features = {
        name: read_lines(HERE / "models" / name / "features.txt")
        for name in cfg["families"]
    }
    policy = cfg["signal_policy"]
    gate_source = str(policy["gate_source_feature"])
    all_features = sorted(set().union(*family_features.values(), {gate_source}))

    print(f"[load] {args.input} {args.start_date} ~ {args.end_date}")
    frame = read_input_range(
        args.input,
        args.start_date,
        args.end_date,
        all_features,
    )
    dates = sorted(frame["trade_date"].astype(str).unique().tolist())
    print(
        f"[rows] {len(frame)} dates={len(dates)} "
        f"date_min={dates[0]} date_max={dates[-1]}"
    )

    ranked, policy = build_ranked(frame, cfg, family_features)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for index, date in enumerate(dates, start=1):
        summary = write_one_date(
            ranked_range=ranked,
            date=date,
            out_dir=args.out_dir,
            save_ranked=args.save_ranked,
            policy=policy,
            gate_source=gate_source,
        )
        summaries.append(summary)
        print(
            f"[{index}/{len(dates)}] {date} "
            f"universe={summary['universe_rows']} signals={summary['signal_rows']} "
            f"production={summary['production_signal_rows']}"
        )

    summary_path = args.out_dir / f"range_generation_summary_{dates[0]}_{dates[-1]}.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf_8_sig")
    print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()
