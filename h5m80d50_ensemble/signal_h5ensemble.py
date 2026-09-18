from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl


HERE = Path(__file__).resolve().parent


def read_lines(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def normalize_date(value: object) -> str:
    return str(value).replace("-", "")[:8]


def load_day(path: Path, requested_date: str | None, columns: list[str]) -> tuple[pd.DataFrame, str]:
    schema = pl.read_parquet_schema(path)
    required = ["trade_date", "ts_code", *columns]
    missing = sorted(set(required) - set(schema.names()))
    if missing:
        raise KeyError(f"Input lacks {len(missing)} required columns: {missing[:20]}")
    lf = pl.scan_parquet(path).select(required).with_columns(
        pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", "").alias("trade_date"),
        pl.col("ts_code").cast(pl.Utf8),
    )
    date = normalize_date(requested_date) if requested_date else str(lf.select(pl.col("trade_date").max()).collect().item())
    frame = lf.filter(pl.col("trade_date") == date).collect().to_pandas()
    if frame.empty:
        raise ValueError(f"No rows for trade_date={date}")
    if frame[["trade_date", "ts_code"]].duplicated().any():
        raise ValueError(f"Duplicate trade_date/ts_code rows for {date}")
    return frame, date


def predict_family(frame: pd.DataFrame, family: str, seeds: list[int]) -> pd.DataFrame:
    family_dir = HERE / "models" / family
    features = read_lines(family_dir / "features.txt")
    x = frame[features].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    predictions = []
    for seed in seeds:
        model_path = family_dir / f"seed{seed}" / "model.txt"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = lgb.Booster(model_file=str(model_path))
        if model.feature_name() != features:
            raise ValueError(f"{family}/seed{seed}: model feature schema differs from features.txt")
        predictions.append(model.predict(x, num_iteration=model.current_iteration()))
    matrix = np.column_stack(predictions)
    return pd.DataFrame({
        f"{family}_pred": matrix.mean(axis=1),
        f"{family}_seed_std": matrix.std(axis=1),
        f"{family}_seed_min": matrix.min(axis=1),
        f"{family}_seed_max": matrix.max(axis=1),
    }, index=frame.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate transparent F150/C185/F220 ensemble signals.")
    parser.add_argument("--input", type=Path, required=True, help="Feature parquet, usually processed/train_v5b/train_v5b.parquet")
    parser.add_argument("--trade-date", help="Feature date T (YYYYMMDD); default is latest date in input")
    parser.add_argument("--out-dir", type=Path, default=HERE / "signals")
    parser.add_argument("--save-ranked", action="store_true", help="Save all stocks with scores and ranks")
    args = parser.parse_args()

    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    family_features = {name: read_lines(HERE / "models" / name / "features.txt") for name in cfg["families"]}
    policy = cfg["signal_policy"]
    gate_source = str(policy["gate_source_feature"])
    all_features = sorted(set().union(*family_features.values(), {gate_source}))
    frame, date = load_day(args.input, args.trade_date, all_features)

    ranked = frame[["trade_date", "ts_code"]].copy()
    for family in cfg["families"]:
        pred = predict_family(frame, family, cfg["seeds"])
        ranked = pd.concat([ranked, pred], axis=1)
        ranked[f"{family}_rank"] = ranked[f"{family}_pred"].rank(method="first", ascending=False).astype(int)
        ranked[f"{family}_selected"] = ranked[f"{family}_rank"] <= int(cfg["families"][family]["top_k"])

    selected_cols = [f"{x}_selected" for x in cfg["families"]]
    ranked["family_count"] = ranked[selected_cols].sum(axis=1).astype(int)
    ranked["ensemble_pred"] = ranked[[f"{x}_pred" for x in cfg["families"]]].mean(axis=1)
    ranked["ensemble_seed_std"] = ranked[[f"{x}_seed_std" for x in cfg["families"]]].mean(axis=1)
    ranked["legacy_signal_tag"] = np.select(
        [ranked.family_count == 3, ranked.family_count == 2, ranked.family_count == 1],
        ["CONSENSUS_3", "CONSENSUS_2", "FAMILY_ONLY"], default="",
    )
    ranked["family_tags"] = ranked.apply(
        lambda row: "|".join(name for name in cfg["families"] if row[f"{name}_selected"]), axis=1
    )
    valid_gate_values = frame[gate_source].replace([np.inf, -np.inf], np.nan)
    valid_count = int(valid_gate_values.notna().sum())
    ranked[gate_source] = valid_gate_values
    ranked[str(policy["gate_feature"])] = (
        valid_gate_values.rank(method="average", ascending=True) / valid_count
        if valid_count else np.nan
    )
    threshold = float(policy["gate_threshold"])
    ranked["production_gate_pass"] = ranked[str(policy["gate_feature"])].le(threshold).fillna(False)
    ranked["production_signal"] = (
        ranked[f"{policy['family']}_rank"].le(int(policy["top_k"]))
        & ranked["production_gate_pass"]
    )
    ranked["ensemble_signal"] = ranked["family_count"].ge(1)
    ranked["output_signal"] = ranked["production_signal"] | ranked["ensemble_signal"]
    ranked["signal_tag"] = ranked.apply(
        lambda row: "|".join(
            tag for tag in [
                str(policy["production_tag"]) if row["production_signal"] else "",
                str(row["legacy_signal_tag"]) if row["ensemble_signal"] else "",
            ] if tag
        ),
        axis=1,
    )
    ranked["gate_threshold"] = threshold
    ranked = ranked.sort_values(
        ["production_signal", f"{policy['family']}_rank", "family_count", "ensemble_pred", "ts_code"],
        ascending=[False, True, False, False, True],
    )
    signals = ranked[ranked.output_signal].copy()
    production_signals = ranked[ranked.production_signal].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    signal_path = args.out_dir / f"signals_{date}.csv"
    signals.to_csv(signal_path, index=False)
    signals.to_csv(args.out_dir / "signals_latest.csv", index=False)
    production_path = args.out_dir / f"production_signals_{date}.csv"
    production_signals.to_csv(production_path, index=False)
    production_signals.to_csv(args.out_dir / "production_signals_latest.csv", index=False)
    if args.save_ranked:
        ranked.to_parquet(args.out_dir / f"ranked_{date}.parquet", index=False)

    summary = {
        "trade_date": date, "universe_rows": len(ranked), "signal_rows": len(signals),
        "ensemble_signal_rows": int(ranked.ensemble_signal.sum()),
        "production_policy": str(policy["production_tag"]),
        "f220_top1_rows": int((ranked.F220_rank == 1).sum()),
        "gate_pass_rows": int(ranked.production_gate_pass.sum()),
        "production_signal_rows": int(ranked.production_signal.sum()),
        "consensus_3_rows": int((ranked.legacy_signal_tag == "CONSENSUS_3").sum()),
        "consensus_2_rows": int((ranked.legacy_signal_tag == "CONSENSUS_2").sum()),
        "family_only_rows": int((ranked.legacy_signal_tag == "FAMILY_ONLY").sum()),
        "gate_source_nonnull_rows": valid_count,
        "gate_threshold": threshold,
    }
    (args.out_dir / f"summary_{date}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[SUMMARY] " + json.dumps(summary))
    if production_signals.empty:
        print("[SIGNALS] F220 Top1 did not pass the late/early gate.")
    else:
        print(production_signals[[
            "trade_date", "ts_code", "signal_tag", "F220_pred", "F220_rank",
            str(policy["gate_feature"]), "gate_threshold", "production_gate_pass",
        ]].to_string(index=False))
    print(f"[SAVE] {signal_path}")
    print(f"[SAVE] {production_path}")


if __name__ == "__main__":
    main()
