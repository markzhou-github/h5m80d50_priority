#!/usr/bin/env python3
"""Generate frozen H5 competing-direction Top3 signals for one feature date."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl


HERE = Path(__file__).resolve().parent
EPS = 1e-6


def lines(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def predict(frame: pd.DataFrame, model_root: Path, features: list[str], seeds: list[int]) -> np.ndarray:
    x = frame[features].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    values = []
    for seed in seeds:
        model = lgb.Booster(model_file=str(model_root / f"seed{seed}" / "model.txt"))
        if model.feature_name() != features:
            raise ValueError(f"Feature schema mismatch: {model_root.name}/seed{seed}")
        values.append(model.predict(x, num_iteration=model.current_iteration()))
    return np.column_stack(values).mean(axis=1)


def platt(values: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    clipped = np.clip(values, EPS, 1 - EPS)
    logits = np.log(clipped / (1 - clipped))
    z = parameters["coefficient"] * logits + parameters["intercept"]
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--trade-date")
    parser.add_argument("--out-dir", type=Path, default=HERE / "signals")
    parser.add_argument("--save-ranked", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    calibration = json.loads((HERE / "calibration.json").read_text(encoding="utf-8"))

    feature_lists: dict[str, list[str]] = {}
    for family in cfg["stage1_families"]:
        feature_lists[f"stage1_{family}"] = lines(HERE / "models/stage1" / family / "features.txt")
    for head in ["up", "down", "intensity", "direction"]:
        feature_lists[f"competing_{head}"] = lines(HERE / "models/competing" / head / "features.txt")
    required = sorted(set().union(*feature_lists.values()))

    scan = pl.scan_parquet(args.input).with_columns(
        pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", "").alias("trade_date"),
        pl.col("ts_code").cast(pl.Utf8),
    )
    date = str(args.trade_date).replace("-", "") if args.trade_date else scan.select(pl.col("trade_date").max()).collect().item()
    schema = scan.collect_schema()
    missing = sorted(set(required + ["trade_date", "ts_code"]) - set(schema.names()))
    if missing:
        raise KeyError(f"Input lacks {len(missing)} required columns: {missing[:30]}")
    frame = scan.select(["trade_date", "ts_code", *required]).filter(pl.col("trade_date") == date).collect().to_pandas()
    if frame.empty:
        raise ValueError(f"No rows for trade_date={date}")

    ranked = frame[["trade_date", "ts_code"]].copy()
    for family, spec in cfg["stage1_families"].items():
        ranked[f"{family}_pred"] = predict(
            frame, HERE / "models/stage1" / family,
            feature_lists[f"stage1_{family}"], cfg["stage1_seeds"],
        )
        ranked[f"{family}_rank"] = ranked[f"{family}_pred"].rank(method="first", ascending=False).astype(int)
        ranked[f"{family}_selected"] = ranked[f"{family}_rank"] <= int(spec["top_k"])
    selected_columns = [f"{family}_selected" for family in cfg["stage1_families"]]
    ranked["family_count"] = ranked[selected_columns].sum(axis=1).astype(int)
    ranked["stage1_candidate"] = ranked.family_count >= 1
    ranked["stage1_ensemble_pred"] = ranked[[f"{x}_pred" for x in cfg["stage1_families"]]].mean(axis=1)

    for head in ["up", "down", "intensity", "direction"]:
        ranked[f"p_{head}"] = predict(
            frame, HERE / "models/competing" / head,
            feature_lists[f"competing_{head}"], cfg["competing_seeds"],
        )
    ranked["p_up_calibrated"] = platt(ranked.p_up.to_numpy(), calibration["up"])
    ranked["p_down_calibrated"] = platt(ranked.p_down.to_numpy(), calibration["down"])
    ranked["direction_ratio"] = ranked.p_up_calibrated / (
        ranked.p_up_calibrated + ranked.p_down_calibrated + EPS
    )
    ranked["conditional_score"] = ranked.p_intensity * ranked.p_direction

    candidates = ranked[ranked.stage1_candidate].copy().sort_values(
        ["conditional_score", "stage1_ensemble_pred", "ts_code"], ascending=[False, False, True]
    )
    candidates["conditional_rank"] = np.arange(1, len(candidates) + 1)
    policy = cfg["policy"]
    candidates["direction_gate_pass"] = candidates.direction_ratio >= float(policy["direction_ratio_threshold"])
    candidates["production_signal"] = (
        candidates.conditional_rank <= int(policy["top_k"])
    ) & candidates.direction_gate_pass
    candidates["signal_tag"] = np.where(
        candidates.production_signal, "H5_COMPETING_DIRECTION_TOP3", "WATCHLIST"
    )
    signals = candidates[candidates.production_signal].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    signals.to_csv(args.out_dir / f"signals_{date}.csv", index=False)
    signals.to_csv(args.out_dir / "signals_latest.csv", index=False)
    candidates.to_csv(args.out_dir / f"watchlist_{date}.csv", index=False)
    if args.save_ranked:
        ranked.to_parquet(args.out_dir / f"ranked_{date}.parquet", index=False)
    summary = {
        "trade_date": date, "universe_rows": len(ranked),
        "stage1_candidates": len(candidates), "signals": len(signals),
        "policy": policy,
    }
    (args.out_dir / f"summary_{date}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[SUMMARY] " + json.dumps(summary))
    print(signals[["trade_date", "ts_code", "signal_tag", "conditional_rank", "conditional_score", "direction_ratio"]].to_string(index=False) if len(signals) else "[SIGNALS] No signals")


if __name__ == "__main__":
    main()
