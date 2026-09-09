#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl


HERE = Path(__file__).resolve().parent
SEEDS = (20260801, 20260811, 20260821)
EXPERTS = ("top1500_lr20_l2_12", "top1600_leaf600")
DERIVED_RISK_FEATURES = {
    "sig_vote_mean", "sig_vote_min", "sig_rank_mean", "sig_rank_max",
    "expert_top3_overlap",
    *{
        f"{expert}_{suffix}"
        for expert in EXPERTS
        for suffix in ("top3_score_mean", "score_gap_3_10", "seed_dispersion")
    },
}


def lines(path: Path) -> list[str]:
    return [x for x in path.read_text(encoding="utf-8").splitlines() if x]


@dataclass
class SignalContext:
    feature_map: dict[str, list[str]]
    risk_features: list[str]
    models: dict[tuple[str, int], lgb.Booster]
    risk_model: object


def load_context(input_path: Path) -> SignalContext:
    schema = pl.read_parquet_schema(input_path)
    feature_map = {expert: lines(HERE / "features" / f"{expert}.txt") for expert in EXPERTS}
    risk_features = lines(HERE / "risk/day_risk_features.txt")
    required = list(dict.fromkeys(
        [x for values in feature_map.values() for x in values]
        + [x for x in risk_features if x not in DERIVED_RISK_FEATURES]
    ))
    missing = [x for x in required if x not in schema]
    if missing:
        raise KeyError(f"Prediction data lacks {len(missing)} model features: {missing[:20]}")
    models = {
        (expert, seed): lgb.Booster(
            model_file=str(HERE / "models" / expert / f"seed_{seed}" / "model.txt")
        )
        for expert in EXPERTS for seed in SEEDS
    }
    return SignalContext(
        feature_map=feature_map,
        risk_features=risk_features,
        models=models,
        risk_model=joblib.load(HERE / "risk/day_risk_logit.joblib"),
    )


def generate_for_date(
    input_path: Path, date: str, context: SignalContext, history: list[float]
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = list(dict.fromkeys(
        [x for values in context.feature_map.values() for x in values]
        + [x for x in context.risk_features if x not in DERIVED_RISK_FEATURES]
    ))
    frame = (
        pl.scan_parquet(input_path)
        .filter(pl.col("trade_date").cast(pl.Utf8) == date)
        .select("ts_code", pl.col("trade_date").cast(pl.Utf8), *required)
        .collect().to_pandas()
    )
    if frame.empty:
        raise ValueError(f"No rows for trade_date={date}")

    rank_cols = []
    for expert in EXPERTS:
        X = frame[context.feature_map[expert]].replace([np.inf, -np.inf], np.nan).astype(np.float32)
        expert_ranks = []
        for seed in SEEDS:
            model = context.models[(expert, seed)]
            pred = f"pred__{expert}__{seed}"
            rank = f"rank__{expert}__{seed}"
            frame[pred] = model.predict(X, num_iteration=model.best_iteration)
            frame[rank] = frame[pred].rank(ascending=False, method="average")
            rank_cols.append(rank)
            expert_ranks.append(rank)
        frame[f"avg_rank__{expert}"] = frame[expert_ranks].mean(axis=1)
    frame["pooled_avg_rank"] = frame[rank_cols].mean(axis=1)
    frame["vote3"] = sum((frame[col] <= 3).astype("int8") for col in rank_cols)
    ranked = frame.sort_values(["vote3", "pooled_avg_rank", "ts_code"], ascending=[False, True, True])
    top3 = ranked.head(3).copy()

    row = {
        "sig_vote_mean": top3.vote3.mean(), "sig_vote_min": top3.vote3.min(),
        "sig_rank_mean": top3.pooled_avg_rank.mean(), "sig_rank_max": top3.pooled_avg_rank.max(),
    }
    expert_top1 = {}
    expert_top3 = {}
    for expert in EXPERTS:
        ordered = frame.sort_values([f"avg_rank__{expert}", "ts_code"])
        expert_top1[expert] = set(ordered.head(1).ts_code)
        expert_top3[expert] = set(ordered.head(3).ts_code)
        values = [f"pred__{expert}__{seed}" for seed in SEEDS]
        score = frame[values].mean(axis=1).sort_values(ascending=False)
        row[f"{expert}_top3_score_mean"] = score.head(3).mean()
        row[f"{expert}_score_gap_3_10"] = score.iloc[2] - score.iloc[9]
        row[f"{expert}_seed_dispersion"] = top3[values].std(axis=1).mean()
    row["expert_top3_overlap"] = len(expert_top3[EXPERTS[0]] & expert_top3[EXPERTS[1]])
    for feature in context.risk_features:
        if feature not in row:
            if feature not in frame:
                raise KeyError(f"Prediction data lacks day-risk feature: {feature}")
            row[feature] = pd.to_numeric(frame[feature], errors="coerce").mean()
    risk = float(context.risk_model.predict_proba(pd.DataFrame([row])[context.risk_features])[:, 1][0])
    threshold = float(np.quantile(history, 0.50))
    accepted = risk <= threshold
    strict = expert_top1[EXPERTS[0]] & expert_top1[EXPERTS[1]]
    top3["signal_tag"] = np.where(top3.ts_code.isin(strict), "HIGH_CONF_STRICT_TOP1", "STANDARD_TOP3")
    top3["day_gate"] = "KEEP50_ACCEPT" if accepted else "KEEP50_REJECT"
    top3["priority"] = np.where(
        accepted, np.where(top3.ts_code.isin(strict), 1, 2),
        np.where(top3.ts_code.isin(strict), 3, 4),
    )
    top3["priority_name"] = np.select(
        [
            accepted & top3.ts_code.isin(strict),
            accepted & ~top3.ts_code.isin(strict),
            ~accepted & top3.ts_code.isin(strict),
        ],
        ["P1_HIGH_CONF_ACCEPT", "P2_STANDARD_ACCEPT", "WATCH_HIGH_CONF_DAY_REJECT"],
        default="WATCH_STANDARD_DAY_REJECT",
    )
    top3["bad_day_risk"] = risk
    top3["bad_day_threshold"] = threshold
    columns = [
        "trade_date", "ts_code", "priority", "priority_name", "signal_tag", "day_gate",
        "vote3", "pooled_avg_rank", "bad_day_risk", "bad_day_threshold",
    ]
    diagnostic = {
        "trade_date": date, "rows": len(frame), "accepted": bool(accepted),
        "bad_day_risk": risk, "bad_day_threshold": threshold,
        "strict_top1_count": int(top3.ts_code.isin(strict).sum()),
        "top3_vote_mean": float(top3.vote3.mean()),
        "top3_vote_min": int(top3.vote3.min()),
    }
    return top3[columns], diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen H3M55D15 consensus signals")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--trade-date", help="YYYYMMDD; defaults to latest input date")
    parser.add_argument("--out-dir", type=Path, default=HERE / "signals")
    args = parser.parse_args()
    date = args.trade_date or str(
        pl.scan_parquet(args.input).select(pl.col("trade_date").max()).collect().item()
    )
    context = load_context(args.input)
    history = pd.read_csv(HERE / "risk/risk_score_history.csv")["bad_risk_logit"].dropna().astype(float).tolist()
    top3, diagnostic = generate_for_date(args.input, date, context, history)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"signals_{date}.csv"
    top3.to_csv(output, index=False)
    top3.to_csv(args.out_dir / "signals_latest.csv", index=False)
    pd.DataFrame([diagnostic]).to_csv(args.out_dir / f"diagnostic_{date}.csv", index=False)
    print(top3.to_string(index=False))
    print(f"[SAVE] {output}")


if __name__ == "__main__":
    main()
