#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    market_features: list[str]
    market_model: object
    market_threshold: float
    earliest_date: str


def load_context(input_path: Path) -> SignalContext:
    schema = pl.read_parquet_schema(input_path)
    feature_map = {expert: lines(HERE / "features" / f"{expert}.txt") for expert in EXPERTS}
    risk_features = lines(HERE / "risk/day_risk_features.txt")
    market_protocol = json.loads((HERE / 'risk/market_protocol.json').read_text())
    market_features = market_protocol['features']
    required = list(dict.fromkeys(
        [x for values in feature_map.values() for x in values]
        + [x for x in risk_features if x not in DERIVED_RISK_FEATURES]
        + market_features
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
    for (expert, seed), model in models.items():
        if model.feature_name() != feature_map[expert]:
            raise ValueError(f'Model feature order mismatch: {expert}/{seed}')
    return SignalContext(
        feature_map=feature_map,
        risk_features=risk_features,
        models=models,
        risk_model=joblib.load(HERE / "risk/day_risk_logit.joblib"),
        market_features=market_features,
        market_model=joblib.load(HERE / 'risk/market_loss_logit.joblib'),
        market_threshold=float(market_protocol['threshold']),
        earliest_date=str(market_protocol['development_last_date']),
    )


def generate_for_date(
    input_path: Path, date: str, context: SignalContext, history: list[float],
    *, historical_scoring: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = list(dict.fromkeys(
        [x for values in context.feature_map.values() for x in values]
        + [x for x in context.risk_features if x not in DERIVED_RISK_FEATURES]
        + context.market_features
    ))
    frame = (
        pl.scan_parquet(input_path)
        .filter(pl.col("trade_date").cast(pl.Utf8) == date)
        .select("ts_code", pl.col("trade_date").cast(pl.Utf8), *required)
        .collect().to_pandas()
    )
    if date <= context.earliest_date and not historical_scoring:
        raise ValueError('Date is not after the frozen development period; not a historical replay model')
    if len(frame) < 10 or frame.ts_code.isna().any() or frame.ts_code.duplicated().any():
        raise ValueError(f'Need at least 10 unique non-null stocks for {date}')
    if not history or not np.isfinite(history).all():
        raise ValueError('Missing/nonfinite earlier-gate calibration history')

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
    frame['vote1'] = sum((frame[col] <= 1).astype('int8') for col in rank_cols)
    pooled_top1 = frame.sort_values(['vote1', 'pooled_avg_rank', 'ts_code'], ascending=[False, True, True]).iloc[0].ts_code
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
    for feature in set(context.risk_features + context.market_features):
        if feature not in row:
            if feature not in frame:
                raise KeyError(f"Prediction data lacks day-risk feature: {feature}")
            row[feature] = pd.to_numeric(frame[feature], errors="coerce").mean()
    risk = float(context.risk_model.predict_proba(pd.DataFrame([row])[context.risk_features])[:, 1][0])
    threshold50, threshold60 = np.quantile(history, [.5, .6])
    market_risk = float(context.market_model.predict_proba(pd.DataFrame([row])[context.market_features])[0, 1])
    top3 = assign_priorities(top3, pooled_top1, risk <= threshold50, risk <= threshold60,
                             market_risk <= context.market_threshold)
    top3['earlier_risk'] = risk
    top3['earlier_threshold50'] = threshold50
    top3['earlier_threshold60'] = threshold60
    top3['market_risk'] = market_risk
    top3['market_threshold'] = context.market_threshold
    columns = ['trade_date', 'ts_code', 'priority', 'priority_name', 'signal_tag',
               'pooled_top1', 'earlier_keep50', 'earlier_keep60', 'current_selection',
               'vote1', 'vote3', 'pooled_avg_rank', 'earlier_risk', 'earlier_threshold50',
               'earlier_threshold60', 'market_risk', 'market_threshold']
    diagnostic = {
        "trade_date": date, "rows": len(frame), 'p1': int(top3.priority.eq(1).sum()),
        'p2': int(top3.priority.eq(2).sum()), 'bad_day_risk': risk,
        'market_risk': market_risk, 'threshold50': threshold50, 'threshold60': threshold60,
        "top3_vote_mean": float(top3.vote3.mean()),
        "top3_vote_min": int(top3.vote3.min()),
    }
    return top3[columns], diagnostic


def assign_priorities(top3, pooled_top1, keep50, keep60, market_accept):
    if keep50 and not keep60:
        raise ValueError('keep50 must be nested in keep60')
    top3 = top3.copy()
    top3['pooled_top1'] = top3.ts_code.eq(pooled_top1)
    top3['earlier_keep50'] = bool(keep50)
    top3['earlier_keep60'] = bool(keep60)
    top3['current_selection'] = (~top3.pooled_top1) & bool(market_accept)
    p1 = top3.current_selection & bool(keep50)
    p2 = (top3.current_selection | bool(keep60)) & ~p1
    top3['priority'] = np.select([p1, p2], [1, 2], default=0)
    top3['priority_name'] = np.select([p1, p2], ['P1', 'P2'], default='NO_SIGNAL')
    top3['signal_tag'] = np.select([p1, p2 & top3.current_selection & bool(keep60),
        p2 & top3.current_selection, p2], ['KEEP50_AND_CURRENT', 'KEEP60_AND_CURRENT',
        'CURRENT_ONLY', 'KEEP60_ONLY'], default='REJECTED')
    return top3


def read_history(path):
    seed = pd.read_csv(HERE / 'risk/dated_history.csv', dtype={'trade_date': str})
    if path.exists():
        seed = pd.concat([seed, pd.read_csv(path, dtype={'trade_date': str})], ignore_index=True)
    if seed.groupby('trade_date').bad_risk_logit.nunique().gt(1).any():
        raise ValueError('Conflicting dated risk scores; use a clean history for a different model/data version')
    return seed.drop_duplicates('trade_date').sort_values('trade_date')


def record_history(history, date, risk, path):
    existing = history[history.trade_date.eq(date)]
    if len(existing) and not np.allclose(existing.bad_risk_logit, risk, rtol=1e-6, atol=1e-8):
        raise ValueError(f'Risk differs from saved history on {date}; refusing inconsistent replay')
    if existing.empty:
        history = pd.concat([history, pd.DataFrame([dict(trade_date=date, bad_risk_logit=risk)])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    history.sort_values('trade_date').to_csv(tmp, index=False)
    tmp.replace(path)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen H3M55D15 consensus signals")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--trade-date", help="YYYYMMDD; defaults to latest input date")
    parser.add_argument("--out-dir", type=Path, default=HERE / "signals")
    parser.add_argument('--history-file', type=Path, default=HERE / 'state/risk_history.csv')
    args = parser.parse_args()
    date = args.trade_date or str(
        pl.scan_parquet(args.input).select(pl.col("trade_date").max()).collect().item()
    )
    context = load_context(args.input)
    history = read_history(args.history_file)
    top3, diagnostic = generate_for_date(args.input, date, context,
        history.loc[history.trade_date.lt(date), 'bad_risk_logit'].tolist())
    record_history(history, date, diagnostic['bad_day_risk'], args.history_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"signals_{date}.csv"
    top3[top3.priority.gt(0)].to_csv(output, index=False)
    top3[top3.priority.gt(0)].to_csv(args.out_dir / "signals_latest.csv", index=False)
    top3.to_csv(args.out_dir / f'candidates_{date}.csv', index=False)
    pd.DataFrame([diagnostic]).to_csv(args.out_dir / f"diagnostic_{date}.csv", index=False)
    print(top3.to_string(index=False))
    print(f"[SAVE] {output}")


if __name__ == "__main__":
    main()
