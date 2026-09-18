#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen logistic day-risk artifacts")
    parser.add_argument(
        "--test-work", type=Path,
        default=ROOT / "complete_features_h3m55d15/work/day_risk_filter",
    )
    parser.add_argument(
        "--oos-work", type=Path,
        default=ROOT / "complete_features_h3m55d15/work/day_risk_locked_oos",
    )
    parser.add_argument("--out-dir", type=Path, default=HERE / "risk")
    args = parser.parse_args()

    feature_source = args.test_work / "regime_features.txt"
    dataset_source = args.test_work / "date_regime_dataset.parquet"
    decision_source = args.oos_work / "locked_oos_day_risk_decisions.csv"
    for path in (feature_source, dataset_source, decision_source):
        if not path.exists():
            raise FileNotFoundError(path)

    features = [x for x in feature_source.read_text(encoding="utf-8").splitlines() if x]
    train = pd.read_parquet(dataset_source).sort_values("trade_date")
    missing = [feature for feature in features if feature not in train.columns]
    if missing:
        raise KeyError(f"Day-risk dataset lacks features: {missing}")

    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(
            C=0.10, class_weight="balanced", max_iter=3000, random_state=20260719
        ),
    )
    model.fit(train[features], train["bad_day"])
    train_scores = model.predict_proba(train[features])[:, 1].tolist()

    decisions = pd.read_csv(decision_source)
    required = {"trade_date", "bad_risk_logit"}
    if not required.issubset(decisions.columns):
        raise KeyError(f"OOS decisions lack columns: {sorted(required - set(decisions.columns))}")
    oos_scores = (
        decisions[["trade_date", "bad_risk_logit"]]
        .drop_duplicates("trade_date")
        .sort_values("trade_date")["bad_risk_logit"]
        .dropna().astype(float).tolist()
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_dir / "day_risk_logit.joblib")
    (args.out_dir / "day_risk_features.txt").write_text(
        "\n".join(features) + "\n", encoding="utf-8"
    )
    pd.DataFrame({"bad_risk_logit": train_scores + oos_scores}).to_csv(
        args.out_dir / "risk_score_history.csv", index=False
    )
    dated = pd.concat([
        pd.DataFrame({'trade_date': train.trade_date.astype(str), 'bad_risk_logit': train_scores}),
        decisions[['trade_date', 'bad_risk_logit']].assign(trade_date=lambda x: x.trade_date.astype(str)),
    ], ignore_index=True)
    if dated.groupby('trade_date').bad_risk_logit.nunique().gt(1).any():
        raise ValueError('Inconsistent historical risk scores')
    dated.drop_duplicates('trade_date').sort_values('trade_date').to_csv(
        args.out_dir / 'dated_history.csv', index=False)
    frozen = ROOT / 'complete_features_h3m55d15/raw_loss_filter/frozen_oos'
    shutil.copy2(frozen / 'market_loss_logit.joblib', args.out_dir / 'market_loss_logit.joblib')
    shutil.copy2(frozen / 'frozen_protocol.json', args.out_dir / 'market_protocol.json')
    print(f"[features] {len(features)}")
    print(f"[history] train={len(train_scores)} oos={len(oos_scores)} total={len(train_scores)+len(oos_scores)}")
    print(f"[save] {args.out_dir}")


if __name__ == "__main__":
    main()
