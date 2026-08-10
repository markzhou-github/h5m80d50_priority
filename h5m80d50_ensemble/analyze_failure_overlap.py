from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
import polars as pl


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_ensemble(model_root: Path, source_name: str, seeds: list[int], split: str) -> pd.DataFrame:
    parts = []
    for seed in seeds:
        path = model_root / source_name / f"seed{seed}" / f"pred_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        part = (
            pl.read_parquet(path, columns=["trade_date", "ts_code", "target", "pred"])
            .with_columns(
                pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", ""),
                pl.col("ts_code").cast(pl.Utf8),
            )
            .to_pandas()
        )
        part["seed"] = seed
        parts.append(part)
    return (
        pd.concat(parts, ignore_index=True)
        .groupby(["trade_date", "ts_code"], as_index=False)
        .agg(target=("target", "first"), pred=("pred", "mean"))
    )


def topk(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    return (
        frame.sort_values(["trade_date", "pred", "ts_code"], ascending=[True, False, True])
        .groupby("trade_date", sort=False)
        .head(k)
        .copy()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure whether ensemble false positives cluster on the same dates.")
    parser.add_argument("--model-root", type=Path, default=ROOT / "retrain_robust_v5b/work/final_oos/models")
    parser.add_argument(
        "--completion-report", type=Path,
        default=ROOT / "retrain_robust_v5b/work/final_oos/reports/final_oos_ensemble_selected_trades.csv",
        help="Frozen report used only to identify OOS dates whose forward returns are complete.",
    )
    parser.add_argument("--splits", nargs="+", default=["valid", "oos"])
    parser.add_argument("--out-dir", type=Path, default=HERE / "benchmark_reports/failure_overlap")
    args = parser.parse_args()

    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    completion_report = pd.read_csv(args.completion_report, dtype={"trade_date": str})
    completed_oos_dates = set(completion_report.loc[completion_report.ret_raw.notna(), "trade_date"].astype(str))
    summary_rows, pair_rows, daily_rows, signal_rows = [], [], [], []

    for split in args.splits:
        selected_by_family: dict[str, pd.DataFrame] = {}
        for family, spec in cfg["families"].items():
            frame = load_ensemble(args.model_root, spec["source_name"], cfg["seeds"], split)
            if split == "oos":
                frame = frame[frame.trade_date.isin(completed_oos_dates)].copy()
            selected = topk(frame, int(spec["top_k"]))
            selected["family"] = family
            selected["is_fp"] = selected.target.eq(0)
            selected_by_family[family] = selected
            signal_rows.append(selected.assign(split=split))

            day = selected.groupby("trade_date", as_index=False).agg(
                picks=("ts_code", "size"), tp=("target", "sum"), fp=("is_fp", "sum")
            )
            day["any_fp"] = day.fp.gt(0)
            day["all_fp"] = day.tp.eq(0)
            day["family"] = family
            day["split"] = split
            daily_rows.append(day)
            summary_rows.append({
                "split": split, "family": family, "top_k": int(spec["top_k"]),
                "dates": len(day), "signals": len(selected), "precision": selected.target.mean(),
                "any_fp_days": int(day.any_fp.sum()), "any_fp_day_rate": day.any_fp.mean(),
                "all_fp_days": int(day.all_fp.sum()), "all_fp_day_rate": day.all_fp.mean(),
            })

        for left, right in itertools.combinations(selected_by_family, 2):
            a, b = selected_by_family[left], selected_by_family[right]
            ad = a.groupby("trade_date").target.agg(lambda x: bool((x == 0).any()))
            bd = b.groupby("trade_date").target.agg(lambda x: bool((x == 0).any()))
            aa = a.groupby("trade_date").target.agg(lambda x: bool((x == 0).all()))
            ba = b.groupby("trade_date").target.agg(lambda x: bool((x == 0).all()))
            any_a, any_b = set(ad[ad].index), set(bd[bd].index)
            all_a, all_b = set(aa[aa].index), set(ba[ba].index)
            fp_a = set(map(tuple, a.loc[a.is_fp, ["trade_date", "ts_code"]].to_numpy()))
            fp_b = set(map(tuple, b.loc[b.is_fp, ["trade_date", "ts_code"]].to_numpy()))
            pair_rows.append({
                "split": split, "left": left, "right": right,
                "left_any_fp_days": len(any_a), "right_any_fp_days": len(any_b),
                "shared_any_fp_days": len(any_a & any_b),
                "shared_any_fp_over_left": len(any_a & any_b) / len(any_a) if any_a else 0,
                "shared_any_fp_over_right": len(any_a & any_b) / len(any_b) if any_b else 0,
                "shared_all_fp_days": len(all_a & all_b),
                "left_fp_signals": len(fp_a), "right_fp_signals": len(fp_b),
                "same_stock_fp_signals": len(fp_a & fp_b),
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    pairs = pd.DataFrame(pair_rows)
    daily = pd.concat(daily_rows, ignore_index=True)
    signals = pd.concat(signal_rows, ignore_index=True)
    summary.to_csv(args.out_dir / "failure_summary.csv", index=False)
    pairs.to_csv(args.out_dir / "failure_pair_overlap.csv", index=False)
    daily.to_csv(args.out_dir / "failure_daily.csv", index=False)
    signals.to_parquet(args.out_dir / "selected_signals.parquet", index=False)
    print("[FAILURE SUMMARY]")
    print(summary.to_string(index=False))
    print("\n[PAIR OVERLAP]")
    print(pairs.to_string(index=False))
    print(f"\n[SAVE] {args.out_dir}")


if __name__ == "__main__":
    main()
