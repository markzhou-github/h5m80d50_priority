#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl

from generate_signals import HERE, generate_for_date, load_context, read_history, record_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen H3M55D15 signals for a date range")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--start-date", required=True, help="Inclusive YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="Inclusive YYYYMMDD")
    parser.add_argument("--out-dir", type=Path, default=HERE / "signals_range")
    parser.add_argument("--combined-out", type=Path)
    parser.add_argument('--history-file', type=Path, default=HERE / 'state/risk_history.csv')
    args = parser.parse_args()
    if args.start_date > args.end_date:
        raise ValueError("--start-date must not be after --end-date")

    dates = (
        pl.scan_parquet(args.input)
        .select(pl.col("trade_date").cast(pl.Utf8).alias("trade_date"))
        .filter(pl.col("trade_date").is_between(args.start_date, args.end_date, closed="both"))
        .unique().sort("trade_date").collect()["trade_date"].to_list()
    )
    if not dates:
        raise ValueError(f"No dates in requested range {args.start_date}..{args.end_date}")
    context = load_context(args.input)
    history = read_history(args.history_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    signals, diagnostics = [], []
    for index, date in enumerate(dates, 1):
        candidates, diagnostic = generate_for_date(args.input, date, context,
            history.loc[history.trade_date.lt(date), 'bad_risk_logit'].tolist())
        history = record_history(history, date, diagnostic['bad_day_risk'], args.history_file)
        candidates.to_csv(args.out_dir / f'candidates_{date}.csv', index=False)
        selected = candidates[candidates.priority.gt(0)]
        selected.to_csv(args.out_dir / f"signals_{date}.csv", index=False)
        signals.append(selected)
        diagnostics.append(diagnostic)
        print(
            f"[{index}/{len(dates)}] {date} P1={diagnostic['p1']} P2={diagnostic['p2']}",
            flush=True,
        )
    combined = pd.concat(signals, ignore_index=True)
    combined_out = args.combined_out or args.out_dir / f"signals_{dates[0]}_{dates[-1]}.csv"
    combined_out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_out, index=False)
    combined.to_csv(args.out_dir / "signals_latest_range.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(
        args.out_dir / f"diagnostics_{dates[0]}_{dates[-1]}.csv", index=False
    )
    print(f"[dates] {len(dates)} [rows] {len(combined)}")
    print(f"[SAVE] {combined_out}")


if __name__ == "__main__":
    main()
