from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl

from generate_signals import (
    add_base_selections,
    add_predictions,
    add_signal_models,
    add_watchlist,
    diagnostic_columns,
    diagnostic_summary,
    load_config,
    load_models,
    normalize_keys,
    signal_columns,
)


PACKAGE_DIR = Path(__file__).resolve().parent


def normalize_date(x: str) -> str:
    return str(x).replace("-", "")


def read_input_range(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    start = normalize_date(start_date)
    end = normalize_date(end_date)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        lf = (
            pl.scan_parquet(str(path))
            .with_columns(pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", "").alias("trade_date"))
            .filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))
        )
        out = lf.collect().to_pandas()
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(path, dtype={"trade_date": str, "ts_code": str})
        df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "", regex=False)
        out = df[df["trade_date"].between(start, end)].copy()
    else:
        raise ValueError(f"Unsupported input file type: {path}")

    if out.empty:
        raise ValueError(f"No rows for date range {start} ~ {end} in {path}")
    return out


def write_one_date(df: pd.DataFrame, date: str, out_dir: Path, save_ranked: bool, diagnostic_top_n: int) -> dict[str, object]:
    part = df[df["trade_date"].astype(str).eq(date)].copy()
    if part.empty:
        return {"trade_date": date, "rows": 0, "signal_rows": 0, "watchlist_rows": 0}

    signals = part[part["signal_type"].ne("")].sort_values(
        ["signal_type", "priority", "avg_pred_rank", "ts_code"],
        ascending=[True, True, True, True],
    )
    trade_signals = part[part["priority"] > 0].sort_values(["priority", "avg_pred_rank", "ts_code"])
    watchlist = part[part["watchlist_signal"]].sort_values(["avg_pred_rank", "ts_code"])

    signal_path = out_dir / f"signals_h5m80d50_priority_{date}.csv"
    latest_path = out_dir / "signals_latest.csv"
    watchlist_path = out_dir / f"watchlist_h5m80d50_priority_{date}.csv"
    summary_path = out_dir / f"signal_summary_{date}.csv"
    diag_path = out_dir / f"diagnostic_summary_{date}.csv"
    diag_ranked_path = out_dir / f"diagnostic_top{diagnostic_top_n}_{date}.csv"

    signals[signal_columns(signals)].to_csv(signal_path, index=False, encoding="utf_8_sig")
    signals[signal_columns(signals)].to_csv(latest_path, index=False, encoding="utf_8_sig")
    watchlist[signal_columns(watchlist)].to_csv(watchlist_path, index=False, encoding="utf_8_sig")

    summary = (
        trade_signals.groupby(["trade_date", "priority", "priority_name"], as_index=False)
        .agg(signal_count=("ts_code", "size"), avg_pred_mean=("avg_pred", "mean"))
        .sort_values(["trade_date", "priority"])
    )
    summary.to_csv(summary_path, index=False, encoding="utf_8_sig")

    diag = diagnostic_summary(part)
    diag.to_csv(diag_path, index=False, encoding="utf_8_sig")

    diag_ranked = part.sort_values(["avg_pred_rank", "ts_code"]).head(diagnostic_top_n)
    diag_ranked[diagnostic_columns(diag_ranked)].to_csv(diag_ranked_path, index=False, encoding="utf_8_sig")

    if save_ranked:
        ranked_path = out_dir / f"ranked_h5m80d50_priority_{date}.csv"
        ranked_cols = signal_columns(part) + [c for c in part.columns if c.startswith("pred_") or c.startswith("rank_")]
        part[ranked_cols].sort_values(["avg_pred_rank", "ts_code"]).to_csv(ranked_path, index=False, encoding="utf_8_sig")

    return {
        "trade_date": date,
        "rows": len(part),
        "signal_rows": len(signals),
        "trade_rows": len(trade_signals),
        "watchlist_rows": len(watchlist),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate h5m80d50 P1/P2/P3 signals for a date range.")
    parser.add_argument("--input", type=Path, required=True, help="Prediction feature parquet/csv.")
    parser.add_argument("--start-date", required=True, help="YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD.")
    parser.add_argument("--out-dir", type=Path, default=PACKAGE_DIR / "signals")
    parser.add_argument("--allow-missing-features", action="store_true")
    parser.add_argument("--save-ranked", action="store_true")
    parser.add_argument("--diagnostic-top-n", type=int, default=30)
    parser.add_argument("--watchlist-top-n", type=int, default=3)
    parser.add_argument("--watchlist-vote-min", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config()
    bundles = load_models(cfg)
    seed_names = [b.name for b in bundles]

    print(f"[load] {args.input} {args.start_date} ~ {args.end_date}")
    df = normalize_keys(read_input_range(args.input, args.start_date, args.end_date))
    dates = sorted(df["trade_date"].astype(str).unique().tolist())
    print(f"[rows] {len(df)} dates={len(dates)} date_min={dates[0]} date_max={dates[-1]}")

    print("[predict] start")
    df = add_predictions(df, bundles, args.allow_missing_features)
    df = add_base_selections(df, cfg, seed_names)
    df = add_signal_models(df, cfg)
    df = add_watchlist(df, args.watchlist_top_n, args.watchlist_vote_min)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, date in enumerate(dates, start=1):
        rec = write_one_date(df, date, args.out_dir, args.save_ranked, args.diagnostic_top_n)
        rows.append(rec)
        print(
            f"[{i}/{len(dates)}] {date} rows={rec['rows']} "
            f"trade={rec['trade_rows']} watchlist={rec['watchlist_rows']}"
        )

    summary = pd.DataFrame(rows)
    summary_path = args.out_dir / f"range_generation_summary_{dates[0]}_{dates[-1]}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf_8_sig")
    print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()
