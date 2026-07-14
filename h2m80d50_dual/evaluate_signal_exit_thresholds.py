from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parent
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed production signals across precomputed target/exit thresholds."
    )
    parser.add_argument("--signals-dir", type=Path, default=PACKAGE_DIR / "signals")
    parser.add_argument("--pattern", default="signals_*.csv")
    parser.add_argument("--targets", type=Path, default=Path("processed/train_v5b/multi_targets_v5b.parquet"))
    parser.add_argument("--out", type=Path, default=PACKAGE_DIR / "reports" / "signal_exit_threshold_sweep.csv")
    parser.add_argument("--details-out", type=Path, default=PACKAGE_DIR / "reports" / "signal_exit_threshold_details.csv")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--layer",
        choices=["all", "layer1", "layer2", "strict", "strong"],
        default="all",
        help="Which generated signal subset to evaluate.",
    )
    parser.add_argument("--min-trades", type=int, default=1)
    return parser.parse_args()


def parse_suffix(suffix: str) -> dict[str, float | int | str]:
    match = re.fullmatch(r"h(\d+)m(\d+)d(\d+)", suffix)
    if not match:
        return {"suffix": suffix, "horizon_days": np.nan, "take_profit_pct": np.nan, "stop_loss_pct": np.nan}
    horizon = int(match.group(1))
    max_code = int(match.group(2))
    dd_code = int(match.group(3))
    return {
        "suffix": suffix,
        "horizon_days": horizon,
        "take_profit_pct": max_code / 10.0,
        "stop_loss_pct": -dd_code / 10.0,
    }


def read_signals(signals_dir: Path, pattern: str, start_date: str | None, end_date: str | None, layer: str) -> pd.DataFrame:
    frames = []
    for path in sorted(signals_dir.glob(pattern)):
        date_part = path.stem.split("_")[-1]
        if start_date and date_part < start_date:
            continue
        if end_date and date_part > end_date:
            continue
        frame = pd.read_csv(path, dtype={"trade_date": str, "ts_code": str})
        if frame.empty:
            continue
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No non-empty signal files matched {signals_dir / pattern}")
    signals = pd.concat(frames, ignore_index=True)

    for col in ["layer2_signal", "layer2_m1_signal", "layer2_m3_signal"]:
        if col not in signals.columns:
            signals[col] = False
        signals[col] = signals[col].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])

    if layer == "layer1":
        signals = signals[signals["signal_layer"].isin(["overlap_core", "strong_only", "strict_only"])].copy()
    elif layer == "layer2":
        signals = signals[signals["layer2_signal"]].copy()
    elif layer == "strict":
        signals = signals[signals["signal_layer"].isin(["overlap_core", "strict_only"])].copy()
    elif layer == "strong":
        signals = signals[signals["signal_layer"].isin(["overlap_core", "strong_only"])].copy()

    if signals.empty:
        raise ValueError(f"No signals left after layer filter: {layer}")
    return signals


def target_suffixes(path: Path) -> list[str]:
    cols = pd.read_parquet(path, engine="pyarrow").columns.tolist()
    suffixes = []
    for col in cols:
        match = re.fullmatch(r"target_(h\d+m\d+d\d+)", col)
        if match:
            suffixes.append(match.group(1))
    return suffixes


def load_targets(path: Path, suffixes: list[str]) -> pd.DataFrame:
    cols = ["sample_id", "ts_code", "trade_date"]
    for suffix in suffixes:
        cols.extend(
            [
                f"target_{suffix}",
                f"ret_raw_{suffix}",
                f"ret_adj_{suffix}",
                f"exit_type_{suffix}",
                f"hold_days_{suffix}",
            ]
        )
    target = pd.read_parquet(path, columns=cols, engine="pyarrow")
    target["trade_date"] = target["trade_date"].astype(str)
    target["ts_code"] = target["ts_code"].astype(str)
    return target


def max_drawdown(daily_ret: pd.Series) -> float:
    if daily_ret.empty:
        return np.nan
    equity = (1.0 + daily_ret.fillna(0.0)).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def profit_factor(ret: pd.Series) -> float:
    pos = ret[ret > 0].sum()
    neg = -ret[ret < 0].sum()
    return float(pos / neg) if neg > EPS else np.inf


def summarize(joined: pd.DataFrame, suffix: str) -> dict[str, float | int | str]:
    ret_raw = pd.to_numeric(joined[f"ret_raw_{suffix}"], errors="coerce")
    ret_adj = pd.to_numeric(joined[f"ret_adj_{suffix}"], errors="coerce")
    target = pd.to_numeric(joined[f"target_{suffix}"], errors="coerce")
    exit_type = joined[f"exit_type_{suffix}"]
    daily_raw = joined.assign(_ret=ret_raw).groupby("trade_date")["_ret"].mean()
    rec: dict[str, float | int | str] = parse_suffix(suffix)
    rec.update(
        {
            "trades": int(len(joined)),
            "dates": int(joined["trade_date"].nunique()),
            "avg_picks_per_day": float(len(joined) / max(joined["trade_date"].nunique(), 1)),
            "label_precision": float(target.mean()),
            "raw_win_rate": float((ret_raw > 0).mean()),
            "raw_tp_rate": float((ret_raw >= rec["take_profit_pct"] / 100.0).mean()),
            "avg_trade_ret_raw": float(ret_raw.mean()),
            "median_trade_ret_raw": float(ret_raw.median()),
            "avg_trade_ret_adj": float(ret_adj.mean()),
            "profit_factor_raw": profit_factor(ret_raw),
            "max_drawdown_raw": max_drawdown(daily_raw),
            "avg_daily_signal_ret_raw": float(daily_raw.mean()),
            "profit_exit_rate": float(exit_type.astype(str).str.contains("profit", na=False).mean()),
            "stop_exit_rate": float(exit_type.astype(str).str.contains("stop", na=False).mean()),
            "expiry_exit_rate": float(exit_type.astype(str).str.contains("expiry", na=False).mean()),
        }
    )
    return rec


def main() -> None:
    args = parse_args()
    suffixes = target_suffixes(args.targets)
    signals = read_signals(args.signals_dir, args.pattern, args.start_date, args.end_date, args.layer)
    targets = load_targets(args.targets, suffixes)
    keys = ["sample_id"] if "sample_id" in signals.columns and "sample_id" in targets.columns else ["trade_date", "ts_code"]
    joined = signals.merge(targets, on=keys, how="left", suffixes=("", "_target"))

    rows = []
    for suffix in suffixes:
        if joined[f"ret_raw_{suffix}"].notna().sum() < args.min_trades:
            continue
        rows.append(summarize(joined, suffix))
    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["avg_trade_ret_raw", "profit_factor_raw", "label_precision", "max_drawdown_raw"],
        ascending=[False, False, False, False],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("[TOP BY RAW AVG TRADE RETURN]")
    cols = [
        "suffix",
        "horizon_days",
        "take_profit_pct",
        "stop_loss_pct",
        "trades",
        "dates",
        "label_precision",
        "raw_win_rate",
        "raw_tp_rate",
        "avg_trade_ret_raw",
        "profit_factor_raw",
        "max_drawdown_raw",
        "profit_exit_rate",
        "stop_exit_rate",
        "expiry_exit_rate",
    ]
    print(out[cols].head(30).to_string(index=False))
    print(f"[SAVE] {args.out}")

    detail_cols = [
        "sample_id",
        "trade_date",
        "ts_code",
        "signal_layer",
        "avg_rank",
        "layer2_signal",
        "prob_good_day",
        "ixic_swing",
    ]
    detail_cols = [c for c in detail_cols if c in joined.columns]
    details = joined[detail_cols].copy()
    args.details_out.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.details_out, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {args.details_out}")


if __name__ == "__main__":
    main()
