from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import generate_signals2 as gs


PACKAGE_DIR = Path(__file__).resolve().parent
EPS = 1e-12


def normalize_date(x: Any) -> str:
    return str(x).replace("-", "")[:8]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate production signals over a date window using raw OHLC exits.")
    p.add_argument("--input", type=Path, default=PACKAGE_DIR / "processed" / "train_v5b" / "0709.parquet")
    p.add_argument("--daily-dir", type=Path, default=PACKAGE_DIR / "processed" / "daily" / "merged")
    p.add_argument("--out-dir", type=Path, default=PACKAGE_DIR / "reports" / "prediction_window_raw")
    p.add_argument("--start-date", default="20260604")
    p.add_argument("--end-date", default="20260706")
    p.add_argument("--layer", choices=["all", "layer1", "layer2"], default="all")
    p.add_argument("--take-profit", type=float, default=None, help="Override config raw_exit_rule.take_profit.")
    p.add_argument("--stop-loss", type=float, default=None, help="Override config raw_exit_rule.stop_loss.")
    p.add_argument("--max-hold-days", type=int, default=None, help="Override config raw_exit_rule.max_hold_days.")
    p.add_argument("--allow-missing-features", action="store_true")
    return p.parse_args()


def required_columns(
    cfg: dict[str, Any],
    family: list[gs.ModelBundle],
    seeds: list[gs.ModelBundle],
    day: gs.ModelBundle | None,
) -> list[str]:
    cols: set[str] = {"sample_id", "ts_code", "trade_date", "ixic_swing", "csi1500_mcap_oc_ret"}
    for bundle in family + seeds:
        cols.update(bundle.features)
    if day is not None:
        cols.update(day.features)
    return sorted(cols)


def load_window_panel(path: Path, cols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    import polars as pl

    schema = pl.scan_parquet(str(path)).collect_schema().names()
    existing = [c for c in cols if c in schema]
    missing = sorted(set(cols) - set(existing))
    if missing:
        print(f"[WARN] missing input columns={len(missing)} first={missing[:15]}", flush=True)
    lf = (
        pl.scan_parquet(str(path))
        .with_columns(pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", "").alias("trade_date"))
        .filter((pl.col("trade_date") >= start_date) & (pl.col("trade_date") <= end_date))
        .select(existing)
    )
    df = lf.collect().to_pandas()
    if df.empty:
        raise ValueError(f"No rows in {path} for {start_date}~{end_date}")
    return gs.normalize_keys(df)


def score_signals(df: pd.DataFrame, cfg: dict[str, Any], allow_missing: bool) -> pd.DataFrame:
    family, seed_bundles, day_bundle = gs.load_all_models(cfg, include_seed=True, include_day=True)
    scored = gs.add_family_predictions(df, family, allow_missing)
    scored = gs.add_seed_cv(scored, seed_bundles, allow_missing)
    scored = gs.add_day_regime(scored, day_bundle, allow_missing)
    scored = gs.add_signal_tags(scored, cfg)
    return scored


def filter_signal_layer(scored: pd.DataFrame, layer: str) -> pd.DataFrame:
    if layer == "layer1":
        mask = scored["strict_signal"].astype(bool) | scored["strong_signal"].astype(bool)
    elif layer == "layer2":
        mask = scored["layer2_signal"].astype(bool)
    else:
        mask = scored["signal_layer"].ne("none")
    return scored.loc[mask].copy()


def load_stock_daily(daily_dir: Path, ts_code: str) -> pd.DataFrame | None:
    path = daily_dir / f"{ts_code}.all.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"trade_date": str, "ts_code": str})
    if df.empty:
        return None
    df["trade_date"] = df["trade_date"].map(normalize_date)
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("trade_date").reset_index(drop=True)


def simulate_one(
    stock_df: pd.DataFrame,
    signal_date: str,
    take_profit: float,
    stop_loss: float,
    horizon: int,
) -> dict[str, Any] | None:
    dates = stock_df["trade_date"].astype(str).to_numpy()
    pos = np.flatnonzero(dates > signal_date)
    if len(pos) == 0:
        return None

    entry_idx = int(pos[0])
    entry_row = stock_df.iloc[entry_idx]
    entry_price = float(entry_row["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    tp_price = entry_price * (1.0 + take_profit)
    stop_price = entry_price * (1.0 + stop_loss)
    end_idx = min(entry_idx + horizon - 1, len(stock_df) - 1)

    # China A-share T+1: cannot sell on entry day. If entry close violates stop,
    # the executable exit is next trading day's open.
    entry_close = float(entry_row["close"])
    if np.isfinite(entry_close) and entry_close <= stop_price and entry_idx + 1 < len(stock_df):
        exit_idx = entry_idx + 1
        exit_row = stock_df.iloc[exit_idx]
        exit_price = float(exit_row["open"])
        reason = "t1_close_stop_sell_t2_open"
        basis = "open"
    else:
        exit_idx = end_idx
        exit_row = stock_df.iloc[exit_idx]
        exit_price = float(exit_row["close"])
        reason = "force_exit_horizon_close"
        basis = "close"

        for idx in range(entry_idx + 1, end_idx + 1):
            row = stock_df.iloc[idx]
            open_price = float(row["open"])
            high_price = float(row["high"])
            close_price = float(row["close"])
            if np.isfinite(open_price) and open_price >= tp_price:
                exit_idx = idx
                exit_price = open_price
                reason = "tp_open"
                basis = "open"
                break
            if np.isfinite(high_price) and high_price >= tp_price:
                exit_idx = idx
                exit_price = tp_price
                reason = "tp_intraday"
                basis = "take_profit"
                break
            if np.isfinite(close_price) and close_price <= stop_price:
                exit_idx = idx
                exit_price = close_price
                reason = "stop_close"
                basis = "close"
                break

    if not np.isfinite(exit_price) or exit_price <= 0:
        return None
    ret = exit_price / entry_price - 1.0
    return {
        "entry_trade_date": str(stock_df.iloc[entry_idx]["trade_date"]),
        "exit_trade_date": str(stock_df.iloc[exit_idx]["trade_date"]),
        "holding_days": int(exit_idx - entry_idx),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_price_basis": basis,
        "exit_reason": reason,
        "exit_return_raw": ret,
        "target_raw": int(reason.startswith("tp_")),
    }


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def profit_factor(returns: pd.Series) -> float:
    pos = returns[returns > 0].sum()
    neg = -returns[returns < 0].sum()
    if neg <= EPS:
        return np.inf if pos > 0 else np.nan
    return float(pos / neg)


def summarize(trades: pd.DataFrame, scored: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame([{
            "start_date": start_date,
            "end_date": end_date,
            "panel_dates": int(scored["trade_date"].nunique()),
            "signal_dates": 0,
            "trades": 0,
        }])
    r = pd.to_numeric(trades["exit_return_raw"], errors="coerce")
    daily = trades.groupby("trade_date")["exit_return_raw"].mean().sort_index()
    by_layer = (
        trades.groupby("signal_layer")
        .agg(
            trades=("ts_code", "size"),
            signal_dates=("trade_date", "nunique"),
            precision=("target_raw", "mean"),
            win_rate=("exit_return_raw", lambda s: float((s > 0).mean())),
            avg_trade_ret=("exit_return_raw", "mean"),
            median_trade_ret=("exit_return_raw", "median"),
        )
        .reset_index()
    )
    overall = pd.DataFrame([{
        "signal_layer": "ALL",
        "trades": int(len(trades)),
        "signal_dates": int(trades["trade_date"].nunique()),
        "precision": float(trades["target_raw"].mean()),
        "win_rate": float((r > 0).mean()),
        "avg_trade_ret": float(r.mean()),
        "median_trade_ret": float(r.median()),
    }])
    out = pd.concat([overall, by_layer], ignore_index=True)
    out.insert(0, "start_date", start_date)
    out.insert(1, "end_date", end_date)
    out.insert(2, "panel_dates", int(scored["trade_date"].nunique()))
    out["avg_picks_per_signal_day"] = out["trades"] / out["signal_dates"].replace(0, np.nan)
    out["profit_factor"] = out["signal_layer"].map(
        lambda x: profit_factor(r) if x == "ALL" else profit_factor(trades.loc[trades["signal_layer"].eq(x), "exit_return_raw"])
    )
    out["max_drawdown"] = out["signal_layer"].map(
        lambda x: max_drawdown(daily) if x == "ALL" else max_drawdown(
            trades.loc[trades["signal_layer"].eq(x)].groupby("trade_date")["exit_return_raw"].mean().sort_index()
        )
    )
    return out


def main() -> None:
    args = parse_args()
    start_date = normalize_date(args.start_date)
    end_date = normalize_date(args.end_date)
    cfg = gs.load_config()
    exit_cfg = cfg.get("raw_exit_rule", {})
    take_profit = float(args.take_profit if args.take_profit is not None else exit_cfg.get("take_profit", 0.08))
    stop_loss = float(args.stop_loss if args.stop_loss is not None else exit_cfg.get("stop_loss", -0.05))
    horizon = int(args.max_hold_days if args.max_hold_days is not None else exit_cfg.get("max_hold_days", 2))

    family, seeds, day = gs.load_all_models(cfg, include_seed=True, include_day=True)
    cols = required_columns(cfg, family, seeds, day)
    print(f"[load] {args.input} {start_date}~{end_date} cols={len(cols)}", flush=True)
    panel = load_window_panel(args.input, cols, start_date, end_date)
    print(f"[panel] rows={len(panel)} dates={panel['trade_date'].nunique()} stocks={panel['ts_code'].nunique()}", flush=True)

    scored = score_signals(panel, cfg, args.allow_missing_features)
    signals = filter_signal_layer(scored, args.layer)
    print(f"[signals] layer={args.layer} rows={len(signals)} dates={signals['trade_date'].nunique() if not signals.empty else 0}", flush=True)

    daily_cache: dict[str, pd.DataFrame | None] = {}
    trade_rows = []
    for _, row in signals.sort_values(["trade_date", "signal_priority", "avg_rank", "ts_code"]).iterrows():
        ts_code = str(row["ts_code"])
        if ts_code not in daily_cache:
            daily_cache[ts_code] = load_stock_daily(args.daily_dir, ts_code)
        stock_df = daily_cache[ts_code]
        if stock_df is None:
            continue
        sim = simulate_one(stock_df, str(row["trade_date"]), take_profit, stop_loss, horizon)
        if sim is None:
            continue
        trade_rows.append({
            "trade_date": str(row["trade_date"]),
            "ts_code": ts_code,
            "signal_layer": row["signal_layer"],
            "signal_priority": int(row["signal_priority"]),
            "avg_rank": float(row["avg_rank"]),
            "score_mean": float(row["score_mean"]),
            "prob_good_day": float(row["prob_good_day"]) if pd.notna(row["prob_good_day"]) else np.nan,
            **sim,
        })
    trades = pd.DataFrame(trade_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rule_tag = f"tp{int(round(take_profit * 1000)):03d}_sl{int(round(abs(stop_loss) * 1000)):03d}_h{horizon}"
    detail_path = args.out_dir / f"window_trades_{start_date}_{end_date}_{args.layer}_{rule_tag}.csv"
    summary_path = args.out_dir / f"window_summary_{start_date}_{end_date}_{args.layer}_{rule_tag}.csv"
    context_path = args.out_dir / f"window_context_{start_date}_{end_date}_{args.layer}_{rule_tag}.csv"

    if not trades.empty:
        trades.to_csv(detail_path, index=False, encoding="utf_8_sig")
    summary = summarize(trades, scored, start_date, end_date)
    summary.insert(3, "take_profit", take_profit)
    summary.insert(4, "stop_loss", stop_loss)
    summary.insert(5, "max_hold_days", horizon)
    summary.to_csv(summary_path, index=False, encoding="utf_8_sig")
    scored.groupby("trade_date").agg(
        rows=("ts_code", "size"),
        signals=("signal_layer", lambda s: int((s != "none").sum())),
        layer1=("strict_signal", lambda s: int(s.astype(bool).sum())),
        strong=("strong_signal", lambda s: int(s.astype(bool).sum())),
        layer2=("layer2_signal", lambda s: int(s.astype(bool).sum())),
        prob_good_day=("prob_good_day", "first"),
        ixic_swing=("ixic_swing", "first"),
        csi1500_mcap_oc_ret=("csi1500_mcap_oc_ret", "first"),
    ).reset_index().to_csv(context_path, index=False, encoding="utf_8_sig")

    print("[SUMMARY]")
    display_cols = [
        "signal_layer", "trades", "signal_dates", "precision", "win_rate",
        "avg_trade_ret", "profit_factor", "max_drawdown", "avg_picks_per_signal_day",
    ]
    print(summary[display_cols].to_string(index=False))
    print(f"[SAVE] {summary_path}")
    print(f"[SAVE] {detail_path if not trades.empty else '(no trades detail)'}")
    print(f"[SAVE] {context_path}")


if __name__ == "__main__":
    main()
