#!/usr/bin/env python
# coding: utf-8
"""Build v5b intraday features from per-stock 1-minute bars.

Input files are expected at data/raw/{ts_code}.parquet. The new raw files store
volume in shares and amount in yuan, so VWAP is amount / vol.

Default output is one daily feature parquet per stock:
  processed/minute_features_v5b/by_stock/{ts_code}.parquet

Panel output is available through --output-mode chunked or --output-mode memory.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view


RAW_DIR = Path("data/raw")
OUT_DIR = Path("processed/minute_features_v5b")
BY_STOCK_DIRNAME = "by_stock"
CHUNK_DIRNAME = "chunks"
PANEL_FILENAME = "minute_features_panel.parquet"
EPS = 1e-12


FEATURE_COLUMNS = [
    "first_1m_ret", "first_1m_range", "first_1m_amount_ratio", "first_1m_vwap_ratio",
    "first_5m_ret", "first_5m_range", "first_5m_amount_ratio", "first_5m_vwap_ratio",
    "first_10m_ret", "first_10m_range", "first_10m_amount_ratio", "first_10m_vwap_ratio",
    "first_15m_ret", "first_15m_range", "first_15m_amount_ratio", "first_15m_vwap_ratio",
    "first_30m_ret", "first_30m_range", "first_30m_amount_ratio", "first_30m_vwap_ratio",
    "first_60m_ret", "first_60m_range", "first_60m_amount_ratio", "first_60m_vwap_ratio",
    "morning_ret", "morning_range", "morning_amount_ratio", "morning_vwap_ratio",
    "afternoon_ret", "afternoon_range", "afternoon_amount_ratio", "afternoon_vwap_ratio",
    "morning_total_amount_ratio", "morning_total_volume_ratio",
    "last_60m_ret", "last_60m_range", "last_60m_amount_ratio", "last_60m_vwap_ratio",
    "last_30m_ret", "last_30m_range", "last_30m_amount_ratio", "last_30m_vwap_ratio",
    "last_15m_ret", "last_15m_range", "last_15m_amount_ratio", "last_15m_vwap_ratio",
    "last_10m_ret", "last_10m_range", "last_10m_amount_ratio", "last_10m_vwap_ratio",
    "last_5m_ret", "last_5m_range", "last_5m_amount_ratio", "last_5m_vwap_ratio",
    "max_5m_ret", "min_5m_ret", "max_10m_ret", "min_10m_ret", "max_15m_ret", "min_15m_ret",
    "max_5m_range", "mean_5m_range", "max_15m_range", "mean_15m_range",
    "realized_vol_5m", "late_realized_vol_5m", "trend_efficiency", "intraday_sign_changes",
    "intraday_return_skew", "up_bar_ratio", "down_bar_ratio", "late_up_bar_ratio",
    "max_consecutive_up_bars", "max_consecutive_down_bars", "pct_bars_above_vwap",
    "vwap_cross_count", "mean_vwap_distance", "amount_concentration_top3",
    "intraday_max_drawdown", "drawdown_duration", "minute_of_high", "minute_of_low",
    "minute_of_high_raw", "minute_of_low_raw",
    "pct_time_above_vwap", "max_vwap_distance", "std_vwap_distance",
    "last30_vwap_distance", "last60_vwap_distance", "close_vwap_distance", "vwap_recovery_ratio",
    "morning_efficiency", "afternoon_efficiency", "morning_afternoon_corr", "realized_kurtosis",
    "buy_volume_ratio", "sell_volume_ratio", "buy_pressure", "afternoon_volume_share",
    "volume_curve_skew", "variance_top1_share", "variance_top3_share",
    "variance_top5_share", "variance_top10_share", "variance_top20_share",
    "gap_fill_ratio", "gap_persistence", "gap_same_side_ratio",
    "gap_fill_time", "gap_fill_time_raw", "first_reversal_minute",
    "realized_vol_1m", "late_realized_vol_1m", "realized_up_semivar",
    "realized_down_semivar", "realized_semivar_imbalance",
    "bipower_variation", "jump_variation", "jump_variation_share",
    "first30_vol_share", "morning_vol_share", "afternoon_vol_share", "last30_vol_share", "last60_vol_share",
    "return_autocorr_1", "return_autocorr_5", "return_autocorr_10", "bar_entropy", "hurst_intraday",
    "max_1m_ret", "minute_of_max_1m_ret", "minute_of_max_1m_ret_raw",
    "max_1m_amount", "minute_of_max_amount", "minute_of_max_amount_raw",
    "max_1m_volume", "minute_of_max_volume", "minute_of_max_volume_raw",
    "max_1m_range", "minute_of_max_range", "minute_of_max_range_raw",
    "start_minute_longest_up_run", "start_minute_longest_up_run_raw",
    "start_minute_longest_down_run", "start_minute_longest_down_run_raw",
    "late_max_5m_drop", "late_max_5m_range", "afternoon_minus_morning_ret", "last60_minus_first60_ret",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 1-minute intraday features per stock.")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--stocks", nargs="*", default=[], help="Optional stock list. Accepts 600004.SH or 600004.")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--start-date", default=None, help="Optional YYYYMMDD inclusive.")
    p.add_argument("--end-date", default=None, help="Optional YYYYMMDD inclusive.")
    p.add_argument("--save-panel", action="store_true", help="Also save combined panel parquet.")
    p.add_argument(
        "--output-mode",
        choices=["by_stock", "chunked", "memory"],
        default="by_stock",
        help=(
            "by_stock writes one feature file per stock. chunked writes chunk parquet files plus one panel. "
            "memory keeps all stock feature frames in memory and writes one panel."
        ),
    )
    p.add_argument("--stock-chunk-size", type=int, default=150, help="Stocks per chunk for --output-mode chunked.")
    p.add_argument("--panel-name", default=PANEL_FILENAME, help="Final panel parquet name for chunked/memory/save-panel.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing per-stock feature files.")
    p.add_argument("--min-continuous-bars", type=int, default=180)
    p.add_argument("--report-name", default="minute_feature_build_summary.csv")
    return p.parse_args()


def normalize_ts_code(code: str) -> str:
    code = str(code).strip().upper()
    if not code or "." in code:
        return code
    return f"{code}.SH" if code.startswith(("6", "9")) else f"{code}.SZ"


def stock_paths(raw_dir: Path, stocks: list[str]) -> list[Path]:
    if not stocks:
        return sorted(raw_dir.glob("*.parquet"))
    paths = []
    for stock in stocks:
        path = raw_dir / f"{normalize_ts_code(stock)}.parquet"
        if path.exists():
            paths.append(path)
        else:
            print(f"[warn] missing raw file: {path}", flush=True)
    return paths


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def safe_div(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(float(denominator)) <= EPS:
        return np.nan
    return float(numerator) / float(denominator)


def safe_mean(values: np.ndarray) -> float:
    v = finite(values)
    return float(v.mean()) if len(v) else np.nan


def safe_std(values: np.ndarray) -> float:
    v = finite(values)
    return float(v.std(ddof=0)) if len(v) else np.nan


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    n = min(len(left), len(right))
    if n < 5:
        return np.nan
    a = np.asarray(left[:n], dtype=float)
    b = np.asarray(right[:n], dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 5 or a.std(ddof=0) <= EPS or b.std(ddof=0) <= EPS:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def normalized_pos(pos: int | None, length: int) -> float:
    if pos is None or length <= 1:
        return np.nan
    return float(pos) / float(length - 1)


def longest_true_run(values: np.ndarray) -> tuple[int, float]:
    best_len = cur_len = 0
    best_start: int | None = None
    cur_start: int | None = None
    for i, value in enumerate(values):
        if bool(value):
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
            cur_start = None
    return int(best_len), normalized_pos(best_start, len(values)) if best_start is not None else np.nan


def realized_vol(values: np.ndarray) -> float:
    v = finite(values)
    return float(math.sqrt(float(np.sum(v ** 2)))) if len(v) else np.nan


def realized_vol_share(values: np.ndarray, idx: np.ndarray, full_realized_vol: float) -> float:
    if not np.isfinite(full_realized_vol) or full_realized_vol <= EPS or len(idx) == 0:
        return np.nan
    part = realized_vol(values[idx])
    return safe_div(part, full_realized_vol)


def sign_change_count(values: np.ndarray) -> int:
    signs = np.sign(finite(values))
    signs = signs[signs != 0]
    return int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0


def autocorr(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag + 2:
        return np.nan
    return safe_corr(values[lag:], values[:-lag])


def bar_entropy(values: np.ndarray) -> float:
    signs = np.sign(finite(values))
    if len(signs) == 0:
        return np.nan
    counts = np.array([(signs > 0).sum(), (signs < 0).sum(), (signs == 0).sum()], dtype=float)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log(probs)).sum() / np.log(3.0))


def hurst_intraday(close: np.ndarray) -> float:
    values = finite(close)
    if len(values) < 40:
        return np.nan
    xs, ys = [], []
    for lag in [1, 2, 4, 8, 16]:
        if len(values) <= lag:
            continue
        std = float(np.std(values[lag:] - values[:-lag]))
        if std > EPS:
            xs.append(math.log(float(lag)))
            ys.append(math.log(std))
    return float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 3 else np.nan


def pct_change(close: np.ndarray) -> np.ndarray:
    out = np.full(len(close), np.nan, dtype=float)
    if len(close) > 1:
        denom = close[:-1]
        out[1:] = np.where(np.abs(denom) > EPS, close[1:] / denom - 1.0, np.nan)
    return out


def rolling_return(close: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(close), np.nan, dtype=float)
    if len(close) > n:
        denom = close[:-n]
        out[n:] = np.where(np.abs(denom) > EPS, close[n:] / denom - 1.0, np.nan)
    return out


def rolling_range(high: np.ndarray, low: np.ndarray, open_: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(high), np.nan, dtype=float)
    if len(high) >= n:
        hi = sliding_window_view(high, n).max(axis=1)
        lo = sliding_window_view(low, n).min(axis=1)
        op = open_[n - 1:]
        out[n - 1:] = np.where(np.abs(op) > EPS, (hi - lo) / op, np.nan)
    return out


def trend_efficiency(open_: np.ndarray, close: np.ndarray) -> float:
    if len(close) == 0 or not np.isfinite(open_[0]) or abs(open_[0]) <= EPS:
        return np.nan
    rets = pct_change(close)
    rets[0] = close[0] / open_[0] - 1.0
    denom = np.nansum(np.abs(rets))
    total = close[-1] / open_[0] - 1.0
    return float(abs(total) / denom) if denom > EPS else 0.0


def window_stats(
    row: dict[str, Any],
    prefix: str,
    idx: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    vol: np.ndarray,
    amount: np.ndarray,
    total_amount: float,
    daily_vwap: float,
) -> None:
    if len(idx) == 0:
        return
    amt = float(np.nansum(amount[idx]))
    volume = float(np.nansum(vol[idx]))
    vwap = safe_div(amt, volume)
    first_open = float(open_[idx[0]])
    last_close = float(close[idx[-1]])
    row[f"{prefix}_ret"] = last_close / first_open - 1.0 if abs(first_open) > EPS else np.nan
    row[f"{prefix}_range"] = safe_div(float(np.nanmax(high[idx]) - np.nanmin(low[idx])), first_open)
    row[f"{prefix}_amount_ratio"] = safe_div(amt, total_amount)
    row[f"{prefix}_vwap_ratio"] = vwap / daily_vwap - 1.0 if np.isfinite(vwap) and np.isfinite(daily_vwap) else np.nan


def build_one_day(
    ts_code: str,
    trade_date: str,
    raw_day: pl.DataFrame,
    min_continuous_bars: int,
    prev_close: float | None = None,
) -> dict[str, Any] | None:
    raw_day = raw_day.sort("datetime")
    all_time = raw_day.get_column("minute_time").to_numpy()
    raw_open = raw_day.get_column("open").cast(pl.Float64).to_numpy()
    raw_close = raw_day.get_column("close").cast(pl.Float64).to_numpy()
    raw_high = raw_day.get_column("high").cast(pl.Float64).to_numpy()
    raw_low = raw_day.get_column("low").cast(pl.Float64).to_numpy()
    raw_vol = raw_day.get_column("vol").cast(pl.Float64).to_numpy()
    raw_amount = raw_day.get_column("amount").cast(pl.Float64).to_numpy()

    raw_valid = (
        np.isfinite(raw_open) & np.isfinite(raw_close) & np.isfinite(raw_high) & np.isfinite(raw_low)
        & np.isfinite(raw_vol) & np.isfinite(raw_amount)
        & (raw_open > 0) & (raw_close > 0) & (raw_high > 0) & (raw_low > 0)
    )
    cont = raw_valid & (all_time != "09:30")
    if int(cont.sum()) < min_continuous_bars:
        return None

    minute_time = all_time[cont]
    open_ = raw_open[cont]
    close = raw_close[cont]
    high = raw_high[cont]
    low = raw_low[cont]
    vol = raw_vol[cont]
    amount = raw_amount[cont]
    n_bars = len(close)
    idx_all = np.arange(n_bars)
    norm_minute = idx_all.astype(float) / max(n_bars - 1, 1)

    total_amount = float(np.nansum(amount))
    total_vol = float(np.nansum(vol))
    daily_vwap = safe_div(total_amount, total_vol)
    row: dict[str, Any] = {
        "ts_code": ts_code,
        "trade_date": str(trade_date),
        "minute_bar_count": int(n_bars),
        "minute_total_amount": total_amount,
        "minute_total_vol": total_vol,
        "minute_daily_vwap": daily_vwap,
    }

    windows = [
        ("first_1m", idx_all[:1]),
        ("first_5m", idx_all[:5]),
        ("first_10m", idx_all[:10]),
        ("first_15m", idx_all[:15]),
        ("first_30m", idx_all[:30]),
        ("first_60m", idx_all[:60]),
        ("morning", idx_all[minute_time <= "11:30"]),
        ("afternoon", idx_all[minute_time >= "13:00"]),
        ("last_60m", idx_all[-60:]),
        ("last_30m", idx_all[-30:]),
        ("last_15m", idx_all[-15:]),
        ("last_10m", idx_all[-10:]),
        ("last_5m", idx_all[-5:]),
    ]
    for prefix, idx in windows:
        window_stats(row, prefix, idx, open_, close, high, low, vol, amount, total_amount, daily_vwap)

    all_valid_time = all_time[raw_valid]
    all_amount_valid = raw_amount[raw_valid]
    all_vol_valid = raw_vol[raw_valid]
    morning_total = all_valid_time <= "11:30"
    row["morning_total_amount_ratio"] = safe_div(float(np.nansum(all_amount_valid[morning_total])), float(np.nansum(all_amount_valid)))
    row["morning_total_volume_ratio"] = safe_div(float(np.nansum(all_vol_valid[morning_total])), float(np.nansum(all_vol_valid)))

    first_ret = close[0] / open_[0] - 1.0 if abs(open_[0]) > EPS else np.nan
    path_ret = pct_change(close)
    path_ret[0] = first_ret
    bar_range = np.full(n_bars, np.nan, dtype=float)
    valid_open = np.abs(open_) > EPS
    bar_range[valid_open] = (high[valid_open] - low[valid_open]) / open_[valid_open]
    cum_vol = np.cumsum(vol)
    cum_amount = np.cumsum(amount)
    cum_vwap = np.full(n_bars, np.nan, dtype=float)
    valid_cum_vol = cum_vol > EPS
    cum_vwap[valid_cum_vol] = cum_amount[valid_cum_vol] / cum_vol[valid_cum_vol]
    close_minus_vwap = close - cum_vwap

    for n in [5, 10, 15]:
        r = rolling_return(close, n)
        row[f"max_{n}m_ret"] = float(np.nanmax(r)) if np.isfinite(r).any() else np.nan
        row[f"min_{n}m_ret"] = float(np.nanmin(r)) if np.isfinite(r).any() else np.nan
    range_5m = rolling_range(high, low, open_, 5)
    range_15m = rolling_range(high, low, open_, 15)
    row["max_5m_range"] = float(np.nanmax(range_5m)) if np.isfinite(range_5m).any() else np.nan
    row["mean_5m_range"] = safe_mean(range_5m)
    row["max_15m_range"] = float(np.nanmax(range_15m)) if np.isfinite(range_15m).any() else np.nan
    row["mean_15m_range"] = safe_mean(range_15m)

    path_abs_sum = float(np.nansum(np.abs(path_ret)))
    full_path_ret = close[-1] / open_[0] - 1.0 if abs(open_[0]) > EPS else np.nan
    row["realized_vol_5m"] = safe_std(path_ret)
    row["realized_vol_1m"] = realized_vol(path_ret)
    up_semivar = float(np.nansum(np.square(np.clip(path_ret, 0.0, None))))
    down_semivar = float(np.nansum(np.square(np.clip(path_ret, None, 0.0))))
    row["realized_up_semivar"] = up_semivar
    row["realized_down_semivar"] = down_semivar
    row["realized_semivar_imbalance"] = safe_div(up_semivar - down_semivar, up_semivar + down_semivar)
    row["trend_efficiency"] = abs(full_path_ret) / path_abs_sum if np.isfinite(full_path_ret) and path_abs_sum > EPS else 0.0
    row["intraday_sign_changes"] = sign_change_count(path_ret)
    pr = finite(path_ret)
    if len(pr) >= 3 and pr.std(ddof=0) > EPS:
        z = (pr - pr.mean()) / pr.std(ddof=0)
        row["intraday_return_skew"] = float(np.mean(z ** 3))
    else:
        row["intraday_return_skew"] = np.nan
    if len(pr) >= 4 and pr.std(ddof=0) > EPS:
        z = (pr - pr.mean()) / pr.std(ddof=0)
        row["realized_kurtosis"] = float(np.mean(z ** 4) - 3.0)
    else:
        row["realized_kurtosis"] = np.nan
    row["up_bar_ratio"] = float(np.nanmean(path_ret > 0))
    row["down_bar_ratio"] = float(np.nanmean(path_ret < 0))
    row["pct_bars_above_vwap"] = float(np.nanmean(close > cum_vwap))
    row["pct_time_above_vwap"] = row["pct_bars_above_vwap"]
    above = np.sign(finite(close_minus_vwap))
    row["vwap_cross_count"] = int(np.sum(above[1:] != above[:-1])) if len(above) > 1 else 0
    vwap_distance = np.full(n_bars, np.nan, dtype=float)
    valid_cum_vwap = np.abs(cum_vwap) > EPS
    vwap_distance[valid_cum_vwap] = close_minus_vwap[valid_cum_vwap] / cum_vwap[valid_cum_vwap]
    abs_vwap_distance = np.abs(vwap_distance)
    row["mean_vwap_distance"] = safe_mean(abs_vwap_distance)
    row["max_vwap_distance"] = float(np.nanmax(abs_vwap_distance)) if np.isfinite(abs_vwap_distance).any() else np.nan
    row["std_vwap_distance"] = safe_std(vwap_distance)
    row["last30_vwap_distance"] = safe_mean(abs_vwap_distance[idx_all[-30:]])
    row["last60_vwap_distance"] = safe_mean(abs_vwap_distance[idx_all[-60:]])
    row["close_vwap_distance"] = close[-1] / daily_vwap - 1.0 if np.isfinite(daily_vwap) and abs(daily_vwap) > EPS else np.nan
    close_abs_vwap_distance = abs(row["close_vwap_distance"]) if np.isfinite(row["close_vwap_distance"]) else np.nan
    row["vwap_recovery_ratio"] = (
        safe_div(row["max_vwap_distance"] - close_abs_vwap_distance, row["max_vwap_distance"])
        if np.isfinite(row["max_vwap_distance"]) and np.isfinite(close_abs_vwap_distance)
        else np.nan
    )
    row["amount_concentration_top3"] = safe_div(float(np.nansum(np.sort(amount)[-3:])), total_amount)

    up_flags = np.nan_to_num(path_ret, nan=0.0) > 0
    down_flags = np.nan_to_num(path_ret, nan=0.0) < 0
    row["max_consecutive_up_bars"], row["start_minute_longest_up_run"] = longest_true_run(up_flags)
    row["max_consecutive_down_bars"], row["start_minute_longest_down_run"] = longest_true_run(down_flags)
    row["start_minute_longest_up_run_raw"] = (
        int(round(row["start_minute_longest_up_run"] * (n_bars - 1)))
        if np.isfinite(row["start_minute_longest_up_run"])
        else np.nan
    )
    row["start_minute_longest_down_run_raw"] = (
        int(round(row["start_minute_longest_down_run"] * (n_bars - 1)))
        if np.isfinite(row["start_minute_longest_down_run"])
        else np.nan
    )

    running_peak = np.maximum.accumulate(close)
    drawdown = np.where(running_peak > EPS, close / running_peak - 1.0, np.nan)
    row["intraday_max_drawdown"] = float(np.nanmin(drawdown)) if np.isfinite(drawdown).any() else np.nan
    row["drawdown_duration"], _ = longest_true_run(close < running_peak)
    high_pos = int(np.nanargmax(high))
    low_pos = int(np.nanargmin(low))
    row["minute_of_high"] = normalized_pos(high_pos, n_bars)
    row["minute_of_low"] = normalized_pos(low_pos, n_bars)
    row["minute_of_high_raw"] = high_pos
    row["minute_of_low_raw"] = low_pos

    row["max_1m_ret"] = float(np.nanmax(path_ret)) if np.isfinite(path_ret).any() else np.nan
    max_ret_pos = int(np.nanargmax(path_ret)) if np.isfinite(path_ret).any() else None
    row["minute_of_max_1m_ret"] = normalized_pos(max_ret_pos, n_bars) if max_ret_pos is not None else np.nan
    row["minute_of_max_1m_ret_raw"] = max_ret_pos if max_ret_pos is not None else np.nan
    row["max_1m_amount"] = float(np.nanmax(amount)) if np.isfinite(amount).any() else np.nan
    max_amount_pos = int(np.nanargmax(amount)) if np.isfinite(amount).any() else None
    row["minute_of_max_amount"] = normalized_pos(max_amount_pos, n_bars) if max_amount_pos is not None else np.nan
    row["minute_of_max_amount_raw"] = max_amount_pos if max_amount_pos is not None else np.nan
    row["max_1m_volume"] = float(np.nanmax(vol)) if np.isfinite(vol).any() else np.nan
    max_volume_pos = int(np.nanargmax(vol)) if np.isfinite(vol).any() else None
    row["minute_of_max_volume"] = normalized_pos(max_volume_pos, n_bars) if max_volume_pos is not None else np.nan
    row["minute_of_max_volume_raw"] = max_volume_pos if max_volume_pos is not None else np.nan
    row["max_1m_range"] = float(np.nanmax(bar_range)) if np.isfinite(bar_range).any() else np.nan
    max_range_pos = int(np.nanargmax(bar_range)) if np.isfinite(bar_range).any() else None
    row["minute_of_max_range"] = normalized_pos(max_range_pos, n_bars) if max_range_pos is not None else np.nan
    row["minute_of_max_range_raw"] = max_range_pos if max_range_pos is not None else np.nan

    buy_vol = float(np.nansum(vol[path_ret > 0]))
    sell_vol = float(np.nansum(vol[path_ret < 0]))
    row["buy_volume_ratio"] = safe_div(buy_vol, total_vol)
    row["sell_volume_ratio"] = safe_div(sell_vol, total_vol)
    row["buy_pressure"] = row["buy_volume_ratio"] - row["sell_volume_ratio"] if np.isfinite(row["buy_volume_ratio"]) and np.isfinite(row["sell_volume_ratio"]) else np.nan
    morning_mask = minute_time <= "11:30"
    afternoon_mask = minute_time >= "13:00"
    row["afternoon_volume_share"] = safe_div(float(np.nansum(vol[afternoon_mask])), total_vol)
    row["volume_curve_skew"] = safe_div(float(np.nansum(norm_minute * vol)), total_vol)
    sq_ret = finite(path_ret ** 2)
    total_var = float(np.nansum(sq_ret))
    row["variance_top1_share"] = safe_div(float(np.nansum(np.sort(sq_ret)[-1:])), total_var)
    row["variance_top3_share"] = safe_div(float(np.nansum(np.sort(sq_ret)[-3:])), total_var)
    row["variance_top5_share"] = safe_div(float(np.nansum(np.sort(sq_ret)[-5:])), total_var)
    row["variance_top10_share"] = safe_div(float(np.nansum(np.sort(sq_ret)[-10:])), total_var)
    row["variance_top20_share"] = safe_div(float(np.nansum(np.sort(sq_ret)[-20:])), total_var)
    abs_ret = np.abs(finite(path_ret))
    row["bipower_variation"] = (
        float((math.pi / 2.0) * np.sum(abs_ret[1:] * abs_ret[:-1]))
        if len(abs_ret) >= 2
        else np.nan
    )
    row["jump_variation"] = max(0.0, total_var - row["bipower_variation"]) if np.isfinite(row["bipower_variation"]) else np.nan
    row["jump_variation_share"] = safe_div(row["jump_variation"], total_var)
    row["first30_vol_share"] = realized_vol_share(path_ret, idx_all[:30], row["realized_vol_1m"])
    row["morning_vol_share"] = realized_vol_share(path_ret, idx_all[morning_mask], row["realized_vol_1m"])
    row["afternoon_vol_share"] = realized_vol_share(path_ret, idx_all[afternoon_mask], row["realized_vol_1m"])
    row["last30_vol_share"] = realized_vol_share(path_ret, idx_all[-30:], row["realized_vol_1m"])
    row["last60_vol_share"] = realized_vol_share(path_ret, idx_all[-60:], row["realized_vol_1m"])
    row["return_autocorr_1"] = autocorr(path_ret, 1)
    row["return_autocorr_5"] = autocorr(path_ret, 5)
    row["return_autocorr_10"] = autocorr(path_ret, 10)
    row["bar_entropy"] = bar_entropy(path_ret)
    row["hurst_intraday"] = hurst_intraday(close)

    row["morning_efficiency"] = trend_efficiency(open_[morning_mask], close[morning_mask])
    row["afternoon_efficiency"] = trend_efficiency(open_[afternoon_mask], close[afternoon_mask])
    row["morning_afternoon_corr"] = safe_corr(pct_change(close[morning_mask])[1:], pct_change(close[afternoon_mask])[1:])

    if prev_close is not None and np.isfinite(prev_close) and prev_close > EPS:
        gap = open_[0] / prev_close - 1.0
        if abs(gap) > EPS:
            if gap > 0:
                filled = max(0.0, open_[0] - float(np.nanmin(low)))
                same_side = close >= prev_close
                reversed_mask = close <= prev_close
            else:
                filled = max(0.0, float(np.nanmax(high)) - open_[0])
                same_side = close <= prev_close
                reversed_mask = close >= prev_close
            fill_mask = (low <= prev_close) if gap > 0 else (high >= prev_close)
            row["gap_fill_ratio"] = min(1.0, safe_div(filled, abs(open_[0] - prev_close)))
            row["gap_same_side_ratio"] = float(np.mean(same_side))
            row["gap_persistence"] = row["gap_same_side_ratio"]
            if fill_mask.any():
                fill_pos = int(np.argmax(fill_mask))
                row["gap_fill_time"] = normalized_pos(fill_pos, n_bars)
                row["gap_fill_time_raw"] = fill_pos
            else:
                row["gap_fill_time"] = np.nan
                row["gap_fill_time_raw"] = np.nan
            row["first_reversal_minute"] = normalized_pos(int(np.argmax(reversed_mask)), n_bars) if reversed_mask.any() else 1.0
        else:
            row["gap_fill_ratio"] = 0.0
            row["gap_same_side_ratio"] = np.nan
            row["gap_persistence"] = np.nan
            row["gap_fill_time"] = np.nan
            row["gap_fill_time_raw"] = np.nan
            row["first_reversal_minute"] = np.nan

    late_idx = idx_all[-60:]
    late_close = close[late_idx]
    late_open = open_[late_idx]
    late_high = high[late_idx]
    late_low = low[late_idx]
    late_ret = pct_change(late_close)
    late_ret[0] = late_close[0] / late_open[0] - 1.0 if len(late_close) and abs(late_open[0]) > EPS else np.nan
    row["late_up_bar_ratio"] = float(np.nanmean(late_ret > 0))
    row["late_realized_vol_5m"] = safe_std(late_ret)
    row["late_realized_vol_1m"] = realized_vol(late_ret)
    late_drop = rolling_return(late_close, 5)
    late_range = rolling_range(late_high, late_low, late_open, 5)
    row["late_max_5m_drop"] = float(np.nanmin(late_drop)) if np.isfinite(late_drop).any() else np.nan
    row["late_max_5m_range"] = float(np.nanmax(late_range)) if np.isfinite(late_range).any() else np.nan
    row["afternoon_minus_morning_ret"] = (
        row.get("afternoon_ret", np.nan) - row.get("morning_ret", np.nan)
        if np.isfinite(row.get("afternoon_ret", np.nan)) and np.isfinite(row.get("morning_ret", np.nan))
        else np.nan
    )
    row["last60_minus_first60_ret"] = (
        row.get("last_60m_ret", np.nan) - row.get("first_60m_ret", np.nan)
        if np.isfinite(row.get("last_60m_ret", np.nan)) and np.isfinite(row.get("first_60m_ret", np.nan))
        else np.nan
    )

    for col in FEATURE_COLUMNS:
        row.setdefault(col, np.nan)
    return row


def read_stock_file(path: Path, start_date: str | None, end_date: str | None) -> pl.DataFrame:
    schema = pl.read_parquet_schema(path)
    required = {"ts_code", "trade_time", "open", "close", "high", "low", "vol", "amount", "trade_date"}
    missing = sorted(required - set(schema))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df = pl.read_parquet(path, columns=[c for c in schema if c in required or c == "year"])
    if df.is_empty():
        return df
    df = df.with_columns(pl.col("trade_date").cast(pl.Utf8))
    if start_date:
        df = df.filter(pl.col("trade_date") >= str(start_date))
    if end_date:
        df = df.filter(pl.col("trade_date") <= str(end_date))
    if df.is_empty():
        return df

    dtype = df.schema["trade_time"]
    if dtype == pl.Datetime:
        dt_expr = pl.col("trade_time").alias("datetime")
    else:
        dt_expr = pl.col("trade_time").cast(pl.Utf8).str.to_datetime(strict=False).alias("datetime")
    df = (
        df.with_columns(dt_expr)
        .filter(pl.col("datetime").is_not_null())
        .with_columns(pl.col("datetime").dt.strftime("%H:%M").alias("minute_time"))
        .sort(["trade_date", "datetime"])
    )
    return df


def empty_output() -> pl.DataFrame:
    return pl.DataFrame({"ts_code": pl.Series([], dtype=pl.Utf8), "trade_date": pl.Series([], dtype=pl.Utf8)})


def nan_to_null(df: pl.DataFrame) -> pl.DataFrame:
    float_cols = [c for c, dtype in zip(df.columns, df.dtypes) if dtype in (pl.Float32, pl.Float64)]
    if not float_cols:
        return df
    return df.with_columns(
        pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
        for c in float_cols
    )


def build_stock_features(
    path: Path,
    start_date: str | None,
    end_date: str | None,
    min_continuous_bars: int,
) -> tuple[dict[str, Any], pl.DataFrame]:
    ts_code = path.stem
    try:
        df = read_stock_file(path, start_date, end_date)
        if df.is_empty():
            return {"ts_code": ts_code, "status": "empty", "rows": 0}, empty_output()
        ts_from_data = df.select(pl.col("ts_code").drop_nulls().first()).item() or ts_code
        rows = []
        skipped_days = 0
        prev_close: float | None = None
        for key, day in df.group_by("trade_date", maintain_order=True):
            trade_date = key[0] if isinstance(key, tuple) else key
            row = build_one_day(str(ts_from_data), str(trade_date), day, min_continuous_bars, prev_close)
            cleaned = (
                day.filter(pl.col("minute_time") != "09:30")
                .sort("datetime")
                .select(pl.col("close").cast(pl.Float64))
                .drop_nulls()
            )
            if cleaned.height:
                candidate = cleaned.get_column("close")[-1]
                if candidate is not None and np.isfinite(candidate) and float(candidate) > EPS:
                    prev_close = float(candidate)
            if row is None:
                skipped_days += 1
            else:
                rows.append(row)

        out = pl.DataFrame(rows, infer_schema_length=None) if rows else empty_output()
        if rows:
            out = out.select(
                ["ts_code", "trade_date", "minute_bar_count", "minute_total_amount", "minute_total_vol", "minute_daily_vwap"]
                + FEATURE_COLUMNS
            )
            out = nan_to_null(out)
        return {
            "ts_code": ts_code,
            "status": "ok",
            "rows": out.height,
            "skipped_days": skipped_days,
            "date_min": out.get_column("trade_date").min() if out.height and "trade_date" in out.columns else None,
            "date_max": out.get_column("trade_date").max() if out.height and "trade_date" in out.columns else None,
        }, out
    except Exception as exc:  # noqa: BLE001
        return {"ts_code": ts_code, "status": "failed", "rows": 0, "error": repr(exc)}, empty_output()


def process_stock(
    path: Path,
    out_dir: Path,
    start_date: str | None,
    end_date: str | None,
    overwrite: bool,
    min_continuous_bars: int,
    write_by_stock: bool = True,
    return_data: bool = False,
) -> dict[str, Any]:
    ts_code = path.stem
    by_stock_dir = out_dir / BY_STOCK_DIRNAME
    out_path = by_stock_dir / f"{ts_code}.parquet"
    if write_by_stock and out_path.exists() and not overwrite:
        return {"ts_code": ts_code, "status": "skipped_exists", "rows": None, "path": str(out_path)}

    summary, out = build_stock_features(path, start_date, end_date, min_continuous_bars)
    summary["path"] = str(out_path) if write_by_stock else ""
    if write_by_stock:
        out.write_parquet(out_path)
    if return_data:
        summary["data"] = out
    return summary


def concat_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    frames = [df for df in frames if df is not None and not df.is_empty()]
    if not frames:
        return empty_output()
    return pl.concat(frames, how="vertical_relaxed").sort(["ts_code", "trade_date"])


def chunks(items: list[Path], chunk_size: int) -> list[list[Path]]:
    if chunk_size <= 0:
        raise ValueError("--stock-chunk-size must be positive")
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def process_panel_batch(
    paths: list[Path],
    start_date: str | None,
    end_date: str | None,
    min_continuous_bars: int,
    progress_prefix: str = "",
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    frames: list[pl.DataFrame] = []
    for idx, path in enumerate(paths, start=1):
        result = process_stock(path, Path("."), start_date, end_date, True, min_continuous_bars, False, True)
        data = result.pop("data", None)
        if data is not None and not data.is_empty():
            frames.append(data)
        summaries.append(result)
        print(
            f"{progress_prefix}[{idx}/{len(paths)}] {path.stem} {result.get('status')} rows={result.get('rows')}",
            flush=True,
        )
    return {
        "summaries": summaries,
        "frame": concat_frames(frames),
        "stock_count": len(paths),
        "row_count": int(sum(int(s.get("rows") or 0) for s in summaries)),
    }


def process_paths_for_panel(
    paths: list[Path],
    start_date: str | None,
    end_date: str | None,
    min_continuous_bars: int,
    workers: int,
    task_chunk_size: int,
    progress_prefix: str = "",
) -> tuple[list[dict[str, Any]], list[pl.DataFrame]]:
    summaries: list[dict[str, Any]] = []
    frames: list[pl.DataFrame] = []

    if workers <= 1:
        for idx, path in enumerate(paths, start=1):
            result = process_stock(path, Path("."), start_date, end_date, True, min_continuous_bars, False, True)
            data = result.pop("data", None)
            if data is not None and not data.is_empty():
                frames.append(data)
            summaries.append(result)
            print(
                f"{progress_prefix}[{idx}/{len(paths)}] {path.stem} {result.get('status')} rows={result.get('rows')}",
                flush=True,
            )
    else:
        path_batches = chunks(paths, task_chunk_size)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    process_panel_batch,
                    batch,
                    start_date,
                    end_date,
                    min_continuous_bars,
                    f"{progress_prefix}[batch {batch_idx}/{len(path_batches)}] ",
                )
                for batch_idx, batch in enumerate(path_batches, start=1)
            ]
            for idx, fut in enumerate(as_completed(futures), start=1):
                result = fut.result()
                batch_summaries = result["summaries"]
                frame = result["frame"]
                if frame is not None and not frame.is_empty():
                    frames.append(frame)
                summaries.extend(batch_summaries)
                ok_count = sum(1 for s in batch_summaries if s.get("status") == "ok")
                fail_count = sum(1 for s in batch_summaries if s.get("status") == "failed")
                print(
                    f"{progress_prefix}[batch {idx}/{len(futures)}] stocks={result['stock_count']} "
                    f"ok={ok_count} failed={fail_count} rows={result['row_count']}",
                    flush=True,
                )
    return summaries, frames


def write_memory_panel(
    paths: list[Path],
    out_dir: Path,
    panel_name: str,
    start_date: str | None,
    end_date: str | None,
    min_continuous_bars: int,
    workers: int,
    task_chunk_size: int,
) -> tuple[list[dict[str, Any]], Path]:
    summaries, frames = process_paths_for_panel(paths, start_date, end_date, min_continuous_bars, workers, task_chunk_size)
    panel = concat_frames(frames)
    panel_path = out_dir / panel_name
    panel.write_parquet(panel_path)
    return summaries, panel_path


def write_chunked_panel(
    paths: list[Path],
    out_dir: Path,
    panel_name: str,
    start_date: str | None,
    end_date: str | None,
    min_continuous_bars: int,
    workers: int,
    chunk_size: int,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], Path, list[Path]]:
    chunk_dir = out_dir / CHUNK_DIRNAME
    chunk_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict[str, Any]] = []
    chunk_paths: list[Path] = []
    path_chunks = chunks(paths, chunk_size)
    for chunk_idx, chunk_paths_in in enumerate(path_chunks, start=1):
        chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.parquet"
        if chunk_path.exists() and not overwrite:
            print(f"[chunk {chunk_idx}/{len(path_chunks)}] skip existing {chunk_path}", flush=True)
            chunk_paths.append(chunk_path)
            continue
        print(f"[chunk {chunk_idx}/{len(path_chunks)}] stocks={len(chunk_paths_in)}", flush=True)
        summaries, frames = process_paths_for_panel(
            chunk_paths_in,
            start_date,
            end_date,
            min_continuous_bars,
            workers,
            max(1, math.ceil(len(chunk_paths_in) / max(workers, 1))),
            progress_prefix=f"[chunk {chunk_idx}/{len(path_chunks)}] ",
        )
        all_summaries.extend(summaries)
        chunk_panel = concat_frames(frames)
        chunk_panel.write_parquet(chunk_path)
        chunk_paths.append(chunk_path)
        print(f"[chunk {chunk_idx}/{len(path_chunks)}] saved rows={chunk_panel.height} path={chunk_path}", flush=True)

    panel_path = out_dir / panel_name
    scans = [pl.scan_parquet(p) for p in chunk_paths if p.exists()]
    if scans:
        pl.concat(scans, how="vertical_relaxed").collect().sort(["ts_code", "trade_date"]).write_parquet(panel_path)
    else:
        empty_output().write_parquet(panel_path)
    return all_summaries, panel_path, chunk_paths


def save_panel(out_dir: Path, summaries: list[dict[str, Any]], panel_name: str = PANEL_FILENAME) -> Path:
    frames = []
    for row in summaries:
        if row.get("status") not in {"ok", "skipped_exists"}:
            continue
        path = row.get("path")
        if path and Path(path).exists():
            frames.append(pl.scan_parquet(path))
    panel_path = out_dir / panel_name
    if frames:
        pl.concat(frames, how="vertical_relaxed").collect().sort(["ts_code", "trade_date"]).write_parquet(panel_path)
    else:
        empty_output().write_parquet(panel_path)
    return panel_path


def main() -> None:
    args = parse_args()
    started = time.time()
    by_stock_dir = args.out_dir / BY_STOCK_DIRNAME
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.output_mode == "by_stock":
        by_stock_dir.mkdir(parents=True, exist_ok=True)

    paths = stock_paths(args.raw_dir, args.stocks)
    panel_path = args.out_dir / args.panel_name
    if args.output_mode in {"chunked", "memory"} and panel_path.exists() and not args.overwrite:
        raise FileExistsError(f"{panel_path} exists. Pass --overwrite to rebuild it.")

    print(f"Minute feature build range: {args.start_date or 'min'} ~ {args.end_date or 'max'}", flush=True)
    print(f"Input files: {len(paths)}", flush=True)
    print(f"Output mode: {args.output_mode}", flush=True)
    if args.output_mode == "by_stock":
        print(f"Output by-stock dir: {by_stock_dir}", flush=True)
    else:
        print(f"Output panel: {panel_path}", flush=True)
    print("Engine: polars+numpy", flush=True)

    summaries: list[dict[str, Any]] = []
    chunk_paths: list[Path] = []
    if args.output_mode == "memory":
        summaries, panel_path = write_memory_panel(
            paths,
            args.out_dir,
            args.panel_name,
            args.start_date,
            args.end_date,
            args.min_continuous_bars,
            args.workers,
            args.stock_chunk_size,
        )
    elif args.output_mode == "chunked":
        summaries, panel_path, chunk_paths = write_chunked_panel(
            paths,
            args.out_dir,
            args.panel_name,
            args.start_date,
            args.end_date,
            args.min_continuous_bars,
            args.workers,
            args.stock_chunk_size,
            args.overwrite,
        )
    else:
        if args.workers <= 1:
            for idx, path in enumerate(paths, start=1):
                result = process_stock(path, args.out_dir, args.start_date, args.end_date, args.overwrite, args.min_continuous_bars)
                summaries.append(result)
                print(f"[{idx}/{len(paths)}] {path.stem} {result.get('status')} rows={result.get('rows')}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [
                    ex.submit(process_stock, path, args.out_dir, args.start_date, args.end_date, args.overwrite, args.min_continuous_bars)
                    for path in paths
                ]
                for idx, fut in enumerate(as_completed(futures), start=1):
                    result = fut.result()
                    summaries.append(result)
                    print(f"[{idx}/{len(paths)}] {result.get('ts_code')} {result.get('status')} rows={result.get('rows')}", flush=True)

    summary_path = args.out_dir / args.report_name
    pl.DataFrame(summaries, infer_schema_length=None).write_csv(summary_path)
    saved_panel_path = None
    if args.output_mode == "by_stock" and args.save_panel:
        saved_panel_path = save_panel(args.out_dir, summaries, args.panel_name)
    elif args.output_mode in {"chunked", "memory"}:
        saved_panel_path = panel_path
    meta = {
        "raw_dir": str(args.raw_dir),
        "out_dir": str(args.out_dir),
        "stock_count": len(paths),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "output_mode": args.output_mode,
        "stock_chunk_size": args.stock_chunk_size if args.output_mode == "chunked" else None,
        "save_panel": args.save_panel,
        "panel_path": str(saved_panel_path) if saved_panel_path else None,
        "chunk_count": len(chunk_paths) if chunk_paths else None,
        "engine": "polars+numpy",
        "vol_unit": "shares",
        "amount_unit": "yuan",
        "excluded_continuous_bar": "09:30",
        "feature_count": len(FEATURE_COLUMNS),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (args.out_dir / "minute_feature_build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ok_count = sum(1 for x in summaries if x.get("status") in {"ok", "skipped_exists"})
    fail_count = sum(1 for x in summaries if x.get("status") == "failed")
    print("[done]", flush=True)
    print(f"ok_or_skipped={ok_count} failed={fail_count} elapsed_seconds={meta['elapsed_seconds']}", flush=True)
    print("[SAVE]", summary_path, flush=True)
    if saved_panel_path:
        print("[SAVE]", saved_panel_path, flush=True)


if __name__ == "__main__":
    main()
