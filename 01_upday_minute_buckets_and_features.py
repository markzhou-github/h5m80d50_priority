#!/usr/bin/env python
# coding: utf-8
"""Upday minute raw bucket files and build minute feature bucket files.

New minute architecture:
  - raw minute data is stored in 20 deterministic stock-code buckets
  - minute features are also stored by bucket
  - minute features have no cross-sectional features and no lags

Bucket files:
  data/minute_raw_buckets/bucket_00.parquet
  processed/minute_feature_buckets/bucket_00.parquet

This script can:
  1. download recent 1-minute data by stock from Tushare
  2. merge/overwrite the upday window inside raw bucket files
  3. immediately build minute feature bucket files from the updated raw bucket

Use --no-download to rebuild feature buckets from existing raw buckets.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed as thread_as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import tushare as ts
from numpy.lib.stride_tricks import sliding_window_view

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TUSHARE_TOKEN  # noqa: E402

# ---------------------------------------------------------------------------
# Embedded minute upday / feature helpers.
# Kept self-contained so this production script does not depend on old minute scripts.
# ---------------------------------------------------------------------------
def format_elapsed(started: float) -> str:
    elapsed = max(0.0, time.time() - started)
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    return f"{elapsed / 60.0:.1f}m"


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

EPS = 1e-12
EXPECTED_COLS = ["ts_code", "trade_time", "open", "close", "high", "low", "vol", "amount"]
THREAD_LOCAL = threading.local()

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
    # Direction-preserving counterparts to unsigned intensity features.
    "intraday_direction", "signed_trend_efficiency", "signed_realized_vol_1m",
    "up_down_bar_imbalance", "late_up_down_bar_imbalance", "late_signed_trend_efficiency",
    "mean_signed_vwap_distance", "max_positive_vwap_distance", "max_negative_vwap_distance",
    "last30_signed_vwap_distance", "last60_signed_vwap_distance", "vwap_side_imbalance",
    "high_low_excursion_imbalance", "drawup_drawdown_imbalance",
    "signed_jump_variation", "positive_jump_share", "negative_jump_share",
    "large_bar_return_balance", "amount_weighted_return", "volume_return_corr",
    "morning_signed_efficiency", "afternoon_signed_efficiency",
]

def normalize_ts_code(code: str) -> str:
    code = str(code).strip().upper()
    if not code or "." in code:
        return code
    return f"{code}.SH" if code.startswith(("6", "9")) else f"{code}.SZ"

def normalize_trade_date(value: str | int | pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace("-", "")
    if not text:
        return None
    if len(text) != 8:
        raise ValueError(f"date must be YYYYMMDD or YYYY-MM-DD, got {value!r}")
    datetime.strptime(text, "%Y%m%d")
    return text

def make_tushare_pro(token: str):
    token = str(token or "").strip()
    if not token:
        raise ValueError("Tushare token is empty. Pass --token or set TUSHARE_TOKEN in config.py.")
    return ts.pro_api(token)

def get_thread_pro(token: str):
    pro = getattr(THREAD_LOCAL, "pro", None)
    if pro is None:
        pro = make_tushare_pro(token)
        THREAD_LOCAL.pro = pro
    return pro

def local_last_trade_date(pro) -> str:
    today = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=45)).strftime("%Y%m%d")
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=today)
    open_days = cal[cal["is_open"] == 1]["cal_date"].astype(str).sort_values()
    if open_days.empty:
        raise ValueError("No open trade day found.")
    return open_days.iloc[-1]

def local_trade_date_before(pro, end_date: str, trade_days: int) -> str:
    if trade_days < 1:
        raise ValueError("trade_days must be >= 1")
    end = normalize_trade_date(end_date)
    end_ts = datetime.strptime(end, "%Y%m%d")
    start = (end_ts - timedelta(days=max(90, trade_days * 5))).strftime("%Y%m%d")
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=(end_ts - timedelta(days=1)).strftime("%Y%m%d"))
    open_days = cal[cal["is_open"] == 1]["cal_date"].astype(str).sort_values(ascending=False)
    if len(open_days) < trade_days:
        raise ValueError(f"Only found {len(open_days)} open trade days before {end}")
    return open_days.iloc[trade_days - 1]

def get_config_end_date(token: str, pro=None) -> str:
    try:
        from config_date import end_date as config_end_date  # noqa: PLC0415

        return normalize_trade_date(config_end_date)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not import config_date.end_date ({exc}); using local Tushare calendar.", flush=True)
        return local_last_trade_date(pro or make_tushare_pro(token))

def resolve_window(
    start_upday: str | None,
    end_upday: str | None,
    lookback_trade_days: int,
    token: str,
    pro=None,
) -> tuple[str, str]:
    need_calendar = end_upday is None or start_upday is None
    pro = (pro or make_tushare_pro(token)) if need_calendar else pro
    end = normalize_trade_date(end_upday) if end_upday else get_config_end_date(token, pro)
    if start_upday:
        start = normalize_trade_date(start_upday)
    elif lookback_trade_days == 1:
        start = end
    else:
        # Inclusive download window: lookback_trade_days=20 means 20 trade
        # dates total, including end_upday, so start is 19 open days before end.
        start = local_trade_date_before(pro, end, trade_days=lookback_trade_days - 1)
    if start > end:
        raise ValueError(f"start_upday {start} cannot be after end_upday {end}")
    return start, end

def upday_to_datetime_window(start_upday: str, end_upday: str, start_time: str, end_time: str) -> tuple[str, str]:
    return (
        f"{start_upday[:4]}-{start_upday[4:6]}-{start_upday[6:]} {start_time}",
        f"{end_upday[:4]}-{end_upday[4:6]}-{end_upday[6:]} {end_time}",
    )

def standardize_1min_frame(df: pd.DataFrame, ts_code: str | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=EXPECTED_COLS + ["trade_date"])
    out = df.copy()
    if "ts_code" not in out.columns and ts_code:
        out["ts_code"] = ts_code
    missing = [c for c in EXPECTED_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing 1min columns: {missing}")
    out = out[EXPECTED_COLS].copy()
    out["ts_code"] = out["ts_code"].astype(str).map(normalize_ts_code)
    out["trade_time"] = pd.to_datetime(out["trade_time"], errors="coerce")
    out = out.dropna(subset=["trade_time"])
    for col in ["open", "close", "high", "low", "vol", "amount"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = (
        out.sort_values(["ts_code", "trade_time"])
        .drop_duplicates(["ts_code", "trade_time"], keep="last")
        .reset_index(drop=True)
    )
    out["trade_date"] = out["trade_time"].dt.strftime("%Y%m%d")
    return out

def download_one_stock_1min_tushare(
    pro,
    ts_code: str,
    start_dt: str,
    end_dt: str,
    retry: int,
    sleep_sec: float,
    overlap_minutes: int,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_dt)
    cur_end = pd.Timestamp(end_dt)
    all_parts: list[pd.DataFrame] = []

    while cur_end >= start_ts:
        start_str = start_ts.strftime("%Y-%m-%d %H:%M:%S")
        end_str = cur_end.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts_code}] request {start_str} -> {end_str}", flush=True)

        df = None
        last_err = None
        for attempt in range(1, retry + 1):
            try:
                df = pro.stk_mins(ts_code=ts_code, freq="1min", start_date=start_str, end_date=end_str)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                wait = sleep_sec * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                print(f"[{ts_code}] attempt {attempt}/{retry} failed: {exc}; sleep {wait:.2f}s", flush=True)
                time.sleep(wait)
        if df is None:
            raise RuntimeError(f"[{ts_code}] failed after {retry} retries: {last_err}")
        if df.empty:
            print(f"[{ts_code}] empty result, stop.", flush=True)
            break

        part = standardize_1min_frame(df, ts_code)
        if part.empty:
            print(f"[{ts_code}] all rows invalid after parsing, stop.", flush=True)
            break
        all_parts.append(part)

        first_time = part["trade_time"].iloc[0]
        last_time = part["trade_time"].iloc[-1]
        print(f"[{ts_code}] got {len(part):,} rows, {first_time} -> {last_time}", flush=True)

        if first_time <= start_ts:
            break
        next_end = first_time + pd.Timedelta(minutes=overlap_minutes)
        if next_end >= cur_end:
            next_end = first_time - pd.Timedelta(minutes=1)
        cur_end = next_end
        time.sleep(sleep_sec + random.uniform(0.0, 0.2))

    if not all_parts:
        return pd.DataFrame(columns=EXPECTED_COLS + ["trade_date"])
    return standardize_1min_frame(pd.concat(all_parts, ignore_index=True), ts_code)

def process_download_stock(
    idx: int,
    total: int,
    ts_code: str,
    args: argparse.Namespace,
    token: str,
    start_dt: str,
    end_dt: str,
) -> dict[str, Any]:
    print("=" * 100, flush=True)
    print(f"[download {idx}/{total}] {ts_code}", flush=True)
    try:
        pro = get_thread_pro(token)
        data = download_one_stock_1min_tushare(
            pro=pro,
            ts_code=ts_code,
            start_dt=start_dt,
            end_dt=end_dt,
            retry=args.retry,
            sleep_sec=args.sleep_sec,
            overlap_minutes=args.overlap_minutes,
        )
        return {
            "ts_code": ts_code,
            "status": "downloaded",
            "rows": len(data),
            "date_min": None if data.empty else data["trade_date"].min(),
            "date_max": None if data.empty else data["trade_date"].max(),
            "data": data,
        }
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[{ts_code}] FAILED: {exc}", flush=True)
        traceback.print_exc()
        return {"ts_code": ts_code, "status": "failed", "rows": 0, "message": str(exc), "data": pd.DataFrame()}

def download_upday_frame(
    stock_list: list[str],
    args: argparse.Namespace,
    token: str,
    start_dt: str,
    end_dt: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    started = time.time()
    rows = []
    frames = []
    if not args.download:
        return pd.DataFrame(columns=EXPECTED_COLS + ["trade_date"]), rows
    if args.download_workers <= 1:
        for i, ts_code in enumerate(stock_list, 1):
            res = process_download_stock(i, len(stock_list), ts_code, args, token, start_dt, end_dt)
            data = res.pop("data", pd.DataFrame())
            if not data.empty:
                frames.append(data)
            rows.append(res)
            if i == 1 or i % 25 == 0 or i == len(stock_list):
                ok_count = sum(1 for row in rows if row.get("status") == "downloaded")
                fail_count = sum(1 for row in rows if row.get("status") == "failed")
                row_count = sum(int(row.get("rows") or 0) for row in rows)
                print(
                    f"[download-progress] {i}/{len(stock_list)} ok={ok_count} "
                    f"failed={fail_count} rows={row_count:,} elapsed={format_elapsed(started)}",
                    flush=True,
                )
    else:
        with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
            futures = [
                executor.submit(process_download_stock, i, len(stock_list), ts_code, args, token, start_dt, end_dt)
                for i, ts_code in enumerate(stock_list, 1)
            ]
            for done_count, fut in enumerate(thread_as_completed(futures), 1):
                res = fut.result()
                data = res.pop("data", pd.DataFrame())
                if not data.empty:
                    frames.append(data)
                rows.append(res)
                if done_count == 1 or done_count % 25 == 0 or done_count == len(futures):
                    ok_count = sum(1 for row in rows if row.get("status") == "downloaded")
                    fail_count = sum(1 for row in rows if row.get("status") == "failed")
                    row_count = sum(int(row.get("rows") or 0) for row in rows)
                    print(
                        f"[download-progress] {done_count}/{len(futures)} ok={ok_count} "
                        f"failed={fail_count} rows={row_count:,} elapsed={format_elapsed(started)}",
                        flush=True,
                    )
    downloaded = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EXPECTED_COLS + ["trade_date"])
    return downloaded, rows

def clean_raw_panel(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    return (
        df.with_columns(
            pl.col("ts_code").cast(pl.Utf8),
            pl.col("trade_time").cast(pl.Datetime),
            pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", ""),
            pl.col(["open", "close", "high", "low", "vol", "amount"]).cast(pl.Float64),
        )
        .drop_nulls(["ts_code", "trade_time", "trade_date"])
        .with_columns(pl.col("trade_time").alias("datetime"))
        .with_columns(pl.col("datetime").dt.strftime("%H:%M").alias("minute_time"))
        .unique(["ts_code", "trade_time"], keep="last")
        .sort(["ts_code", "trade_date", "datetime"])
    )

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
    direction = float(np.sign(full_path_ret)) if np.isfinite(full_path_ret) else np.nan
    row["intraday_direction"] = direction
    row["signed_trend_efficiency"] = (
        full_path_ret / path_abs_sum if np.isfinite(full_path_ret) and path_abs_sum > EPS else 0.0
    )
    row["signed_realized_vol_1m"] = (
        direction * row["realized_vol_1m"]
        if np.isfinite(direction) and np.isfinite(row["realized_vol_1m"])
        else np.nan
    )
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
    row["up_down_bar_imbalance"] = row["up_bar_ratio"] - row["down_bar_ratio"]
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
    row["mean_signed_vwap_distance"] = safe_mean(vwap_distance)
    row["max_positive_vwap_distance"] = (
        float(np.nanmax(vwap_distance)) if np.isfinite(vwap_distance).any() else np.nan
    )
    row["max_negative_vwap_distance"] = (
        float(np.nanmin(vwap_distance)) if np.isfinite(vwap_distance).any() else np.nan
    )
    row["last30_signed_vwap_distance"] = safe_mean(vwap_distance[idx_all[-30:]])
    row["last60_signed_vwap_distance"] = safe_mean(vwap_distance[idx_all[-60:]])
    row["vwap_side_imbalance"] = float(
        np.nanmean(close > cum_vwap) - np.nanmean(close < cum_vwap)
    )
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
    upside_excursion = float(np.nanmax(high / open_[0] - 1.0)) if abs(open_[0]) > EPS else np.nan
    downside_excursion = float(np.nanmin(low / open_[0] - 1.0)) if abs(open_[0]) > EPS else np.nan
    excursion_denom = max(upside_excursion, 0.0) + max(-downside_excursion, 0.0)
    row["high_low_excursion_imbalance"] = safe_div(
        max(upside_excursion, 0.0) - max(-downside_excursion, 0.0), excursion_denom
    )
    running_trough = np.minimum.accumulate(close)
    drawup = np.where(running_trough > EPS, close / running_trough - 1.0, np.nan)
    max_drawup = float(np.nanmax(drawup)) if np.isfinite(drawup).any() else np.nan
    max_drawdown_abs = (
        abs(row["intraday_max_drawdown"])
        if np.isfinite(row["intraday_max_drawdown"])
        else np.nan
    )
    row["drawup_drawdown_imbalance"] = safe_div(
        max_drawup - max_drawdown_abs, max_drawup + max_drawdown_abs
    )
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
    row["signed_jump_variation"] = (
        row["jump_variation"] * row["realized_semivar_imbalance"]
        if np.isfinite(row["jump_variation"]) and np.isfinite(row["realized_semivar_imbalance"])
        else np.nan
    )
    row["positive_jump_share"] = safe_div(up_semivar, total_var)
    row["negative_jump_share"] = safe_div(down_semivar, total_var)
    finite_idx = np.flatnonzero(np.isfinite(path_ret))
    if len(finite_idx):
        top_idx = finite_idx[np.argsort(np.abs(path_ret[finite_idx]))[-min(10, len(finite_idx)):]]
        row["large_bar_return_balance"] = safe_div(
            float(np.sum(path_ret[top_idx])), float(np.sum(np.abs(path_ret[top_idx])))
        )
    else:
        row["large_bar_return_balance"] = np.nan
    valid_amount_ret = np.isfinite(path_ret) & np.isfinite(amount)
    row["amount_weighted_return"] = safe_div(
        float(np.sum(path_ret[valid_amount_ret] * amount[valid_amount_ret])),
        float(np.sum(amount[valid_amount_ret])),
    )
    row["volume_return_corr"] = safe_corr(path_ret, vol)
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
    row["morning_signed_efficiency"] = (
        np.sign(row["morning_ret"]) * row["morning_efficiency"]
        if np.isfinite(row.get("morning_ret", np.nan)) and np.isfinite(row["morning_efficiency"])
        else np.nan
    )
    row["afternoon_signed_efficiency"] = (
        np.sign(row["afternoon_ret"]) * row["afternoon_efficiency"]
        if np.isfinite(row.get("afternoon_ret", np.nan)) and np.isfinite(row["afternoon_efficiency"])
        else np.nan
    )
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
    late_down_bar_ratio = float(np.nanmean(late_ret < 0))
    row["late_up_down_bar_imbalance"] = row["late_up_bar_ratio"] - late_down_bar_ratio
    row["late_realized_vol_5m"] = safe_std(late_ret)
    row["late_realized_vol_1m"] = realized_vol(late_ret)
    late_abs_sum = float(np.nansum(np.abs(late_ret)))
    late_total_ret = (
        late_close[-1] / late_open[0] - 1.0
        if len(late_close) and abs(late_open[0]) > EPS
        else np.nan
    )
    row["late_signed_trend_efficiency"] = (
        late_total_ret / late_abs_sum
        if np.isfinite(late_total_ret) and late_abs_sum > EPS
        else 0.0
    )
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

def build_stock_days_from_panel(stock_df: pl.DataFrame, min_continuous_bars: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    summaries = []
    if stock_df.is_empty():
        return summaries, rows
    stock_df = stock_df.sort(["trade_date", "datetime"])
    ts_code_value = stock_df.select(pl.col("ts_code").drop_nulls().first()).item()
    ts_code = str(ts_code_value) if ts_code_value is not None else ""
    prev_close: float | None = None
    for key, day in stock_df.group_by("trade_date", maintain_order=True):
        trade_date = key[0] if isinstance(key, tuple) else key
        row = build_one_day(str(ts_code), str(trade_date), day, min_continuous_bars, prev_close)
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
            summaries.append({"ts_code": str(ts_code), "trade_date": str(trade_date), "status": "skipped_low_bars"})
        else:
            rows.append(row)
            summaries.append({"ts_code": str(ts_code), "trade_date": str(trade_date), "status": "ok"})
    return summaries, rows

def nan_to_null(df: pl.DataFrame) -> pl.DataFrame:
    float_cols = [c for c, dtype in zip(df.columns, df.dtypes) if dtype in (pl.Float32, pl.Float64)]
    if not float_cols:
        return df
    return df.with_columns(
        pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
        for c in float_cols
    )


RAW_BUCKET_DIR = PROJECT_ROOT / "data" / "minute_raw_buckets"
FEATURE_BUCKET_DIR = PROJECT_ROOT / "processed" / "minute_feature_buckets"
REPORT_DIR = PROJECT_ROOT / "processed" / "minute_bucket_upday_report"
CON_FILE = PROJECT_ROOT / "csi1500con.csv"
RAW_COLS = EXPECTED_COLS + ["trade_date"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upday minute bucket raw files and build minute feature buckets.")
    p.add_argument("--con-file", type=Path, default=CON_FILE)
    p.add_argument("--raw-bucket-dir", type=Path, default=RAW_BUCKET_DIR)
    p.add_argument("--feature-bucket-dir", type=Path, default=FEATURE_BUCKET_DIR)
    p.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    p.add_argument("--bucket-id-col", default="minute_bucket_id")
    p.add_argument("--bucket-file-col", default="minute_bucket_file")
    p.add_argument("--stock-col", default=None)

    p.add_argument("--start-upday", default=None, help="YYYYMMDD inclusive download/update start.")
    p.add_argument("--end-upday", default=None, help="YYYYMMDD inclusive download/update end. Defaults to config_date.end_date.")
    p.add_argument("--lookback-trade-days", type=int, default=30)
    p.add_argument("--source-start-date", default=None, help="YYYYMMDD inclusive feature source start.")
    p.add_argument("--source-end-date", default=None, help="YYYYMMDD inclusive feature source end. Defaults to end-upday.")
    p.add_argument("--start-time", default="09:30:00")
    p.add_argument("--end-time", default="15:30:00")

    p.add_argument("--stocks", nargs="*", default=[], help="Optional stock subset.")
    p.add_argument("--max-stocks", type=int, default=None)
    p.add_argument("--token", default=None)
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--download-workers", type=int, default=4)
    p.add_argument("--retry", type=int, default=5)
    p.add_argument("--sleep-sec", type=float, default=0.35)
    p.add_argument("--overlap-minutes", type=int, default=10)

    p.add_argument("--raw-compression", default="zstd", choices=["zstd", "snappy", "lz4", "uncompressed"])
    p.add_argument("--feature-compression", default="zstd", choices=["zstd", "snappy", "lz4", "uncompressed"])
    p.add_argument("--min-continuous-bars", type=int, default=180)
    p.add_argument("--overwrite-features", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-raw-write", action="store_true", help="Build feature buckets but do not write raw buckets.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def atomic_write_parquet(df: pl.DataFrame, path: Path, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp_path, compression=compression)
    os.replace(tmp_path, path)


def detect_stock_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"--stock-col {requested!r} not found in {df.columns.tolist()}")
        return requested
    for col in ["con_code", "ts_code", "stock_code", "code"]:
        if col in df.columns:
            return col
    raise ValueError("Cannot detect stock column. Pass --stock-col.")


def load_constituents(args: argparse.Namespace) -> pd.DataFrame:
    con = pd.read_csv(args.con_file, dtype=str)
    stock_col = detect_stock_col(con, args.stock_col)
    required = [args.bucket_id_col, args.bucket_file_col]
    missing = [c for c in required if c not in con.columns]
    if missing:
        raise ValueError(f"{args.con_file} missing bucket columns {missing}. Run assign_minute_buckets.py first.")
    con["ts_code"] = con[stock_col].map(normalize_ts_code)
    con = con[con["ts_code"] != ""].drop_duplicates("ts_code", keep="last").copy()
    con[args.bucket_id_col] = pd.to_numeric(con[args.bucket_id_col], errors="raise").astype("int16")
    con[args.bucket_file_col] = con[args.bucket_file_col].astype(str)

    if args.stocks:
        selected = {normalize_ts_code(s) for s in args.stocks}
        con = con[con["ts_code"].isin(selected)].copy()
    con = con.sort_values("ts_code")
    if args.max_stocks:
        con = con.head(args.max_stocks).copy()
    return con[["ts_code", args.bucket_id_col, args.bucket_file_col]].copy()


def read_raw_bucket(path: Path) -> pl.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame()
    schema = pl.read_parquet_schema(path)
    missing = sorted(set(RAW_COLS) - set(schema))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return clean_raw_panel(pl.read_parquet(path, columns=RAW_COLS))


def raw_for_save(raw: pl.DataFrame) -> pl.DataFrame:
    if raw.is_empty():
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in RAW_COLS})
    return (
        raw.select(RAW_COLS)
        .with_columns(pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", ""))
        .sort(["ts_code", "trade_time"])
    )


def downloaded_bucket_frame(downloaded: pd.DataFrame, bucket_stocks: set[str]) -> pl.DataFrame:
    if downloaded.empty:
        return pl.DataFrame()
    pdf = downloaded[downloaded["ts_code"].isin(bucket_stocks)].copy()
    if pdf.empty:
        return pl.DataFrame()
    return clean_raw_panel(pl.from_pandas(pdf))


def merge_raw_bucket(
    old_raw: pl.DataFrame,
    new_raw: pl.DataFrame,
    update_stocks: set[str],
    start_upday: str,
    end_upday: str,
) -> pl.DataFrame:
    frames = []
    if not old_raw.is_empty():
        if update_stocks:
            old_raw = old_raw.filter(
                ~(
                    pl.col("ts_code").is_in(list(update_stocks))
                    & (pl.col("trade_date") >= start_upday)
                    & (pl.col("trade_date") <= end_upday)
                )
            )
        frames.append(old_raw)
    if not new_raw.is_empty():
        frames.append(new_raw)
    if not frames:
        return pl.DataFrame()
    return clean_raw_panel(pl.concat(frames, how="diagonal_relaxed"))


def build_feature_bucket(raw: pl.DataFrame, source_start: str, source_end: str, min_continuous_bars: int) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    if raw.is_empty():
        return pl.DataFrame(), []
    source = raw.filter((pl.col("trade_date") >= source_start) & (pl.col("trade_date") <= source_end))
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    if source.is_empty():
        return pl.DataFrame(), summaries
    for stock_df in source.sort(["ts_code", "trade_date", "datetime"]).partition_by("ts_code", maintain_order=True):
        batch_summaries, batch_rows = build_stock_days_from_panel(stock_df, min_continuous_bars)
        summaries.extend(batch_summaries)
        rows.extend(batch_rows)
    if not rows:
        return pl.DataFrame(), summaries
    feature = pl.DataFrame(rows, infer_schema_length=None).select(
        ["ts_code", "trade_date", "minute_bar_count", "minute_total_amount", "minute_total_vol", "minute_daily_vwap"]
        + FEATURE_COLUMNS
    )
    return nan_to_null(feature).sort(["ts_code", "trade_date"]), summaries


def write_empty_feature(path: Path, compression: str) -> None:
    empty = pl.DataFrame({"ts_code": pl.Series([], dtype=pl.Utf8), "trade_date": pl.Series([], dtype=pl.Utf8)})
    atomic_write_parquet(empty, path, compression)


def process_bucket(
    bucket_file: str,
    bucket_stocks: pd.DataFrame,
    downloaded: pd.DataFrame,
    args: argparse.Namespace,
    start_upday: str,
    end_upday: str,
    source_start: str,
    source_end: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.time()
    stock_set = set(bucket_stocks["ts_code"].tolist())
    raw_path = args.raw_bucket_dir / bucket_file
    feature_path = args.feature_bucket_dir / bucket_file
    print(f"[bucket {bucket_file}] stocks={len(stock_set)}", flush=True)

    old_raw = read_raw_bucket(raw_path)
    new_raw = downloaded_bucket_frame(downloaded, stock_set)
    update_stocks = set(new_raw.get_column("ts_code").unique().to_list()) if not new_raw.is_empty() else set()
    merged = merge_raw_bucket(old_raw, new_raw, update_stocks, start_upday, end_upday)

    if not args.skip_raw_write:
        atomic_write_parquet(raw_for_save(merged), raw_path, args.raw_compression)

    feature, feature_summaries = build_feature_bucket(merged, source_start, source_end, args.min_continuous_bars)
    if feature.is_empty():
        write_empty_feature(feature_path, args.feature_compression)
    else:
        atomic_write_parquet(feature, feature_path, args.feature_compression)

    result = {
        "bucket_file": bucket_file,
        "stock_count": len(stock_set),
        "old_raw_rows": old_raw.height,
        "download_rows": new_raw.height,
        "merged_raw_rows": merged.height,
        "feature_rows": feature.height,
        "raw_path": str(raw_path),
        "feature_path": str(feature_path),
        "raw_date_min": merged.get_column("trade_date").min() if not merged.is_empty() else None,
        "raw_date_max": merged.get_column("trade_date").max() if not merged.is_empty() else None,
        "feature_date_min": feature.get_column("trade_date").min() if not feature.is_empty() else None,
        "feature_date_max": feature.get_column("trade_date").max() if not feature.is_empty() else None,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    print(
        f"[bucket {bucket_file}] old={old_raw.height:,} new={new_raw.height:,} "
        f"merged={merged.height:,} feature={feature.height:,} elapsed={format_elapsed(started)}",
        flush=True,
    )
    for row in feature_summaries:
        row["bucket_file"] = bucket_file
    return result, feature_summaries


def main() -> None:
    args = parse_args()
    started = time.time()
    token = args.token or TUSHARE_TOKEN
    needs_calendar = args.end_upday is None or args.start_upday is None
    pro = make_tushare_pro(token) if args.download and needs_calendar else None
    start_upday, end_upday = resolve_window(args.start_upday, args.end_upday, args.lookback_trade_days, token, pro)
    source_end = normalize_trade_date(args.source_end_date) if args.source_end_date else end_upday
    source_start = normalize_trade_date(args.source_start_date) if args.source_start_date else start_upday
    if source_start > source_end:
        raise ValueError(f"source-start-date {source_start} cannot be after source-end-date {source_end}")
    start_dt, end_dt = upday_to_datetime_window(start_upday, end_upday, args.start_time, args.end_time)

    con = load_constituents(args)
    stock_list = con["ts_code"].tolist()
    bucket_files = sorted(con[args.bucket_file_col].drop_duplicates().tolist())

    args.raw_bucket_dir.mkdir(parents=True, exist_ok=True)
    args.feature_bucket_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    print(f"[upday-window] {start_upday} ~ {end_upday}", flush=True)
    print(f"[source-window] {source_start} ~ {source_end}", flush=True)
    print(f"[datetime] {start_dt} ~ {end_dt}", flush=True)
    print(f"[stocks] {len(stock_list)}", flush=True)
    print(f"[buckets] {len(bucket_files)}", flush=True)
    print(f"[raw-bucket-dir] {args.raw_bucket_dir}", flush=True)
    print(f"[feature-bucket-dir] {args.feature_bucket_dir}", flush=True)

    if args.dry_run:
        return

    downloaded, download_rows = download_upday_frame(stock_list, args, token, start_dt, end_dt)
    print(f"[download] rows={len(downloaded):,}", flush=True)

    bucket_rows = []
    feature_rows = []
    for i, bucket_file in enumerate(bucket_files, 1):
        bucket_stocks = con[con[args.bucket_file_col] == bucket_file]
        print(f"[bucket-progress] {i}/{len(bucket_files)} {bucket_file}", flush=True)
        bucket_result, bucket_feature_rows = process_bucket(
            bucket_file=bucket_file,
            bucket_stocks=bucket_stocks,
            downloaded=downloaded,
            args=args,
            start_upday=start_upday,
            end_upday=end_upday,
            source_start=source_start,
            source_end=source_end,
        )
        bucket_rows.append(bucket_result)
        feature_rows.extend(bucket_feature_rows)

    download_report = args.report_dir / "minute_upday_download_summary.csv"
    bucket_report = args.report_dir / "minute_upday_bucket_summary.csv"
    feature_report = args.report_dir / "minute_upday_feature_summary.csv"
    pl.DataFrame(download_rows, infer_schema_length=None).write_csv(download_report)
    pl.DataFrame(bucket_rows, infer_schema_length=None).write_csv(bucket_report)
    pl.DataFrame(feature_rows, infer_schema_length=None).write_csv(feature_report)

    print("[done]", flush=True)
    print(f"elapsed_seconds={time.time() - started:.3f}", flush=True)
    print("[SAVE]", download_report, flush=True)
    print("[SAVE]", bucket_report, flush=True)
    print("[SAVE]", feature_report, flush=True)


if __name__ == "__main__":
    main()
