#!/usr/bin/env python
# coding: utf-8
"""
Upday CSI1500 margin_detail downloader.

Downloads date-first, then appends/updates existing per-stock margin_detail CSVs.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import tushare as ts

from config import STOCK_DATA_DIR, TUSHARE_TOKEN
from config_date import resolve_upday_window, trade_dates_between
from download_csi1500_margin_detail import REQUEST_INTERVAL, RETRY_DELAY, normalize


PROJECT_ROOT = Path(__file__).resolve().parent
UNIVERSE_CSV = PROJECT_ROOT / "csi1500con.csv"
BASE_DIR = Path(STOCK_DATA_DIR)
MARGIN_DIR = BASE_DIR / "margin_detail"
REPORT_DIR = BASE_DIR / "report"
MAX_RETRIES = 5


def load_stock_set() -> set[str]:
    df = pd.read_csv(UNIVERSE_CSV, dtype={"con_code": str})
    if "con_code" not in df.columns:
        raise ValueError(f"Missing con_code column in {UNIVERSE_CSV}")
    return set(df["con_code"].dropna().astype(str).str.strip())


def make_tushare_pro():
    token = str(TUSHARE_TOKEN or "").strip()
    if not token:
        raise ValueError("Tushare token is empty. Set TUSHARE_TOKEN in config.py.")
    ts.set_token(token)
    return ts.pro_api(token)


def call_with_retry(desc: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = fn()
            return pd.DataFrame() if df is None else df
        except Exception as exc:
            print(f"      [ERROR] {desc}, attempt {attempt}/{MAX_RETRIES}: {exc}", flush=True)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return pd.DataFrame()


def append_update_csv(path: Path, new_df: pd.DataFrame, keys: list[str]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    old_rows = 0
    if path.exists():
        old_df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        old_rows = len(old_df)
    else:
        old_df = pd.DataFrame()

    if new_df.empty:
        if not path.exists():
            old_df.to_csv(path, index=False, encoding="utf_8_sig")
        return {"old_rows": old_rows, "new_rows": 0, "final_rows": len(old_df), "status": "no_new_rows"}

    combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    for key in keys:
        if key in combined.columns:
            combined[key] = combined[key].astype(str)
    combined = combined.drop_duplicates(keys, keep="last")
    if "trade_date" in combined.columns:
        combined = combined.sort_values(["trade_date", "ts_code"])
    combined.to_csv(path, index=False, encoding="utf_8_sig")
    return {"old_rows": old_rows, "new_rows": len(new_df), "final_rows": len(combined), "status": "updated"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-upday", default=None)
    parser.add_argument("--end-upday", default=None)
    args = parser.parse_args()

    start_upday, end_upday = resolve_upday_window(args.start_upday, args.end_upday)
    trade_dates = trade_dates_between(start_upday, end_upday)
    stock_set = load_stock_set()

    pro = make_tushare_pro()

    MARGIN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Upday margin_detail download window: {start_upday} ~ {end_upday}")
    print(f"Trade dates: {len(trade_dates)}")
    print(f"Universe stock count: {len(stock_set)}")

    downloaded_parts = []
    for i, trade_date in enumerate(trade_dates, 1):
        print(f"[margin_detail {i}/{len(trade_dates)}] {trade_date}", flush=True)
        df = call_with_retry(
            f"{trade_date} margin_detail",
            lambda: pro.margin_detail(trade_date=trade_date),
        )
        df = normalize(df)
        if not df.empty and "ts_code" in df.columns:
            df = df[df["ts_code"].astype(str).isin(stock_set)].copy()
        downloaded_parts.append(df)
        print(f"    downloaded_rows={len(df)}", flush=True)
        time.sleep(REQUEST_INTERVAL)

    if downloaded_parts:
        window_df = pd.concat(downloaded_parts, ignore_index=True, sort=False)
    else:
        window_df = pd.DataFrame()

    summary = []
    touched = 0
    if not window_df.empty and "ts_code" in window_df.columns:
        window_df = normalize(window_df)
        for ts_code, g in window_df.groupby("ts_code", sort=True):
            out_path = MARGIN_DIR / f"{ts_code}.margin_detail.csv"
            result = append_update_csv(out_path, g, keys=["ts_code", "trade_date"])
            touched += 1
            summary.append({
                "trade_date": f"{start_upday}-{end_upday}",
                "ts_code": ts_code,
                **result,
                "output_file": str(out_path),
            })
    else:
        summary.append({
            "trade_date": f"{start_upday}-{end_upday}",
            "ts_code": "",
            "old_rows": 0,
            "new_rows": 0,
            "final_rows": 0,
            "status": "empty_window",
            "output_file": "",
        })
    print(f"    window_rows={len(window_df)} touched_stocks={touched}", flush=True)

    report_path = REPORT_DIR / f"download_csi1500_margin_detail_upday_{start_upday}_{end_upday}.csv"
    pd.DataFrame(summary).to_csv(report_path, index=False, encoding="utf_8_sig")
    print(f"\n[DONE] report saved: {report_path}")


if __name__ == "__main__":
    main()
