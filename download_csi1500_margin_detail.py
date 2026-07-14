#!/usr/bin/env python
# coding: utf-8
"""
Historical CSI1500 margin_detail downloader.

This is intentionally separate because margin_detail is published later than
the other daily datasets.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import pandas as pd
import tushare as ts

from config import STOCK_DATA_DIR, TUSHARE_TOKEN
from config_date import history_start_date, refresh_start_date, end_date


PROJECT_ROOT = Path(__file__).resolve().parent
UNIVERSE_CSV = PROJECT_ROOT / "csi1500con.csv"
BASE_DIR = Path(STOCK_DATA_DIR)
MARGIN_DIR = BASE_DIR / "margin_detail"
REPORT_DIR = BASE_DIR / "report"

Start_date = history_start_date
End_date = end_date

REQUEST_INTERVAL = 0.4
MAX_RETRIES = 5
RETRY_DELAY = 2


def load_stock_list() -> list[str]:
    df = pd.read_csv(UNIVERSE_CSV, dtype={"con_code": str})
    if "con_code" not in df.columns:
        raise ValueError(f"Missing con_code column in {UNIVERSE_CSV}")
    return sorted(df["con_code"].dropna().astype(str).str.strip().unique())


def call_with_retry(desc: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = fn()
            return pd.DataFrame() if df is None else df
        except Exception as exc:
            print(f"      [ERROR] {desc}, attempt {attempt}/{MAX_RETRIES}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return pd.DataFrame()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ("ts_code", "trade_date"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    if "trade_date" in out.columns:
        out = out.sort_values("trade_date")
    return out


def main() -> None:
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    MARGIN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    stock_list = load_stock_list()
    print(f"Historical margin_detail range: {Start_date} ~ {End_date}")
    print(f"Refresh start imported for future use: {refresh_start_date}")
    print(f"CSI1500 stock count: {len(stock_list)}")
    print(f"Output directory: {MARGIN_DIR}")

    summary = []
    for i, ts_code in enumerate(stock_list, 1):
        print(f"[margin_detail {i}/{len(stock_list)}] {ts_code}")
        df = call_with_retry(
            f"{ts_code} margin_detail",
            lambda: pro.margin_detail(ts_code=ts_code, start_date=Start_date, end_date=End_date),
        )
        df = normalize(df)
        out_path = MARGIN_DIR / f"{ts_code}.margin_detail.csv"
        df.to_csv(out_path, index=False, encoding="utf_8_sig")
        summary.append({
            "dataset": "margin_detail",
            "ts_code": ts_code,
            "rows": len(df),
            "output_file": str(out_path),
            "status": "saved_empty" if df.empty else "saved",
        })
        if i < len(stock_list):
            time.sleep(REQUEST_INTERVAL)

    report_path = REPORT_DIR / "download_csi1500_margin_detail_summary.csv"
    pd.DataFrame(summary).to_csv(report_path, index=False, encoding="utf_8_sig")
    print(f"\n[DONE] report saved: {report_path}")


if __name__ == "__main__":
    main()
