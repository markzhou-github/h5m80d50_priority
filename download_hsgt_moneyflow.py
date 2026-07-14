#!/usr/bin/env python
# coding: utf-8
"""
Download Tushare moneyflow_hsgt data for the market panel.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import tushare as ts

from config import STOCK_INDEX_DIR, TUSHARE_TOKEN
from config_date import history_start_date, refresh_start_date, end_date


OUT_DIR = Path(STOCK_INDEX_DIR)
OUT_PATH = OUT_DIR / "moneyflow_hsgt.csv"
REPORT_DIR = OUT_DIR / "report"

# Historical phase. Keep refresh_start_date imported for future maintenance.
Start_date = history_start_date
End_date = end_date

MAX_RETRIES = 5
RETRY_DELAY = 2

HSGT_COLUMNS = [
    "trade_date",
    "ggt_ss",
    "ggt_sz",
    "hgt",
    "sgt",
    "north_money",
    "south_money",
]


def download_moneyflow_hsgt() -> pd.DataFrame:
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = pro.moneyflow_hsgt(start_date=Start_date, end_date=End_date)
            if df is None:
                return pd.DataFrame(columns=HSGT_COLUMNS)
            return df
        except Exception as exc:
            print(f"[ERROR] moneyflow_hsgt attempt {attempt}/{MAX_RETRIES}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return pd.DataFrame(columns=HSGT_COLUMNS)


def normalize_hsgt(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=HSGT_COLUMNS)

    out = df.copy()
    for col in HSGT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[HSGT_COLUMNS].copy()
    out["trade_date"] = out["trade_date"].astype(str)
    for col in [c for c in HSGT_COLUMNS if c != "trade_date"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Download moneyflow_hsgt: {Start_date} ~ {End_date}")
    print(f"Refresh start imported for future use: {refresh_start_date}")
    df = normalize_hsgt(download_moneyflow_hsgt())
    df.to_csv(OUT_PATH, index=False, encoding="utf_8_sig")

    report = pd.DataFrame([{
        "dataset": "moneyflow_hsgt",
        "rows": len(df),
        "first_trade_date": "" if df.empty else df["trade_date"].min(),
        "last_trade_date": "" if df.empty else df["trade_date"].max(),
        "output_file": str(OUT_PATH),
        "status": "saved_empty" if df.empty else "saved",
    }])
    report_path = REPORT_DIR / "download_hsgt_moneyflow_summary.csv"
    report.to_csv(report_path, index=False, encoding="utf_8_sig")

    print(f"[SAVE] {OUT_PATH}")
    print(f"[SAVE] {report_path}")


if __name__ == "__main__":
    main()
