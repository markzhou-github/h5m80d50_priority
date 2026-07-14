#!/usr/bin/env python
# coding: utf-8
"""
download_ashare_indexes.py

Task 1: download domestic A-share market/index data.
Outputs under STOCK_INDEX_DIR / INDEX_DIR:
    <ts_code>.index_dailybasic.csv
    <ts_code>.idxfactor.csv
    mktdc.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import tushare as ts

PROJ_ROOT = Path.cwd().parent
CONFIG_DIR = PROJ_ROOT / "config"
if str(CONFIG_DIR) not in sys.path:
    sys.path.append(str(CONFIG_DIR))

from config_date import history_start_date, end_date
from config import STOCK_INDEX_DIR, INDEX_DIR, MKTDC_CSV, TUSHARE_TOKEN

STOCK_INDEX_DIR = Path(STOCK_INDEX_DIR)
INDEX_DIR = Path(INDEX_DIR)
STOCK_INDEX_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

IDX_FACTOR_SCALES = {"vol": 100, "amount": 1000}

DEFAULT_INDEX_DAILYBASIC_LIST = [
    "000300.SH", "000905.SH", "000852.SH", "000985.CSI",
]
DEFAULT_INDEX_FACTOR_LIST = [
    "000300.SH", "000905.SH", "000852.SH", "000985.CSI",
]


def apply_scaling(df: pd.DataFrame, scale_dict: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col, multiplier in scale_dict.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * multiplier
    return out


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf_8_sig")


def download_single_index_dailybasic(ts_code: str, start_date: str, end_date: str,
                                     max_retries: int = 5, retry_delay: int = 2) -> pd.DataFrame:
    for attempt in range(max_retries):
        try:
            df = pro.index_dailybasic(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                print(f"[WARN] index_dailybasic empty: {ts_code}")
                return pd.DataFrame()
            df["ts_code"] = df["ts_code"].astype(str)
            df["trade_date"] = df["trade_date"].astype(str)
            return df.sort_values("trade_date").reset_index(drop=True)
        except Exception as e:
            print(f"[ERROR] index_dailybasic {ts_code} attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return pd.DataFrame()


def download_index_dailybasic_all(start_date: str = history_start_date, end_date: str = end_date,
                                  data_dir: Path = STOCK_INDEX_DIR,
                                  index_list: Optional[List[str]] = None,
                                  max_retries: int = 5, retry_delay: int = 2,
                                  request_interval: float = 0.4) -> None:
    if index_list is None:
        index_list = DEFAULT_INDEX_DAILYBASIC_LIST
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Download A-share index_dailybasic")
    print(f"Range: {start_date} ~ {end_date}")
    print(f"Output: {data_dir}")
    print("=" * 70)
    for i, ts_code in enumerate(index_list, 1):
        print(f"[{i}/{len(index_list)}] {ts_code} index_dailybasic")
        df = download_single_index_dailybasic(ts_code, start_date, end_date, max_retries, retry_delay)
        if df.empty:
            print(f"  [SKIP] no data: {ts_code}")
        else:
            out_path = data_dir / f"{ts_code}.index_dailybasic.csv"
            save_csv(df, out_path)
            print(f"  [SAVE] {len(df)} rows -> {out_path}")
        if i < len(index_list):
            time.sleep(request_interval)


def download_index_factor(ts_code: str, start_date: str, end_date: str,
                          max_retries: int = 5, retry_delay: int = 2) -> pd.DataFrame:
    for attempt in range(max_retries):
        try:
            df = pro.idx_factor_pro(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                print(f"[WARN] idx_factor_pro empty: {ts_code}")
                return pd.DataFrame()
            df = apply_scaling(df, IDX_FACTOR_SCALES)
            df["ts_code"] = df["ts_code"].astype(str)
            df["trade_date"] = df["trade_date"].astype(str)
            return df.sort_values("trade_date").reset_index(drop=True)
        except Exception as e:
            print(f"[ERROR] idx_factor_pro {ts_code} attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return pd.DataFrame()


def index_factor_download(index_list: Optional[List[str]] = None, start_date: str = history_start_date,
                          end_date: str = end_date, data_dir: Path = STOCK_INDEX_DIR,
                          request_interval: float = 0.4, max_retries: int = 5,
                          retry_delay: int = 2) -> None:
    if index_list is None:
        index_list = DEFAULT_INDEX_FACTOR_LIST
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Download A-share idx_factor_pro")
    print(f"Range: {start_date} ~ {end_date}")
    print(f"Output: {data_dir}")
    print("=" * 70)
    for i, ts_code in enumerate(index_list, 1):
        print(f"[{i}/{len(index_list)}] {ts_code} idx_factor_pro")
        df = download_index_factor(ts_code, start_date, end_date, max_retries, retry_delay)
        if df.empty:
            print(f"  [SKIP] no data: {ts_code}")
        else:
            out_path = data_dir / f"{ts_code}.idxfactor.csv"
            save_csv(df, out_path)
            print(f"  [SAVE] {len(df)} rows -> {out_path}")
        if i < len(index_list):
            time.sleep(request_interval)


def download_market_moneyflow(start_date: str = history_start_date, end_date: str = end_date,
                              max_retries: int = 5, retry_delay: int = 2) -> pd.DataFrame:
    for attempt in range(max_retries):
        try:
            df = pro.moneyflow_mkt_dc(start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                print(f"[WARN] moneyflow_mkt_dc empty: {start_date} ~ {end_date}")
                return pd.DataFrame()
            df["trade_date"] = df["trade_date"].astype(str)
            return df.sort_values("trade_date").reset_index(drop=True)
        except Exception as e:
            print(f"[ERROR] moneyflow_mkt_dc attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return pd.DataFrame()


def download_all_ashare_market_data() -> None:
    download_index_dailybasic_all()
    index_factor_download()
    df_mkt = download_market_moneyflow(history_start_date, end_date)
    if df_mkt.empty:
        print("[FAIL] moneyflow_mkt_dc unavailable")
    else:
        save_csv(df_mkt, Path(MKTDC_CSV))
        print(f"[SAVE] moneyflow_mkt_dc {len(df_mkt)} rows -> {MKTDC_CSV}")


if __name__ == "__main__":
    download_all_ashare_market_data()
