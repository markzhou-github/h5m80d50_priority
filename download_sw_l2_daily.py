#!/usr/bin/env python
# coding: utf-8
"""
Download all Shenwan L2 index daily data into one CSV file.

Unit normalization:
  - vol: 万股 -> 股
  - amount: 万元 -> 元
  - float_mv: 万元 -> 元
  - total_mv: 万元 -> 元
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import tushare as ts

from config import STOCK_INDEX_DIR, TUSHARE_TOKEN
from config_date import history_start_date, refresh_start_date, end_date


PROJECT_ROOT = Path(__file__).resolve().parent
SW_L2_CSV = PROJECT_ROOT / "sw_l2_si.csv"
OUT_DIR = Path(STOCK_INDEX_DIR)
REPORT_DIR = OUT_DIR / "report"
OUT_PATH = OUT_DIR / "sw_l2_daily.csv"

# Historical phase. Keep refresh_start_date imported for future maintenance.
Start_date = history_start_date
End_date = end_date

REQUEST_INTERVAL = 0.4
MAX_RETRIES = 5
RETRY_DELAY = 2

FIELDS = [
    "ts_code",
    "trade_date",
    "name",
    "open",
    "low",
    "high",
    "close",
    "change",
    "pct_change",
    "vol",
    "amount",
    "pe",
    "pb",
    "float_mv",
    "total_mv",
]

UNIT_SCALE = {
    "vol": 10000,
    "amount": 10000,
    "float_mv": 10000,
    "total_mv": 10000,
}


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str)


def load_sw_l2_index_list() -> pd.DataFrame:
    df = read_csv_with_fallback(SW_L2_CSV)
    required = {"index_code", "industry_code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {SW_L2_CSV}: {sorted(missing)}")

    keep_cols = [
        "index_code",
        "industry_name",
        "level",
        "industry_code",
        "is_pub",
        "parent_code",
        "src",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()
    out["index_code"] = out["index_code"].astype(str).str.strip()
    out["industry_code"] = out["industry_code"].astype(str).str.strip().str.zfill(6)
    out = out[out["index_code"].ne("")]
    return out.drop_duplicates("index_code", keep="last").sort_values("index_code").reset_index(drop=True)


def download_one_index(pro, ts_code: str) -> pd.DataFrame:
    fields = ",".join(FIELDS)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = pro.sw_daily(ts_code=ts_code, start_date=Start_date, end_date=End_date, fields=fields)
            return pd.DataFrame(columns=FIELDS) if df is None else df
        except Exception as exc:
            print(f"      [ERROR] {ts_code} sw_daily attempt {attempt}/{MAX_RETRIES}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return pd.DataFrame(columns=FIELDS)


def normalize_sw_daily(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=FIELDS)

    out = df.copy()
    for col in FIELDS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[FIELDS].copy()
    out["ts_code"] = out["ts_code"].fillna(ts_code).astype(str)
    out["trade_date"] = out["trade_date"].astype(str)
    out["name"] = out["name"].astype(str)

    numeric_cols = [c for c in FIELDS if c not in ("ts_code", "trade_date", "name")]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col, multiplier in UNIT_SCALE.items():
        out[col] = out[col] * multiplier

    return (
        out.sort_values(["ts_code", "trade_date"])
        .drop_duplicates(["ts_code", "trade_date"], keep="last")
        .reset_index(drop=True)
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    index_list = load_sw_l2_index_list()
    print(f"Download SW L2 daily: {Start_date} ~ {End_date}")
    print(f"Refresh start imported for future use: {refresh_start_date}")
    print(f"SW L2 index count: {len(index_list)}")
    print(f"Output file: {OUT_PATH}")

    frames = []
    summary = []
    for i, row in index_list.iterrows():
        ts_code = str(row["index_code"])
        print(f"[{i + 1}/{len(index_list)}] {ts_code}")
        df = normalize_sw_daily(download_one_index(pro, ts_code), ts_code)
        frames.append(df)
        summary.append({
            "ts_code": ts_code,
            "industry_code": row.get("industry_code", ""),
            "industry_name": row.get("industry_name", ""),
            "rows": len(df),
            "first_trade_date": "" if df.empty else df["trade_date"].min(),
            "last_trade_date": "" if df.empty else df["trade_date"].max(),
            "status": "saved_empty" if df.empty else "downloaded",
        })
        if i < len(index_list) - 1:
            time.sleep(REQUEST_INTERVAL)

    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FIELDS)
    all_df = all_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    all_df.to_csv(OUT_PATH, index=False, encoding="utf_8_sig")

    summary_path = REPORT_DIR / "download_sw_l2_daily_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False, encoding="utf_8_sig")

    print(f"[SAVE] {OUT_PATH}")
    print(f"[SAVE] {summary_path}")
    print(f"[DONE] rows={len(all_df)}")


if __name__ == "__main__":
    main()
