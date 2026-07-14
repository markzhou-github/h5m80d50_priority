#!/usr/bin/env python
# coding: utf-8
"""
Historical CSI1500 daily/interday downloader, excluding margin_detail.

Outputs per-stock CSV files under processed/daily:
  stkfactor, moneyflow, cyq_perf, auction_o, auction_c, limit
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
REPORT_DIR = BASE_DIR / "report"

# Historical phase. Keep refresh_start_date imported for future maintenance.
Start_date = history_start_date
End_date = end_date

REQUEST_INTERVAL = 0.4
MAX_RETRIES = 5
RETRY_DELAY = 2

STK_FACTOR_SCALES = {
    "vol": 100,
    "amount": 1000,
    "total_share": 10000,
    "float_share": 10000,
    "free_share": 10000,
    "total_mv": 10000,
    "circ_mv": 10000,
}

MONEYFLOW_SCALES = {
    "net_amount": 10000,
    "buy_elg_amount": 10000,
    "buy_lg_amount": 10000,
    "buy_md_amount": 10000,
    "buy_sm_amount": 10000,
}

LIMIT_COLUMNS = [
    "trade_date",
    "ts_code",
    "name",
    "industry",
    "close",
    "pct_chg",
    "amount",
    "limit_amount",
    "float_mv",
    "total_mv",
    "turnover_ratio",
    "fd_amount",
    "first_time",
    "last_time",
    "open_times",
    "up_stat",
    "limit_times",
    "limit",
]


def load_stock_list() -> list[str]:
    df = pd.read_csv(UNIVERSE_CSV, dtype={"con_code": str})
    if "con_code" not in df.columns:
        raise ValueError(f"Missing con_code column in {UNIVERSE_CSV}")
    return sorted(df["con_code"].dropna().astype(str).str.strip().unique())


def apply_scaling(df: pd.DataFrame, scales: dict[str, float]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col, multiplier in scales.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * multiplier
    return out


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


def normalize_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ("ts_code", "trade_date"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    if "trade_date" in out.columns:
        out = out.sort_values("trade_date")
    return out


def save_csv(df: pd.DataFrame, path: Path, empty_columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty and empty_columns is not None:
        df = pd.DataFrame(columns=empty_columns)
    df.to_csv(path, index=False, encoding="utf_8_sig")


def download_one_dataset(pro, ts_code: str, dataset: str) -> pd.DataFrame:
    if dataset == "stkfactor":
        df = call_with_retry(
            f"{ts_code} stk_factor_pro",
            lambda: pro.stk_factor_pro(ts_code=ts_code, start_date=Start_date, end_date=End_date),
        )
        return normalize_date_cols(apply_scaling(df, STK_FACTOR_SCALES))

    if dataset == "moneyflow":
        df = call_with_retry(
            f"{ts_code} moneyflow_dc",
            lambda: pro.moneyflow_dc(ts_code=ts_code, start_date=Start_date, end_date=End_date),
        )
        return normalize_date_cols(apply_scaling(df, MONEYFLOW_SCALES))

    if dataset == "cyq_perf":
        df = call_with_retry(
            f"{ts_code} cyq_perf",
            lambda: pro.cyq_perf(ts_code=ts_code, start_date=Start_date, end_date=End_date),
        )
        return normalize_date_cols(df)

    if dataset == "auction_o":
        df = call_with_retry(
            f"{ts_code} stk_auction_o",
            lambda: pro.stk_auction_o(ts_code=ts_code, start_date=Start_date, end_date=End_date),
        )
        return normalize_date_cols(df)

    if dataset == "auction_c":
        df = call_with_retry(
            f"{ts_code} stk_auction_c",
            lambda: pro.stk_auction_c(ts_code=ts_code, start_date=Start_date, end_date=End_date),
        )
        return normalize_date_cols(df)

    if dataset == "limit":
        # Empty result is normal: most stocks are not in limit status most days.
        df = call_with_retry(
            f"{ts_code} limit_list_d",
            lambda: pro.limit_list_d(ts_code=ts_code, start_date=Start_date, end_date=End_date),
        )
        return normalize_date_cols(df)

    raise ValueError(f"Unsupported dataset: {dataset}")


def main() -> None:
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    stock_list = load_stock_list()
    print(f"Historical daily download range: {Start_date} ~ {End_date}")
    print(f"Refresh start imported for future use: {refresh_start_date}")
    print(f"CSI1500 stock count: {len(stock_list)}")
    print(f"Output directory: {BASE_DIR}")

    datasets = ["stkfactor", "moneyflow", "cyq_perf", "auction_o", "auction_c", "limit"]
    suffix = {
        "stkfactor": "stkfactor",
        "moneyflow": "moneyflow",
        "cyq_perf": "cyq_perf",
        "auction_o": "auction_o",
        "auction_c": "auction_c",
        "limit": "limit",
    }

    summary = []
    for dataset in datasets:
        data_dir = BASE_DIR / dataset
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Download {dataset} ===")

        for i, ts_code in enumerate(stock_list, 1):
            print(f"[{dataset} {i}/{len(stock_list)}] {ts_code}")
            df = download_one_dataset(pro, ts_code, dataset)
            out_path = data_dir / f"{ts_code}.{suffix[dataset]}.csv"
            empty_cols = LIMIT_COLUMNS if dataset == "limit" else None
            save_csv(df, out_path, empty_columns=empty_cols)
            summary.append({
                "dataset": dataset,
                "ts_code": ts_code,
                "rows": len(df),
                "output_file": str(out_path),
                "status": "saved_empty" if df.empty else "saved",
            })
            if i < len(stock_list):
                time.sleep(REQUEST_INTERVAL)

    report_path = REPORT_DIR / "download_csi1500_daily_summary.csv"
    pd.DataFrame(summary).to_csv(report_path, index=False, encoding="utf_8_sig")
    print(f"\n[DONE] report saved: {report_path}")


if __name__ == "__main__":
    main()
