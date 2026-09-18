#!/usr/bin/env python
# coding: utf-8
"""
Production upday downloader for CSI1500 daily/interday stock data.

This script refreshes recent stock daily source files date-first:
one Tushare request per dataset per trade date, then merge into per-stock CSVs.

Default window is the last 5 open A-share trade dates ending at config_date.end_date.
Margin detail is intentionally excluded and should be refreshed by a separate script.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import tushare as ts

from config import STOCK_DATA_DIR, TUSHARE_TOKEN
from config_date import end_date, normalize_trade_date, trade_date_before, trade_dates_between
from download_csi1500_daily import (
    LIMIT_COLUMNS,
    MONEYFLOW_SCALES,
    REQUEST_INTERVAL,
    RETRY_DELAY,
    STK_FACTOR_SCALES,
    apply_scaling,
    normalize_date_cols,
)


PROJECT_ROOT = Path(__file__).resolve().parent
UNIVERSE_CSV = PROJECT_ROOT / "csi1500con.csv"
BASE_DIR = Path(STOCK_DATA_DIR)
REPORT_DIR = BASE_DIR / "report"

MAX_RETRIES = 5
DEFAULT_LOOKBACK_TRADE_DAYS = 30

DATASETS = ["stkfactor", "moneyflow", "cyq_perf", "auction_o", "auction_c", "limit"]
SUFFIX = {
    "stkfactor": "stkfactor",
    "moneyflow": "moneyflow",
    "cyq_perf": "cyq_perf",
    "auction_o": "auction_o",
    "auction_c": "auction_c",
    "limit": "limit",
}


def make_tushare_pro():
    token = str(TUSHARE_TOKEN or "").strip()
    if not token:
        raise ValueError("Tushare token is empty. Set TUSHARE_TOKEN in config.py.")
    ts.set_token(token)
    return ts.pro_api(token)


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
            print(f"      [ERROR] {desc}, attempt {attempt}/{MAX_RETRIES}: {exc}", flush=True)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return pd.DataFrame()


def download_one_day(pro, dataset: str, trade_date: str) -> pd.DataFrame:
    if dataset == "stkfactor":
        df = call_with_retry(
            f"{trade_date} stk_factor_pro",
            lambda: pro.stk_factor_pro(trade_date=trade_date),
        )
        return normalize_date_cols(apply_scaling(df, STK_FACTOR_SCALES))

    if dataset == "moneyflow":
        df = call_with_retry(
            f"{trade_date} moneyflow_dc",
            lambda: pro.moneyflow_dc(trade_date=trade_date),
        )
        return normalize_date_cols(apply_scaling(df, MONEYFLOW_SCALES))

    if dataset == "cyq_perf":
        df = call_with_retry(
            f"{trade_date} cyq_perf",
            lambda: pro.cyq_perf(trade_date=trade_date),
        )
        return normalize_date_cols(df)

    if dataset == "auction_o":
        df = call_with_retry(
            f"{trade_date} stk_auction_o",
            lambda: pro.stk_auction_o(trade_date=trade_date),
        )
        return normalize_date_cols(df)

    if dataset == "auction_c":
        df = call_with_retry(
            f"{trade_date} stk_auction_c",
            lambda: pro.stk_auction_c(trade_date=trade_date),
        )
        return normalize_date_cols(df)

    if dataset == "limit":
        df = call_with_retry(
            f"{trade_date} limit_list_d",
            lambda: pro.limit_list_d(trade_date=trade_date),
        )
        return normalize_date_cols(df)

    raise ValueError(f"Unsupported dataset: {dataset}")


def normalize_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = df.copy()
    for key in keys:
        if key in out.columns:
            out[key] = out[key].astype(str).str.strip()
    return out


def cell_equal(left, right) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False

    left_s = str(left).strip()
    right_s = str(right).strip()
    if left_s == right_s:
        return True
    if left_s == "" and right_s == "":
        return True

    try:
        left_f = float(left_s)
        right_f = float(right_s)
        if math.isfinite(left_f) and math.isfinite(right_f):
            return math.isclose(left_f, right_f, rel_tol=1e-10, abs_tol=1e-8)
    except ValueError:
        pass
    return False


def row_changed(old_row: pd.Series, new_row: pd.Series, compare_cols: list[str]) -> bool:
    for col in compare_cols:
        old_val = old_row[col] if col in old_row.index else pd.NA
        new_val = new_row[col] if col in new_row.index else pd.NA
        if not cell_equal(old_val, new_val):
            return True
    return False


def count_changes(old_df: pd.DataFrame, new_df: pd.DataFrame, keys: list[str]) -> tuple[int, int]:
    if new_df.empty:
        return 0, 0
    if old_df.empty:
        return len(new_df), 0

    old = normalize_keys(old_df, keys).set_index(keys, drop=False)
    new = normalize_keys(new_df, keys).set_index(keys, drop=False)
    old_index = set(old.index)

    new_key_count = 0
    changed_overlap_count = 0
    compare_cols = sorted(set(old.columns).union(new.columns) - set(keys))
    for idx, new_row in new.iterrows():
        if idx not in old_index:
            new_key_count += 1
        else:
            old_row = old.loc[idx]
            if isinstance(old_row, pd.DataFrame):
                old_row = old_row.iloc[-1]
            if row_changed(old_row, new_row, compare_cols):
                changed_overlap_count += 1
    return new_key_count, changed_overlap_count


def merge_update_frame(old_df: pd.DataFrame, new_df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    combined = normalize_keys(combined, keys)
    combined = combined.drop_duplicates(keys, keep="last")
    sort_cols = [c for c in ["trade_date", "ts_code"] if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols)
    return combined


def read_existing_csv(path: Path, empty_columns: list[str] | None = None) -> tuple[pd.DataFrame, bool]:
    if path.exists():
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig"), True
    if empty_columns is not None:
        return pd.DataFrame(columns=empty_columns), False
    return pd.DataFrame(), False


def update_stock_file(
    path: Path,
    new_df: pd.DataFrame,
    keys: list[str],
    empty_columns: list[str] | None = None,
    dry_run: bool = False,
    skip_save_if_unchanged: bool = False,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    old_df, existed = read_existing_csv(path, empty_columns=empty_columns)
    old_rows = len(old_df)

    if new_df.empty:
        if not existed and not dry_run:
            old_df.to_csv(path, index=False, encoding="utf_8_sig")
        return {
            "old_rows": old_rows,
            "download_rows": 0,
            "new_key_rows": 0,
            "changed_overlap_rows": 0,
            "final_rows": len(old_df),
            "status": "created_empty" if not existed else "no_new_rows",
        }

    new_df = normalize_keys(new_df, keys)
    new_key_rows, changed_overlap_rows = count_changes(old_df, new_df, keys)
    if skip_save_if_unchanged and new_key_rows == 0 and changed_overlap_rows == 0 and existed:
        return {
            "old_rows": old_rows,
            "download_rows": len(new_df),
            "new_key_rows": 0,
            "changed_overlap_rows": 0,
            "final_rows": old_rows,
            "status": "skipped_unchanged",
        }

    final_df = merge_update_frame(old_df, new_df, keys)
    if not dry_run:
        final_df.to_csv(path, index=False, encoding="utf_8_sig")
    return {
        "old_rows": old_rows,
        "download_rows": len(new_df),
        "new_key_rows": new_key_rows,
        "changed_overlap_rows": changed_overlap_rows,
        "final_rows": len(final_df),
        "status": "dry_run_update" if dry_run else "updated",
    }


def prune_limit_window_for_non_limit_stocks(
    data_dir: Path,
    stock_list: list[str],
    trade_dates: list[str],
    returned_stock_set: set[str],
    dry_run: bool = False,
) -> list[dict]:
    """Remove stale limit rows in the refresh window for stocks absent from limit_list_d."""
    rows = []
    date_set = set(trade_dates)
    for ts_code in stock_list:
        if ts_code in returned_stock_set:
            continue
        path = data_dir / f"{ts_code}.limit.csv"
        if not path.exists():
            continue

        old_df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        if old_df.empty or "trade_date" not in old_df.columns:
            continue

        old_rows = len(old_df)
        keep_mask = ~old_df["trade_date"].astype(str).isin(date_set)
        final_df = old_df[keep_mask].copy()
        removed = old_rows - len(final_df)
        if removed == 0:
            continue

        if not dry_run:
            final_df.to_csv(path, index=False, encoding="utf_8_sig")
        rows.append({
            "dataset": "limit",
            "trade_date": ",".join(trade_dates),
            "ts_code": ts_code,
            "old_rows": old_rows,
            "download_rows": 0,
            "new_key_rows": 0,
            "changed_overlap_rows": 0,
            "final_rows": len(final_df),
            "status": "dry_run_prune_stale_limit" if dry_run else "pruned_stale_limit",
            "output_file": str(path),
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-upday", default=None, help="YYYYMMDD. Overrides lookback window start.")
    parser.add_argument("--end-upday", default=None, help="YYYYMMDD. Defaults to config_date.end_date.")
    parser.add_argument("--lookback-trade-days", type=int, default=DEFAULT_LOOKBACK_TRADE_DAYS)
    parser.add_argument("--datasets", nargs="*", default=DATASETS, choices=DATASETS)
    parser.add_argument("--workers", type=int, default=1, help="Dataset-level parallel workers. One worker handles one data source.")
    parser.add_argument("--dry-run", action="store_true", help="Download and compare, but do not write stock files.")
    parser.add_argument(
        "--skip-save-if-unchanged",
        action="store_true",
        help="Skip rewriting a stock file when downloaded rows are identical to existing rows.",
    )
    parser.add_argument(
        "--no-prune-limit-window",
        action="store_true",
        help="Do not remove old limit rows in the refresh window for stocks absent from the latest limit download.",
    )
    return parser.parse_args()


def resolve_refresh_window(
    start_upday: str | None,
    end_upday: str | None,
    lookback_trade_days: int,
) -> tuple[str, str]:
    """Resolve an inclusive refresh window with exactly lookback_trade_days by default."""
    if lookback_trade_days < 1:
        raise ValueError("--lookback-trade-days must be >= 1")

    resolved_end = normalize_trade_date(end_upday) if end_upday else end_date
    if start_upday:
        resolved_start = normalize_trade_date(start_upday)
    elif lookback_trade_days == 1:
        resolved_start = resolved_end
    else:
        resolved_start = trade_date_before(resolved_end, trade_days=lookback_trade_days - 1)

    if resolved_start > resolved_end:
        raise ValueError(f"start_upday {resolved_start} cannot be after end_upday {resolved_end}")
    return resolved_start, resolved_end


def process_dataset(
    dataset: str,
    args: argparse.Namespace,
    trade_dates: list[str],
    stock_list: list[str],
    stock_set: set[str],
    start_upday: str,
    end_upday: str,
) -> list[dict]:
    pro = make_tushare_pro()
    data_dir = BASE_DIR / dataset
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Upday download {dataset} ===", flush=True)

    downloaded_parts = []
    for i, trade_date in enumerate(trade_dates, 1):
        print(f"[{dataset} {i}/{len(trade_dates)}] {trade_date}", flush=True)
        df = download_one_day(pro, dataset, trade_date)
        if not df.empty and "ts_code" in df.columns:
            df = df[df["ts_code"].astype(str).isin(stock_set)].copy()
        downloaded_parts.append(df)
        print(f"    [{dataset}] downloaded_rows={len(df)}", flush=True)
        time.sleep(REQUEST_INTERVAL)

    if downloaded_parts:
        day_df = pd.concat(downloaded_parts, ignore_index=True, sort=False)
    else:
        day_df = pd.DataFrame()

    summary = []
    touched = 0
    status_counts: dict[str, int] = {}
    returned_stock_set: set[str] = set()

    if not day_df.empty and "ts_code" in day_df.columns:
        day_df = normalize_date_cols(day_df)
        returned_stock_set = set(day_df["ts_code"].astype(str))
        for ts_code, g in day_df.groupby("ts_code", sort=True):
            out_path = data_dir / f"{ts_code}.{SUFFIX[dataset]}.csv"
            empty_cols = LIMIT_COLUMNS if dataset == "limit" else None
            result = update_stock_file(
                out_path,
                g,
                keys=["ts_code", "trade_date"],
                empty_columns=empty_cols,
                dry_run=args.dry_run,
                skip_save_if_unchanged=args.skip_save_if_unchanged,
            )
            touched += 1
            status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
            summary.append({
                "dataset": dataset,
                "trade_date": f"{start_upday}-{end_upday}",
                "ts_code": ts_code,
                **result,
                "output_file": str(out_path),
            })
    else:
        summary.append({
            "dataset": dataset,
            "trade_date": f"{start_upday}-{end_upday}",
            "ts_code": "",
            "old_rows": 0,
            "download_rows": 0,
            "new_key_rows": 0,
            "changed_overlap_rows": 0,
            "final_rows": 0,
            "status": "empty_window",
            "output_file": "",
        })
        status_counts["empty_window"] = 1

    if dataset == "limit" and not args.no_prune_limit_window:
        prune_rows = prune_limit_window_for_non_limit_stocks(
            data_dir=data_dir,
            stock_list=stock_list,
            trade_dates=trade_dates,
            returned_stock_set=returned_stock_set,
            dry_run=args.dry_run,
        )
        summary.extend(prune_rows)
        for row in prune_rows:
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    print(
        f"    [{dataset}] window_rows={len(day_df)} touched_stocks={touched} status_counts={status_counts}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    start_upday, end_upday = resolve_refresh_window(
        args.start_upday,
        args.end_upday,
        args.lookback_trade_days,
    )
    trade_dates = trade_dates_between(start_upday, end_upday)
    stock_list = load_stock_list()
    stock_set = set(stock_list)

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Production daily upday window: {start_upday} ~ {end_upday}", flush=True)
    print(f"Trade dates: {len(trade_dates)} {trade_dates}", flush=True)
    print(f"Universe stock count: {len(stock_list)}", flush=True)
    print(f"Dataset workers: {args.workers}", flush=True)
    print(f"Output directory: {BASE_DIR}", flush=True)
    if args.dry_run:
        print("DRY RUN: no stock files will be written", flush=True)

    summary = []
    if args.workers == 1 or len(args.datasets) <= 1:
        for dataset in args.datasets:
            summary.extend(process_dataset(dataset, args, trade_dates, stock_list, stock_set, start_upday, end_upday))
    else:
        max_workers = min(args.workers, len(args.datasets))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(process_dataset, dataset, args, trade_dates, stock_list, stock_set, start_upday, end_upday): dataset
                for dataset in args.datasets
            }
            for future in as_completed(future_map):
                summary.extend(future.result())

    report_path = REPORT_DIR / f"download_csi1500_daily_upday_{start_upday}_{end_upday}.csv"
    pd.DataFrame(summary).to_csv(report_path, index=False, encoding="utf_8_sig")
    print(f"\n[DONE] report saved: {report_path}", flush=True)


if __name__ == "__main__":
    main()
