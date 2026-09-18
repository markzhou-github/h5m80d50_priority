#!/usr/bin/env python
# coding: utf-8
"""Update per-stock 1-minute raw data for the latest upday window.

Default behavior:
  - stock universe: csi1500con.csv / con_code
  - output: data/raw/{ts_code}.parquet
  - end date: config_date.end_date
  - start date: 30 trading days before end date, using config_date.trade_date_before
  - merge policy: newly downloaded rows overwrite old rows on overlapping trade_time

The output is intentionally one parquet per stock because build_minute_features_v5b.py
discovers raw inputs with data/raw/*.parquet.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import tushare as ts

from config import TUSHARE_TOKEN


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CON_FILE = PROJECT_ROOT / "csi1500con.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "raw"
EXPECTED_COLS = ["ts_code", "trade_time", "open", "close", "high", "low", "vol", "amount"]
THREAD_LOCAL = threading.local()


def normalize_trade_date(value: str | int | pd.Timestamp) -> str:
    text = str(value).strip().replace("-", "")
    if len(text) != 8:
        raise ValueError(f"date must be YYYYMMDD or YYYY-MM-DD, got {value!r}")
    datetime.strptime(text, "%Y%m%d")
    return text


def make_tushare_pro(token: str):
    token = str(token or "").strip()
    if not token:
        raise ValueError("Tushare token is empty. Pass --token or set TUSHARE_TOKEN in config.py.")
    ts.set_token(token)
    return ts.pro_api(token)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/update Tushare 1-minute stock data for an upday window.")
    parser.add_argument("--con-file", type=Path, default=DEFAULT_CON_FILE, help="CSV with con_code stock list.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output folder for {ts_code}.parquet.")
    parser.add_argument("--start-upday", default=None, help="YYYYMMDD inclusive. Defaults to lookback before end-upday.")
    parser.add_argument("--end-upday", default=None, help="YYYYMMDD inclusive. Defaults to config_date.end_date.")
    parser.add_argument("--lookback-trade-days", type=int, default=20, help="Trading-day lookback if start-upday omitted.")
    parser.add_argument("--stocks", nargs="*", default=[], help="Optional subset. Accepts 600004.SH or 600004.")
    parser.add_argument("--max-stocks", type=int, default=None, help="Process only first N stocks after filtering.")
    parser.add_argument("--token", default=None, help="Optional Tushare token override.")
    parser.add_argument("--retry", type=int, default=5)
    parser.add_argument("--sleep-sec", type=float, default=0.35, help="Base sleep between API calls.")
    parser.add_argument("--overlap-minutes", type=int, default=10, help="Backward pagination overlap.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel stock workers. Default keeps old sequential behavior.")
    parser.add_argument("--start-time", default="09:30:00", help="Intraday start time for the upday window.")
    parser.add_argument("--end-time", default="15:30:00", help="Intraday end time for the upday window.")
    parser.add_argument("--skip-if-complete", action="store_true", help="Skip stock if existing file appears to cover window.")
    parser.add_argument("--min-daily-rows", type=int, default=200, help="Daily row sanity threshold for skip-if-complete.")
    parser.add_argument("--report-file", type=Path, default=None, help="Default: out-dir/download_1min_upday_summary.csv")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved work plan without downloading.")
    return parser.parse_args()


def normalize_ts_code(code: str) -> str:
    text = str(code).strip().upper()
    if not text:
        return text
    if "." in text:
        return text
    return f"{text}.SH" if text.startswith(("6", "9")) else f"{text}.SZ"


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
    start = normalize_trade_date(start_upday) if start_upday else local_trade_date_before(pro, end, trade_days=lookback_trade_days)
    if start > end:
        raise ValueError(f"start_upday {start} cannot be after end_upday {end}")
    return start, end


def upday_to_datetime_window(
    start_upday: str,
    end_upday: str,
    start_time: str = "09:30:00",
    end_time: str = "15:30:00",
) -> tuple[str, str]:
    return (
        f"{start_upday[:4]}-{start_upday[4:6]}-{start_upday[6:]} {start_time}",
        f"{end_upday[:4]}-{end_upday[4:6]}-{end_upday[6:]} {end_time}",
    )


def get_thread_pro(token: str):
    pro = getattr(THREAD_LOCAL, "pro", None)
    if pro is None:
        pro = make_tushare_pro(token)
        THREAD_LOCAL.pro = pro
    return pro


def read_stock_list(con_file: Path, stocks: list[str], max_stocks: int | None) -> list[str]:
    if stocks:
        selected = sorted({normalize_ts_code(s) for s in stocks if str(s).strip()})
    else:
        con = pd.read_csv(con_file, dtype=str)
        if "con_code" not in con.columns:
            raise ValueError(f"Column con_code not found in {con_file}")
        selected = (
            con["con_code"]
            .dropna()
            .map(normalize_ts_code)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
    if max_stocks is not None:
        selected = selected[:max_stocks]
    return selected


def read_existing_1min_stock(out_dir: Path, ts_code: str) -> pd.DataFrame:
    path = out_dir / f"{ts_code}.parquet"
    if path.exists() and path.stat().st_size > 0:
        return pd.read_parquet(path)

    # Legacy compatibility only. New saves always write out_dir/{ts_code}.parquet.
    yearly_dir = out_dir / ts_code
    if yearly_dir.exists():
        files = sorted(yearly_dir.glob(f"{ts_code}_*.parquet"))
        if files:
            return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.DataFrame()


def standardize_1min_frame(df: pd.DataFrame, ts_code: str | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=EXPECTED_COLS + ["trade_date", "year"])
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
    out["year"] = out["trade_time"].dt.year.astype("int16")
    return out


def existing_covers_window(
    out_dir: Path,
    ts_code: str,
    start_dt: str,
    end_dt: str,
    min_daily_rows: int,
) -> bool:
    df = read_existing_1min_stock(out_dir, ts_code)
    if df.empty or "trade_time" not in df.columns:
        return False
    tt = pd.to_datetime(df["trade_time"], errors="coerce").dropna()
    if tt.empty:
        return False

    start_date = pd.Timestamp(start_dt).strftime("%Y%m%d")
    end_date = pd.Timestamp(end_dt).strftime("%Y%m%d")
    tmp = pd.DataFrame({"trade_time": tt})
    tmp["trade_date"] = tmp["trade_time"].dt.strftime("%Y%m%d")
    daily_rows = tmp.groupby("trade_date").size()
    if daily_rows.empty or daily_rows.median() < min_daily_rows:
        return False

    # Upday refresh must confirm the actual end_upday is present.  A lag
    # tolerance can silently skip stale files, which defeats overwrite updates.
    if daily_rows.index.min() > start_date:
        return False
    if daily_rows.index.max() < end_date:
        return False
    if int(daily_rows.get(end_date, 0)) < min_daily_rows:
        return False
    return True


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
        return pd.DataFrame(columns=EXPECTED_COLS + ["trade_date", "year"])
    return standardize_1min_frame(pd.concat(all_parts, ignore_index=True), ts_code)


def merge_old_new(old_df: pd.DataFrame, new_df: pd.DataFrame, ts_code: str) -> tuple[pd.DataFrame, int, int]:
    old = standardize_1min_frame(old_df, ts_code) if not old_df.empty else pd.DataFrame()
    new = standardize_1min_frame(new_df, ts_code) if not new_df.empty else pd.DataFrame()
    if old.empty:
        return new, len(new), 0
    if new.empty:
        return old, 0, 0

    old_keys = set(zip(old["ts_code"], old["trade_time"]))
    new_keys = set(zip(new["ts_code"], new["trade_time"]))
    overlap = len(old_keys & new_keys)
    new_only = len(new_keys - old_keys)

    # New rows come last, so new upday data overwrites old rows in overlaps.
    merged = standardize_1min_frame(pd.concat([old, new], ignore_index=True), ts_code)
    return merged, new_only, overlap


def save_single_stock_parquet(df: pd.DataFrame, out_dir: Path, ts_code: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = standardize_1min_frame(df, ts_code)
    path = out_dir / f"{ts_code}.parquet"
    out.to_parquet(path, index=False)
    return path


def summarize_frame(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0, "date_min": None, "date_max": None}
    tt = pd.to_datetime(df["trade_time"], errors="coerce")
    return {
        "rows": int(len(df)),
        "date_min": None if tt.dropna().empty else tt.min().strftime("%Y%m%d"),
        "date_max": None if tt.dropna().empty else tt.max().strftime("%Y%m%d"),
    }


def process_one_stock(
    i: int,
    total: int,
    ts_code: str,
    args: argparse.Namespace,
    token: str,
    start_dt: str,
    end_dt: str,
) -> dict:
    print("=" * 100, flush=True)
    print(f"[{i}/{total}] {ts_code}", flush=True)
    try:
        if args.skip_if_complete and existing_covers_window(
            args.out_dir,
            ts_code,
            start_dt,
            end_dt,
            min_daily_rows=args.min_daily_rows,
        ):
            print(f"[{ts_code}] existing data covers window. Skip.", flush=True)
            old = read_existing_1min_stock(args.out_dir, ts_code)
            summary = summarize_frame(old)
            return {"ts_code": ts_code, "status": "skip_complete", "new_rows": 0, "overlap_rows": 0, **summary}

        old = read_existing_1min_stock(args.out_dir, ts_code)
        if old.empty:
            print(f"[{ts_code}] no existing data.", flush=True)
        else:
            old_summary = summarize_frame(old)
            print(
                f"[{ts_code}] existing rows={old_summary['rows']:,} "
                f"{old_summary['date_min']} -> {old_summary['date_max']}",
                flush=True,
            )

        pro = get_thread_pro(token)
        new = download_one_stock_1min_tushare(
            pro=pro,
            ts_code=ts_code,
            start_dt=start_dt,
            end_dt=end_dt,
            retry=args.retry,
            sleep_sec=args.sleep_sec,
            overlap_minutes=args.overlap_minutes,
        )
        merged, new_only, overlap = merge_old_new(old, new, ts_code)
        path = save_single_stock_parquet(merged, args.out_dir, ts_code)
        summary = summarize_frame(merged)
        print(
            f"[{ts_code}] saved rows={summary['rows']:,} new={new_only:,} overlap={overlap:,} -> {path}",
            flush=True,
        )
        time.sleep(args.sleep_sec + random.uniform(0.0, 0.2))
        return {"ts_code": ts_code, "status": "updated", "new_rows": new_only, "overlap_rows": overlap, **summary}
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[{ts_code}] FAILED: {exc}", flush=True)
        traceback.print_exc()
        return {
            "ts_code": ts_code,
            "status": "failed",
            "new_rows": 0,
            "overlap_rows": 0,
            "rows": 0,
            "date_min": None,
            "date_max": None,
            "message": str(exc),
        }


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    token = args.token or TUSHARE_TOKEN
    pro = None if args.dry_run and args.end_upday and args.start_upday else make_tushare_pro(token)
    start_upday, end_upday = resolve_window(args.start_upday, args.end_upday, args.lookback_trade_days, token, pro)
    start_dt, end_dt = upday_to_datetime_window(start_upday, end_upday, args.start_time, args.end_time)
    stock_list = read_stock_list(args.con_file, args.stocks, args.max_stocks)
    report_file = args.report_file or (args.out_dir / "download_1min_upday_summary.csv")

    print(f"[window] start_upday={start_upday} end_upday={end_upday}", flush=True)
    print(f"[datetime] start_dt={start_dt} end_dt={end_dt}", flush=True)
    print(f"[stocks] count={len(stock_list)}", flush=True)
    print(f"[workers] {args.workers}", flush=True)
    print(f"[out] {args.out_dir}", flush=True)

    if args.dry_run:
        return

    pro = pro or make_tushare_pro(token)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.perf_counter()
    if args.workers == 1:
        for i, ts_code in enumerate(stock_list, 1):
            rows.append(process_one_stock(i, len(stock_list), ts_code, args, token, start_dt, end_dt))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(process_one_stock, i, len(stock_list), ts_code, args, token, start_dt, end_dt): ts_code
                for i, ts_code in enumerate(stock_list, 1)
            }
            for future in as_completed(future_map):
                rows.append(future.result())

    report = pd.DataFrame(rows)
    report["start_upday"] = start_upday
    report["end_upday"] = end_upday
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_file, index=False, encoding="utf_8_sig")
    print(f"[SAVE] {report_file}", flush=True)
    print(report["status"].value_counts(dropna=False).to_string(), flush=True)


if __name__ == "__main__":
    main()
