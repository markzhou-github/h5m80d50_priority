#!/usr/bin/env python
# coding: utf-8
"""
Merge downloaded CSI1500 daily/interday files after margin_detail is available.

This version uses Polars for faster repeated per-stock joins.
"""
from __future__ import annotations

from pathlib import Path

import pandas_market_calendars as mcal
import polars as pl

from config import STOCK_DATA_DIR
from config_date import history_start_date, refresh_start_date, end_date


PROJECT_ROOT = Path(__file__).resolve().parent
UNIVERSE_CSV = PROJECT_ROOT / "csi1500con.csv"
BASE_DIR = Path(STOCK_DATA_DIR)
MERGED_DIR = BASE_DIR / "merged"
REPORT_DIR = BASE_DIR / "report"

Start_date = history_start_date
End_date = end_date

KEYS = ["ts_code", "trade_date"]

LIMIT_FILL_DEFAULTS = {
#    "limit_amount": 0.0,
#    "fd_amount": 0.0,
#    "open_times": 0,
    "limit_times": 0,
#    "limit": 0,
}


def build_sse_calendar(start_date: str = Start_date, end_date: str = End_date) -> list[str]:
    sse = mcal.get_calendar("SSE")
    days = sse.valid_days(start_date, end_date)
    return days.strftime("%Y%m%d").to_list()


def empty_key_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={"ts_code": pl.Utf8, "trade_date": pl.Utf8})


def load_stock_list() -> list[str]:
    df = pl.read_csv(UNIVERSE_CSV, schema_overrides={"con_code": pl.Utf8})
    if "con_code" not in df.columns:
        raise ValueError(f"Missing con_code column in {UNIVERSE_CSV}")
    return (
        df.select(pl.col("con_code").str.strip_chars())
        .drop_nulls()
        .unique()
        .sort("con_code")
        .get_column("con_code")
        .to_list()
    )


def read_dataset(path: Path) -> pl.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return empty_key_frame()
    try:
        df = pl.read_csv(path, schema_overrides={"ts_code": pl.Utf8, "trade_date": pl.Utf8})
    except Exception as exc:
        print(f"      [WARN] failed to read {path.name}: {exc}")
        return empty_key_frame()
    if df.is_empty():
        return empty_key_frame()
    return df.with_columns(
        pl.col("ts_code").cast(pl.Utf8),
        pl.col("trade_date").cast(pl.Utf8),
    )


def join_optional(base: pl.DataFrame, path: Path, suffix: str) -> pl.DataFrame:
    df = read_dataset(path)
    if df.is_empty():
        return base

    cols = [c for c in df.columns if c in KEYS or c not in base.columns]
    if cols == KEYS:
        return base

    return base.join(df.select(cols), on=KEYS, how="left", suffix=suffix)


def join_auction(base: pl.DataFrame, path: Path, kind: str) -> pl.DataFrame:
    df = read_dataset(path)
    if df.is_empty():
        return base

    rename = {
        "open": f"open_auction_{kind}",
        "high": f"high_auction_{kind}",
        "low": f"low_auction_{kind}",
        "close": f"close_auction_{kind}",
        "vol": f"vol_auction_{kind}",
        "amount": f"amount_auction_{kind}",
        "vwap": f"vwap_auction_{kind}",
    }
    df = df.rename({old: new for old, new in rename.items() if old in df.columns})
    cols = [c for c in df.columns if c in KEYS or c not in base.columns]
    if cols == KEYS:
        return base

    return base.join(df.select(cols), on=KEYS, how="left")


def add_missing_limit_defaults(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for col, value in LIMIT_FILL_DEFAULTS.items():
        if col in df.columns:
            exprs.append(pl.col(col).fill_null(value))
        else:
            exprs.append(pl.lit(value).alias(col))
    return df.with_columns(exprs)


def join_limit(base: pl.DataFrame, path: Path) -> pl.DataFrame:
    df = read_dataset(path)
    keep = [
        "limit_amount",
        "float_mv",
        "fd_amount",
        "first_time",
        "last_time",
        "open_times",
        "up_stat",
        "limit_times",
        "limit",
    ]

    if df.is_empty():
        return add_missing_limit_defaults(base)

    cols = [c for c in KEYS + keep if c in df.columns]
    out = base.join(df.select(cols), on=KEYS, how="left", suffix="_limit")
    return add_missing_limit_defaults(out)


def add_invalid4train_flag(
    df: pl.DataFrame,
    market_calendar: list[str],
    ipo_buffer_days: int = 20,
    retire_buffer_days: int = 20,
    retire_grace_days: int = 30,
    pre_suspend_buffer_days: int = 10,
    resume_buffer_days: int = 10,
) -> pl.DataFrame:
    if df.is_empty():
        return df.with_columns(pl.lit(0).alias("invalid4train"))

    sorted_df = df.sort("trade_date")
    trade_dates = sorted_df.get_column("trade_date").cast(pl.Utf8).to_list()
    invalid = [0] * len(trade_dates)
    calendar_pos = {date: i for i, date in enumerate(market_calendar)}

    if not market_calendar:
        return sorted_df.with_columns(pl.Series("invalid4train", invalid))

    first_market_date = market_calendar[0]
    last_market_date = market_calendar[-1]

    # Grace period: if the stock's last date is still within the recent
    # market window, do not treat it as delisted/frozen.
    effective_last_market_date = market_calendar[
        max(0, len(market_calendar) - 1 - retire_grace_days)
    ]

    if trade_dates[0] > first_market_date:
        for i in range(min(ipo_buffer_days, len(invalid))):
            invalid[i] = 1

    if trade_dates[-1] < effective_last_market_date:
        start = max(0, len(invalid) - retire_buffer_days)
        for i in range(start, len(invalid)):
            invalid[i] = 1
            
    for i in range(len(trade_dates) - 1):
        current_pos = calendar_pos.get(trade_dates[i])
        next_pos = calendar_pos.get(trade_dates[i + 1])
        if current_pos is None or next_pos is None:
            continue
        if next_pos - current_pos <= 1:
            continue

        pre_start = max(0, i - pre_suspend_buffer_days + 1)
        for j in range(pre_start, i + 1):
            invalid[j] = 1

        resume_end = min(len(invalid), i + 1 + resume_buffer_days)
        for j in range(i + 1, resume_end):
            invalid[j] = 1

    return sorted_df.with_columns(pl.Series("invalid4train", invalid))


def merge_one_stock(ts_code: str, market_calendar: list[str]) -> tuple[dict, pl.DataFrame | None]:
    stkfactor_file = BASE_DIR / "stkfactor" / f"{ts_code}.stkfactor.csv"
    base = read_dataset(stkfactor_file)
    if base.is_empty():
        return {"ts_code": ts_code, "rows": 0, "status": "missing_stkfactor"}, None

    merged = base.sort("trade_date")
    merged = join_optional(merged, BASE_DIR / "moneyflow" / f"{ts_code}.moneyflow.csv", suffix="_mf")
    merged = join_optional(merged, BASE_DIR / "margin_detail" / f"{ts_code}.margin_detail.csv", suffix="_mg")
    merged = join_optional(merged, BASE_DIR / "cyq_perf" / f"{ts_code}.cyq_perf.csv", suffix="_cyq")
    merged = join_auction(merged, BASE_DIR / "auction_o" / f"{ts_code}.auction_o.csv", kind="o")
    merged = join_auction(merged, BASE_DIR / "auction_c" / f"{ts_code}.auction_c.csv", kind="c")
    merged = join_limit(merged, BASE_DIR / "limit" / f"{ts_code}.limit.csv")
    merged = add_invalid4train_flag(merged, market_calendar)

    return {
        "ts_code": ts_code,
        "rows": merged.height,
        "invalid4train_rows": merged.select(pl.col("invalid4train").sum()).item(),
        "first_trade_date": merged.select(pl.col("trade_date").min()).item(),
        "last_trade_date": merged.select(pl.col("trade_date").max()).item(),
        "status": "merged",
    }, merged


def main() -> None:
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    stock_list = load_stock_list()
    print(f"Merge range reference: {Start_date} ~ {End_date}")
    print(f"Refresh start imported for future use: {refresh_start_date}")
    print(f"CSI1500 stock count: {len(stock_list)}")
    print(f"Output directory: {MERGED_DIR}")
    market_calendar = build_sse_calendar()
    print(f"SSE trading days: {len(market_calendar)}")

    summary = []
    for i, ts_code in enumerate(stock_list, 1):
        print(f"[merge {i}/{len(stock_list)}] {ts_code}")
        info, merged = merge_one_stock(ts_code, market_calendar)
        if merged is not None:
            out_path = MERGED_DIR / f"{ts_code}.all.csv"
            merged.write_csv(out_path, include_bom=True)
            info["output_file"] = str(out_path)
        summary.append(info)

    report_path = REPORT_DIR / "merge_csi1500_daily_summary.csv"
    pl.DataFrame(summary).write_csv(report_path, include_bom=True)
    print(f"\n[DONE] report saved: {report_path}")


if __name__ == "__main__":
    main()
