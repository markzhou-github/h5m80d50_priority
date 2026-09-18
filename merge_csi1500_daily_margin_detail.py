#!/usr/bin/env python
# coding: utf-8
"""Refresh margin_detail columns inside existing CSI1500 merged daily files.

Use this after the delayed margin_detail download finishes. It avoids rerunning
the full interday merge and only replaces margin_detail-sourced columns in each
processed/daily/merged/{ts_code}.all.csv file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from config import STOCK_DATA_DIR


PROJECT_ROOT = Path(__file__).resolve().parent
UNIVERSE_CSV = PROJECT_ROOT / "csi1500con.csv"
BASE_DIR = Path(STOCK_DATA_DIR)
MERGED_DIR = BASE_DIR / "merged"
MARGIN_DIR = BASE_DIR / "margin_detail"
REPORT_DIR = BASE_DIR / "report"

KEYS = ["ts_code", "trade_date"]


def normalize_date(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace("-", "")
    return text or None


def empty_key_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={"ts_code": pl.Utf8, "trade_date": pl.Utf8})


def load_stock_list(max_stocks: int | None = None) -> list[str]:
    df = pl.read_csv(UNIVERSE_CSV, schema_overrides={"con_code": pl.Utf8})
    if "con_code" not in df.columns:
        raise ValueError(f"Missing con_code column in {UNIVERSE_CSV}")
    stocks = (
        df.select(pl.col("con_code").str.strip_chars())
        .drop_nulls()
        .unique()
        .sort("con_code")
        .get_column("con_code")
        .to_list()
    )
    return stocks[:max_stocks] if max_stocks is not None else stocks


def read_csv(path: Path) -> pl.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return empty_key_frame()
    try:
        df = pl.read_csv(path, schema_overrides={"ts_code": pl.Utf8, "trade_date": pl.Utf8})
    except Exception as exc:
        print(f"      [WARN] failed to read {path.name}: {exc}", flush=True)
        return empty_key_frame()
    if df.is_empty():
        return empty_key_frame()
    return df.with_columns(
        pl.col("ts_code").cast(pl.Utf8),
        pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", ""),
    )


def date_filter(df: pl.DataFrame, start_date: str | None, end_date: str | None) -> pl.DataFrame:
    if df.is_empty():
        return df
    out = df
    if start_date:
        out = out.filter(pl.col("trade_date") >= start_date)
    if end_date:
        out = out.filter(pl.col("trade_date") <= end_date)
    return out


def stock_date_coverage(ts_code: str) -> dict:
    merged_path = MERGED_DIR / f"{ts_code}.all.csv"
    margin_path = MARGIN_DIR / f"{ts_code}.margin_detail.csv"
    row = {
        "ts_code": ts_code,
        "merged_rows": 0,
        "merged_last_trade_date": None,
        "margin_rows": 0,
        "margin_last_trade_date": None,
        "status": "ok",
    }
    for label, path in [("merged", merged_path), ("margin", margin_path)]:
        if not path.exists() or path.stat().st_size == 0:
            row["status"] = f"missing_{label}"
            continue
        try:
            stats = (
                pl.scan_csv(path, schema_overrides={"trade_date": pl.Utf8})
                .select(
                    pl.len().alias("rows"),
                    pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", "").max().alias("last_trade_date"),
                )
                .collect()
                .row(0, named=True)
            )
        except Exception as exc:
            print(f"      [WARN] failed to scan {path.name}: {exc}", flush=True)
            row["status"] = f"bad_{label}"
            continue
        row[f"{label}_rows"] = stats["rows"]
        row[f"{label}_last_trade_date"] = stats["last_trade_date"]
    return row


def resolve_common_end_date(stocks: list[str], mode: str) -> tuple[str | None, list[dict]]:
    coverage = [stock_date_coverage(ts_code) for ts_code in stocks]
    dates: list[str] = []
    for row in coverage:
        if mode in ("merged", "both") and row.get("merged_last_trade_date"):
            dates.append(str(row["merged_last_trade_date"]))
        if mode in ("margin", "both") and row.get("margin_last_trade_date"):
            dates.append(str(row["margin_last_trade_date"]))
    common_end = min(dates) if dates else None
    for row in coverage:
        row["common_end_date"] = common_end
        row["merged_after_common_end"] = (
            "" if not common_end or not row.get("merged_last_trade_date") else int(str(row["merged_last_trade_date"]) > common_end)
        )
        row["margin_after_common_end"] = (
            "" if not common_end or not row.get("margin_last_trade_date") else int(str(row["margin_last_trade_date"]) > common_end)
        )
    return common_end, coverage


def merge_margin_one_stock(
    ts_code: str,
    start_date: str | None,
    end_date: str | None,
    dry_run: bool,
) -> dict:
    merged_path = MERGED_DIR / f"{ts_code}.all.csv"
    margin_path = MARGIN_DIR / f"{ts_code}.margin_detail.csv"

    merged = read_csv(merged_path)
    if merged.is_empty():
        return {"ts_code": ts_code, "status": "missing_merged", "rows": 0}

    margin = read_csv(margin_path)
    if margin.is_empty():
        return {"ts_code": ts_code, "status": "missing_margin_detail", "rows": merged.height}

    margin = date_filter(margin, start_date, end_date)
    if margin.is_empty():
        return {"ts_code": ts_code, "status": "empty_margin_window", "rows": merged.height}

    if end_date:
        merged = merged.filter(pl.col("trade_date") <= end_date)

    margin_cols = [c for c in margin.columns if c not in KEYS]
    old_cols = [c for c in margin_cols if c in merged.columns]
    base = merged.drop(old_cols) if old_cols else merged
    out = (
        base.join(margin.select(KEYS + margin_cols), on=KEYS, how="left")
        .sort("trade_date")
    )

    if not dry_run:
        out.write_csv(merged_path, include_bom=True)

    return {
        "ts_code": ts_code,
        "status": "updated",
        "rows": out.height,
        "first_trade_date": out.select(pl.col("trade_date").min()).item(),
        "last_trade_date": out.select(pl.col("trade_date").max()).item(),
        "margin_rows_used": margin.height,
        "margin_columns_replaced": len(old_cols),
        "margin_columns_joined": len(margin_cols),
        "output_file": str(merged_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=None, help="Optional YYYYMMDD lower bound for margin rows to use.")
    parser.add_argument("--end-date", default=None, help="Optional YYYYMMDD final merged cutoff.")
    parser.add_argument(
        "--common-end-mode",
        choices=["none", "merged", "margin", "both"],
        default="none",
        help=(
            "When --end-date is not provided, resolve common cutoff from merged files, margin files, or both. "
            "Default none patches margin columns without cutting merged dates."
        ),
    )
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start_date = normalize_date(args.start_date)
    end_date = normalize_date(args.end_date)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stocks = load_stock_list(args.max_stocks)
    common_end, coverage = resolve_common_end_date(stocks, args.common_end_mode)
    if end_date is None and args.common_end_mode != "none":
        end_date = common_end

    coverage_path = REPORT_DIR / "merge_csi1500_daily_margin_detail_date_coverage.csv"
    pl.DataFrame(coverage).write_csv(coverage_path, include_bom=True)

    print(f"Margin-detail refresh stocks: {len(stocks)}")
    print(f"Source dirs: merged={MERGED_DIR} margin_detail={MARGIN_DIR}")
    print(f"Window: start={start_date or 'min'} end={end_date or 'max'} common_end_mode={args.common_end_mode}")
    print(f"Date coverage report: {coverage_path}")
    if args.dry_run:
        print("[dry-run] no merged files will be written")

    summary = []
    for i, ts_code in enumerate(stocks, 1):
        print(f"[margin merge {i}/{len(stocks)}] {ts_code}", flush=True)
        summary.append(merge_margin_one_stock(ts_code, start_date, end_date, args.dry_run))

    report_path = REPORT_DIR / "merge_csi1500_daily_margin_detail_summary.csv"
    pl.DataFrame(summary).write_csv(report_path, include_bom=True)
    updated = sum(1 for row in summary if row.get("status") == "updated")
    print(f"\n[DONE] updated={updated}/{len(summary)} report saved: {report_path}")


if __name__ == "__main__":
    main()
