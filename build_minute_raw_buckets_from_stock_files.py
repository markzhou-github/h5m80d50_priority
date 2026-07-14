#!/usr/bin/env python
# coding: utf-8
"""Build minute raw bucket parquet files from existing per-stock minute files.

This is the one-time/bootstrap converter for the new minute architecture.

Input:
  data/raw/{ts_code}.parquet
  csi1500con.csv with minute_bucket_id and minute_bucket_file

Output:
  data/minute_raw_buckets/bucket_00.parquet ... bucket_19.parquet

Rows are filtered by start/end date and sorted by ts_code + trade_time.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CON_FILE = PROJECT_ROOT / "csi1500con.csv"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "minute_raw_buckets"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "processed" / "minute_raw_buckets_report"

RAW_COLS = ["ts_code", "trade_time", "open", "close", "high", "low", "vol", "amount", "trade_date"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build minute raw bucket files from per-stock raw parquet files.")
    p.add_argument("--con-file", type=Path, default=DEFAULT_CON_FILE)
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--start-date", default=None, help="YYYYMMDD inclusive.")
    p.add_argument("--end-date", default=None, help="YYYYMMDD inclusive.")
    p.add_argument("--stocks", nargs="*", default=[], help="Optional stock subset.")
    p.add_argument("--max-stocks", type=int, default=None)
    p.add_argument("--bucket-id-col", default="minute_bucket_id")
    p.add_argument("--bucket-file-col", default="minute_bucket_file")
    p.add_argument("--stock-col", default=None, help="Auto-detects con_code or ts_code if omitted.")
    p.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "lz4", "uncompressed"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--infer-bucket-if-missing", action="store_true")
    return p.parse_args()


def normalize_date(value: str | int | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.replace("-", "")


def normalize_ts_code(value: object) -> str:
    text = str(value).strip().upper()
    if not text or text.lower() == "nan":
        return ""
    if len(text) == 6 and text.isdigit():
        return f"{text}.SH" if text.startswith(("6", "9")) else f"{text}.SZ"
    return text


def infer_bucket_id(ts_code: str) -> int:
    number = "".join(ch for ch in ts_code if ch.isdigit())[:6]
    if len(number) != 6:
        raise ValueError(f"Cannot infer bucket id from {ts_code!r}")
    return int(number[-2:]) // 5


def detect_stock_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"--stock-col {requested!r} not found in {df.columns.tolist()}")
        return requested
    for col in ["con_code", "ts_code", "stock_code", "code"]:
        if col in df.columns:
            return col
    raise ValueError("Cannot detect stock column. Pass --stock-col.")


def load_constituents(args: argparse.Namespace) -> pd.DataFrame:
    con = pd.read_csv(args.con_file, dtype=str)
    stock_col = detect_stock_col(con, args.stock_col)
    con["ts_code"] = con[stock_col].map(normalize_ts_code)
    con = con[con["ts_code"] != ""].drop_duplicates("ts_code", keep="last").copy()

    if args.bucket_id_col not in con.columns or args.bucket_file_col not in con.columns:
        if not args.infer_bucket_if_missing:
            raise ValueError(
                f"{args.con_file} must contain {args.bucket_id_col!r} and {args.bucket_file_col!r}. "
                "Run assign_minute_buckets.py first, or pass --infer-bucket-if-missing."
            )
        con[args.bucket_id_col] = con["ts_code"].map(infer_bucket_id).astype("int16")
        con[args.bucket_file_col] = con[args.bucket_id_col].map(lambda x: f"bucket_{int(x):02d}.parquet")

    con[args.bucket_id_col] = pd.to_numeric(con[args.bucket_id_col], errors="raise").astype("int16")
    con[args.bucket_file_col] = con[args.bucket_file_col].astype(str)

    if args.stocks:
        selected = {normalize_ts_code(s) for s in args.stocks}
        con = con[con["ts_code"].isin(selected)].copy()
    if args.max_stocks:
        con = con.sort_values("ts_code").head(args.max_stocks).copy()
    return con[["ts_code", args.bucket_id_col, args.bucket_file_col]].copy()


def read_stock_minute_file(path: Path, ts_code: str, start_date: str | None, end_date: str | None) -> pl.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame()
    schema = pl.read_parquet_schema(path)
    missing = sorted(set(RAW_COLS) - set(schema))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = pl.read_parquet(path, columns=RAW_COLS)
    if df.is_empty():
        return df
    df = df.with_columns(
        pl.lit(ts_code).alias("ts_code"),
        pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", "").alias("trade_date"),
    )
    if start_date:
        df = df.filter(pl.col("trade_date") >= start_date)
    if end_date:
        df = df.filter(pl.col("trade_date") <= end_date)
    if df.is_empty():
        return df

    if df.schema["trade_time"] != pl.Datetime:
        df = df.with_columns(pl.col("trade_time").cast(pl.Utf8).str.to_datetime(strict=False).alias("trade_time"))

    return (
        df.select(RAW_COLS)
        .drop_nulls(["ts_code", "trade_time", "trade_date"])
        .with_columns(
            pl.col("open").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("vol").cast(pl.Float64),
            pl.col("amount").cast(pl.Float64),
        )
    )


def atomic_write_parquet(df: pl.DataFrame, path: Path, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp_path, compression=compression)
    os.replace(tmp_path, path)


def format_elapsed(started: float) -> str:
    elapsed = max(0.0, time.time() - started)
    return f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed / 60.0:.1f}m"


def build_bucket(
    bucket_file: str,
    bucket_stocks: pd.DataFrame,
    source_dir: Path,
    out_dir: Path,
    start_date: str | None,
    end_date: str | None,
    compression: str,
    progress_every: int,
) -> dict[str, Any]:
    started = time.time()
    frames: list[pl.DataFrame] = []
    rows: list[dict[str, Any]] = []
    stocks = bucket_stocks["ts_code"].tolist()
    print(f"[bucket {bucket_file}] stocks={len(stocks)}", flush=True)

    for i, ts_code in enumerate(stocks, 1):
        path = source_dir / f"{ts_code}.parquet"
        try:
            df = read_stock_minute_file(path, ts_code, start_date, end_date)
            if not df.is_empty():
                frames.append(df)
            rows.append(
                {
                    "bucket_file": bucket_file,
                    "ts_code": ts_code,
                    "status": "ok" if not df.is_empty() else "empty",
                    "rows": df.height,
                    "date_min": df.get_column("trade_date").min() if not df.is_empty() else None,
                    "date_max": df.get_column("trade_date").max() if not df.is_empty() else None,
                    "message": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "bucket_file": bucket_file,
                    "ts_code": ts_code,
                    "status": "failed",
                    "rows": 0,
                    "date_min": None,
                    "date_max": None,
                    "message": str(exc),
                }
            )
        if i == 1 or i % progress_every == 0 or i == len(stocks):
            loaded_rows = sum(int(r["rows"] or 0) for r in rows)
            print(
                f"[bucket {bucket_file}] {i}/{len(stocks)} loaded_rows={loaded_rows:,} "
                f"elapsed={format_elapsed(started)}",
                flush=True,
            )

    out_path = out_dir / bucket_file
    if frames:
        bucket = (
            pl.concat(frames, how="diagonal_relaxed")
            .unique(["ts_code", "trade_time"], keep="last")
            .sort(["ts_code", "trade_time"])
        )
    else:
        bucket = pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in RAW_COLS})

    atomic_write_parquet(bucket, out_path, compression)
    print(f"[bucket {bucket_file}] saved rows={bucket.height:,} path={out_path}", flush=True)

    return {
        "bucket_file": bucket_file,
        "bucket_path": str(out_path),
        "stock_count": len(stocks),
        "rows": bucket.height,
        "date_min": bucket.get_column("trade_date").min() if not bucket.is_empty() else None,
        "date_max": bucket.get_column("trade_date").max() if not bucket.is_empty() else None,
        "elapsed_seconds": round(time.time() - started, 3),
        "stock_rows": rows,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    start_date = normalize_date(args.start_date)
    end_date = normalize_date(args.end_date)
    if start_date and end_date and start_date > end_date:
        raise ValueError(f"--start-date must be <= --end-date, got {start_date} > {end_date}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    con = load_constituents(args)
    bucket_files = sorted(con[args.bucket_file_col].drop_duplicates().tolist())
    existing_outputs = [args.out_dir / b for b in bucket_files if (args.out_dir / b).exists()]
    if existing_outputs and not args.overwrite:
        sample = ", ".join(str(p) for p in existing_outputs[:5])
        raise FileExistsError(f"Bucket output files already exist. Pass --overwrite. Examples: {sample}")

    print(f"[range] {start_date or 'min'} ~ {end_date or 'max'}", flush=True)
    print(f"[stocks] {len(con)}", flush=True)
    print(f"[buckets] {len(bucket_files)}", flush=True)
    print(f"[source] {args.source_dir}", flush=True)
    print(f"[out] {args.out_dir}", flush=True)

    bucket_summaries = []
    stock_summaries = []
    for bucket_file in bucket_files:
        bucket_stocks = con[con[args.bucket_file_col] == bucket_file].sort_values("ts_code")
        result = build_bucket(
            bucket_file=bucket_file,
            bucket_stocks=bucket_stocks,
            source_dir=args.source_dir,
            out_dir=args.out_dir,
            start_date=start_date,
            end_date=end_date,
            compression=args.compression,
            progress_every=max(1, args.progress_every),
        )
        stock_rows = result.pop("stock_rows")
        bucket_summaries.append(result)
        stock_summaries.extend(stock_rows)

    bucket_report = args.report_dir / "minute_raw_bucket_summary.csv"
    stock_report = args.report_dir / "minute_raw_bucket_stock_summary.csv"
    pl.DataFrame(bucket_summaries, infer_schema_length=None).write_csv(bucket_report)
    pl.DataFrame(stock_summaries, infer_schema_length=None).write_csv(stock_report)

    print("[done]", flush=True)
    print(f"elapsed_seconds={time.time() - started:.3f}", flush=True)
    print("[SAVE]", bucket_report, flush=True)
    print("[SAVE]", stock_report, flush=True)


if __name__ == "__main__":
    main()
