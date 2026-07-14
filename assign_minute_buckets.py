#!/usr/bin/env python
# coding: utf-8
"""Assign deterministic minute-data buckets to stocks in csi1500con.csv.

Bucket rule:
  stock number ending 00-04 -> bucket_00
  stock number ending 05-09 -> bucket_01
  ...
  stock number ending 95-99 -> bucket_19

The bucket id is computed as int(last_two_digits) // 5.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("csi1500con.csv")
DEFAULT_BUCKET_FILE_TEMPLATE = "bucket_{bucket_id:02d}.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assign minute bucket columns to csi1500con.csv.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input csi1500 constituent CSV.")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV. Defaults to <input stem>_with_minute_buckets.csv unless --inplace is used.",
    )
    p.add_argument("--inplace", action="store_true", help="Overwrite input CSV atomically.")
    p.add_argument(
        "--stock-col",
        default=None,
        help="Stock code column. Auto-detects con_code, ts_code, or stock_code if omitted.",
    )
    p.add_argument("--bucket-digit-span", type=int, default=5, help="Last-two-digit span per bucket. Default 5 -> 20 buckets.")
    p.add_argument("--bucket-count", type=int, default=20, help="Expected bucket count. Default 20.")
    p.add_argument("--bucket-id-col", default="minute_bucket_id")
    p.add_argument("--bucket-file-col", default="minute_bucket_file")
    p.add_argument("--bucket-file-template", default=DEFAULT_BUCKET_FILE_TEMPLATE)
    p.add_argument("--encoding", default="utf-8-sig", help="CSV encoding for output. Default utf-8-sig.")
    return p.parse_args()


def detect_stock_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"--stock-col {requested!r} not found. Available columns: {df.columns.tolist()}")
        return requested
    for col in ["con_code", "ts_code", "stock_code", "code"]:
        if col in df.columns:
            return col
    raise ValueError("Could not auto-detect stock code column. Pass --stock-col.")


def normalize_ts_code(value: object) -> str:
    text = str(value).strip().upper()
    if not text or text.lower() == "nan":
        return ""
    if re.fullmatch(r"\d{6}", text):
        suffix = ".SH" if text.startswith(("6", "9")) else ".SZ"
        return f"{text}{suffix}"
    return text


def stock_number(ts_code: str) -> str:
    match = re.search(r"(\d{6})", ts_code)
    if not match:
        raise ValueError(f"Cannot extract 6-digit stock number from {ts_code!r}")
    return match.group(1)


def assign_bucket(ts_code: str, bucket_digit_span: int, bucket_count: int) -> int:
    if bucket_digit_span <= 0:
        raise ValueError("--bucket-digit-span must be positive.")
    number = stock_number(ts_code)
    last_two = int(number[-2:])
    bucket_id = last_two // bucket_digit_span
    if bucket_id < 0 or bucket_id >= bucket_count:
        raise ValueError(
            f"Bucket id out of range for {ts_code}: last_two={last_two}, "
            f"bucket_id={bucket_id}, bucket_count={bucket_count}"
        )
    return int(bucket_id)


def atomic_write_csv(df: pd.DataFrame, path: Path, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False, encoding=encoding)
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, dtype=str)
    stock_col = detect_stock_col(df, args.stock_col)

    normalized = df[stock_col].map(normalize_ts_code)
    missing = normalized.eq("")
    if missing.any():
        examples = df.loc[missing, stock_col].head(10).tolist()
        raise ValueError(f"Found empty/invalid stock codes in {stock_col}: {examples}")

    bucket_ids = normalized.map(lambda x: assign_bucket(x, args.bucket_digit_span, args.bucket_count)).astype("int16")
    df[args.bucket_id_col] = bucket_ids
    df[args.bucket_file_col] = bucket_ids.map(lambda x: args.bucket_file_template.format(bucket_id=int(x)))

    out_path = args.input if args.inplace else args.output
    if out_path is None:
        out_path = args.input.with_name(f"{args.input.stem}_with_minute_buckets{args.input.suffix}")

    summary = (
        df.groupby(args.bucket_id_col, dropna=False)
        .size()
        .rename("stock_count")
        .reset_index()
        .sort_values(args.bucket_id_col)
    )

    atomic_write_csv(df, out_path, args.encoding)
    print(f"[SAVE] {out_path}")
    print(f"[stock_col] {stock_col}")
    print(f"[rows] {len(df)}")
    print("[bucket_summary]")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
