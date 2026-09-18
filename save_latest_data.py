#!/usr/bin/env python3
# save latest trade_date data for future audit
# python save_latest_data.py --parquet-path processed/train_v5b/train_v5b.parquet --prefix train --output-dir audit

import argparse
from datetime import datetime
from pathlib import Path

import polars as pl

def valid_date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--latest-date must be a valid date in YYYYMMDD format"
        ) from exc

    return value
    
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export rows with the latest trade_date from a parquet file."
    )
    parser.add_argument(
        "--latest-date",
        type=valid_date,
        default=None,
        help="Date to export in YYYYMMDD format. Defaults to the latest date in the parquet file.",
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        help="Path to the source parquet file.",
    )
    parser.add_argument(
        "--prefix",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the parquet file's directory.",
    )
    args = parser.parse_args()

    if not args.parquet_path.is_file():
        parser.error(f"Parquet file does not exist: {args.parquet_path}")

    output_dir = args.output_dir or args.parquet_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    source = pl.scan_parquet(args.parquet_path).with_columns(
        pl.col("trade_date")
        .cast(pl.Utf8)
        .str.replace_all("-", "")
        .str.slice(0, 8)
        .alias("trade_date")
    )

    if args.latest_date:
        latest_trade_date = args.latest_date
    else:
        latest_trade_date = (
            source.select(pl.col("trade_date").max())
            .collect(engine="streaming")
            .item()
        )

        if latest_trade_date is None:
            raise ValueError(f"No rows found in {args.parquet_path}")

    save_date = datetime.now().strftime("%Y%m%d")
    output_path = output_dir / (
        f"{args.prefix}_{latest_trade_date}_{save_date}.csv"
    )

    (
        source
        .filter(pl.col("trade_date") == latest_trade_date)
        .sink_csv(output_path)
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()