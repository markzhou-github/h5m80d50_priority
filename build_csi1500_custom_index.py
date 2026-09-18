#!/usr/bin/env python
# coding: utf-8
"""
Build custom CSI1500 equal-weight and market-cap-weighted daily indexes.

Source:
  processed/daily/merged/{ts_code}.all.csv

Output:
  processed/index/csi1500_custom_index.csv

Rules:
  - Use only rows with invalid4train == 0.
  - Equal-weight index uses the simple average of valid stock returns.
  - Market-cap-weighted index uses circ_mv weights.
  - Index levels start at 1.0 for easier debugging/plotting.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from config import STOCK_DATA_DIR, STOCK_INDEX_DIR
from config_date import history_start_date, refresh_start_date, end_date


PROJECT_ROOT = Path(__file__).resolve().parent
UNIVERSE_CSV = PROJECT_ROOT / "csi1500con.csv"
MERGED_DIR = Path(STOCK_DATA_DIR) / "merged"
OUT_DIR = Path(STOCK_INDEX_DIR)
REPORT_DIR = OUT_DIR / "report"
OUT_PATH = OUT_DIR / "csi1500_custom_index.csv"
SUMMARY_PATH = REPORT_DIR / "build_csi1500_custom_index_summary.csv"
MISSING_PATH = REPORT_DIR / "build_csi1500_custom_index_missing_files.csv"

# Historical phase. Keep refresh_start_date imported for future maintenance.
Start_date = history_start_date
End_date = end_date

BASE_LEVEL = 1.0

NEEDED_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "pct_chg",
    "vol",
    "amount",
    "circ_mv",
    "invalid4train",
    "limit",
]


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


def read_stock_file(path: Path, ts_code: str) -> pl.DataFrame:
    with path.open("r", encoding="utf-8-sig") as f:
        available_columns = f.readline().strip().split(",")

    read_columns = [col for col in NEEDED_COLUMNS if col in available_columns]
    schema_overrides = {
        "ts_code": pl.Utf8,
        "trade_date": pl.Utf8,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "pre_close": pl.Float64,
        "pct_chg": pl.Float64,
        "vol": pl.Float64,
        "amount": pl.Float64,
        "circ_mv": pl.Float64,
        "invalid4train": pl.Int64,
        "limit": pl.Utf8,
    }
    df = pl.read_csv(
        path,
        columns=read_columns,
        schema_overrides={k: v for k, v in schema_overrides.items() if k in read_columns},
        null_values=["", "NA", "N/A", "nan", "None"],
    )
    if "invalid4train" not in df.columns:
        df = df.with_columns(pl.lit(0).alias("invalid4train"))

    for col in NEEDED_COLUMNS:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))

    return (
        df.select(NEEDED_COLUMNS)
        .with_columns(
            pl.lit(ts_code).alias("ts_code"),
            pl.col("trade_date").cast(pl.Utf8),
            pl.col("open").cast(pl.Float64, strict=False),
            pl.col("high").cast(pl.Float64, strict=False),
            pl.col("low").cast(pl.Float64, strict=False),
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("pre_close").cast(pl.Float64, strict=False),
            pl.col("pct_chg").cast(pl.Float64, strict=False),
            pl.col("vol").cast(pl.Float64, strict=False),
            pl.col("amount").cast(pl.Float64, strict=False),
            pl.col("circ_mv").cast(pl.Float64, strict=False),
            pl.col("invalid4train").cast(pl.Int64, strict=False).fill_null(0),
            pl.col("limit").cast(pl.Utf8, strict=False),
        )
        .filter((pl.col("trade_date") >= Start_date) & (pl.col("trade_date") <= End_date))
    )


def load_all_stock_rows(stock_list: list[str]) -> tuple[pl.DataFrame, list[dict]]:
    frames = []
    missing = []
    for i, ts_code in enumerate(stock_list, 1):
        path = MERGED_DIR / f"{ts_code}.all.csv"
        print(f"[load {i}/{len(stock_list)}] {ts_code}")
        if not path.exists():
            missing.append({"ts_code": ts_code, "missing_file": str(path)})
            continue
        frames.append(read_stock_file(path, ts_code))

    if not frames:
        return pl.DataFrame(), missing
    return pl.concat(frames, how="vertical"), missing


def prepare_valid_rows(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("invalid4train") == 0)
        .with_columns(
            pl.when(pl.col("pre_close") > 0).then(pl.col("open") / pl.col("pre_close") - 1).otherwise(None).alias("open_ret"),
            pl.when(pl.col("pre_close") > 0).then(pl.col("high") / pl.col("pre_close") - 1).otherwise(None).alias("high_ret"),
            pl.when(pl.col("pre_close") > 0).then(pl.col("low") / pl.col("pre_close") - 1).otherwise(None).alias("low_ret"),
            pl.when(pl.col("pct_chg").is_not_null()).then(pl.col("pct_chg") / 100.0)
            .when(pl.col("pre_close") > 0).then(pl.col("close") / pl.col("pre_close") - 1)
            .otherwise(None)
            .alias("close_ret"),
        )
        .filter(
            pl.col("trade_date").is_not_null()
            & pl.col("open_ret").is_not_null()
            & pl.col("high_ret").is_not_null()
            & pl.col("low_ret").is_not_null()
            & pl.col("close_ret").is_not_null()
        )
    )


def build_daily_returns(valid: pl.DataFrame, universe_count: int) -> pl.DataFrame:
    valid = valid.with_columns(
        pl.when((pl.col("circ_mv").is_not_null()) & (pl.col("circ_mv") > 0))
        .then(pl.col("circ_mv"))
        .otherwise(None)
        .alias("mcap_weight_base")
    )

    ew = valid.group_by("trade_date").agg(
        pl.col("open_ret").mean().alias("csi1500_ew_open_ret"),
        pl.col("high_ret").mean().alias("csi1500_ew_high_ret"),
        pl.col("low_ret").mean().alias("csi1500_ew_low_ret"),
        pl.col("close_ret").mean().alias("csi1500_ew_close_ret"),
        pl.col("close_ret").median().alias("csi1500_ret_median"),
        pl.col("close_ret").std().alias("csi1500_ret_std"),
        (pl.col("close_ret") > 0).mean().alias("csi1500_up_ratio"),
        (pl.col("close_ret") < 0).mean().alias("csi1500_down_ratio"),
        (pl.col("close_ret") > 0.02).mean().alias("csi1500_gt_2pct_ratio"),
        (pl.col("close_ret") > 0.03).mean().alias("csi1500_gt_3pct_ratio"),
        (pl.col("close_ret") > 0.05).mean().alias("csi1500_gt_5pct_ratio"),
        (pl.col("close_ret") < -0.02).mean().alias("csi1500_lt_minus_2pct_ratio"),
        ((pl.col("limit") == "U") | (pl.col("close_ret") >= 0.095)).sum().alias("csi1500_limit_up_count"),
        ((pl.col("limit") == "D") | (pl.col("close_ret") <= -0.095)).sum().alias("csi1500_limit_down_count"),
        pl.len().alias("csi1500_valid_count"),
        pl.col("vol").sum().alias("csi1500_total_vol"),
        pl.col("amount").sum().alias("csi1500_total_amount"),
        pl.col("circ_mv").sum().alias("csi1500_total_circ_mv"),
    )

    weighted_source = valid.filter(pl.col("mcap_weight_base").is_not_null()).with_columns(
        pl.col("mcap_weight_base").sum().over("trade_date").alias("mcap_sum")
    ).with_columns(
        (pl.col("mcap_weight_base") / pl.col("mcap_sum")).alias("mcap_weight")
    )

    mcap = weighted_source.group_by("trade_date").agg(
        (pl.col("open_ret") * pl.col("mcap_weight")).sum().alias("csi1500_mcap_open_ret"),
        (pl.col("high_ret") * pl.col("mcap_weight")).sum().alias("csi1500_mcap_high_ret"),
        (pl.col("low_ret") * pl.col("mcap_weight")).sum().alias("csi1500_mcap_low_ret"),
        (pl.col("close_ret") * pl.col("mcap_weight")).sum().alias("csi1500_mcap_close_ret"),
        pl.len().alias("csi1500_mcap_valid_count"),
        pl.col("mcap_weight").max().alias("csi1500_mcap_max_weight"),
        pl.col("mcap_weight").sort(descending=True).head(10).sum().alias("csi1500_mcap_top10_weight"),
    )

    out = ew.join(mcap, on="trade_date", how="left")
    out = out.with_columns(
        (pl.col("csi1500_valid_count") / pl.lit(universe_count)).alias("csi1500_coverage_ratio"),
        (pl.col("csi1500_mcap_valid_count") / pl.lit(universe_count)).alias("csi1500_mcap_coverage_ratio"),
    )
    return out.sort("trade_date")


def add_index_levels(df: pl.DataFrame) -> pl.DataFrame:
    rows = df.to_dicts()
    ew_prev_close = BASE_LEVEL
    mcap_prev_close = BASE_LEVEL

    for row in rows:
        ew_open = ew_prev_close * (1 + row["csi1500_ew_open_ret"])
        ew_high = ew_prev_close * (1 + row["csi1500_ew_high_ret"])
        ew_low = ew_prev_close * (1 + row["csi1500_ew_low_ret"])
        ew_close = ew_prev_close * (1 + row["csi1500_ew_close_ret"])

        mcap_open = mcap_prev_close * (1 + row["csi1500_mcap_open_ret"])
        mcap_high = mcap_prev_close * (1 + row["csi1500_mcap_high_ret"])
        mcap_low = mcap_prev_close * (1 + row["csi1500_mcap_low_ret"])
        mcap_close = mcap_prev_close * (1 + row["csi1500_mcap_close_ret"])

        row["csi1500_ew_open"] = ew_open
        row["csi1500_ew_high"] = max(ew_high, ew_open, ew_close)
        row["csi1500_ew_low"] = min(ew_low, ew_open, ew_close)
        row["csi1500_ew_close"] = ew_close

        row["csi1500_mcap_open"] = mcap_open
        row["csi1500_mcap_high"] = max(mcap_high, mcap_open, mcap_close)
        row["csi1500_mcap_low"] = min(mcap_low, mcap_open, mcap_close)
        row["csi1500_mcap_close"] = mcap_close

        ew_prev_close = ew_close
        mcap_prev_close = mcap_close

    return pl.DataFrame(rows)


def save_report(rows: list[dict], path: Path) -> None:
    if rows:
        pl.DataFrame(rows).write_csv(path, include_bom=True)
    else:
        path.write_text("", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    stock_list = load_stock_list()
    raw, missing = load_all_stock_rows(stock_list)
    if raw.is_empty():
        raise RuntimeError(f"No source rows loaded from {MERGED_DIR}")

    valid = prepare_valid_rows(raw)
    daily = build_daily_returns(valid, universe_count=len(stock_list))
    result = add_index_levels(daily)
    result.write_csv(OUT_PATH, include_bom=True)
    save_report(missing, MISSING_PATH)

    summary = [{
        "start_date": Start_date,
        "end_date": End_date,
        "universe_count": len(stock_list),
        "source_rows": raw.height,
        "valid_rows": valid.height,
        "index_rows": result.height,
        "first_trade_date": result.select(pl.col("trade_date").min()).item(),
        "last_trade_date": result.select(pl.col("trade_date").max()).item(),
        "missing_files": len(missing),
        "output_file": str(OUT_PATH),
    }]
    save_report(summary, SUMMARY_PATH)

    print(f"[SAVE] {OUT_PATH}")
    print(f"[SAVE] {SUMMARY_PATH}")
    print(f"[SAVE] {MISSING_PATH}")
    print(pl.DataFrame(summary))


if __name__ == "__main__":
    main()
