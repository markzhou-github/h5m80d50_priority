#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

MODEL_CONFIG = {
    "H2M80D50": {
        "horizon": 2,
        "gain_profit": 0.08,
        "cut_loss": 0.05,
    },
    "H5M80D50": {
        "horizon": 5,
        "gain_profit": 0.08,
        "cut_loss": 0.05,
    },
}

PRICE_COLUMNS = ("open", "close", "high")
MAX_PRICE_DAY = 5


@dataclass
class StockData:
    data: pd.DataFrame
    date_to_position: dict[str, int]


def normalize_date(value) -> Optional[str]:
    """
    Convert values such as:
        20260717
        20260717.0
        2026-07-17

    into:
        20260717
    """
    if pd.isna(value):
        return None

    value_str = str(value).strip()

    if value_str.endswith(".0") and value_str[:-2].isdigit():
        value_str = value_str[:-2]

    parsed = pd.to_datetime(value_str, errors="coerce")

    if pd.isna(parsed):
        return None

    return parsed.strftime("%Y%m%d")


def normalize_date_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_date).astype("string")


def load_default_dates() -> tuple[str, str]:
    """
    Load defaults from config.py.

    Expected config.py variables:

        START_DATE = "20260717"
        END_DATE = "20260731"
    """
    try:
        from config_date import End_date
        Start_date = End_date
    except ImportError as exc:
        raise RuntimeError(
            "start_date or end_date was not supplied, but config.py "
            "could not be imported."
        ) from exc
    except AttributeError as exc:
        raise RuntimeError(
            "config.py must contain START_DATE and END_DATE."
        ) from exc

    start_date = normalize_date(Start_date)
    end_date = normalize_date(End_date)

    if start_date is None or end_date is None:
        raise ValueError(
            "START_DATE and END_DATE in config.py must be valid dates."
        )

    return start_date, end_date


def parse_model(model: str) -> dict:
    model = model.strip().upper()

    if model not in MODEL_CONFIG:
        valid_models = ", ".join(MODEL_CONFIG)
        raise ValueError(
            f"Invalid model: {model!r}. Valid models: {valid_models}"
        )

    return MODEL_CONFIG[model]


class SourceFileResolver:
    """
    Find the source CSV whose filename contains ts_code.

    Results are cached so the directory is not searched repeatedly for
    the same stock.
    """

    def __init__(self, source_dir: Path):
        self.source_dir = source_dir

        if not source_dir.exists():
            raise FileNotFoundError(
                f"Source directory does not exist: {source_dir}"
            )

        if not source_dir.is_dir():
            raise NotADirectoryError(
                f"source_dir is not a directory: {source_dir}"
            )

        self.csv_files = list(source_dir.rglob("*.csv"))

        if not self.csv_files:
            raise FileNotFoundError(
                f"No CSV files found under: {source_dir}"
            )

        self._cache: dict[str, Path] = {}

    def find(self, ts_code: str) -> Path:
        ts_code = str(ts_code).strip()

        if ts_code in self._cache:
            return self._cache[ts_code]

        ts_code_lower = ts_code.lower()

        # First try the exact ts_code, for example 000001.SZ.
        matches = [
            path
            for path in self.csv_files
            if ts_code_lower in path.stem.lower()
        ]

        # If no exact match exists, try common sanitized filename forms.
        if not matches:
            alternate_codes = {
                ts_code_lower.replace(".", "_"),
                ts_code_lower.replace(".", "-"),
                ts_code_lower.replace(".", ""),
            }

            matches = [
                path
                for path in self.csv_files
                if any(
                    code and code in path.stem.lower()
                    for code in alternate_codes
                )
            ]

        if not matches:
            raise FileNotFoundError(
                f"No source CSV filename contains ts_code {ts_code!r}"
            )

        if len(matches) > 1:
            matched_names = "\n".join(f"  - {path}" for path in matches[:10])

            raise RuntimeError(
                f"Multiple source files match ts_code {ts_code!r}:\n"
                f"{matched_names}"
            )

        self._cache[ts_code] = matches[0]
        return matches[0]


@lru_cache(maxsize=256)
def load_stock_data(file_path: str) -> StockData:
    """
    Load and cache one stock source CSV.
    """
    path = Path(file_path)

    data = pd.read_csv(path, dtype={"trade_date": "string"})
    data.columns = data.columns.str.strip()

    required_columns = {"trade_date", *PRICE_COLUMNS}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    data["trade_date"] = normalize_date_series(data["trade_date"])

    for column in PRICE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = (
        data.dropna(subset=["trade_date"])
        .sort_values("trade_date")
        .drop_duplicates(subset=["trade_date"], keep="last")
        .reset_index(drop=True)
    )

    date_to_position = {
        trade_date: position
        for position, trade_date in enumerate(data["trade_date"])
    }

    return StockData(
        data=data,
        date_to_position=date_to_position,
    )


def extract_t0_to_t5(
    stock_data: StockData,
    trade_date: str,
) -> Optional[dict[str, float]]:
    """
    Treat trade_date as T0 and extract open/close/high from T0 through T5.

    T1, T2, etc. are based on the next rows in the source data, so they
    represent actual trading days rather than calendar days.
    """
    t0_position = stock_data.date_to_position.get(trade_date)

    if t0_position is None:
        return None

    result: dict[str, float] = {}

    for day in range(MAX_PRICE_DAY + 1):
        source_position = t0_position + day

        for price_column in PRICE_COLUMNS:
            output_column = f"T{day}_{price_column}"

            if source_position >= len(stock_data.data):
                result[output_column] = np.nan
            else:
                result[output_column] = stock_data.data.at[
                    source_position,
                    price_column,
                ]

    return result


def is_valid_price(value) -> bool:
    return pd.notna(value) and np.isfinite(float(value))


def calculate_performance(
    prices: dict[str, float],
    horizon: int,
    gain_profit: float,
    cut_loss: float,
) -> tuple[float, float, float, Optional[int]]:
    """
    Sell logic
    ----------

    Buy:
        buy = T1_open

    T1:
        If T1_close < (1 - cut_loss) * buy:
            sell at T2_open

    T2 through the horizon day:
        1. If open >= (1 + gain_profit) * buy:
               sell at the actual open
        2. Otherwise, if high >= (1 + gain_profit) * buy:
               sell at exactly the take-profit price
        3. Otherwise, if close < (1 - cut_loss) * buy:
               sell at that day's close
        4. If it is the final horizon day:
               sell at that day's close
    """
    buy = prices.get("T1_open", np.nan)

    if not is_valid_price(buy) or float(buy) <= 0:
        return np.nan, np.nan, np.nan, None

    buy = float(buy)

    gain_price = buy * (1.0 + gain_profit)
    loss_price = buy * (1.0 - cut_loss)

    sell = np.nan

    # -------------------------------------------------------------
    # Special T1 rule:
    # A stop-loss signal at T1 close is executed at T2 open.
    # -------------------------------------------------------------
    t1_close = prices.get("T1_close", np.nan)

    if is_valid_price(t1_close) and float(t1_close) < loss_price:
        t2_open = prices.get("T2_open", np.nan)

        if is_valid_price(t2_open):
            sell = float(t2_open)

    # -------------------------------------------------------------
    # T2 through the model horizon.
    # -------------------------------------------------------------
    if not is_valid_price(sell):
        for day in range(2, horizon + 1):
            day_open = prices.get(f"T{day}_open", np.nan)
            day_high = prices.get(f"T{day}_high", np.nan)
            day_close = prices.get(f"T{day}_close", np.nan)

            # Take profit at the opening price if the stock gaps up.
            if (
                is_valid_price(day_open)
                and float(day_open) >= gain_price
            ):
                sell = float(day_open)
                break

            # Take profit intraday.
            if (
                is_valid_price(day_high)
                and float(day_high) >= gain_price
            ):
                sell = gain_price
                break

            # From T2 onward, stop loss is executed at the same day's close.
            if (
                is_valid_price(day_close)
                and float(day_close) < loss_price
            ):
                sell = float(day_close)
                break

            # No trigger by the final horizon day.
            if day == horizon and is_valid_price(day_close):
                sell = float(day_close)
                break

    if not is_valid_price(sell):
        return buy, np.nan, np.nan, None

    sell = float(sell)
    trade_return = sell / buy - 1.0

    # A sale at exactly 1.08 * buy should be target = 1.
    target = int(trade_return >= gain_profit)

    return buy, sell, trade_return, target


def ensure_output_columns(audit_data: pd.DataFrame) -> None:
    """
    Create required output columns when they do not already exist.
    """
    for day in range(MAX_PRICE_DAY + 1):
        for price_column in PRICE_COLUMNS:
            column = f"T{day}_{price_column}"

            if column not in audit_data.columns:
                audit_data[column] = np.nan

    for column in ("buy", "sell", "return"):
        if column not in audit_data.columns:
            audit_data[column] = np.nan

    if "target" not in audit_data.columns:
        audit_data["target"] = pd.Series(
            pd.NA,
            index=audit_data.index,
            dtype="Int64",
        )
    else:
        audit_data["target"] = pd.to_numeric(
            audit_data["target"],
            errors="coerce",
        ).astype("Int64")


def process_audit_file(
    audit_file: Path,
    source_dir: Path,
    model: str,
    start_date: str,
    end_date: str,
) -> None:
    model_config = parse_model(model)

    horizon = model_config["horizon"]
    gain_profit = model_config["gain_profit"]
    cut_loss = model_config["cut_loss"]

    if not audit_file.exists():
        raise FileNotFoundError(
            f"Audit file does not exist: {audit_file}"
        )

    audit_data = pd.read_csv(
        audit_file,
        dtype={
            "trade_date": "string",
            "ts_code": "string",
        },
    )
    audit_data.columns = audit_data.columns.str.strip()

    required_audit_columns = {"trade_date", "ts_code"}
    missing_columns = required_audit_columns - set(audit_data.columns)

    if missing_columns:
        raise ValueError(
            f"Audit file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    normalized_audit_dates = normalize_date_series(
        audit_data["trade_date"]
    )

    selected_mask = (
        normalized_audit_dates.notna()
        & (normalized_audit_dates >= start_date)
        & (normalized_audit_dates <= end_date)
    )

    selected_indices = audit_data.index[selected_mask]

    ensure_output_columns(audit_data)

    resolver = SourceFileResolver(source_dir)

    processed_count = 0
    missing_t0_count = 0
    incomplete_trade_count = 0
    error_count = 0

    for row_number, row_index in enumerate(selected_indices, start=1):
        ts_code = audit_data.at[row_index, "ts_code"]
        trade_date = normalized_audit_dates.at[row_index]

        if pd.isna(ts_code) or not str(ts_code).strip():
            print(
                f"Warning: row {row_index} has an empty ts_code."
            )
            error_count += 1
            continue

        try:
          #  source_file = resolver.find(str(ts_code))
            source_file = str(source_dir) + '/' + ts_code + '.all.csv'
            print('source_file', source_file)
            stock_data = load_stock_data(str(source_file))
            
            

            prices = extract_t0_to_t5(
                stock_data=stock_data,
                trade_date=str(trade_date),
            )

            if prices is None:
                print(
                    f"Warning: T0 {trade_date} was not found for "
                    f"{ts_code}."
                )
                missing_t0_count += 1
                continue

            # Fill T0 through T5 open, close and high.
            for column, value in prices.items():
                audit_data.at[row_index, column] = value

            buy, sell, trade_return, target = calculate_performance(
                prices=prices,
                horizon=horizon,
                gain_profit=gain_profit,
                cut_loss=cut_loss,
            )

            audit_data.at[row_index, "buy"] = buy
            audit_data.at[row_index, "sell"] = sell
            audit_data.at[row_index, "return"] = trade_return

            if target is None:
                audit_data.at[row_index, "target"] = pd.NA
                incomplete_trade_count += 1
            else:
                audit_data.at[row_index, "target"] = target

            processed_count += 1

        except Exception as exc:
            print(
                f"Error processing row {row_index}, "
                f"ts_code={ts_code}, trade_date={trade_date}: {exc}"
            )
            error_count += 1

        if row_number % 100 == 0:
            print(
                f"Processed {row_number:,} / "
                f"{len(selected_indices):,} selected rows..."
            )

    # Write to a temporary file before replacing the original file.
    temporary_file = audit_file.with_suffix(
        audit_file.suffix + ".tmp"
    )

    audit_data.to_csv(
        temporary_file,
        index=False,
        na_rep="",
    )

    os.replace(temporary_file, audit_file)

    print()
    print(f"Audit file saved: {audit_file}")
    print(f"Model: {model}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Selected rows: {len(selected_indices):,}")
    print(f"Processed rows: {processed_count:,}")
    print(f"T0 not found: {missing_t0_count:,}")
    print(f"Incomplete trades: {incomplete_trade_count:,}")
    print(f"Errors: {error_count:,}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill T0-T5 prices and calculate model performance "
            "in an audit CSV."
        )
    )

    parser.add_argument(
        "audit_file",
        type=Path,
        help="Path to the audit CSV file.",
    )

    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing per-stock source CSV files.",
    )

    parser.add_argument(
        "model",
        type=str,
        choices=sorted(MODEL_CONFIG),
        help="Model name, such as H2M80D50 or H5M80D50.",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help=(
            "Start date in YYYYMMDD format. "
            "Defaults to START_DATE from config.py."
        ),
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help=(
            "End date in YYYYMMDD format. "
            "Defaults to END_DATE from config.py."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    start_date = (
        normalize_date(args.start_date)
        if args.start_date is not None
        else None
    )

    end_date = (
        normalize_date(args.end_date)
        if args.end_date is not None
        else None
    )

    if start_date is None or end_date is None:
        default_start_date, default_end_date = load_default_dates()

        if start_date is None:
            start_date = default_start_date

        if end_date is None:
            end_date = default_end_date

    if start_date > end_date:
        raise ValueError(
            f"start_date {start_date} is after end_date {end_date}."
        )

    process_audit_file(
        audit_file=args.audit_file,
        source_dir=args.source_dir,
        model=args.model,
        start_date=start_date,
        end_date=end_date,
    )


if __name__ == "__main__":
    main()