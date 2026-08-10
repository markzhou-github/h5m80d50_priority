#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd


KEY_COLUMNS = ["ts_code", "trade_date"]
PRIORITY_COLUMNS = ["priority_name", "priority"]


def normalize_date(value) -> Optional[str]:
    """
    Convert common date formats to YYYYMMDD.

    Examples:
        20260717
        20260717.0
        2026-07-17
        2026/07/17
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]

    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return None

    return parsed.strftime("%Y%m%d")


def normalize_date_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_date).astype("string")


def read_data_file(file_path: Path) -> pd.DataFrame:
    """
    Read CSV or Parquet based on the file extension.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(
            file_path,
            dtype={
                "ts_code": "string",
                "trade_date": "string",
            },
        )

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)

    raise ValueError(
        f"Unsupported file type: {file_path.suffix}. "
        "Only CSV and Parquet files are supported."
    )


def save_data_file(data: pd.DataFrame, file_path: Path) -> None:
    """
    Save safely by writing to a temporary file first.
    """
    suffix = file_path.suffix.lower()
    temporary_file = file_path.with_suffix(file_path.suffix + ".tmp")

    if suffix == ".csv":
        data.to_csv(
            temporary_file,
            index=False,
            na_rep="",
        )

    elif suffix in {".parquet", ".pq"}:
        data.to_parquet(
            temporary_file,
            index=False,
        )

    else:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}"
        )

    os.replace(temporary_file, file_path)


def standardize_priority_columns(
    data: pd.DataFrame,
    file_description: str,
) -> pd.DataFrame:
    """
    Convert either priority column convention to:

        priority_name
        priority

    Supported source conventions:

        priority_name, priority

    or:

        signal_layer, signal_priority
    """
    data = data.copy()
    data.columns = data.columns.str.strip()

    # Create priority_name from signal_layer when necessary.
    if "priority_name" not in data.columns:
        if "signal_layer" in data.columns:
            data["priority_name"] = data["signal_layer"]
        else:
            raise ValueError(
                f"{file_description} has neither "
                "'priority_name' nor 'signal_layer'."
            )

    # Create priority from signal_priority when necessary.
    if "priority" not in data.columns:
        if "signal_priority" in data.columns:
            data["priority"] = data["signal_priority"]
        else:
            raise ValueError(
                f"{file_description} has neither "
                "'priority' nor 'signal_priority'."
            )

    return data


def prepare_source_data(
    source_data: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    """
    Validate, filter and standardize source data.
    """
    source_data = standardize_priority_columns(
        source_data,
        file_description="Source file",
    )

    required_columns = {
        "ts_code",
        "trade_date",
        "priority_name",
        "priority",
    }

    missing_columns = required_columns - set(source_data.columns)

    if missing_columns:
        raise ValueError(
            f"Source file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    source_data["ts_code"] = (
        source_data["ts_code"]
        .astype("string")
        .str.strip()
    )

    source_data["trade_date"] = normalize_date_series(
        source_data["trade_date"]
    )

    # Remove rows without valid identifying information.
    source_data = source_data.dropna(
        subset=["ts_code", "trade_date"]
    ).copy()

    source_data = source_data[
        source_data["ts_code"].ne("")
    ].copy()

    # Filter dates inclusively.
    if start_date is not None:
        source_data = source_data[
            source_data["trade_date"] >= start_date
        ]

    if end_date is not None:
        source_data = source_data[
            source_data["trade_date"] <= end_date
        ]

    # Keep only the required output columns.
    source_data = source_data[
        [
            "ts_code",
            "trade_date",
            "priority_name",
            "priority",
        ]
    ].copy()

    # Near dates first, far dates later.
    source_data = source_data.sort_values(
        by=["trade_date", "ts_code"],
        ascending=[False, True],
        kind="stable",
    )

    duplicate_count = source_data.duplicated(
        subset=KEY_COLUMNS,
        keep="first",
    ).sum()

    if duplicate_count:
        print(
            f"Warning: source contains {duplicate_count:,} duplicate "
            "ts_code/trade_date rows. The first row is retained."
        )

        source_data = source_data.drop_duplicates(
            subset=KEY_COLUMNS,
            keep="first",
        )

    return source_data.reset_index(drop=True)


def prepare_audit_data(
    audit_data: pd.DataFrame,
) -> pd.DataFrame:
    audit_data = audit_data.copy()
    audit_data.columns = audit_data.columns.str.strip()

    required_columns = {"ts_code", "trade_date"}
    missing_columns = required_columns - set(audit_data.columns)

    if missing_columns:
        raise ValueError(
            f"Audit file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    audit_data["ts_code"] = (
        audit_data["ts_code"]
        .astype("string")
        .str.strip()
    )

    audit_data["trade_date"] = normalize_date_series(
        audit_data["trade_date"]
    )

    # Support an audit file that still uses the old column names.
    if "priority_name" not in audit_data.columns:
        if "signal_layer" in audit_data.columns:
            audit_data["priority_name"] = audit_data["signal_layer"]
        else:
            audit_data["priority_name"] = pd.NA

    if "priority" not in audit_data.columns:
        if "signal_priority" in audit_data.columns:
            audit_data["priority"] = audit_data["signal_priority"]
        else:
            audit_data["priority"] = pd.NA

    return audit_data


def merge_source_into_audit(
    audit_data: pd.DataFrame,
    source_data: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    """
    Merge on ts_code + trade_date.

    Behavior:
    - Source priority values overwrite existing audit priority values.
    - Source rows not found in audit are appended.
    - Other existing audit columns are preserved.
    """
    existing_audit_keys = set(
        zip(
            audit_data["ts_code"],
            audit_data["trade_date"],
        )
    )

    source_keys = set(
        zip(
            source_data["ts_code"],
            source_data["trade_date"],
        )
    )

    updated_count = len(existing_audit_keys & source_keys)
    appended_count = len(source_keys - existing_audit_keys)

    merged = audit_data.merge(
        source_data,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("", "__source"),
        sort=False,
        validate="many_to_one",
    )

    # Source values take priority when a matching source row exists.
    for column in PRIORITY_COLUMNS:
        source_column = f"{column}__source"

        merged[column] = merged[source_column].combine_first(
            merged[column]
        )

        merged = merged.drop(columns=source_column)

    # Sort newest dates first.
    merged = merged.sort_values(
        by=["trade_date", "ts_code"],
        ascending=[False, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    return merged, updated_count, appended_count


def process_files(
    audit_file: Path,
    source_file: Path,
    start_date: Optional[str],
    end_date: Optional[str],
) -> None:
    if not source_file.exists():
        raise FileNotFoundError(
            f"Source file does not exist: {source_file}"
        )

    if start_date is not None:
        start_date = normalize_date(start_date)

        if start_date is None:
            raise ValueError(
                "start_date must be a valid date, such as 20260717."
            )

    if end_date is not None:
        end_date = normalize_date(end_date)

        if end_date is None:
            raise ValueError(
                "end_date must be a valid date, such as 20260717."
            )

    if (
        start_date is not None
        and end_date is not None
        and start_date > end_date
    ):
        raise ValueError(
            f"start_date {start_date} is after end_date {end_date}."
        )

    source_data = read_data_file(source_file)

    source_data = prepare_source_data(
        source_data=source_data,
        start_date=start_date,
        end_date=end_date,
    )

    # Create a new audit file when it does not already exist.
    if audit_file.exists():
        audit_data = read_data_file(audit_file)
        audit_data = prepare_audit_data(audit_data)
    else:
        print(
            f"Audit file does not exist. A new file will be created: "
            f"{audit_file}"
        )

        audit_data = pd.DataFrame(
            columns=[
                "ts_code",
                "trade_date",
                "priority_name",
                "priority",
            ]
        )

    merged_data, updated_count, appended_count = (
        merge_source_into_audit(
            audit_data=audit_data,
            source_data=source_data,
        )
    )

    audit_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_data_file(
        data=merged_data,
        file_path=audit_file,
    )

    source_min_date = (
        source_data["trade_date"].min()
        if not source_data.empty
        else None
    )

    source_max_date = (
        source_data["trade_date"].max()
        if not source_data.empty
        else None
    )

    print()
    print(f"Audit file saved: {audit_file}")
    print(f"Source file: {source_file}")
    print(f"Filtered source rows: {len(source_data):,}")
    print(f"Updated audit keys: {updated_count:,}")
    print(f"Appended audit keys: {appended_count:,}")
    print(f"Final audit rows: {len(merged_data):,}")

    if source_min_date is not None:
        print(
            f"Source date range used: "
            f"{source_min_date} to {source_max_date}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter priority signals from a source file and merge them "
            "into an audit file."
        )
    )

    parser.add_argument(
        "--audit-file",
        "--audit_file",
        dest="audit_file",
        type=Path,
        required=True,
        help="Path to the audit CSV or Parquet file.",
    )

    parser.add_argument(
        "--source-file",
        "--source_file",
        dest="source_file",
        type=Path,
        required=True,
        help="Path to the source CSV or Parquet file.",
    )

    parser.add_argument(
        "--start-date",
        "--start_date",
        dest="start_date",
        default=None,
        help=(
            "Inclusive start date in YYYYMMDD format. "
            "Defaults to the earliest source date."
        ),
    )

    parser.add_argument(
        "--end-date",
        "--end_date",
        dest="end_date",
        default=None,
        help=(
            "Inclusive end date in YYYYMMDD format. "
            "Defaults to the latest source date."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    process_files(
        audit_file=args.audit_file,
        source_file=args.source_file,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()