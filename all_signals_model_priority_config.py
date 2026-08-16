#!/usr/bin/env python3
"""Merge model-specific daily signal files into a single audit file.

Rows are uniquely identified by model, ts_code, and trade_date. Existing rows
receive priority updates only; columns populated by other scripts are preserved.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd


KEY_COLUMNS = ["model", "ts_code", "trade_date"]
PRIORITY_COLUMNS = ["priority_name", "priority"]
NEW_ROW_COLUMNS = KEY_COLUMNS + PRIORITY_COLUMNS


from config_models import MODEL_CONFIGS


def normalize_date(value) -> Optional[str]:
    """Convert a common date representation to YYYYMMDD."""
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
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(
            file_path,
            dtype={"model": "string", "ts_code": "string", "trade_date": "string"},
        )
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    raise ValueError(
        f"Unsupported file type: {file_path.suffix}. "
        "Only CSV and Parquet files are supported."
    )


def save_data_file(data: pd.DataFrame, file_path: Path) -> None:
    """Save through a temporary file and replace the destination atomically."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = file_path.suffix.lower()
    temporary_file = file_path.with_suffix(file_path.suffix + ".tmp")

    if suffix == ".csv":
        data.to_csv(temporary_file, index=False, na_rep="")
    elif suffix in {".parquet", ".pq"}:
        data.to_parquet(temporary_file, index=False)
    else:
        raise ValueError(f"Unsupported output type: {file_path.suffix}")

    os.replace(temporary_file, file_path)


def discover_signal_files(
    model_name: str,
    start_date: str,
    end_date: str,
    signal_dir_override: Optional[Path] = None,
) -> list[tuple[str, Path]]:
    """Find existing model signal files within the inclusive date range."""
    config = MODEL_CONFIGS[model_name]
    signal_dir = (
        signal_dir_override
        if signal_dir_override is not None
        else Path(config["signal_dir"])
    )

    if not signal_dir.exists():
        raise FileNotFoundError(f"Signal directory does not exist: {signal_dir}")

    date_regex = re.compile(config["date_regex"])
    found: list[tuple[str, Path]] = []

    for file_path in signal_dir.glob(config["file_glob"]):
        match = date_regex.match(file_path.name)
        if match is None:
            continue

        file_date = normalize_date(match.group(1))
        if file_date is None:
            print(f"[warning] Could not parse date from {file_path.name}; skipping.")
            continue
        if start_date <= file_date <= end_date:
            found.append((file_date, file_path))

    found.sort(key=lambda item: (item[0], item[1].name))
    return found


def prepare_source_data(
    source_data: pd.DataFrame,
    model_name: str,
    file_date: str,
    source_file: Path,
) -> pd.DataFrame:
    """Standardize one signal file to model/key and priority fields."""
    config = MODEL_CONFIGS[model_name]
    priority_name_column = config["priority_name_column"]
    priority_mapping = config["priority_mapping"]

    data = source_data.copy()
    data.columns = data.columns.str.strip()

    if "ts_code" not in data.columns:
        raise ValueError(f"{source_file} does not contain required column 'ts_code'.")
    if priority_name_column not in data.columns:
        raise ValueError(
            f"{source_file} does not contain priority column "
            f"{priority_name_column!r} required by model {model_name!r}."
        )

    data["model"] = model_name
    data["ts_code"] = data["ts_code"].astype("string").str.strip()
    data = data[data["ts_code"].notna() & data["ts_code"].ne("")].copy()
    data["trade_date"] = file_date
    data["priority_name"] = (
        data[priority_name_column].astype("string").str.strip()
    )
    data["priority"] = data["priority_name"].map(priority_mapping)

    unknown_mask = (
        data["priority_name"].notna()
        & data["priority_name"].ne("")
        & data["priority"].isna()
    )
    if unknown_mask.any():
        unknown_values = sorted(
            data.loc[unknown_mask, "priority_name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            f"{source_file}: model {model_name!r} contains priority names "
            f"not present in priority_mapping: {unknown_values}"
        )

    data = data[NEW_ROW_COLUMNS].copy()
    duplicate_mask = data.duplicated(subset=KEY_COLUMNS, keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        print(
            f"[warning] {source_file.name}: {duplicate_count:,} duplicate "
            "model/ts_code/trade_date rows; keeping the first."
        )
        data = data.drop_duplicates(subset=KEY_COLUMNS, keep="first")

    return data.reset_index(drop=True)


def prepare_audit_data(audit_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize audit keys and ensure script-owned output columns exist."""
    data = audit_data.copy()
    data.columns = data.columns.str.strip()

    missing_columns = set(KEY_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Audit file is missing required columns: {sorted(missing_columns)}"
        )

    data["model"] = data["model"].astype("string").str.strip()
    data["ts_code"] = data["ts_code"].astype("string").str.strip()
    data["trade_date"] = normalize_date_series(data["trade_date"])

    for column in PRIORITY_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA

    # Target belongs to downstream processing. This script only ensures that
    # the column exists; it never assigns or clears target values.
    if "target" not in data.columns:
        data["target"] = pd.NA

    return data


def merge_source_into_audit(
    audit_data: pd.DataFrame,
    source_data: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    """Update priorities for existing full keys and append brand-new keys."""
    data = audit_data.copy()
    source_lookup = source_data.set_index(KEY_COLUMNS)[PRIORITY_COLUMNS]
    audit_keys = pd.MultiIndex.from_frame(data[KEY_COLUMNS])
    source_keys = source_lookup.index
    existing_mask = audit_keys.isin(source_keys)
    updated_count = int(existing_mask.sum())

    # Preserve every existing column except the two priority fields.
    if existing_mask.any():
        matching_values = source_lookup.reindex(audit_keys[existing_mask])
        for column in PRIORITY_COLUMNS:
            data.loc[existing_mask, column] = matching_values[column].to_numpy()

    # Detect additions with the complete model + ts_code + trade_date key.
    new_mask = ~source_keys.isin(audit_keys)
    new_source = source_data.loc[new_mask].copy()
    appended_count = len(new_source)

    if not new_source.empty:
        new_rows = pd.DataFrame(
            pd.NA,
            index=range(len(new_source)),
            columns=data.columns,
        )
        for column in NEW_ROW_COLUMNS:
            new_rows[column] = new_source[column].to_numpy()
        data = pd.concat([data, new_rows], ignore_index=True)

    return data, updated_count, appended_count


def process_files(
    audit_file: Path,
    model_name: str,
    start_date: str,
    end_date: str,
    signal_dir_override: Optional[Path] = None,
) -> None:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_name {model_name!r}. "
            f"Available models: {sorted(MODEL_CONFIGS)}"
        )

    normalized_start = normalize_date(start_date)
    normalized_end = normalize_date(end_date)
    if normalized_start is None:
        raise ValueError("Invalid start_date.")
    if normalized_end is None:
        raise ValueError("Invalid end_date.")
    if normalized_start > normalized_end:
        raise ValueError(
            f"start_date {normalized_start} is after end_date {normalized_end}."
        )

    signal_files = discover_signal_files(
        model_name=model_name,
        start_date=normalized_start,
        end_date=normalized_end,
        signal_dir_override=signal_dir_override,
    )
    config = MODEL_CONFIGS[model_name]
    signal_dir = signal_dir_override or Path(config["signal_dir"])

    print("=" * 80)
    print(f"Model:        {model_name}")
    print(f"Signal dir:   {signal_dir}")
    print(f"Date range:   {normalized_start} -> {normalized_end}")
    print(f"Files found:  {len(signal_files)}")
    print("=" * 80)

    if not signal_files:
        print("No matching signal files found. Nothing to do.")
        return

    if audit_file.exists():
        audit_data = prepare_audit_data(read_data_file(audit_file))
    else:
        print(f"Audit file does not exist. A new file will be created: {audit_file}")
        audit_data = pd.DataFrame(columns=NEW_ROW_COLUMNS + ["target"])

    total_source_rows = 0
    total_updated = 0
    total_appended = 0

    for index, (file_date, source_file) in enumerate(signal_files, start=1):
        print(f"\n[{index}/{len(signal_files)}] {file_date}  {source_file.name}")
        source_data = prepare_source_data(
            source_data=read_data_file(source_file),
            model_name=model_name,
            file_date=file_date,
            source_file=source_file,
        )
        audit_data, updated_count, appended_count = merge_source_into_audit(
            audit_data=audit_data,
            source_data=source_data,
        )
        total_source_rows += len(source_data)
        total_updated += updated_count
        total_appended += appended_count
        print(
            f"    rows={len(source_data):,} updated={updated_count:,} "
            f"appended={appended_count:,}"
        )

    audit_data = audit_data.sort_values(
        by=["trade_date", "model", "ts_code"],
        ascending=[False, True, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    save_data_file(audit_data, audit_file)

    print("\n" + "=" * 80)
    print(f"Files processed: {len(signal_files):,}")
    print(f"Source rows:     {total_source_rows:,}")
    print(f"Rows updated:    {total_updated:,}")
    print(f"Rows appended:   {total_appended:,}")
    print(f"Audit rows:      {len(audit_data):,}")
    print(f"Saved:           {audit_file}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-file", "--audit_file", required=True, type=Path)
    parser.add_argument(
        "--model-name",
        "--model_name",
        required=True,
        choices=sorted(MODEL_CONFIGS),
    )
    parser.add_argument("--start-date", "--start_date", required=True)
    parser.add_argument("--end-date", "--end_date", required=True)
    parser.add_argument(
        "--signal-dir",
        "--signal_dir",
        type=Path,
        default=None,
        help="Optional override for the model's configured signal directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_files(
        audit_file=args.audit_file,
        model_name=args.model_name,
        start_date=args.start_date,
        end_date=args.end_date,
        signal_dir_override=args.signal_dir,
    )


if __name__ == "__main__":
    main()
