#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config_date import End_date


PROJECT_ROOT = Path(__file__).resolve().parent

ALL_SIGNALS_SCRIPT = "all_signals_model_priority_config.py"
FILL_TARGETS_SCRIPT = "fill_signals_targets.py"

STOCK_DIR = "processed/daily/merged"
BENCHMARK_FILE = "processed/index/csi1500_custom_index.csv"
BENCHMARK_RETURN_COL = "csi1500_ew_close_ret"


PIPELINES = {
    "h5m80d50_priority": {
        "generate_script": "h5m80d50_priority/generate_signals_range.py",
        "input": "processed/train_v5b/train_v5b.parquet",
        "signal_dir": "signals_h5priority",
        "audit_file": "h5priority.csv",
        "target_file": "h5priority_target.csv",
    },
    "h2m80d50_dual": {
        # Change these two paths if your actual locations differ.
        "generate_script": "h2m80d50_dual/generate_signals_range.py",
        "input": "processed/train_v5b/train_v5b.parquet",
        "signal_dir": "signals_h2dual",
        "audit_file": "h2dual.csv",
        "target_file": "h2dual_target.csv",
    },
    "h5m80d50_ensemble": {
        # Change these two paths if your actual locations differ.
        "generate_script": "h5m80d50_ensemble/generate_signals_range.py",
        "input": "processed/train_v5b/train_v5b.parquet",
        "signal_dir": "signals_h5ensemble",
        "audit_file": "h5ensemble.csv",
        "target_file": "h5ensemble_target.csv",
    },
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_date(value: str) -> str:
    from datetime import datetime

    text = str(value).strip().replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError(
            f"Invalid date {value!r}; expected YYYYMMDD or YYYY-MM-DD"
        )
    datetime.strptime(text, "%Y%m%d")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run after-market signal generation, signal-audit update, "
            "and target update."
        )
    )

    parser.add_argument(
        "--start-date",
        "--start_date",
        dest="start_date",
        default=None,
        help="Inclusive start date. Default: config_date.End_date.",
    )

    parser.add_argument(
        "--end-date",
        "--end_date",
        dest="end_date",
        default=None,
        help="Inclusive end date. Default: config_date.End_date.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(PIPELINES.keys()),
        default=None,
        help="Optional model subset. Default: process all three models.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )

    return parser.parse_args()


def run_command(command: list[str], label: str, dry_run: bool) -> None:
    print()
    print("-" * 100)
    print(label)
    print("-" * 100)
    print(" ".join(command), flush=True)

    if dry_run:
        return

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def run_model_pipeline(
    model_name: str,
    start_date: str,
    end_date: str,
    dry_run: bool,
) -> None:
    cfg = PIPELINES[model_name]

    generate_script = resolve_path(cfg["generate_script"])
    input_file = resolve_path(cfg["input"])
    signal_dir = resolve_path(cfg["signal_dir"])
    audit_file = resolve_path(cfg["audit_file"])
    target_file = resolve_path(cfg["target_file"])

    all_signals_script = resolve_path(ALL_SIGNALS_SCRIPT)
    fill_targets_script = resolve_path(FILL_TARGETS_SCRIPT)
    stock_dir = resolve_path(STOCK_DIR)
    benchmark_file = resolve_path(BENCHMARK_FILE)

    print()
    print("=" * 100)
    print(f"MODEL: {model_name}")
    print(f"DATE RANGE: {start_date} -> {end_date}")
    print("=" * 100)

    # 1. Generate daily signal files.
    run_command(
        [
            sys.executable,
            str(generate_script),
            "--input",
            str(input_file),
            "--out-dir",
            str(signal_dir),
            "--start-date",
            start_date,
            "--end-date",
            end_date,
        ],
        label=f"[{model_name}] 1/3 generate signals",
        dry_run=dry_run,
    )

    # 2. Merge the generated daily signals into the audit file.
    run_command(
        [
            sys.executable,
            str(all_signals_script),
            "--audit-file",
            str(audit_file),
            "--start-date",
            start_date,
            "--end-date",
            end_date,
            "--model-name",
            model_name,
        ],
        label=f"[{model_name}] 2/3 update signal audit",
        dry_run=dry_run,
    )

    # 3. Update targets/open-position information for all pending audit rows.
    run_command(
        [
            sys.executable,
            str(fill_targets_script),
            "--signals",
            str(audit_file),
            "--stock-dir",
            str(stock_dir),
            "--benchmark",
            str(benchmark_file),
            "--out",
            str(target_file),
            "--benchmark-return-col",
            BENCHMARK_RETURN_COL,
        ],
        label=f"[{model_name}] 3/3 fill targets",
        dry_run=dry_run,
    )


def main() -> None:
    args = parse_args()

    # Requested simple behavior:
    #   missing start-date -> End_date
    #   missing end-date   -> End_date
    start_date = normalize_date(
        args.start_date if args.start_date is not None else End_date
    )
    end_date = normalize_date(
        args.end_date if args.end_date is not None else End_date
    )

    if start_date > end_date:
        start_date = end_date
#        raise ValueError(
 #           f"start_date {start_date} cannot be after end_date {end_date}"
#        )

    models = args.models if args.models is not None else list(PIPELINES.keys())

    print("=" * 100)
    print("AFTER MARKET PIPELINE")
    print("=" * 100)
    print(f"start_date : {start_date}")
    print(f"end_date   : {end_date}")
    print(f"models     : {', '.join(models)}")
    print(f"dry_run    : {args.dry_run}")

    completed: list[str] = []

    try:
        for model_name in models:
            run_model_pipeline(
                model_name=model_name,
                start_date=start_date,
                end_date=end_date,
                dry_run=args.dry_run,
            )
            completed.append(model_name)

    except subprocess.CalledProcessError as exc:
        print()
        print("=" * 100)
        print("AFTER MARKET PIPELINE FAILED")
        print("=" * 100)
        print(f"Completed models: {completed}")
        print(f"Failed command return code: {exc.returncode}")
        raise SystemExit(exc.returncode) from exc

    print()
    print("=" * 100)
    print("AFTER MARKET PIPELINE COMPLETE")
    print("=" * 100)
    print(f"Completed models: {', '.join(completed)}")


if __name__ == "__main__":
    main()
