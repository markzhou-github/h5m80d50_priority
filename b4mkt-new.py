#!/usr/bin/env python3
# coding: utf-8
"""Before-market sell-price suggestions for currently held signal positions.

The program:
  1. reads a signal/target CSV or Parquet file;
  2. keeps rows that have already been bought (buy is filled) but not sold
     (sell is empty);
  3. uses MODEL_CONFIGS for the model-specific gain/cut thresholds;
  4. uses the row's cumulative csi1500_ew through the latest completed close;
  5. calculates today's benchmark-adjusted gain/cut prices;
  6. on T+2 morning only, checks the stock's T+1 close. If the T+1 adjusted
     close already breached the stop threshold, cut_price is set to -1,
     meaning "sell at today's open";
  7. prints a compact CSV-like message. No output file is written.

Price formulas:
    gain_price = buy * (1 + csi1500_ew + gain / 100)
    cut_price  = buy * (1 + csi1500_ew + cut  / 100)

Special sentinel:
    cut_price = -1  -> sell at today's open.

Today's unknown CSI1500-EW movement is intentionally out of scope.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from config_date import End_date_global, is_trade_date, normalize_trade_date, trade_dates_between
from config_models import MODEL_CONFIGS


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_STKFACTOR_DIR = PROJECT_ROOT / "processed" / "daily" / "stkfactor"

REQUIRED_SIGNAL_COLUMNS = {
    "model",
    "ts_code",
    "trade_date",
    "priority",
    "buy",
    "sell",
    "csi1500_ew",
}


# =============================================================================
# COMMAND LINE
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print before-market gain/cut sell prices for positions that have "
            "a buy price but no sell price."
        )
    )

    parser.add_argument(
        "--signal-file",
        "--signal_file",
        dest="signal_file",
        type=Path,
        required=True,
        help="Signal/target CSV or Parquet file.",
    )

    parser.add_argument(
        "--stkfactor-dir",
        "--stkfactor_dir",
        dest="stkfactor_dir",
        type=Path,
        default=DEFAULT_STKFACTOR_DIR,
        help=(
            "Directory containing {ts_code}.stkfactor.csv. "
            "Default: processed/daily/stkfactor relative to this script."
        ),
    )

    parser.add_argument(
        "--today",
        default=None,
        help=(
            "Optional YYYYMMDD/ YYYY-MM-DD override for testing. "
            "Default: config_date.End_date_global (China calendar date)."
        ),
    )

    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Number of decimals used when printing prices. Default: 3.",
    )

    return parser.parse_args()


# =============================================================================
# BASIC HELPERS
# =============================================================================


def read_signal_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Signal file does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(
            path,
            dtype={
                "model": "string",
                "ts_code": "string",
                "trade_date": "string",
            },
        )

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported signal file type: {path.suffix}. "
        "Only CSV and Parquet are supported."
    )


def normalize_ts_code(value) -> str:
    return str(value).strip().upper()


def valid_number(value) -> bool:
    return pd.notna(value) and pd.notna(pd.to_numeric(value, errors="coerce"))


def format_value(value, decimals: int) -> str:
    if pd.isna(value):
        return ""

    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return str(value)

    return f"{float(number):.{decimals}f}"


def format_priority(value) -> str:
    if pd.isna(value):
        return ""

    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return str(value)

    number = float(number)
    if number.is_integer():
        return str(int(number))
    return str(number)


# =============================================================================
# TRADING-DAY HELPERS
# =============================================================================


def position_trade_day(signal_date: str, today: str) -> tuple[int, list[str]]:
    """Return the T+n number for today and the inclusive trading-day sequence.

    Example:
        sequence = [T, T+1, T+2]
        returns (2, sequence)

    A return value of 1 means today is the buy day (T+1).
    A return value of 2 means today is T+2.
    """
    days = trade_dates_between(signal_date, today)

    if not days:
        raise ValueError(
            f"No trading dates found between signal_date={signal_date} and today={today}."
        )

    if days[0] != signal_date:
        raise ValueError(
            f"Signal date {signal_date} is not an A-share trading day. "
            f"First trading day in range is {days[0]}."
        )

    if days[-1] != today:
        raise ValueError(
            f"Today {today} is not included as a trading day in the calendar."
        )

    return len(days) - 1, days


# =============================================================================
# T+1 CLOSE CHECK
# =============================================================================


def read_t1_close(
    stkfactor_dir: Path,
    ts_code: str,
    t1_date: str,
) -> float:
    """Read the unadjusted close price for T+1 from the stock stkfactor file."""
    path = stkfactor_dir / f"{ts_code}.stkfactor.csv"

    if not path.exists():
        raise FileNotFoundError(f"stkfactor file does not exist: {path}")

    # Only read the two columns b4mkt needs.
    df = pd.read_csv(
        path,
        usecols=["trade_date", "close"],
        dtype={"trade_date": "string"},
    )

    df["trade_date"] = (
        df["trade_date"]
        .astype("string")
        .str.replace("-", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    row = df.loc[df["trade_date"] == t1_date]

    if row.empty:
        raise ValueError(
            f"T+1 date {t1_date} not found in stkfactor file for {ts_code}: {path}"
        )

    close = pd.to_numeric(row.iloc[-1]["close"], errors="coerce")

    if pd.isna(close) or float(close) <= 0:
        raise ValueError(
            f"Invalid T+1 close for {ts_code} on {t1_date}: {row.iloc[-1]['close']!r}"
        )

    return float(close)


# =============================================================================
# ONE POSITION
# =============================================================================


def calculate_one_position(
    row: pd.Series,
    today: str,
    stkfactor_dir: Path,
) -> dict:
    model = str(row["model"]).strip()
    ts_code = normalize_ts_code(row["ts_code"])
    signal_date = normalize_trade_date(row["trade_date"])

    if model not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model {model!r} for {ts_code} {signal_date}. "
            f"Available models: {sorted(MODEL_CONFIGS)}"
        )

    config = MODEL_CONFIGS[model]

    if "gain" not in config or "cut" not in config:
        raise ValueError(
            f"MODEL_CONFIGS[{model!r}] must contain both 'gain' and 'cut'."
        )

    buy = pd.to_numeric(row["buy"], errors="coerce")
    ew = pd.to_numeric(row["csi1500_ew"], errors="coerce")

    if pd.isna(buy) or float(buy) <= 0:
        raise ValueError(f"Invalid buy price for {model} {ts_code} {signal_date}: {row['buy']!r}")

    if pd.isna(ew):
        raise ValueError(
            f"Missing csi1500_ew for active position {model} {ts_code} {signal_date}."
        )

    buy = float(buy)
    ew = float(ew)
    gain = float(config["gain"]) / 100.0
    cut = float(config["cut"]) / 100.0

    current_day, trade_days = position_trade_day(signal_date, today)

    if current_day < 2:
        # The row has a buy price, but on T+1 there is no normal sell action under
        # the A-share T+1 rule. b4mkt should not produce a sell instruction yet.
        return {
            "skip": True,
            "skip_reason": "T+1 buy day; cannot sell yet",
            "model": model,
            "ts_code": ts_code,
            "signal_date": signal_date,
        }

    gain_price = buy * (1.0 + ew + gain)
    cut_price = buy * (1.0 + ew + cut)

    t1_stop_triggered = False

    # Special rule: on T+2 morning, verify whether T+1 close already breached
    # the adjusted stop threshold. At this moment csi1500_ew in the signal row
    # is the T+1 EW return, so it can be used directly for this check.
    if current_day == 2:
        t1_date = trade_days[1]
        t1_close = read_t1_close(
            stkfactor_dir=stkfactor_dir,
            ts_code=ts_code,
            t1_date=t1_date,
        )

        t1_adj_close_ret = (t1_close / buy - 1.0) - ew

        if t1_adj_close_ret <= cut:
            cut_price = -1.0
            t1_stop_triggered = True

    return {
        "skip": False,
        "model": model,
        "ts_code": ts_code,
        "signal_date": signal_date,
        "priority": row.get("priority", pd.NA),
        "buy": buy,
        "gain_price": gain_price,
        "cut_price": cut_price,
        "current_day": current_day,
        "t1_stop_triggered": t1_stop_triggered,
    }


# =============================================================================
# MESSAGE
# =============================================================================


def build_message(results: list[dict], decimals: int) -> str:
    header = "model,ts_code,signal_date,priority,buy,gain_price,cut_price"
    lines = [header]

    for item in results:
        lines.append(
            ",".join(
                [
                    item["model"],
                    item["ts_code"],
                    item["signal_date"],
                    format_priority(item["priority"]),
                    format_value(item["buy"], decimals),
                    format_value(item["gain_price"], decimals),
                    "-1" if item["cut_price"] == -1 else format_value(item["cut_price"], decimals),
                ]
            )
        )

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    args = parse_args()

    if args.decimals < 0:
        raise ValueError("--decimals must be >= 0")

    today = normalize_trade_date(args.today) if args.today else normalize_trade_date(End_date_global)

    # b4mkt is intended for a trading-day morning. If today is not a trading
    # date, there are no market sell instructions to send.
    if not is_trade_date(today):
        print(f"[b4mkt] {today} is not an A-share trading day. Nothing to do.")
        return

    signal_data = read_signal_file(args.signal_file)
    signal_data.columns = signal_data.columns.str.strip()

    missing = REQUIRED_SIGNAL_COLUMNS - set(signal_data.columns)
    if missing:
        raise ValueError(
            f"Signal file is missing required columns: {sorted(missing)}"
        )

    # Only positions that are already bought and not yet sold.
    active = signal_data.loc[
        signal_data["buy"].notna()
        & pd.to_numeric(signal_data["buy"], errors="coerce").notna()
        & signal_data["sell"].isna()
    ].copy()

    if active.empty:
        print("[b4mkt] No bought-but-unsold positions.")
        return

    results: list[dict] = []
    skipped: list[dict] = []
    errors: list[str] = []

    for _, row in active.iterrows():
        try:
            item = calculate_one_position(
                row=row,
                today=today,
                stkfactor_dir=args.stkfactor_dir,
            )

            if item.get("skip"):
                skipped.append(item)
            else:
                results.append(item)

        except Exception as exc:  # keep other positions usable if one row is bad
            model = str(row.get("model", ""))
            ts_code = str(row.get("ts_code", ""))
            trade_date = str(row.get("trade_date", ""))
            errors.append(
                f"{model},{ts_code},{trade_date}: {exc}"
            )

    # Stable, useful notification ordering: higher priority first, then signal date.
    def sort_key(item: dict):
        p = pd.to_numeric(item.get("priority"), errors="coerce")
        p_key = float(p) if pd.notna(p) else float("inf")
        return (p_key, item["signal_date"], item["model"], item["ts_code"])

    results.sort(key=sort_key)

    if results:
        msg = build_message(results, decimals=args.decimals)
        print(msg)
    else:
        print("[b4mkt] No sell instructions for today.")

    if skipped:
        print()
        print(f"[b4mkt] skipped={len(skipped)}")
        for item in skipped:
            print(
                f"[skip] {item['model']},{item['ts_code']},{item['signal_date']}: "
                f"{item['skip_reason']}"
            )

    if errors:
        print()
        print(f"[b4mkt] errors={len(errors)}")
        for message in errors:
            print(f"[error] {message}")


if __name__ == "__main__":
    main()
