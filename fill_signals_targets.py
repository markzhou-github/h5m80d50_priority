#!/usr/bin/env python
# coding: utf-8
"""Fill missing target/result fields in signals_all.csv.


python fill_signals_targets.py \
    --signals /path/to/signals_all.csv \
    --stock-dir /path/to/stock_csv_folder \
    --benchmark /path/to/csi1500_ew.csv \
    --out /path/to/signals_all_filled.csv

Trading rules
-------------
Signal date is T. Buy at T+1 open. For horizon H, the final day is T+H.

1. T+1 exception:
   If the benchmark-adjusted T+1 close return is <= cut, sell at T+2 open.
2. From T+2 through T+H, in this priority order:
   a. If adjusted open return >= gain, sell at the open.
   b. Else if adjusted high return >= gain, sell at the gain threshold price.
   c. Else if adjusted close return <= cut, sell at the cut threshold price.
3. If nothing triggers by T+H, sell at T+H close.
   On expiry, target is 1 when the adjusted exit return >= gain; otherwise 0.

All stock returns are adjusted by subtracting the compounded CSI1500 equal-
weight daily close return from T+1 through the exit date.

The model column selects horizon, gain, and cut from MODEL_CONFIGS. Configured
gain/cut values are percentages (for example, 8.0 means 8%).

Rows with an existing non-null target are never changed. Open positions are
marked to the latest available close: buy, ret, ret_ew, holding_days, and
csi1500_ew are refreshed while target and sell remain blank. The script can be
run again as new market data arrive.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-12


from config_models import MODEL_CONFIGS


def get_model_parameters(model_name: str) -> tuple[int, float, float]:
    """Return horizon and decimal gain/cut thresholds for one model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model {model_name!r}; available models: "
            f"{sorted(MODEL_CONFIGS)}"
        )

    config = MODEL_CONFIGS[model_name]
    missing = [name for name in ("horizon", "gain", "cut") if name not in config]
    if missing:
        raise ValueError(
            f"MODEL_CONFIGS[{model_name!r}] is missing parameters {missing}"
        )

    horizon_value = to_finite_float(config["horizon"], f"{model_name}.horizon")
    horizon = int(horizon_value)
    if horizon_value != horizon or horizon < 1:
        raise ValueError(
            f"Invalid {model_name}.horizon: {config['horizon']!r}; "
            "expected a positive integer"
        )

    # MODEL_CONFIGS stores percentage points: 8.0 -> 0.08, -5.0 -> -0.05.
    gain = to_finite_float(config["gain"], f"{model_name}.gain") / 100.0
    cut = to_finite_float(config["cut"], f"{model_name}.cut") / 100.0
    return horizon, gain, cut

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill missing targets in signals_all.csv using daily stock and CSI1500 data."
    )
    parser.add_argument(
        "--signals",
        type=Path,
        default=Path("signals_all.csv"),
        help="Input signals CSV. Default: signals_all.csv",
    )
    parser.add_argument(
        "--stock-dir",
        type=Path,
        required=True,
        help="Folder containing one stock CSV per ts_code, e.g. 600000.SH.csv",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="CSI1500 equal-weight daily index CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV. If omitted, overwrite --signals safely.",
    )

    parser.add_argument("--signal-date-col", default="trade_date")
    parser.add_argument("--stock-date-col", default="trade_date")
    parser.add_argument("--benchmark-date-col", default="trade_date")
    parser.add_argument("--open-col", default="open")
    parser.add_argument("--high-col", default="high")
    parser.add_argument("--close-col", default="close")
    parser.add_argument(
        "--benchmark-return-col",
        default=None,
        help=(
            "Benchmark daily close-return column. If omitted, the script tries "
            "csi1500_ew, ret, return, daily_return, pct_chg, and close."
        ),
    )
    parser.add_argument(
        "--benchmark-return-percent",
        action="store_true",
        help="Use when the benchmark return column stores percentages, e.g. 1.2 means 1.2%%.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV encoding used for reading and writing. Default: utf-8-sig",
    )
    return parser.parse_args()


def normalize_date_series(s: pd.Series) -> pd.Series:
    """Convert common YYYYMMDD/date formats to pandas normalized timestamps."""
    text = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], errors="coerce")
    return parsed.dt.normalize()


def to_finite_float(value: Any, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}: {value!r}") from exc
    if not math.isfinite(x):
        raise ValueError(f"Invalid {name}: {value!r}")
    return x


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults: dict[str, Any] = {
        "target": np.nan,
        "ret": np.nan,
        "ret_ew": np.nan,
        "buy": np.nan,
        "sell": np.nan,
        "holding_days": np.nan,
        "exit_reason": pd.NA,
        "csi1500_ew": np.nan,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def detect_benchmark_return(
    benchmark: pd.DataFrame,
    requested_col: str | None,
    date_col: str,
) -> tuple[pd.Series, str]:
    """Return decimal daily benchmark returns indexed by date."""
    if requested_col is not None:
        if requested_col not in benchmark.columns:
            raise ValueError(
                f"Benchmark return column {requested_col!r} is missing. "
                f"Available columns: {benchmark.columns.tolist()}"
            )
        chosen = requested_col
    else:
        candidates = [
            "csi1500_ew",
            "csi1500_ew_ret",
            "ret",
            "return",
            "daily_return",
            "pct_chg",
        ]
        chosen = next((c for c in candidates if c in benchmark.columns), "")

        # If there is no return column, derive close-to-close returns from close.
        if not chosen and "close" in benchmark.columns:
            chosen = "__derived_from_close__"
        if not chosen:
            raise ValueError(
                "Could not identify the benchmark return column. Pass "
                "--benchmark-return-col. Available columns: "
                f"{benchmark.columns.tolist()}"
            )

    dates = normalize_date_series(benchmark[date_col])
    if dates.isna().any():
        bad = benchmark.loc[dates.isna(), date_col].head(5).tolist()
        raise ValueError(f"Invalid benchmark dates, examples: {bad}")

    if chosen == "__derived_from_close__":
        close = pd.to_numeric(benchmark["close"], errors="coerce")
        values = close.pct_change(fill_method=None)
    else:
        values = pd.to_numeric(benchmark[chosen], errors="coerce")

    out = pd.Series(values.to_numpy(float), index=dates, name="benchmark_return")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out, chosen


def find_stock_file(stock_dir: Path, ts_code: str) -> Path | None:
    candidates = [
        stock_dir / f"{ts_code}.all.csv",
        stock_dir / f"{ts_code.replace('.', '_')}.all.csv",
        stock_dir / f"{ts_code.replace('.', '')}.all.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_stock(
    path: Path,
    encoding: str,
    date_col: str,
    open_col: str,
    high_col: str,
    close_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding, low_memory=False)
    needed = [date_col, open_col, high_col, close_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; columns={df.columns.tolist()}")

    out = pd.DataFrame(
        {
            "trade_date": normalize_date_series(df[date_col]),
            "open": pd.to_numeric(df[open_col], errors="coerce"),
            "high": pd.to_numeric(df[high_col], errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["trade_date"])
    out = out.drop_duplicates("trade_date", keep="last").sort_values("trade_date")
    out = out.set_index("trade_date")
    return out


def compounded_benchmark_return(
    benchmark_returns: pd.Series,
    trading_dates: pd.DatetimeIndex,
    start_pos: int,
    end_pos: int,
) -> float:
    """Compound benchmark daily close returns from start_pos through end_pos."""
    dates = trading_dates[start_pos : end_pos + 1]
    daily = benchmark_returns.reindex(dates)
    if len(daily) != len(dates) or daily.isna().any():
        return np.nan
    values = daily.to_numpy(float)
    if not np.isfinite(values).all() or np.any(values <= -1.0):
        return np.nan
    return float(np.prod(1.0 + values) - 1.0)


def calculate_result(
    signal_date: pd.Timestamp,
    horizon: int,
    gain: float,
    cut: float,
    stock: pd.DataFrame,
    benchmark_returns: pd.Series,
    trading_dates: pd.DatetimeIndex,
    calendar_pos: dict[pd.Timestamp, int],
) -> dict[str, Any] | None:
    """Calculate an exit or return the latest close-to-date open-position mark."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    t_pos = calendar_pos.get(signal_date)
    if t_pos is None:
        # Signal date must be a benchmark trading date.
        return None

    buy_pos = t_pos + 1
    last_pos = t_pos + horizon
    if buy_pos >= len(trading_dates):
        return None

    buy_date = trading_dates[buy_pos]
    if buy_date not in stock.index:
        return None

    buy_price = float(stock.at[buy_date, "open"])
    if not math.isfinite(buy_price) or buy_price <= 0:
        return None

    # Evaluate only the consecutive market data currently available, capped at
    # the model horizon. A future/missing day stops evaluation without shifting
    # the intended execution date.
    available_end_pos = min(last_pos, len(trading_dates) - 1)
    evaluated_end_pos = buy_pos - 1
    for pos in range(buy_pos, available_end_pos + 1):
        date = trading_dates[pos]
        if date not in stock.index:
            break
        prices = stock.loc[date, ["open", "high", "close"]]
        if prices.isna().any():
            break
        if not np.isfinite(prices.to_numpy(dtype=float)).all():
            break
        if pd.isna(benchmark_returns.get(date, np.nan)):
            break
        evaluated_end_pos = pos

    if evaluated_end_pos < buy_pos:
        return None

    def bench_ret_at(pos: int) -> float:
        return compounded_benchmark_return(
            benchmark_returns=benchmark_returns,
            trading_dates=trading_dates,
            start_pos=buy_pos,
            end_pos=pos,
        )

    def adjusted_return(price: float, pos: int) -> tuple[float, float]:
        raw_ret = price / buy_price - 1.0
        bench_ret = bench_ret_at(pos)
        if not math.isfinite(raw_ret) or not math.isfinite(bench_ret):
            return np.nan, np.nan
        return raw_ret, raw_ret - bench_ret

    def result(
        *,
        target: int,
        sell_price: float,
        exit_pos: int,
        exit_reason: str,
    ) -> dict[str, Any] | None:
        raw_ret, adj_ret = adjusted_return(sell_price, exit_pos)
        bench_ret = bench_ret_at(exit_pos)
        if not all(math.isfinite(x) for x in (sell_price, raw_ret, adj_ret, bench_ret)):
            return None
        return {
            "target": int(target),
            "ret": raw_ret,
            "ret_ew": adj_ret,
            "buy": buy_price,
            "sell": sell_price,
            # Inclusive of both the buy date and the sell date.
            "holding_days": exit_pos - buy_pos + 1,
            "exit_reason": exit_reason,
            "csi1500_ew": bench_ret,
        }

    def open_position_mark(mark_pos: int) -> dict[str, Any] | None:
        """Mark an unsold position using the latest available close."""
        mark_date = trading_dates[mark_pos]
        close_price = float(stock.at[mark_date, "close"])
        raw_ret, adj_ret = adjusted_return(close_price, mark_pos)
        bench_ret = bench_ret_at(mark_pos)
        if not all(math.isfinite(x) for x in (close_price, raw_ret, adj_ret, bench_ret)):
            return None
        return {
            "ret": raw_ret,
            "ret_ew": adj_ret,
            "buy": buy_price,
            "holding_days": mark_pos - buy_pos + 1,
            "csi1500_ew": bench_ret,
        }

    # T+1 exception: if adjusted T+1 close breaches cut, sell at T+2 open.
    t1_close = float(stock.at[buy_date, "close"])
    _, t1_close_adj = adjusted_return(t1_close, buy_pos)
    if not math.isfinite(t1_close_adj):
        return None
    if t1_close_adj <= cut + EPS:
        sell_pos = buy_pos + 1
        if sell_pos > last_pos or sell_pos > evaluated_end_pos:
            return open_position_mark(evaluated_end_pos)
        sell_date = trading_dates[sell_pos]
        return result(
            target=0,
            sell_price=float(stock.at[sell_date, "open"]),
            exit_pos=sell_pos,
            exit_reason="t1_stop_t2_open",
        )

    # For horizon=1, expiry is T+1 close immediately after the exception check.
    if horizon == 1:
        target = int(t1_close_adj >= gain - EPS)
        return result(
            target=target,
            sell_price=t1_close,
            exit_pos=buy_pos,
            exit_reason="expiry_gain" if target == 1 else "expiry_close",
        )

    # Normal checks from T+2 through T+H.
    for pos in range(buy_pos + 1, evaluated_end_pos + 1):
        date = trading_dates[pos]
        open_price = float(stock.at[date, "open"])
        high_price = float(stock.at[date, "high"])
        close_price = float(stock.at[date, "close"])
        bench_ret = bench_ret_at(pos)
        if not math.isfinite(bench_ret):
            return None

        _, open_adj = adjusted_return(open_price, pos)
        _, high_adj = adjusted_return(high_price, pos)
        _, close_adj = adjusted_return(close_price, pos)

        # Priority 1: gap/open above take-profit threshold -> sell at actual open.
        if open_adj >= gain - EPS:
            return result(
                target=1,
                sell_price=open_price,
                exit_pos=pos,
                exit_reason="profit_open",
            )

        # Priority 2: intraday high reaches threshold -> execute at threshold.
        if high_adj >= gain - EPS:
            gain_price = buy_price * (1.0 + bench_ret + gain)
            return result(
                target=1,
                sell_price=gain_price,
                exit_pos=pos,
                exit_reason="profit_intraday",
            )

        # Priority 3: close breaches stop -> execute at the cut threshold price.
        if close_adj <= cut + EPS:
            cut_price = buy_price * (1.0 + bench_ret + cut)
            return result(
                target=0,
                sell_price=cut_price,
                exit_pos=pos,
                exit_reason="stop_close",
            )

        # Expiry occurs only after all normal checks on T+H have failed.
        if pos == last_pos:
            expiry_target = int(close_adj >= gain - EPS)
            return result(
                target=expiry_target,
                sell_price=close_price,
                exit_pos=pos,
                exit_reason="expiry_gain" if expiry_target == 1 else "expiry_close",
            )

    return open_position_mark(evaluated_end_pos)


def safe_write_csv(df: pd.DataFrame, out_path: Path, encoding: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    df.to_csv(tmp_path, index=False, encoding=encoding)
    os.replace(tmp_path, out_path)


def main() -> None:
    args = parse_args()

    if not args.signals.exists():
        raise FileNotFoundError(f"Signals file not found: {args.signals}")
    if not args.stock_dir.is_dir():
        raise NotADirectoryError(f"Stock folder not found: {args.stock_dir}")
    if not args.benchmark.exists():
        raise FileNotFoundError(f"Benchmark file not found: {args.benchmark}")

    signals = pd.read_csv(args.signals, encoding=args.encoding, low_memory=False)
    signals = ensure_output_columns(signals)

    required_signal_cols = ["model", "ts_code", args.signal_date_col, "target"]
    missing = [c for c in required_signal_cols if c not in signals.columns]
    if missing:
        raise ValueError(
            f"Signals file is missing columns {missing}; columns={signals.columns.tolist()}"
        )

    signal_dates = normalize_date_series(signals[args.signal_date_col])

    benchmark = pd.read_csv(args.benchmark, encoding=args.encoding, low_memory=False)
    if args.benchmark_date_col not in benchmark.columns:
        raise ValueError(
            f"Benchmark date column {args.benchmark_date_col!r} is missing; "
            f"columns={benchmark.columns.tolist()}"
        )
    benchmark_returns, benchmark_col = detect_benchmark_return(
        benchmark=benchmark,
        requested_col=args.benchmark_return_col,
        date_col=args.benchmark_date_col,
    )
    if args.benchmark_return_percent:
        benchmark_returns = benchmark_returns / 100.0

    # Guard against accidentally treating percentage values as decimal returns.
    finite_abs = benchmark_returns.dropna().abs()
    if not finite_abs.empty and finite_abs.quantile(0.99) > 0.5:
        raise ValueError(
            f"Benchmark returns from {benchmark_col!r} look too large for decimals. "
            "Use --benchmark-return-percent if values such as 1.2 mean 1.2%."
        )

    trading_dates = pd.DatetimeIndex(benchmark_returns.index.unique()).sort_values()
    calendar_pos = {date: i for i, date in enumerate(trading_dates)}

    pending_mask = signals["target"].isna()
    pending_indices = signals.index[pending_mask].tolist()
    stock_cache: dict[str, pd.DataFrame | None] = {}

    filled = 0
    open_updated = 0
    unresolved = 0
    missing_stock_files: set[str] = set()
    errors: list[str] = []
    model_parameters: dict[str, tuple[int, float, float]] = {}

    for count, idx in enumerate(pending_indices, start=1):
        ts_code = str(signals.at[idx, "ts_code"]).strip()
        model_name = str(signals.at[idx, "model"]).strip()
        signal_date = signal_dates.at[idx]

        if (
            not ts_code
            or ts_code.lower() == "nan"
            or not model_name
            or model_name.lower() == "nan"
            or pd.isna(signal_date)
        ):
            unresolved += 1
            continue

        try:
            if model_name not in model_parameters:
                model_parameters[model_name] = get_model_parameters(model_name)
            horizon, gain, cut = model_parameters[model_name]

            if ts_code not in stock_cache:
                stock_path = find_stock_file(args.stock_dir, ts_code)
                if stock_path is None:
                    stock_cache[ts_code] = None
                    missing_stock_files.add(ts_code)
                else:
                    stock_cache[ts_code] = load_stock(
                        path=stock_path,
                        encoding=args.encoding,
                        date_col=args.stock_date_col,
                        open_col=args.open_col,
                        high_col=args.high_col,
                        close_col=args.close_col,
                    )

            stock = stock_cache[ts_code]
            if stock is None:
                unresolved += 1
                continue

            row_result = calculate_result(
                signal_date=signal_date,
                horizon=horizon,
                gain=gain,
                cut=cut,
                stock=stock,
                benchmark_returns=benchmark_returns,
                trading_dates=trading_dates,
                calendar_pos=calendar_pos,
            )
            if row_result is None:
                unresolved += 1
                continue

            for col, value in row_result.items():
                signals.at[idx, col] = value
            if "target" in row_result:
                filled += 1
            else:
                open_updated += 1

        except Exception as exc:  # Continue other signals and report bad rows.
            unresolved += 1
            errors.append(
                f"row={idx}, model={model_name}, ts_code={ts_code}, error={exc}"
            )

        if count % 500 == 0:
            print(
                f"[progress] checked={count}/{len(pending_indices)} "
                f"sold={filled} open_updated={open_updated} "
                f"unresolved={unresolved}",
                flush=True,
            )

    out_path = args.out if args.out is not None else args.signals
    safe_write_csv(signals, out_path, args.encoding)

    print(f"[benchmark] return_column={benchmark_col}")
    print(f"[rows] total={len(signals)} already_filled={int((~pending_mask).sum())}")
    print(
        f"[result] newly_sold={filled} open_updated={open_updated} "
        f"unresolved={unresolved}"
    )
    print(f"[write] {out_path}")

    if missing_stock_files:
        preview = sorted(missing_stock_files)[:20]
        print(
            f"[warning] missing stock files={len(missing_stock_files)}; "
            f"examples={preview}"
        )
    if errors:
        print(f"[warning] row errors={len(errors)}")
        for message in errors[:20]:
            print(f"  {message}")


if __name__ == "__main__":
    main()
