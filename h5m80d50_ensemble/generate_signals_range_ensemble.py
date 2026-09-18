from __future__ import annotations

import argparse
import json
import gc
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import lightgbm as lgb

from generate_signals import HERE, read_lines


def normalize_date(value: object) -> str:
    return str(value).replace("-", "")[:8]


def read_chunks(path: Path, columns: list[str] | None, batch_rows: int):
    if path.suffix.lower() == ".parquet":
        with pq.ParquetFile(path, pre_buffer=False) as pf:
            for batch in pf.iter_batches(columns=columns, batch_size=batch_rows, use_threads=False):
                yield batch.to_pandas()
    elif path.suffix.lower() in {".csv", ".txt"}:
        with pd.read_csv(path, dtype={"trade_date": str, "ts_code": str},
                         usecols=columns, chunksize=batch_rows) as chunks:
            yield from chunks
    else:
        raise ValueError(f"Unsupported input file: {path}")


def date_arg(value: str) -> str:
    value = str(value).replace("-", "")
    try:
        pd.to_datetime(value, format="%Y%m%d", errors="raise")
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("Use a valid YYYYMMDD date") from exc
    if len(value) != 8 or not value.isdigit():
        raise argparse.ArgumentTypeError("Use YYYYMMDD")
    return value


def normalize_dates(values: pd.Series) -> pd.Series:
    dates = values.astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    if not dates.str.fullmatch(r"\d{8}").all():
        raise ValueError("trade_date must contain non-null YYYYMMDD integers/strings or YYYY-MM-DD strings")
    return dates


def inspect_dates(path: Path, start: str, end: str) -> tuple[list[str], bool]:
    """Read only the date column in bounded batches; never load the wide panel."""
    found = set()
    ordered = True
    previous = None
    for chunk in read_chunks(path, ["trade_date"], 65536):
        values = normalize_dates(chunk["trade_date"])
        if values.empty:
            continue
        ordered = ordered and values.is_monotonic_increasing and (previous is None or previous <= values.iloc[0])
        previous = values.iloc[-1]
        found.update(values[values.between(start, end)].unique().tolist())
    return sorted(found), bool(ordered)


def iter_sorted_days(path: Path, columns: list[str] | None, start: str, end: str,
                     batch_rows: int, max_day_rows: int):
    """Single wide scan. Complete dates may span read batches and row groups."""
    active = None
    pieces = []
    count = 0
    for pdf in read_chunks(path, columns, batch_rows):
        pdf["trade_date"] = normalize_dates(pdf["trade_date"])
        values = pdf["trade_date"].to_numpy()
        boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            date = values[left]
            if active is not None and date != active:
                ready = pd.concat(pieces, ignore_index=True)
                pieces = []
                count = 0
                yield active, ready
                del ready
                active = None
            if date < start:
                continue
            if date > end:
                return
            active = date
            count += right - left
            if count > max_day_rows:
                raise ValueError(f"{date}: more than {max_day_rows} rows; inspect duplicates or increase --max-day-rows")
            # Copy only the fragment so a retained day cannot pin old batches.
            pieces.append(pdf.iloc[left:right].copy())
        del pdf
    if pieces:
        yield active, pd.concat(pieces, ignore_index=True)


def iter_filtered_days(path: Path, columns: list[str] | None, dates: list[str], max_day_rows: int):
    """Unsorted fallback: one full date at a time; potentially many disk scans."""
    if path.suffix.lower() in {".csv", ".txt"}:
        for date in dates:
            parts = []
            count = 0
            for chunk in read_chunks(path, columns, 2048):
                normalized = normalize_dates(chunk["trade_date"])
                part = chunk.loc[normalized.eq(date)].copy()
                if part.empty:
                    continue
                part["trade_date"] = date
                count += len(part)
                if count > max_day_rows:
                    raise ValueError(f"{date}: exceeds --max-day-rows")
                parts.append(part)
            if parts:
                yield date, pd.concat(parts, ignore_index=True)
        return
    source = pl.scan_parquet(path).select(columns).with_columns(
        pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", "").str.slice(0, 8))
    for date in dates:
        frame = source.filter(pl.col("trade_date") == date).limit(max_day_rows + 1).collect(engine="streaming")
        if frame.height > max_day_rows:
            raise ValueError(f"{date}: exceeds --max-day-rows")
        yield date, frame.to_pandas()
        del frame


def rss_message() -> str:
    try:
        import psutil
        return f"rss={psutil.Process().memory_info().rss / 2**30:.2f}GiB"
    except ImportError:
        return "rss=unavailable (optional: install psutil)"


def configure_model_cache(max_models: int = 32) -> None:
    global _load_model
    previous = globals().get("_load_model")
    if previous is not None:
        previous.cache_clear()
    @lru_cache(maxsize=max_models)
    def load_model(path: str):
        return lgb.Booster(model_file=path)
    _load_model = load_model


configure_model_cache()


def predict_family(frame: pd.DataFrame, family: str, seeds: list[int]) -> pd.DataFrame:
    family_dir = HERE / "models" / family
    features = read_lines(family_dir / "features.txt")
    x = frame[features].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    predictions = []
    for seed in seeds:
        model_path = family_dir / f"seed{seed}" / "model.txt"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = _load_model(str(model_path.resolve()))
        if model.feature_name() != features:
            raise ValueError(f"{family}/seed{seed}: model feature schema differs from features.txt")
        predictions.append(model.predict(x, num_iteration=model.current_iteration()))
    matrix = np.column_stack(predictions)
    return pd.DataFrame({
        f"{family}_pred": matrix.mean(axis=1), f"{family}_seed_std": matrix.std(axis=1),
        f"{family}_seed_min": matrix.min(axis=1), f"{family}_seed_max": matrix.max(axis=1),
    }, index=frame.index)


def build_ranked(
    frame: pd.DataFrame,
    cfg: dict[str, object],
    family_features: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Predict the range and apply every rank/gate independently by date."""
    families = cfg["families"]
    policy = cfg["signal_policy"]
    gate_source = str(policy["gate_source_feature"])

    ranked = frame[["trade_date", "ts_code"]].copy()
    for family in families:
        print(f"[predict] family={family}")
        pred = predict_family(frame, family, cfg["seeds"])
        ranked = pd.concat([ranked, pred], axis=1)
        ranked[f"{family}_rank"] = (
            ranked.groupby("trade_date", sort=False)[f"{family}_pred"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        ranked[f"{family}_selected"] = (
            ranked[f"{family}_rank"] <= int(families[family]["top_k"])
        )

    selected_cols = [f"{name}_selected" for name in families]
    ranked["family_count"] = ranked[selected_cols].sum(axis=1).astype(int)
    ranked["ensemble_pred"] = ranked[
        [f"{name}_pred" for name in families]
    ].mean(axis=1)
    ranked["ensemble_seed_std"] = ranked[
        [f"{name}_seed_std" for name in families]
    ].mean(axis=1)
    ranked["legacy_signal_tag"] = np.select(
        [ranked.family_count == 3, ranked.family_count == 2, ranked.family_count == 1],
        ["CONSENSUS_3", "CONSENSUS_2", "FAMILY_ONLY"],
        default="",
    )
    ranked["family_tags"] = ranked.apply(
        lambda row: "|".join(
            name for name in families if row[f"{name}_selected"]
        ),
        axis=1,
    )

    valid_gate_values = frame[gate_source].replace([np.inf, -np.inf], np.nan)
    ranked[gate_source] = valid_gate_values
    gate_rank = valid_gate_values.groupby(frame["trade_date"], sort=False).rank(
        method="average",
        ascending=True,
    )
    valid_count_by_date = valid_gate_values.notna().groupby(
        frame["trade_date"], sort=False
    ).transform("sum")
    ranked[str(policy["gate_feature"])] = gate_rank / valid_count_by_date.replace(0, np.nan)

    threshold = float(policy["gate_threshold"])
    ranked["production_gate_pass"] = (
        ranked[str(policy["gate_feature"])].le(threshold).fillna(False)
    )
    ranked["production_signal"] = (
        ranked[f"{policy['family']}_rank"].le(int(policy["top_k"]))
        & ranked["production_gate_pass"]
    )
    ranked["ensemble_signal"] = ranked["family_count"].ge(1)
    ranked["output_signal"] = ranked["production_signal"] | ranked["ensemble_signal"]
    ranked["signal_tag"] = ranked.apply(
        lambda row: "|".join(
            tag
            for tag in [
                str(policy["production_tag"]) if row["production_signal"] else "",
                str(row["legacy_signal_tag"]) if row["ensemble_signal"] else "",
            ]
            if tag
        ),
        axis=1,
    )
    ranked["gate_threshold"] = threshold
    return ranked, policy


def write_one_date(
    ranked_range: pd.DataFrame,
    date: str,
    out_dir: Path,
    save_ranked: bool,
    policy: dict[str, object],
    gate_source: str,
) -> dict[str, object]:
    """Write the same artifacts produced by generate_signals.py for one date."""
    ranked = ranked_range[ranked_range["trade_date"].eq(date)].copy()
    ranked = ranked.sort_values(
        [
            "production_signal",
            f"{policy['family']}_rank",
            "family_count",
            "ensemble_pred",
            "ts_code",
        ],
        ascending=[False, True, False, False, True],
    )
    signals = ranked[ranked.output_signal].copy()
    production_signals = ranked[ranked.production_signal].copy()

    signal_path = out_dir / f"signals_{date}.csv"
    production_path = out_dir / f"production_signals_{date}.csv"
    signals.to_csv(signal_path, index=False)
    signals.to_csv(out_dir / "signals_latest.csv", index=False)
    production_signals.to_csv(production_path, index=False)
    production_signals.to_csv(out_dir / "production_signals_latest.csv", index=False)
    if save_ranked:
        ranked.to_parquet(out_dir / f"ranked_{date}.parquet", index=False)

    summary = {
        "trade_date": date,
        "universe_rows": len(ranked),
        "signal_rows": len(signals),
        "ensemble_signal_rows": int(ranked.ensemble_signal.sum()),
        "production_policy": str(policy["production_tag"]),
        "f220_top1_rows": int((ranked.F220_rank == 1).sum()),
        "gate_pass_rows": int(ranked.production_gate_pass.sum()),
        "production_signal_rows": int(ranked.production_signal.sum()),
        "consensus_3_rows": int((ranked.legacy_signal_tag == "CONSENSUS_3").sum()),
        "consensus_2_rows": int((ranked.legacy_signal_tag == "CONSENSUS_2").sum()),
        "family_only_rows": int((ranked.legacy_signal_tag == "FAMILY_ONLY").sum()),
        "gate_source_nonnull_rows": int(ranked[gate_source].notna().sum()),
        "gate_threshold": float(policy["gate_threshold"]),
    }
    (out_dir / f"summary_{date}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate transparent F150/C185/F220 ensemble signals for a date range."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Feature parquet, usually processed/train_v5b/train_v5b.parquet",
    )
    parser.add_argument("--start-date", required=True, help="First feature date (YYYYMMDD).")
    parser.add_argument("--end-date", required=True, help="Last feature date (YYYYMMDD).")
    parser.add_argument("--out-dir", type=Path, default=HERE / "signals")
    parser.add_argument(
        "--save-ranked",
        action="store_true",
        help="Save all stocks with scores and ranks for every processed date.",
    )
    parser.add_argument("--read-batch-rows", type=int, default=2048)
    parser.add_argument("--max-day-rows", type=int, default=20000)
    parser.add_argument("--read-mode", choices=["auto", "sorted", "filtered"], default="auto")
    parser.add_argument("--model-cache-size", type=int, default=32)
    args = parser.parse_args()
    if args.input.suffix.lower() != ".parquet":
        parser.error("Input must be a parquet file")
    if min(args.read_batch_rows, args.max_day_rows) < 1 or args.model_cache_size < 0:
        parser.error("Read sizes must be positive and model-cache-size nonnegative")
    configure_model_cache(args.model_cache_size)

    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    family_features = {
        name: read_lines(HERE / "models" / name / "features.txt")
        for name in cfg["families"]
    }
    policy = cfg["signal_policy"]
    gate_source = str(policy["gate_source_feature"])
    all_features = sorted(set().union(*family_features.values(), {gate_source}))

    start_date = date_arg(args.start_date)
    end_date = date_arg(args.end_date)
    if start_date > end_date:
        raise ValueError("start-date must be <= end-date")
    required = list(dict.fromkeys(["trade_date", "ts_code", *all_features]))
    schema = pl.read_parquet_schema(args.input)
    missing = sorted(set(required) - set(schema.names()))
    if missing:
        raise KeyError(f"Input lacks {len(missing)} required columns: {missing[:20]}")
    print("[inspect] reading date column only", flush=True)
    dates, ordered = inspect_dates(args.input, start_date, end_date)
    if not dates:
        raise ValueError("No rows in requested range")
    if args.read_mode == "sorted" and not ordered:
        raise ValueError("Input is not date-sorted; use auto or filtered")
    mode = "sorted" if ordered and args.read_mode != "filtered" else "filtered"
    print(f"[plan] dates={len(dates)} features={len(all_features)} reader={mode} "
          f"model_cache_limit={args.model_cache_size}", flush=True)
    days = (iter_sorted_days(args.input, required, start_date, end_date,
                             args.read_batch_rows, args.max_day_rows)
            if mode == "sorted" else iter_filtered_days(args.input, required, dates, args.max_day_rows))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    summary_path = args.out_dir / f"range_generation_summary_{dates[0]}_{dates[-1]}.csv"
    for index, (date, frame) in enumerate(days, 1):
        t0=time.perf_counter()
        frame["trade_date"] = normalize_dates(frame["trade_date"])
        frame["ts_code"] = frame["ts_code"].astype(str)
        if frame.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError(f"Duplicate trade_date/ts_code rows on {date}")
        ranked, day_policy = build_ranked(frame, cfg, family_features)
        summary = write_one_date(ranked, date, args.out_dir, args.save_ranked, day_policy, gate_source)
        summaries.append(summary)
        temporary=summary_path.with_suffix(".csv.partial")
        pd.DataFrame(summaries).to_csv(temporary,index=False,encoding="utf_8_sig")
        temporary.replace(summary_path)
        del frame, ranked
        gc.collect()
        print(f"[{index}/{len(dates)}] {date} universe={summary['universe_rows']} "
              f"signals={summary['signal_rows']} production={summary['production_signal_rows']} "
              f"seconds={time.perf_counter()-t0:.2f} {rss_message()}",flush=True)
    if [x["trade_date"] for x in summaries] != dates:
        raise RuntimeError("Written dates differ from preflight; input may have changed")

    summary_path = args.out_dir / f"range_generation_summary_{dates[0]}_{dates[-1]}.csv"
    print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()
