from __future__ import annotations

import argparse
import json
import pathlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_DIR / "config.json"
EPS = 1e-9
LOW_TURNOVER_THRESHOLD = 0.20042194092827004


@dataclass
class ModelBundle:
    name: str
    model: lgb.Booster
    features: list[str]
    best_iteration: int | None


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_meta(path: Path) -> dict[str, Any]:
    class CompatUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if module == "pathlib" and name in {"WindowsPath", "PosixPath"}:
                return Path
            if module == "pathlib._local" and name in {"WindowsPath", "PosixPath"}:
                return Path
            if module == "pathlib._local" and hasattr(pathlib, name):
                return getattr(pathlib, name)
            return super().find_class(module, name)

    with path.open("rb") as f:
        meta = CompatUnpickler(f).load()
    if not isinstance(meta, dict):
        raise TypeError(f"{path} did not contain a metadata dict.")
    return meta


def load_bundle(name: str, model_dir: Path) -> ModelBundle:
    meta_path = model_dir / "model_meta.pkl"
    model_path = model_dir / "model.txt"
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing model files under {model_dir}: need model.txt and model_meta.pkl")
    meta = load_meta(meta_path)
    features = meta.get("features") or meta.get("feature_names") or meta.get("feature_columns")
    if not features:
        raise ValueError(f"No feature list found in {meta_path}")
    best_iteration = meta.get("best_iteration")
    if isinstance(best_iteration, np.integer):
        best_iteration = int(best_iteration)
    if not best_iteration or best_iteration <= 0:
        best_iteration = None
    return ModelBundle(
        name=name,
        model=lgb.Booster(model_file=str(model_path)),
        features=list(features),
        best_iteration=best_iteration,
    )


def load_models(cfg: dict[str, Any]) -> list[ModelBundle]:
    bundles = []
    for item in cfg["seed_models"]:
        model_dir = PACKAGE_DIR / item["dir"]
        bundles.append(load_bundle(item["name"], model_dir))
    return bundles


def read_input(path: Path, trade_date: str | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import polars as pl

            lf = pl.scan_parquet(str(path)).with_columns(
                pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", "").alias("trade_date")
            )
            if trade_date:
                date = str(trade_date).replace("-", "")
            else:
                date = str(lf.select(pl.col("trade_date").max()).collect().item())
            out = lf.filter(pl.col("trade_date") == date).collect().to_pandas()
        except ModuleNotFoundError:
            if not trade_date:
                raise ModuleNotFoundError("Polars is required to infer max trade_date. Pass --trade-date or install polars.")
            date = str(trade_date).replace("-", "")
            out = pd.read_parquet(path, filters=[("trade_date", "==", int(date) if date.isdigit() else date)])
        if out.empty:
            raise ValueError(f"No rows for trade_date={date} in {path}")
        return out
    if suffix in {".csv", ".txt"}:
        df = pd.read_csv(path, dtype={"trade_date": str, "ts_code": str})
        if trade_date:
            date = str(trade_date).replace("-", "")
        else:
            date = str(df["trade_date"].astype(str).str.replace("-", "", regex=False).max())
        return df[df["trade_date"].astype(str).str.replace("-", "", regex=False).eq(date)].copy()
    raise ValueError(f"Unsupported input file type: {path}")


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    if "trade_date" not in df.columns or "ts_code" not in df.columns:
        raise ValueError("Input must contain trade_date and ts_code columns.")
    out = df.copy()
    out["trade_date"] = out["trade_date"].astype(str).str.replace("-", "", regex=False)
    out["ts_code"] = out["ts_code"].astype(str)
    return out


def ensure_features(df: pd.DataFrame, features: list[str], model_name: str, allow_missing: bool) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing and not allow_missing:
        preview = ", ".join(missing[:20])
        raise ValueError(f"{model_name} missing {len(missing)} features. First missing: {preview}")
    cols = []
    for col in features:
        if col in df.columns:
            cols.append(pd.to_numeric(df[col], errors="coerce").rename(col))
        else:
            cols.append(pd.Series(np.nan, index=df.index, name=col))
    return pd.concat(cols, axis=1).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def add_predictions(df: pd.DataFrame, bundles: list[ModelBundle], allow_missing: bool) -> pd.DataFrame:
    out = df.copy()
    pred_cols = []
    rank_cols = []
    for bundle in bundles:
        x = ensure_features(out, bundle.features, bundle.name, allow_missing)
        pred_col = f"pred_{bundle.name}"
        rank_col = f"rank_{bundle.name}"
        out[pred_col] = bundle.model.predict(x, num_iteration=bundle.best_iteration)
        out[rank_col] = out.groupby("trade_date")[pred_col].rank(method="first", ascending=False)
        pred_cols.append(pred_col)
        rank_cols.append(rank_col)
    out["avg_pred"] = out[pred_cols].mean(axis=1)
    out["avg_rank_raw"] = out[rank_cols].mean(axis=1)
    out["avg_pred_rank"] = out.groupby("trade_date")["avg_pred"].rank(method="first", ascending=False)
    out["avg_rank"] = out.groupby("trade_date")["avg_rank_raw"].rank(method="first", ascending=True)
    out["seed_pred_std"] = out[pred_cols].std(axis=1, ddof=0)
    out["seed_pred_cv"] = out["seed_pred_std"] / out["avg_pred"].abs().clip(lower=EPS)
    return out


def add_base_selections(df: pd.DataFrame, cfg: dict[str, Any], seed_names: list[str]) -> pd.DataFrame:
    out = df.copy()
    vote_top_n = int(cfg["ranking"]["vote_top_n"])
    vote_min = int(cfg["ranking"]["vote_min"])
    avg_pred_top_n = int(cfg["ranking"]["avg_pred_top_n"])

    out["avg_pred_top1"] = out["avg_pred_rank"] <= avg_pred_top_n
    vote_flags = []
    for seed in seed_names:
        rank_col = f"rank_{seed}"
        flag_col = f"in_top_{vote_top_n}_{seed}"
        out[flag_col] = out[rank_col] <= vote_top_n
        vote_flags.append(flag_col)
    out["vote4_top3_count"] = out[vote_flags].sum(axis=1)

    vote3_selected = []
    for _, part in out.groupby("trade_date", sort=True):
        pick = part[part["vote4_top3_count"] >= 3].sort_values(
            ["avg_rank_raw", "avg_pred_rank", "ts_code"], ascending=[True, True, True]
        ).head(vote_top_n)
        vote3_selected.append(pick[["trade_date", "ts_code"]])
    if vote3_selected:
        vote3_keys = pd.concat(vote3_selected, ignore_index=True)
        vote3_keys["vote3_top3"] = True
        out = out.merge(vote3_keys, on=["trade_date", "ts_code"], how="left")
    else:
        out["vote3_top3"] = False
    out["vote3_top3"] = out["vote3_top3"].fillna(False).astype(bool)

    selected = []
    for _, part in out.groupby("trade_date", sort=True):
        pick = part[part["vote4_top3_count"] >= vote_min].sort_values(
            ["vote4_top3_count", "avg_rank_raw"], ascending=[False, True]
        ).head(vote_top_n)
        selected.append(pick[["trade_date", "ts_code"]])
    if selected:
        selected_keys = pd.concat(selected, ignore_index=True)
        selected_keys["vote4_strict_top3"] = True
        out = out.merge(selected_keys, on=["trade_date", "ts_code"], how="left")
    else:
        out["vote4_strict_top3"] = False
    out["vote4_strict_top3"] = out["vote4_strict_top3"].fillna(False).astype(bool)
    return out


def eval_conditions(df: pd.DataFrame, conditions: list[list[Any]]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, op, threshold in conditions:
        if col not in df.columns:
            raise ValueError(f"Missing regime feature: {col}")
        values = pd.to_numeric(df[col], errors="coerce")
        if op == "<=":
            mask &= values <= float(threshold)
        elif op == ">=":
            mask &= values >= float(threshold)
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return mask.fillna(False)


def eval_condition_groups(df: pd.DataFrame, condition_groups: list[list[list[Any]]]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for group in condition_groups:
        mask |= eval_conditions(df, group)
    return mask.fillna(False)


def add_signal_models(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    defs = cfg["signal_models"]
    # Base signals first.
    for name, spec in defs.items():
        if "base" not in spec:
            continue
        base = out[spec["base"]].astype(bool)
        out[name] = base & eval_conditions(out, spec.get("conditions", []))
    # Composite signals.
    for name, spec in defs.items():
        if spec.get("operator") != "or":
            continue
        mask = pd.Series(False, index=out.index)
        for member in spec["members"]:
            mask |= out[member].astype(bool)
        out[name] = mask
    kept = [
        "avg_hi_hktech40_n22530",
        "vote4_hi_hktech40_n22530",
        "vote4_hi_plus_avg_hi",
        "vote4_hi_plus_avg_broad_hktech20",
    ]
    out["priority_model_count"] = out[kept].sum(axis=1)
    out["priority1a_signal"] = out["priority_model_count"] == 4
    out["priority1_signal"] = out["priority1a_signal"]

    if "turnover_prank_1500" not in out.columns:
        raise ValueError("Missing required production feature: turnover_prank_1500")
    out["priority1b_raw_signal"] = (out["avg_pred_rank"] <= 1) & (
        pd.to_numeric(out["turnover_prank_1500"], errors="coerce") <= LOW_TURNOVER_THRESHOLD
    )
    out["priority1b_signal"] = out["priority1b_raw_signal"] & ~out["priority1a_signal"]
    out["priority2_signal"] = (out["priority_model_count"] >= 1) & ~(out["priority1a_signal"] | out["priority1b_signal"])

    p3 = cfg["priority_rules"]["priority3"]
    out["priority3_raw_signal"] = out[p3["base"]].astype(bool) & eval_condition_groups(out, p3["condition_groups"])
    out["priority3_signal"] = out["priority3_raw_signal"] & ~(
        out["priority1a_signal"] | out["priority1b_signal"] | out["priority2_signal"]
    )

    out["priority"] = np.select(
        [out["priority1a_signal"], out["priority1b_signal"], out["priority2_signal"], out["priority3_signal"]],
        [1, 1, 2, 3],
        default=0,
    )
    out["priority_name"] = np.select(
        [out["priority1a_signal"], out["priority1b_signal"], out["priority2_signal"], out["priority3_signal"]],
        ["P1A_current_consensus", "P1B_low_turnover_exception", "P2_current_broader", "P3_global_china_expansion"],
        default="",
    )
    return out


def add_watchlist(df: pd.DataFrame, watchlist_top_n: int, watchlist_vote_min: int) -> pd.DataFrame:
    out = df.copy()
    out["watchlist_avg_top1"] = out["avg_pred_rank"] <= 1
    out[f"watchlist_avg_top{watchlist_top_n}"] = out["avg_pred_rank"] <= watchlist_top_n
    out[f"watchlist_vote{watchlist_vote_min}_top3"] = out["vote4_top3_count"] >= watchlist_vote_min

    watch_cols = [
        "watchlist_avg_top1",
        f"watchlist_avg_top{watchlist_top_n}",
        f"watchlist_vote{watchlist_vote_min}_top3",
    ]

    reasons = []
    for _, row in out[watch_cols].iterrows():
        active = [col.replace("watchlist_", "") for col in watch_cols if bool(row[col])]
        reasons.append("|".join(active))
    out["watchlist_reason"] = reasons
    out["watchlist_signal"] = (out["priority"] == 0) & out["watchlist_reason"].ne("")

    out["signal_type"] = np.select(
        [out["priority"] > 0, out["watchlist_signal"]],
        ["TRADE", "WATCHLIST"],
        default="",
    )
    out["signal_tag"] = np.select(
        [out["priority"] > 0, out["watchlist_signal"]],
        [out["priority_name"], "WATCHLIST_" + out["watchlist_reason"]],
        default="",
    )
    return out


def signal_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "signal_type", "signal_tag", "watchlist_reason",
        "trade_date", "ts_code", "priority", "priority_name", "priority_model_count",
        "avg_pred", "avg_pred_rank", "avg_rank_raw", "vote4_top3_count",
        "turnover_prank_1500",
        "avg_pred_top1", "vote3_top3", "vote4_strict_top3", "watchlist_signal",
        "avg_hi_hktech40_n22530", "vote4_hi_hktech40_n22530",
        "vote4_hi_plus_avg_hi", "vote4_hi_plus_avg_broad_hktech20",
        "priority1a_signal", "priority1b_signal", "priority1b_raw_signal",
        "priority1_signal", "priority2_signal", "priority3_signal",
        "priority3_raw_signal",
        "hktech_swing", "n225_swing", "spx_swing", "hsi_ret_lag1", "ret_5_rel_csi1500_ew"
    ]
    return [c for c in preferred if c in df.columns]


def diagnostic_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "trade_date", "ts_code", "priority", "priority_name", "priority_model_count",
        "avg_pred", "avg_pred_rank", "avg_rank_raw", "seed_pred_cv",
        "vote4_top3_count", "avg_pred_top1", "vote3_top3", "vote4_strict_top3",
        "avg_hi_hktech40_n22530", "vote4_hi_hktech40_n22530",
        "vote4_hi_plus_avg_hi", "vote4_hi_plus_avg_broad_hktech20",
        "priority1a_signal", "priority1b_signal", "priority1b_raw_signal",
        "priority1_signal", "priority2_signal", "priority3_raw_signal", "priority3_signal",
        "turnover_prank_1500",
        "hktech_swing", "n225_swing", "spx_swing", "hsi_ret_lag1", "ret_5_rel_csi1500_ew",
    ]
    cols = [c for c in preferred if c in df.columns]
    cols.extend([c for c in df.columns if c.startswith("pred_seed") or c.startswith("rank_seed")])
    return cols


def diagnostic_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bool_cols = [
        "avg_pred_top1",
        "vote3_top3",
        "vote4_strict_top3",
        "watchlist_signal",
        "avg_hi_hktech40_n22530",
        "vote4_hi_hktech40_n22530",
        "vote4_hi_plus_avg_hi",
        "vote4_hi_plus_avg_broad_hktech20",
        "priority1a_signal",
        "priority1b_signal",
        "priority1b_raw_signal",
        "priority1_signal",
        "priority2_signal",
        "priority3_raw_signal",
        "priority3_signal",
    ]
    for trade_date, part in df.groupby("trade_date", sort=True):
        row: dict[str, Any] = {"trade_date": trade_date, "rows": len(part)}
        for col in bool_cols:
            if col in part.columns:
                row[f"{col}_count"] = int(part[col].fillna(False).sum())
        for col in ["turnover_prank_1500", "hktech_swing", "n225_swing", "spx_swing", "hsi_ret_lag1", "ret_5_rel_csi1500_ew"]:
            if col in part.columns:
                row[col] = pd.to_numeric(part[col], errors="coerce").dropna().iloc[0] if part[col].notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate h5m80d50 P1/P2/P3 production signals.")
    parser.add_argument("--input", type=Path, required=True, help="Prediction feature parquet/csv.")
    parser.add_argument("--trade-date", default=None, help="YYYYMMDD. Defaults to max date in input.")
    parser.add_argument("--out-dir", type=Path, default=PACKAGE_DIR / "signals")
    parser.add_argument("--allow-missing-features", action="store_true")
    parser.add_argument("--save-ranked", action="store_true")
    parser.add_argument("--diagnostic-top-n", type=int, default=30)
    parser.add_argument("--watchlist-top-n", type=int, default=3)
    parser.add_argument("--watchlist-vote-min", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config()
    bundles = load_models(cfg)
    seed_names = [b.name for b in bundles]

    df = normalize_keys(read_input(args.input, args.trade_date))
    df = add_predictions(df, bundles, args.allow_missing_features)
    df = add_base_selections(df, cfg, seed_names)
    df = add_signal_models(df, cfg)
    df = add_watchlist(df, args.watchlist_top_n, args.watchlist_vote_min)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    date = str(df["trade_date"].max())
    signals = df[df["signal_type"].ne("")].sort_values(
        ["signal_type", "priority", "avg_pred_rank", "ts_code"],
        ascending=[True, True, True, True],
    )
    trade_signals = df[df["priority"] > 0].sort_values(["priority", "avg_pred_rank", "ts_code"])
    watchlist = df[df["watchlist_signal"]].sort_values(["avg_pred_rank", "ts_code"])

    signal_path = args.out_dir / f"signals_h5m80d50_priority_{date}.csv"
    signals[signal_columns(signals)].to_csv(signal_path, index=False, encoding="utf_8_sig")
    latest_path = args.out_dir / "signals_latest.csv"
    signals[signal_columns(signals)].to_csv(latest_path, index=False, encoding="utf_8_sig")

    if args.save_ranked:
        ranked_path = args.out_dir / f"ranked_h5m80d50_priority_{date}.csv"
        ranked_cols = signal_columns(df) + [c for c in df.columns if c.startswith("pred_") or c.startswith("rank_")]
        df[ranked_cols].sort_values(["avg_pred_rank", "ts_code"]).to_csv(ranked_path, index=False, encoding="utf_8_sig")

    summary = (
        trade_signals.groupby(["trade_date", "priority", "priority_name"], as_index=False)
        .agg(signal_count=("ts_code", "size"), avg_pred_mean=("avg_pred", "mean"))
        .sort_values(["trade_date", "priority"])
    )
    summary_path = args.out_dir / f"signal_summary_{date}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf_8_sig")

    diag = diagnostic_summary(df)
    diag_path = args.out_dir / f"diagnostic_summary_{date}.csv"
    diag.to_csv(diag_path, index=False, encoding="utf_8_sig")

    diag_ranked = df.sort_values(["avg_pred_rank", "ts_code"]).head(args.diagnostic_top_n)
    diag_ranked_path = args.out_dir / f"diagnostic_top{args.diagnostic_top_n}_{date}.csv"
    diag_ranked[diagnostic_columns(diag_ranked)].to_csv(diag_ranked_path, index=False, encoding="utf_8_sig")

    watchlist_path = args.out_dir / f"watchlist_h5m80d50_priority_{date}.csv"
    watchlist[signal_columns(watchlist)].to_csv(watchlist_path, index=False, encoding="utf_8_sig")

    print("[SIGNALS]")
    print(summary.to_string(index=False) if not summary.empty else "No signals.")
    print("[WATCHLIST]")
    if watchlist.empty:
        print("No watchlist candidates.")
    else:
        print(watchlist[signal_columns(watchlist)].to_string(index=False))
    print("[DIAGNOSTIC]")
    print(diag.to_string(index=False))
    print(f"[SAVE] {signal_path}")
    print(f"[SAVE] {latest_path}")
    print(f"[SAVE] {watchlist_path}")
    print(f"[SAVE] {diag_path}")
    print(f"[SAVE] {diag_ranked_path}")


if __name__ == "__main__":
    main()
