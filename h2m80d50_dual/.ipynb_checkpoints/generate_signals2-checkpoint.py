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
            if module == "pathlib._local" and hasattr(pathlib, name):
                if name in {"WindowsPath", "PosixPath"}:
                    return Path
                return getattr(pathlib, name)
            return super().find_class(module, name)

    with path.open("rb") as f:
        meta = CompatUnpickler(f).load()
    if not isinstance(meta, dict):
        raise TypeError(f"{path} did not contain a metadata dict.")
    return meta


def load_bundle(name: str, model_dir: Path) -> ModelBundle:
    meta = load_meta(model_dir / "model_meta.pkl")
    features = meta.get("features") or meta.get("feature_names") or meta.get("feature_columns")
    if not features:
        raise ValueError(f"No feature list found in {model_dir / 'model_meta.pkl'}")
    model = lgb.Booster(model_file=str(model_dir / "model.txt"))
    best_iteration = meta.get("best_iteration")
    if isinstance(best_iteration, (np.integer,)):
        best_iteration = int(best_iteration)
    if not best_iteration or best_iteration <= 0:
        best_iteration = None
    return ModelBundle(name=name, model=model, features=list(features), best_iteration=best_iteration)


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
                raise ModuleNotFoundError(
                    "Polars is required to infer max trade_date from a parquet input. "
                    "Install polars or pass --trade-date."
                )
            date = str(trade_date).replace("-", "")
            # Try both string and integer filters because trade_date storage type can differ by build.
            try:
                out = pd.read_parquet(path, filters=[("trade_date", "==", date)])
            except Exception:
                out = pd.DataFrame()
            if out.empty and date.isdigit():
                try:
                    out = pd.read_parquet(path, filters=[("trade_date", "==", int(date))])
                except Exception:
                    out = pd.DataFrame()
        if out.empty:
            raise ValueError(f"No rows for trade_date={date} in {path}")
        return out
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, dtype={"trade_date": str, "ts_code": str})
    raise ValueError(f"Unsupported input file type: {path}")


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    if "trade_date" not in df.columns or "ts_code" not in df.columns:
        raise ValueError("Input must contain trade_date and ts_code columns.")
    out = df.copy()
    out["trade_date"] = out["trade_date"].astype(str).str.replace("-", "", regex=False)
    out["ts_code"] = out["ts_code"].astype(str)
    return out


def select_dates(df: pd.DataFrame, trade_date: str | None) -> pd.DataFrame:
    if trade_date:
        date = str(trade_date).replace("-", "")
    else:
        date = str(df["trade_date"].max())
    out = df[df["trade_date"].eq(date)].copy()
    if out.empty:
        available = sorted(df["trade_date"].dropna().astype(str).unique())
        tail = available[-5:]
        raise ValueError(f"No rows for trade_date={date}. Last available dates: {tail}")
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
    x = pd.concat(cols, axis=1)
    return x.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def predict_bundle(bundle: ModelBundle, df: pd.DataFrame, allow_missing: bool) -> np.ndarray:
    x = ensure_features(df, bundle.features, bundle.name, allow_missing)
    return bundle.model.predict(x, num_iteration=bundle.best_iteration)


def add_family_predictions(df: pd.DataFrame, bundles: list[ModelBundle], allow_missing: bool) -> pd.DataFrame:
    out = df.copy()
    score_cols: list[str] = []
    rank_cols: list[str] = []
    for bundle in bundles:
        score_col = f"pred_{bundle.name}"
        rank_col = f"rank_{bundle.name}"
        out[score_col] = predict_bundle(bundle, out, allow_missing)
        out[rank_col] = out.groupby("trade_date")[score_col].rank(method="first", ascending=False)
        score_cols.append(score_col)
        rank_cols.append(rank_col)

    out["score_mean"] = out[score_cols].mean(axis=1)
    out["score_std"] = out[score_cols].std(axis=1, ddof=0)
    out["family_pred_cv"] = out["score_std"] / out["score_mean"].abs().clip(lower=EPS)
    out["rank_mean"] = out[rank_cols].mean(axis=1)
    out["avg_rank"] = out.groupby("trade_date")["rank_mean"].rank(method="first", ascending=True)
    out["score_rank"] = out.groupby("trade_date")["score_mean"].rank(method="first", ascending=False)
    return out


def add_seed_cv(df: pd.DataFrame, seed_bundles: list[ModelBundle], allow_missing: bool) -> pd.DataFrame:
    out = df.copy()
    if not seed_bundles:
        out["seed_pred_mean"] = np.nan
        out["seed_pred_std"] = np.nan
        out["seed_pred_cv"] = np.nan
        return out

    seed_cols: list[str] = []
    for bundle in seed_bundles:
        col = f"seed_pred_{bundle.name}"
        out[col] = predict_bundle(bundle, out, allow_missing)
        seed_cols.append(col)
    out["seed_pred_mean"] = out[seed_cols].mean(axis=1)
    out["seed_pred_std"] = out[seed_cols].std(axis=1, ddof=0)
    out["seed_pred_cv"] = out["seed_pred_std"] / out["seed_pred_mean"].abs().clip(lower=EPS)
    return out


def add_day_regime(df: pd.DataFrame, bundle: ModelBundle | None, allow_missing: bool) -> pd.DataFrame:
    out = df.copy()
    if bundle is None:
        out["prob_good_day"] = np.nan
        return out

    rows = []
    for trade_date, part in out.groupby("trade_date", sort=True):
        day_row = part.iloc[[0]]
        prob = float(predict_bundle(bundle, day_row, allow_missing)[0])
        rows.append({"trade_date": trade_date, "prob_good_day": prob})
    day_pred = pd.DataFrame(rows)
    return out.merge(day_pred, on="trade_date", how="left")


def add_signal_tags(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    th = cfg["thresholds"]
    required = ["ixic_swing", "csi1500_mcap_oc_ret"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing high-confidence filter columns: {missing}")

    out = df.copy()
    out["is_high_confidence"] = (
        (pd.to_numeric(out["ixic_swing"], errors="coerce") <= th["ixic_swing_max"])
        & (pd.to_numeric(out["csi1500_mcap_oc_ret"], errors="coerce") <= th["csi1500_mcap_oc_ret_max"])
    )
    rank_ok = out["avg_rank"] <= th["base_rank_top_n"]
    out["strict_signal"] = rank_ok & out["is_high_confidence"] & (out["family_pred_cv"] <= th["strict_family_cv_max"])
    out["strict_tight_signal"] = rank_ok & out["is_high_confidence"] & (
        out["family_pred_cv"] <= th["strict_family_cv_tight_max"]
    )
    out["strong_signal"] = (
        rank_ok
        & out["is_high_confidence"]
        & (out["prob_good_day"] >= th["strong_prob_good_day_min"])
        & (out["seed_pred_cv"] <= th["strong_seed_cv_max"])
    )
    out["layer2_m1_signal"] = (
        (out["avg_rank"] <= th["layer2_m1_rank_top_n"])
        & (out["prob_good_day"] >= th["layer2_m1_prob_good_day_min"])
        & (out["family_pred_cv"] <= th["layer2_m1_family_cv_max"])
        & (pd.to_numeric(out["ixic_swing"], errors="coerce") <= th["layer2_m1_ixic_swing_max"])
    )
    out["layer2_m3_signal"] = (
        (out["avg_rank"] <= th["layer2_m3_rank_top_n"])
        & (out["prob_good_day"] >= th["layer2_m3_prob_good_day_min"])
        & (out["seed_pred_cv"] <= th["layer2_m3_seed_cv_max"])
        & (pd.to_numeric(out["ixic_swing"], errors="coerce") <= th["layer2_m3_ixic_swing_max"])
    )
    out["layer2_signal"] = out["layer2_m1_signal"] | out["layer2_m3_signal"]

    out["signal_layer"] = "none"
    out.loc[out["layer2_m1_signal"] & out["layer2_m3_signal"], "signal_layer"] = "layer2_m1_m3_overlap"
    out.loc[out["layer2_m1_signal"] & ~out["layer2_m3_signal"], "signal_layer"] = "layer2_m1_only"
    out.loc[out["layer2_m3_signal"] & ~out["layer2_m1_signal"], "signal_layer"] = "layer2_m3_only"
    out.loc[out["strict_signal"] & out["strong_signal"], "signal_layer"] = "overlap_core"
    out.loc[out["strong_signal"] & ~out["strict_signal"], "signal_layer"] = "strong_only"
    out.loc[out["strict_signal"] & ~out["strong_signal"], "signal_layer"] = "strict_only"
    priority = cfg["priority"]
    out["signal_priority"] = out["signal_layer"].map(priority).fillna(99).astype(int)
    return out


def load_all_models(cfg: dict[str, Any], include_seed: bool, include_day: bool) -> tuple[list[ModelBundle], list[ModelBundle], ModelBundle | None]:
    family = [
        load_bundle(name, PACKAGE_DIR / "models" / "family" / name)
        for name in cfg["family_models"]
    ]
    seeds: list[ModelBundle] = []
    if include_seed:
        for name in cfg.get("seed_models", []):
            seed_dir = PACKAGE_DIR / "models" / "seed" / name
            if seed_dir.exists():
                seeds.append(load_bundle(name, seed_dir))
    day_bundle = None
    if include_day:
        day_name = cfg["day_regime_model"]
        day_dir = PACKAGE_DIR / "models" / "day_regime" / day_name
        if day_dir.exists():
            day_bundle = load_bundle(day_name, day_dir)
    return family, seeds, day_bundle


def write_outputs(df: pd.DataFrame, out_dir: Path, save_ranked: bool, candidate_top_n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for trade_date, part in df.groupby("trade_date", sort=True):
        ranked = part.sort_values(["signal_priority", "avg_rank", "score_rank", "ts_code"]).copy()
        signals = ranked[ranked["signal_layer"].ne("none")].copy()
        signal_path = out_dir / f"signals_{trade_date}.csv"
        signals.to_csv(signal_path, index=False, encoding="utf-8-sig")
        candidates = ranked[ranked["avg_rank"] <= candidate_top_n].copy()
        if candidate_top_n > 0:
            candidates.to_csv(out_dir / f"candidates_top{candidate_top_n}_{trade_date}.csv", index=False, encoding="utf-8-sig")
        if save_ranked:
            ranked.to_csv(out_dir / f"ranked_{trade_date}.csv", index=False, encoding="utf-8-sig")

        rank_ok = part["avg_rank"] <= 7
        high_conf = part["is_high_confidence"].astype(bool)
        family_cv_ok = part["family_pred_cv"] <= 0.074348
        seed_cv_ok = part["seed_pred_cv"] <= 0.06
        prob_good_ok = part["prob_good_day"] >= 0.20
        layer2 = part["layer2_signal"].astype(bool)
        layer2_m1 = part["layer2_m1_signal"].astype(bool)
        layer2_m3 = part["layer2_m3_signal"].astype(bool)
        context = {
            "trade_date": trade_date,
            "row_count": int(len(part)),
            "signal_count": int(len(signals)),
            "overlap_core_count": int((signals["signal_layer"] == "overlap_core").sum()),
            "strong_only_count": int((signals["signal_layer"] == "strong_only").sum()),
            "strict_only_count": int((signals["signal_layer"] == "strict_only").sum()),
            "layer2_signal_count": int(layer2.sum()),
            "layer2_m1_signal_count": int(layer2_m1.sum()),
            "layer2_m3_signal_count": int(layer2_m3.sum()),
            "layer2_m1_m3_overlap_count": int((layer2_m1 & layer2_m3).sum()),
            "layer1_or_layer2_count": int((ranked["signal_layer"].ne("none")).sum()),
            "candidate_top_n": int(candidate_top_n),
            "candidate_count": int(len(candidates)),
            "rank_le_7_count": int(rank_ok.sum()),
            "high_confidence_count": int(high_conf.sum()),
            "rank_le_7_and_high_confidence_count": int((rank_ok & high_conf).sum()),
            "rank_le_7_and_family_cv_ok_count": int((rank_ok & family_cv_ok).sum()),
            "rank_le_7_and_seed_cv_ok_count": int((rank_ok & seed_cv_ok).sum()),
            "rank_le_7_and_prob_good_day_ok_count": int((rank_ok & prob_good_ok).sum()),
            "rank_le_7_full_strict_count": int((rank_ok & high_conf & family_cv_ok).sum()),
            "rank_le_7_full_strong_count": int((rank_ok & high_conf & prob_good_ok & seed_cv_ok).sum()),
            "ixic_swing": None if part["ixic_swing"].isna().all() else float(part["ixic_swing"].dropna().iloc[0]),
            "csi1500_mcap_oc_ret": None if part["csi1500_mcap_oc_ret"].isna().all() else float(part["csi1500_mcap_oc_ret"].dropna().iloc[0]),
            "prob_good_day": None if part["prob_good_day"].isna().all() else float(part["prob_good_day"].dropna().iloc[0]),
            "seed_pred_cv_min_signal": None if signals.empty else float(signals["seed_pred_cv"].min()),
            "family_pred_cv_max_signal": None if signals.empty else float(signals["family_pred_cv"].max()),
        }
        with (out_dir / f"context_{trade_date}.json").open("w", encoding="utf-8") as f:
            json.dump(context, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] {signal_path} rows={len(signals)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate h3m55d15 strict/strong production signals.")
    parser.add_argument("--input", type=Path, required=True, help="Prepared feature panel parquet/csv.")
    parser.add_argument("--trade-date", default=None, help="YYYYMMDD. Defaults to max trade_date in input.")
    parser.add_argument("--out-dir", type=Path, default=PACKAGE_DIR / "signals")
    parser.add_argument("--allow-missing-features", action="store_true")
    parser.add_argument("--save-ranked", action="store_true", help="Also save full ranked universe for audit.")
    parser.add_argument("--candidate-top-n", type=int, default=20, help="Save top-N base candidates for empty-signal diagnostics.")
    parser.add_argument("--skip-seed", action="store_true", help="Disable strong seed-CV layer.")
    parser.add_argument("--skip-day-regime", action="store_true", help="Disable strong day-regime layer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    family, seed_bundles, day_bundle = load_all_models(
        cfg,
        include_seed=not args.skip_seed,
        include_day=not args.skip_day_regime,
    )
    print(f"[models] family={len(family)} seed={len(seed_bundles)} day={day_bundle.name if day_bundle else 'none'}")

    df = normalize_keys(read_input(args.input, args.trade_date))
    if args.input.suffix.lower() != ".parquet":
        df = select_dates(df, args.trade_date)
    print(f"[input] rows={len(df)} trade_date={df['trade_date'].iloc[0]}")

    scored = add_family_predictions(df, family, args.allow_missing_features)
    scored = add_seed_cv(scored, seed_bundles, args.allow_missing_features)
    scored = add_day_regime(scored, day_bundle, args.allow_missing_features)
    scored = add_signal_tags(scored, cfg)
    write_outputs(scored, args.out_dir, args.save_ranked, args.candidate_top_n)


if __name__ == "__main__":
    main()
