#!/usr/bin/env python
# coding: utf-8
"""
download_global_indexes.py

Task 2: download global index data using two-source rule: Tushare + AkShare.
Selection:
    1. If only one source has data, use it.
    2. If both have data, use the source with latest trade_date.
    3. If latest dates tie, use Tushare.
Special:
    XIN9 uses Tushare only for now.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import tushare as ts

try:
    import akshare as ak
except ImportError:
    ak = None

PROJ_ROOT = Path.cwd().parent
CONFIG_DIR = PROJ_ROOT / "config"
if str(CONFIG_DIR) not in sys.path:
    sys.path.append(str(CONFIG_DIR))

from config_date import history_start_date, End_date_global
from config import STOCK_INDEX_DIR, TUSHARE_TOKEN

try:
    from config import GLOBAL_INDEX_DOWNLOAD_SETTINGS
except Exception:
    GLOBAL_INDEX_DOWNLOAD_SETTINGS = {
        "max_retries": 5,
        "retry_delay": 2,
        "request_interval": 0.4,
        "save_candidates": True,
    }

STOCK_INDEX_DIR = Path(STOCK_INDEX_DIR)
STOCK_INDEX_DIR.mkdir(parents=True, exist_ok=True)

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

GLOBAL_INDEX_CONFIGS = {
    "hsi": {
        "ts_code": "HSI", "path": "HSI.index_global.csv", "start_date": "20220101",
        "akshare_sources": [
            {"source": "akshare_hk_sina", "method": "hk_sina", "symbol": "HSI"},
            {"source": "akshare_global_em", "method": "global_em", "symbol": "恒生指数"},
        ],
    },
    "hktech": {
        "ts_code": "HKTECH", "path": "HKTECH.index_global.csv", "start_date": "20220101",
        "akshare_sources": [
            {"source": "akshare_hk_sina", "method": "hk_sina", "symbol": "HSTECH"},
        ],
    },
    "xin9": {
        "ts_code": "XIN9", "path": "XIN9.index_global.csv", "start_date": "20220101",
        "akshare_sources": [],
    },
    "dji": {
        "ts_code": "DJI", "path": "DJI.index_global.csv", "start_date": "20220501",
        "akshare_sources": [
            {"source": "akshare_us_sina", "method": "us_sina", "symbol": ".DJI"},
            {"source": "akshare_global_em", "method": "global_em", "symbol": "道琼斯"},
        ],
    },
    "ftse": {
        "ts_code": "FTSE", "path": "FTSE.index_global.csv", "start_date": "20220101",
        "akshare_sources": [
            {"source": "akshare_global_em", "method": "global_em", "symbol": "英国富时100"},
        ],
    },
    "spx": {
        "ts_code": "SPX", "path": "SPX.index_global.csv", "start_date": "20220101",
        "akshare_sources": [
            {"source": "akshare_global_em", "method": "global_em", "symbol": "标普500"},
        ],
    },
    "ixic": {
        "ts_code": "IXIC", "path": "IXIC.index_global.csv", "start_date": "20220101",
        "akshare_sources": [
            {"source": "akshare_us_sina", "method": "us_sina", "symbol": ".IXIC"},
        ],
    },
    "n225": {
        "ts_code": "N225", "path": "N225.index_global.csv", "start_date": "20220101",
        "akshare_sources": [
            {"source": "akshare_global_em", "method": "global_em", "symbol": "日经225"},
        ],
    },
}

INDEX_GLOBAL_COLUMNS = [
    "ts_code", "trade_date", "open", "close", "high", "low",
    "pre_close", "change", "pct_chg", "swing", "vol",
]


def setting(name: str, default: Any) -> Any:
    return GLOBAL_INDEX_DOWNLOAD_SETTINGS.get(name, default)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf_8_sig")
    tmp.replace(path)


def normalize_trade_date(s: pd.Series) -> pd.Series:
    digits = s.astype(str).str.strip().str.replace(r"\D", "", regex=True).str[:8]
    return pd.to_datetime(digits, format="%Y%m%d", errors="coerce").dt.strftime("%Y%m%d")


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def normalize_global_df(raw: pd.DataFrame, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=INDEX_GLOBAL_COLUMNS)
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {
        "日期": "trade_date", "date": "trade_date", "Date": "trade_date", "trade_date": "trade_date",
        "开盘": "open", "今开": "open", "Open": "open", "open": "open",
        "收盘": "close", "最新价": "close", "Close": "close", "close": "close",
        "最高": "high", "High": "high", "high": "high",
        "最低": "low", "Low": "low", "low": "low",
        "成交量": "vol", "volume": "vol", "Volume": "vol", "vol": "vol",
        "pre_close": "pre_close", "昨收": "pre_close", "昨收价": "pre_close",
        "change": "change", "涨跌额": "change",
        "pct_chg": "pct_chg", "涨跌幅": "pct_chg",
        "swing": "swing", "振幅": "swing",
    }
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})
    if "trade_date" not in df.columns:
        raise ValueError(f"{ts_code}: missing date column; columns={list(raw.columns)}")
    df["trade_date"] = normalize_trade_date(df["trade_date"])
    df = df.dropna(subset=["trade_date"])
    df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
    df = df.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    for col in INDEX_GLOBAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    for col in ["open", "close", "high", "low", "pre_close", "change", "pct_chg", "swing", "vol"]:
        df[col] = numeric(df[col])
    df = df.dropna(subset=["close"])
    if df.empty:
        return pd.DataFrame(columns=INDEX_GLOBAL_COLUMNS)
    df["pre_close"] = df["pre_close"].fillna(df["close"].shift(1))
    df["change"] = df["change"].fillna(df["close"] - df["pre_close"])
    df["pct_chg"] = df["pct_chg"].fillna(df["change"] / df["pre_close"] * 100.0)
    df["swing"] = df["swing"].fillna((df["high"] - df["low"]) / df["pre_close"] * 100.0)
    df["ts_code"] = ts_code
    return df[INDEX_GLOBAL_COLUMNS].sort_values("trade_date").reset_index(drop=True)


def download_with_retry(func, desc: str, max_retries: int, retry_delay: int) -> pd.DataFrame:
    for attempt in range(max_retries):
        try:
            df = func()
            if df is not None and not df.empty:
                return df
            print(f"[WARN] empty: {desc}, attempt {attempt+1}/{max_retries}")
        except Exception as e:
            print(f"[ERROR] {desc}, attempt {attempt+1}/{max_retries}: {e}")
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    return pd.DataFrame()


def download_tushare_global(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    raw = download_with_retry(
        lambda: pro.index_global(ts_code=ts_code, start_date=start_date, end_date=end_date),
        desc=f"tushare index_global {ts_code}",
        max_retries=int(setting("max_retries", 5)),
        retry_delay=int(setting("retry_delay", 2)),
    )
    return normalize_global_df(raw, ts_code, start_date, end_date)


def download_akshare_raw(method: str, symbol: str) -> pd.DataFrame:
    if ak is None:
        raise ImportError("akshare is not installed")
    if method == "hk_sina":
        return ak.stock_hk_index_daily_sina(symbol=symbol)
    if method == "us_sina":
        return ak.stock_us_daily(symbol=symbol, adjust="")
    if method == "global_em":
        return ak.index_global_hist_em(symbol=symbol)
    raise ValueError(f"Unknown AkShare method: {method}")


def download_akshare_global(ts_code: str, method: str, symbol: str,
                            start_date: str, end_date: str) -> pd.DataFrame:
    raw = download_with_retry(
        lambda: download_akshare_raw(method, symbol),
        desc=f"akshare {method} {ts_code}/{symbol}",
        max_retries=int(setting("max_retries", 5)),
        retry_delay=int(setting("retry_delay", 2)),
    )
    return normalize_global_df(raw, ts_code, start_date, end_date)


def summarize_source(ts_code: str, source: str, df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"ts_code": ts_code, "source": source, "selected": False, "rows": 0,
                "first_date": "", "latest_date": "", "selection_rule": ""}
    return {"ts_code": ts_code, "source": source, "selected": False, "rows": int(len(df)),
            "first_date": str(df["trade_date"].min()), "latest_date": str(df["trade_date"].max()),
            "selection_rule": ""}


def choose_source(ts_code: str, candidates: list[tuple[str, pd.DataFrame]]):
    summaries = [summarize_source(ts_code, src, df) for src, df in candidates]
    available = [(src, df, sm) for (src, df), sm in zip(candidates, summaries) if df is not None and not df.empty]
    if not available:
        return "", pd.DataFrame(columns=INDEX_GLOBAL_COLUMNS), summaries
    if len(available) == 1:
        selected_src, selected_df, _ = available[0]
        rule = "only_available_source"
    else:
        def rank(item):
            src, _df, sm = item
            tushare_priority = 1 if src == "tushare" else 0
            return (str(sm["latest_date"]), tushare_priority, int(sm["rows"]))
        selected_src, selected_df, _ = max(available, key=rank)
        rule = "latest_date_then_tushare_tie"
    for s in summaries:
        s["selected"] = s["source"] == selected_src
        if s["selected"]:
            s["selection_rule"] = rule
    return selected_src, selected_df, summaries


def download_global_indices() -> pd.DataFrame:
    data_dir = Path(STOCK_INDEX_DIR)
    candidate_dir = data_dir / "global_source_candidates"
    if bool(setting("save_candidates", True)):
        candidate_dir.mkdir(parents=True, exist_ok=True)
    end_date = str(End_date_global)
    request_interval = float(setting("request_interval", 0.4))
    manifest_rows = []
    print("=" * 70)
    print("Download global indices: Tushare + AkShare")
    print(f"End_date_global: {end_date}")
    print("=" * 70)
    for i, (name, cfg) in enumerate(GLOBAL_INDEX_CONFIGS.items(), 1):
        ts_code = str(cfg["ts_code"]).upper()
        start_date = str(cfg.get("start_date", history_start_date))
        out_path = data_dir / str(cfg["path"])
        print(f"\n[{i}/{len(GLOBAL_INDEX_CONFIGS)}] {ts_code}: {start_date} ~ {end_date}")
        candidates = []
        df_ts = download_tushare_global(ts_code, start_date, end_date)
        candidates.append(("tushare", df_ts))
        if bool(setting("save_candidates", True)) and not df_ts.empty:
            save_csv(df_ts, candidate_dir / f"{ts_code}.tushare.csv")
        for ak_cfg in cfg.get("akshare_sources", []):
            source, method, symbol = str(ak_cfg["source"]), str(ak_cfg["method"]), str(ak_cfg["symbol"])
            df_ak = download_akshare_global(ts_code, method, symbol, start_date, end_date)
            candidates.append((source, df_ak))
            if bool(setting("save_candidates", True)) and not df_ak.empty:
                safe_symbol = symbol.replace(".", "").replace("/", "_").replace("\\", "_").replace(" ", "_")
                save_csv(df_ak, candidate_dir / f"{ts_code}.{source}.{safe_symbol}.csv")
        selected_src, selected_df, summaries = choose_source(ts_code, candidates)
        manifest_rows.extend(summaries)
        if selected_df.empty:
            print(f"  [FAIL] no usable data for {ts_code}")
        else:
            save_csv(selected_df, out_path)
            print(f"  [SAVE] selected={selected_src}, latest={selected_df['trade_date'].max()}, rows={len(selected_df)} -> {out_path}")
        if i < len(GLOBAL_INDEX_CONFIGS):
            time.sleep(request_interval)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = data_dir / "global_index_source_manifest.csv"
    save_csv(manifest, manifest_path)
    selected = manifest[manifest["selected"]].copy()
    print("\nSelected global sources:")
    if not selected.empty:
        print(selected[["ts_code", "source", "rows", "latest_date", "selection_rule"]].to_string(index=False))
    return manifest


if __name__ == "__main__":
    download_global_indices()
