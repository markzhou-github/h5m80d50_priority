#!/usr/bin/env python
# coding: utf-8
"""
build_market_panel.py

Task 3: merge A-share index files, market moneyflow, and selected global
index files into INDEX_DIR / market_panel.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

PROJ_ROOT = Path.cwd().parent
CONFIG_DIR = PROJ_ROOT / "config"
if str(CONFIG_DIR) not in sys.path:
    sys.path.append(str(CONFIG_DIR))

from config_date import history_start_date, end_date
from config import STOCK_INDEX_DIR, INDEX_DIR, MARKET_PANEL_CSV

STOCK_INDEX_DIR = Path(STOCK_INDEX_DIR)
INDEX_DIR = Path(INDEX_DIR)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_INDEX_CONFIGS = {
    "hsi": {"path": "HSI.index_global.csv", "keep_cols": ["pct_chg", "swing", "vol"], "shift_days": 0, "prefix": "hsi_", "start_date": "20220101"},
    "hktech": {"path": "HKTECH.index_global.csv", "keep_cols": ["pct_chg", "swing"], "shift_days": 0, "prefix": "hktech_", "start_date": "20220101"},
    "xin9": {"path": "XIN9.index_global.csv", "keep_cols": ["pct_chg", "swing"], "shift_days": 0, "prefix": "xin9_", "start_date": "20220101"},
    "dji": {"path": "DJI.index_global.csv", "keep_cols": ["pct_chg", "swing", "vol"], "shift_days": 1, "prefix": "dji_", "start_date": "20220501"},
    "ftse": {"path": "FTSE.index_global.csv", "keep_cols": ["pct_chg", "swing", "vol"], "shift_days": 1, "prefix": "ftse_", "start_date": "20220101"},
    "spx": {"path": "SPX.index_global.csv", "keep_cols": ["pct_chg", "swing", "vol"], "shift_days": 1, "prefix": "spx_", "start_date": "20220101"},
    "ixic": {"path": "IXIC.index_global.csv", "keep_cols": ["pct_chg", "swing", "vol"], "shift_days": 1, "prefix": "ixic_", "start_date": "20220101"},
    "n225": {"path": "N225.index_global.csv", "keep_cols": ["pct_chg", "swing", "vol"], "shift_days": 0, "prefix": "n225_", "start_date": "20220101"},
}


def build_sse_calendar(start_date: str = history_start_date, end_date: str = end_date) -> pd.DataFrame:
    sse = mcal.get_calendar("SSE")
    sse_days = sse.valid_days(start_date, end_date)
    return pd.DataFrame({"trade_date": sse_days.strftime("%Y%m%d")}).reset_index(drop=True)


def load_idxfactor(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ts_code": str, "trade_date": str})
    cols_keep = [
        "trade_date", "open", "close", "pct_change", "vol", "amount", "atr_bfq", "lowdays", "topdays",
        "bbi_bfq", "ma_bfq_5", "ma_bfq_20", "ma_bfq_60", "rsi_bfq_6", "rsi_bfq_12",
        "macd_dif_bfq", "macd_dea_bfq", "mtm_bfq", "updays", "downdays",
    ]
    cols_keep = [c for c in cols_keep if c in df.columns]
    out = df[cols_keep].copy()
    out = out.rename(columns={c: f"{prefix}_{c}" for c in cols_keep if c != "trade_date"})
    return out.sort_values("trade_date").reset_index(drop=True)


def load_index_dailybasic(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"trade_date": str})
    cols_keep = ["trade_date", "float_mv", "turnover_rate", "turnover_rate_f", "pe_ttm", "pb"]
    cols_keep = [c for c in cols_keep if c in df.columns]
    out = df[cols_keep].copy()
    out = out.rename(columns={c: f"{prefix}_{c}" for c in cols_keep if c != "trade_date"})
    return out.sort_values("trade_date").reset_index(drop=True)


def load_market_moneyflow(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"trade_date": str})
    cols_keep = [
        "trade_date", "pct_change_sh", "pct_change_sz", "net_amount", "net_amount_rate",
        "buy_elg_amount_rate", "buy_lg_amount_rate", "buy_md_amount_rate", "buy_sm_amount_rate",
    ]
    cols_keep = [c for c in cols_keep if c in df.columns]
    out = df[cols_keep].copy()
    rename_map = {
        "pct_change_sh": "mkt_sh_ret", "pct_change_sz": "mkt_sz_ret", "net_amount": "mkt_net_amount",
        "net_amount_rate": "mkt_net_amount_rate", "buy_elg_amount_rate": "mkt_buy_elg_amount_rate",
        "buy_lg_amount_rate": "mkt_buy_lg_amount_rate", "buy_md_amount_rate": "mkt_buy_md_amount_rate",
        "buy_sm_amount_rate": "mkt_buy_sm_amount_rate",
    }
    out = out.rename(columns=rename_map)
    return out.sort_values("trade_date").reset_index(drop=True)


def load_hsgt_moneyflow(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"trade_date": str})
    cols_keep = [
        "trade_date", "ggt_ss", "ggt_sz", "hgt", "sgt", "north_money", "south_money",
    ]
    cols_keep = [c for c in cols_keep if c in df.columns]
    out = df[cols_keep].copy()
    out = out.rename(columns={c: f"hsgt_{c}" for c in cols_keep if c != "trade_date"})
    return out.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)


def align_hsgt_to_sse_calendar(df_base: pd.DataFrame, df_hsgt: pd.DataFrame,
                               start_date: str | None = None) -> pd.DataFrame:
    merge_cols = ["ggt_ss", "ggt_sz", "hgt", "sgt", "north_money", "south_money"]
    use_cols = ["trade_date"] + [c for c in merge_cols if c in df_hsgt.columns]
    hsgt = df_hsgt[use_cols].copy().dropna(subset=["trade_date"])
    hsgt["trade_date"] = pd.to_datetime(hsgt["trade_date"].astype(str), format="%Y%m%d")
    hsgt = hsgt.sort_values("trade_date").drop_duplicates("trade_date").set_index("trade_date")
    hsgt = hsgt.add_prefix("hsgt_")

    base = df_base.copy()
    base["trade_date"] = pd.to_datetime(base["trade_date"].astype(str), format="%Y%m%d")
    base = base.sort_values("trade_date").set_index("trade_date")

    aligned = base.join(hsgt, how="left").ffill()
    first_hsgt_date = hsgt.index.min()
    final_start = first_hsgt_date
    if start_date is not None:
        final_start = max(first_hsgt_date, pd.to_datetime(start_date, format="%Y%m%d"))

    # Keep pre-HSGT SSE dates in the panel, but leave HSGT columns as NA before first data date.
    hsgt_cols = [c for c in aligned.columns if c.startswith("hsgt_")]
    aligned.loc[aligned.index < final_start, hsgt_cols] = pd.NA

    aligned = aligned.reset_index()
    aligned["trade_date"] = aligned["trade_date"].dt.strftime("%Y%m%d")
    return aligned


def shift_and_merge(df_base: pd.DataFrame, df_global: pd.DataFrame, merge_cols: list[str],
                    shift_days: int, prefix: str, start_date: str | None = None) -> pd.DataFrame:
    use_cols = ["trade_date"] + [c for c in merge_cols if c in df_global.columns]
    g = df_global[use_cols].copy().dropna(subset=["trade_date"])
    g["trade_date"] = pd.to_datetime(g["trade_date"].astype(str), format="%Y%m%d")
    g = g.sort_values("trade_date").drop_duplicates("trade_date")
    g = g.set_index("trade_date")
    g_shifted = g.shift(shift_days).add_prefix(prefix)
    base = df_base.copy()
    base["trade_date"] = pd.to_datetime(base["trade_date"].astype(str), format="%Y%m%d")
    base = base.sort_values("trade_date").set_index("trade_date")
    merged = base.join(g_shifted, how="left").ffill()
    original_start = g_shifted.index.min()
    final_start = max(original_start, pd.to_datetime(start_date, format="%Y%m%d")) if start_date is not None else original_start
    merged = merged[merged.index >= final_start]
    merged = merged.reset_index()
    merged["trade_date"] = merged["trade_date"].dt.strftime("%Y%m%d")
    return merged


def build_market_panel(market_dir: Path = INDEX_DIR) -> pd.DataFrame:
    market_dir = Path(market_dir)
    df_sse = build_sse_calendar()
    df_idx_csi300 = load_idxfactor(market_dir / "000300.SH.idxfactor.csv", "csi300")
    df_idx_csi905 = load_idxfactor(market_dir / "000905.SH.idxfactor.csv", "csi905")
    df_idx_csi852 = load_idxfactor(market_dir / "000852.SH.idxfactor.csv", "csi852")
    df_idx_csi985 = load_idxfactor(market_dir / "000985.CSI.idxfactor.csv", "csi985")
    df_db_csi300 = load_index_dailybasic(market_dir / "000300.SH.index_dailybasic.csv", "csi300")
    df_db_csi905 = load_index_dailybasic(market_dir / "000905.SH.index_dailybasic.csv", "csi905")
    
    market_df = df_sse.copy()

    df_mkt_mf = load_market_moneyflow(market_dir / "mktdc.csv")
 #   df_mkt_mf["mkt_mf_available"] = 1

    market_df = market_df.merge(df_mkt_mf, on="trade_date", how="left")
  #  market_df["mkt_mf_available"] = market_df["mkt_mf_available"].fillna(0).astype("int8")
    
   # df_mkt_mf = load_market_moneyflow(market_dir / "mktdc.csv")
   # market_df = df_mkt_mf.copy()
    hsgt_path = market_dir / "moneyflow_hsgt.csv"
    if hsgt_path.exists():
        df_hsgt_file = pd.read_csv(hsgt_path, dtype={"trade_date": str})
        df_hsgt = align_hsgt_to_sse_calendar(df_sse, df_hsgt_file, start_date=history_start_date)
        market_df = market_df.merge(df_hsgt, on="trade_date", how="left")
    else:
        print(f"[WARN] missing HSGT moneyflow file, skip: {hsgt_path}")
    domestic_to_merge = [df_idx_csi300, df_idx_csi905, df_idx_csi852, df_idx_csi985, df_db_csi300, df_db_csi905]
    for df_other in domestic_to_merge:
        market_df = market_df.merge(df_other, on="trade_date", how="left")
    for name in ["hsi", "hktech", "xin9", "dji", "ftse", "spx", "ixic", "n225"]:
        cfg = GLOBAL_INDEX_CONFIGS[name]
        path = market_dir / cfg["path"]
        if not path.exists():
            print(f"[WARN] missing global file, skip: {path}")
            continue
        df_gfile = pd.read_csv(path, dtype={"trade_date": str})
        df_g = shift_and_merge(df_sse, df_gfile, cfg["keep_cols"], int(cfg["shift_days"]), str(cfg["prefix"]), str(cfg["start_date"]))
        market_df = market_df.merge(df_g, on="trade_date", how="left")
    market_df = market_df.sort_values("trade_date", ascending=False).reset_index(drop=True)
    print(f"[INFO] market panel built: shape={market_df.shape}")
    print(f"[INFO] date range: {market_df['trade_date'].min()} ~ {market_df['trade_date'].max()}")
    return market_df


def main() -> None:
    market_df = build_market_panel(INDEX_DIR)
    out_path = Path(MARKET_PANEL_CSV)
    market_df.to_csv(out_path, index=False, encoding="utf_8_sig")
    print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
