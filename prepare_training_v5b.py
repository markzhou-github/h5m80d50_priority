#!/usr/bin/env python
# coding: utf-8
"""
Build v5b alpha training data.

Design:
  - preflight removes stocks without usable margin_detail files
  - per-stock time-series features and labels are computed before panel merge
  - cross-sectional stock ranks are computed only after clean per-stock rows merge
  - SW L2 code is kept as a categorical feature

Run:
  C:\\Users\\mark_\\anaconda3\\envs\\m1deepl\\python.exe prepare_training_v5b.py --max-stocks 5
  C:\\Users\\mark_\\anaconda3\\envs\\m1deepl\\python.exe prepare_training_v5b.py
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
import polars as pl

from config import STOCK_DATA_DIR, STOCK_INDEX_DIR


PROJECT_ROOT = Path(__file__).resolve().parent
MERGED_DIR = Path(STOCK_DATA_DIR) / "merged"
LIMIT_DIR = Path(STOCK_DATA_DIR) / "limit"
MINUTE_FEATURE_DIR = PROJECT_ROOT / "processed" / "minute_features_v5b" / "by_stock"
SW_L2_MAPPING_PATH = PROJECT_ROOT / "stock_sw_l2_mapping_since_2023.csv"
SW_L2_DAILY_PATH = Path(STOCK_INDEX_DIR) / "sw_l2_daily.csv"
CSI1500_INDEX_PATH = Path(STOCK_INDEX_DIR) / "csi1500_custom_index.csv"
MARKET_PANEL_PATH = Path(STOCK_INDEX_DIR) / "market_panel.csv"

OUT_DIR = PROJECT_ROOT / "processed" / "train_v5b"
SINGLE_DIR = OUT_DIR / "single_stock_features"
CHUNK_DIR = OUT_DIR / "chunk_features"
REPORT_DIR = OUT_DIR / "report"
FINAL_PATH = OUT_DIR / "train_v5b.parquet"
PANEL_NO_CS_PATH = OUT_DIR / "panel_no_cs.parquet"
FEATURE_DICT_PATH = OUT_DIR / "v5b_feature_dictionary.csv"

EPS = 1e-9

warnings.filterwarnings("ignore", category=PerformanceWarning)

G_CSI: pd.DataFrame | None = None
G_MARKET: pd.DataFrame | None = None
G_SW_FEAT: pd.DataFrame | None = None
G_SW_MAPPING: pd.DataFrame | None = None
G_MINUTE_FEATURE_SOURCE: str | None = None
G_MINUTE_FEATURE_BY_STOCK: dict[str, pd.DataFrame] | None = None
G_SOURCE_START_DATE: str | None = None
G_SOURCE_END_DATE: str | None = None
G_OUTPUT_START_DATE: str | None = None
G_OUTPUT_END_DATE: str | None = None

LAGS_FAST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
LAGS_MED = [1, 2, 3, 5, 10, 20]
LAGS_SLOW = [1, 2, 3, 5]
LAGS_SHOCK = [1, 2, 3, 5, 10]
LAGS_3D = [1, 2, 3]
LAGS_TREND = [1, 2, 3, 5]


def normalize_date_arg(value: str | int | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.replace("-", "")


def filter_date_pandas(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    out = df.copy()
    out[date_col] = out[date_col].astype(str).str.replace("-", "", regex=False)
    if start_date:
        out = out[out[date_col] >= start_date]
    if end_date:
        out = out[out[date_col] <= end_date]
    return out.copy()


def filter_date_polars(
    df: pl.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    date_col: str = "trade_date",
) -> pl.DataFrame:
    if df.is_empty() or date_col not in df.columns:
        return df
    out = df.with_columns(pl.col(date_col).cast(pl.Utf8).str.replace_all("-", "").alias(date_col))
    if start_date:
        out = out.filter(pl.col(date_col) >= start_date)
    if end_date:
        out = out.filter(pl.col(date_col) <= end_date)
    return out


BASE_FEATURE_LAGS: dict[str, list[int]] = {
    # Price / VWAP / volatility
    "ret_1": LAGS_FAST,
    "ret_2": LAGS_TREND,
    "ret_3": LAGS_TREND,
    "ret_5": LAGS_TREND,
    "ret_10": LAGS_TREND,
    "ret_20": LAGS_TREND,
    "oc_ret": LAGS_FAST,
    "close_to_high": LAGS_FAST,
    "close_to_low": LAGS_FAST,
    "vwap_z": LAGS_FAST,
    "vwap_close_ratio": LAGS_FAST,
    "vwap_high_ratio": LAGS_FAST,
    "vwap_low_ratio": LAGS_FAST,
    "vwap_log_ret_2": LAGS_TREND,
    "vwap_log_ret_3": LAGS_TREND,
    "vwap_log_ret_5": LAGS_FAST,
    "vwap_log_ret_10": LAGS_TREND,
    "vwap_log_ret_20": LAGS_MED,
    "ret_3_minus_ret_10": LAGS_TREND,
    "ret_5_minus_ret_20": LAGS_TREND,
    "vwap_log_ret_3_minus_10": LAGS_TREND,
    "ret_1_minus_ret_5_avg": LAGS_TREND,
    "vwap_close_z": LAGS_FAST,
    "vwap_imbalance": LAGS_FAST,
    "true_range_pct_z": LAGS_FAST,
    "true_range_mean_5_z": LAGS_FAST,
    "true_range_mean_10_z": LAGS_MED,
    "true_range_mean_20_z": LAGS_MED,
    "ret_std_5_z": LAGS_FAST,
    "ret_std_10_z": LAGS_MED,
    "ret_std_20_z": LAGS_MED,
    # Volume / turnover
    "vol_z": LAGS_FAST,
    "amount_z": LAGS_FAST,
    "vol_rv_5_z": LAGS_FAST,
    "vol_rv_20_z": LAGS_MED,
    "amt_rv_5_z": LAGS_FAST,
    "amt_rv_20_z": LAGS_MED,
    "turnover_f_log": LAGS_FAST,
    "turnover_z_20": LAGS_SHOCK,
    "turnover_prank_60": LAGS_SLOW,
    "turnover_surprise_10": LAGS_SHOCK,
    "turnover_trend_5_20": LAGS_MED,
    "turnover_accel_5_20": LAGS_SLOW,
    "volume_ratio": LAGS_FAST,
    "vol_comp_z": LAGS_MED,
    "turnover_in_compression": LAGS_SHOCK,
    "pressure_build": LAGS_MED,
    "exhaustion_risk": LAGS_MED,
    "turnover_range_pressure": LAGS_MED,
    # Moneyflow / auction / size / chip / margin
    "mf_inst_rate": LAGS_SHOCK,
    "mf_elg_rate": LAGS_SHOCK,
    "mf_lg_rate": LAGS_SHOCK,
    "mf_md_rate": LAGS_SHOCK,
    "mf_sm_rate": LAGS_SHOCK,
    "mf_inst_rate_z": LAGS_SHOCK,
    "mf_elg_rate_z": LAGS_SHOCK,
    "mf_lg_rate_z": LAGS_SHOCK,
    "mf_md_rate_z": LAGS_SHOCK,
    "mf_sm_rate_z": LAGS_SHOCK,
    "mf_price_align_z_20": LAGS_SHOCK,
    "mf_inst_rate_rel_mkt_z20": LAGS_MED,
    "auc_o_ret": LAGS_FAST,
    "auc_o_range": LAGS_FAST,
    "auc_o_vwap_ratio": LAGS_FAST,
    "auc_o_amt_ratio": LAGS_FAST,
    "auc_o_vol_ratio": LAGS_FAST,
    "auc_o_close_position": LAGS_FAST,
    "auc_o_efficiency": LAGS_FAST,
    "auc_o_vs_open": LAGS_FAST,
    "auc_c_ret": LAGS_FAST,
    "auc_c_range": LAGS_FAST,
    "auc_c_vwap_ratio": LAGS_FAST,
    "auc_c_amt_ratio": LAGS_FAST,
    "auc_c_vol_ratio": LAGS_FAST,
    "auc_c_close_position": LAGS_FAST,
    "auc_c_efficiency": LAGS_FAST,
    "auc_c_vs_close": LAGS_FAST,
    "auction_reversal": LAGS_FAST,
    "total_mv_z": LAGS_MED,
    "circ_mv_z": LAGS_MED,
    "winner_rate": LAGS_MED,
    "cost_5pct_z": LAGS_MED,
    "cost_15pct_z": LAGS_MED,
    "cost_50pct_z": LAGS_MED,
    "cost_85pct_z": LAGS_MED,
    "cost_95pct_z": LAGS_MED,
    "weight_avg_z": LAGS_MED,
    "chip_density": LAGS_MED,
    "chip_pressure": LAGS_MED,
    "crowdedness": LAGS_MED,
    "rzye_z": LAGS_MED,
    "rqye_z": LAGS_MED,
    "rzmre_z": LAGS_MED,
    "rqyl_z": LAGS_MED,
    "rzche_z": LAGS_MED,
    "rqchl_z": LAGS_MED,
    "rqmcl_z": LAGS_MED,
    "rzrqye_z": LAGS_MED,
    "margin_pressure": LAGS_MED,
    "short_pressure": LAGS_MED,
    "net_margin_flow": LAGS_MED,
    # CSI1500 stock relative
    "ret_1_rel_csi1500_ew": LAGS_FAST,
    "ret_1_rel_csi1500_mcap": LAGS_FAST,
    "ret_2_rel_csi1500_ew": LAGS_TREND,
    "ret_2_rel_csi1500_mcap": LAGS_TREND,
    "ret_3_rel_csi1500_ew": LAGS_TREND,
    "ret_3_rel_csi1500_mcap": LAGS_TREND,
    "ret_10_rel_csi1500_ew": LAGS_TREND,
    "ret_10_rel_csi1500_mcap": LAGS_TREND,
    "oc_ret_rel_csi1500_ew": LAGS_FAST,
    "oc_ret_rel_csi1500_mcap": LAGS_FAST,
    "ret_5_rel_csi1500_ew": LAGS_MED,
    "ret_5_rel_csi1500_mcap": LAGS_MED,
    "ret_20_rel_csi1500_ew": LAGS_MED,
    "ret_20_rel_csi1500_mcap": LAGS_MED,
    "beta_to_csi1500_ew_20d": LAGS_MED,
    "beta_to_csi1500_mcap_20d": LAGS_MED,
    "vol_over_csi1500_ew_vol": LAGS_MED,
    "vol_over_csi1500_mcap_vol": LAGS_MED,
    "vol_rel_1500": LAGS_FAST,
    "amount_rel_1500": LAGS_FAST,
    "turnover_rel_1500": LAGS_FAST,
    "mf_inst_rate_rel_1500": LAGS_FAST,
    "ret_1_over_mkt_gt5_rate": LAGS_FAST,
    "ret_1_in_weak_mkt": LAGS_FAST,
    "ret_vol_confirm": LAGS_FAST,
    "vol_expand_ret": LAGS_FAST,
    # Market/HSGT/global
    "csi1500_ew_ret": LAGS_SLOW,
    "csi1500_ew_oc_ret": LAGS_SLOW,
    "csi1500_ew_mom_2": LAGS_SLOW,
    "csi1500_ew_mom_3": LAGS_SLOW,
    "csi1500_ew_mom_5": LAGS_SLOW,
    "csi1500_ew_mom_10": LAGS_SLOW,
    "csi1500_ew_mom_20": LAGS_SLOW,
    "csi1500_ew_ret_std_20": LAGS_SLOW,
    "csi1500_ew_up_ratio": LAGS_SLOW,
    "csi1500_ew_down_ratio": LAGS_SLOW,
    "csi1500_ew_gt_2pct_ratio": LAGS_SLOW,
    "csi1500_ew_lt_minus_2pct_ratio": LAGS_SLOW,
    "mkt_pos_rate": LAGS_SLOW,
    "mkt_gt3_rate": LAGS_SLOW,
    "mkt_gt5_rate": LAGS_SLOW,
    "mkt_limit_up_count": LAGS_SLOW,
    "mkt_limit_down_count": LAGS_SLOW,
    "csi1500_coverage_ratio": LAGS_SLOW,
    "csi1500_mcap_ret": LAGS_SLOW,
    "csi1500_mcap_oc_ret": LAGS_SLOW,
    "csi1500_mcap_mom_2": LAGS_SLOW,
    "csi1500_mcap_mom_3": LAGS_SLOW,
    "csi1500_mcap_mom_5": LAGS_SLOW,
    "csi1500_mcap_mom_10": LAGS_SLOW,
    "csi1500_mcap_mom_20": LAGS_SLOW,
    "csi1500_mcap_ret_std_20": LAGS_SLOW,
    "csi1500_mcap_max_weight": LAGS_SLOW,
    "csi1500_mcap_top10_weight": LAGS_SLOW,
    "csi1500_mcap_coverage_ratio": LAGS_SLOW,
    "mkt_ret_sh": LAGS_SLOW,
    "mkt_ret_sz": LAGS_SLOW,
    "mkt_ret_spread": LAGS_SLOW,
    "mkt_mf_inst_rate": LAGS_FAST,
    "mkt_mf_inst_ma_3": LAGS_SLOW,
    "mkt_mf_inst_ma_5": LAGS_SLOW,
    "mkt_mf_inst_ma_10": LAGS_SLOW,
#    "mkt_flow_regime": LAGS_SLOW,
    "mkt_flow_concentration": LAGS_SLOW,
    "mkt_mf_retail_rate": LAGS_SLOW,
    "mkt_mf_inst_z_20": LAGS_SLOW,
    "hsgt_hgt": LAGS_SLOW,
    "hsgt_sgt": LAGS_SLOW,
    "hsgt_north_money": LAGS_SLOW,
    "hsgt_south_money": LAGS_SLOW,
    "hsgt_hgt_minus_sgt": LAGS_SLOW,
    "hsgt_north_money_z_20": LAGS_SLOW,
    "hsgt_north_money_ma_3": LAGS_SLOW,
    "hsgt_north_money_ma_5": LAGS_SLOW,
    "hsgt_north_money_accel_3": LAGS_SLOW,
    "hsgt_south_money_z_20": LAGS_SLOW,
    "hsgt_south_money_ma_3": LAGS_SLOW,
    "hsgt_south_money_ma_5": LAGS_SLOW,
    "hsi_ret": LAGS_SLOW,
    "hsi_swing": LAGS_SLOW,
#    "hsi_vol_z": LAGS_SLOW,
    "hktech_ret": LAGS_SLOW,
    "hktech_swing": LAGS_SLOW,
    "xin9_ret": LAGS_SLOW,
    "xin9_swing": LAGS_SLOW,
    "spx_ret": LAGS_SLOW,
    "spx_swing": LAGS_SLOW,
    "spx_vol_z": LAGS_SLOW,
    "ixic_ret": LAGS_SLOW,
    "ixic_swing": LAGS_SLOW,
    "ixic_vol_z": LAGS_SLOW,
    "n225_ret": LAGS_SLOW,
    "n225_swing": LAGS_SLOW,
#    "n225_vol_z": LAGS_SLOW,
    # Limit
    "fd_ratio": LAGS_SHOCK,
    "fd_ratio_log": LAGS_SHOCK,
#    "fd_ratio_log_delta_1": LAGS_SHOCK,
#    "fd_ratio_log_delta_2": LAGS_SHOCK,
    "first_time_min": LAGS_3D,
    "failure_pressure": LAGS_3D,
    "is_20cm": [],

    # Limit event day counts: already rolling windows, no extra lags
    "limit_up_day_cnt_1": [],
    "limit_up_day_cnt_2": [],
    "limit_up_day_cnt_3": [],
    "limit_up_day_cnt_5": [],
    "limit_up_day_cnt_10": [],

    "broken_board_day_cnt_1": [],
    "broken_board_day_cnt_2": [],
    "broken_board_day_cnt_3": [],
    "broken_board_day_cnt_5": [],
    "broken_board_day_cnt_10": [],

    "limit_down_day_cnt_1": [],
    "limit_down_day_cnt_2": [],
    "limit_down_day_cnt_3": [],
    "limit_down_day_cnt_5": [],
    "limit_down_day_cnt_10": [],

    "sealed_board_day_cnt_1": [],
    "sealed_board_day_cnt_2": [],
    "sealed_board_day_cnt_3": [],
    "sealed_board_day_cnt_5": [],
    "sealed_board_day_cnt_10": [],

    "board_success_ratio_1": [],
    "board_success_ratio_2": [],
    "board_success_ratio_3": [],
    "board_success_ratio_5": [],
    "board_success_ratio_10": [],

    "days_since_last_limit_event_60": [],
    "days_since_last_limit_up_60": [],
    "days_since_last_broken_board_60": [],
    "days_since_last_limit_down_60": [],
    
    # SW L2
    "sw_l2_ret_1": LAGS_FAST,
    "sw_l2_oc_ret": LAGS_FAST,
    "sw_l2_mom_2": LAGS_SLOW,
    "sw_l2_mom_3": LAGS_SLOW,
    "sw_l2_mom_5": LAGS_SLOW,
    "sw_l2_mom_10": LAGS_SLOW,
    "sw_l2_mom_20": LAGS_SLOW,
    "sw_l2_ret_std_5": LAGS_SLOW,
    "sw_l2_ret_std_20": LAGS_SLOW,
    "sw_l2_range_pct": LAGS_SLOW,
    "sw_l2_range_z_20": LAGS_SLOW,
    "sw_l2_amount_z_20": LAGS_SHOCK,
    "sw_l2_vol_z_20": LAGS_SHOCK,
    "sw_l2_turnover_proxy": LAGS_SLOW,
    "sw_l2_turnover_z_20": LAGS_SHOCK,
    "ret_1_rel_sw_l2": LAGS_FAST,
    "ret_2_rel_sw_l2": LAGS_TREND,
    "ret_3_rel_sw_l2": LAGS_TREND,
    "ret_10_rel_sw_l2": LAGS_TREND,
    "oc_ret_rel_sw_l2": LAGS_FAST,
    "ret_5_rel_sw_l2": LAGS_MED,
    "ret_20_rel_sw_l2": LAGS_MED,
    "turnover_rel_sw_l2": LAGS_MED,
    "mf_inst_rate_rel_sw_l2": LAGS_MED,
    "beta_to_sw_l2_20d": LAGS_MED,
    "vol_over_sw_l2_vol": LAGS_MED,
 #   "stock_ind_strength_5": LAGS_MED,
 #   "stock_ind_strength_20": LAGS_MED,
    "sw_l2_ret_rel_csi1500_ew": LAGS_SLOW,
    "sw_l2_ret_rel_csi1500_mcap": LAGS_SLOW,
    "sw_l2_mom5_rel_csi1500_ew": LAGS_SLOW,
    "sw_l2_mom5_rel_csi1500_mcap": LAGS_SLOW,
    "sw_l2_mom20_rel_csi1500_ew": LAGS_SLOW,
    "sw_l2_mom20_rel_csi1500_mcap": LAGS_SLOW,
    "sw_l2_market_beta_ew_20d": LAGS_SLOW,
    "sw_l2_market_beta_mcap_20d": LAGS_SLOW,
    "sw_l2_market_alpha_ew_5": LAGS_SLOW,
    "sw_l2_market_alpha_mcap_5": LAGS_SLOW,

    # SW L2 cross-sectional industry strength
    "sw_l2_ret_1_rank": [],
    "sw_l2_mom_2_rank": [],
    "sw_l2_mom_3_rank": [],
    "sw_l2_mom_5_rank": [],
    "sw_l2_mom_10_rank": [],
    "sw_l2_mom_20_rank": [],

    "sw_l2_ret_rel_mkt_rank": [],
    "sw_l2_mom5_rel_mkt_rank": [],
    "sw_l2_mom20_rel_mkt_rank": [],

    "sw_l2_ret_1_rank_ma5": [],
    "sw_l2_mom_2_rank_ma5": [],
    "sw_l2_mom_3_rank_ma5": [],
    "sw_l2_mom_5_rank_ma5": [],
    "sw_l2_mom_10_rank_ma5": [],
    "sw_l2_mom_20_rank_ma5": [],
    "sw_l2_ret_rel_mkt_rank_ma5": [],
    "sw_l2_mom5_rel_mkt_rank_ma5": [],
    "sw_l2_mom20_rel_mkt_rank_ma5": [],

    "sw_l2_ret_1_rank_accel5": [],
    "sw_l2_mom_2_rank_accel5": [],
    "sw_l2_mom_3_rank_accel5": [],
    "sw_l2_mom_5_rank_accel5": [],
    "sw_l2_mom_10_rank_accel5": [],
    "sw_l2_mom_20_rank_accel5": [],
    "sw_l2_ret_rel_mkt_rank_accel5": [],
    "sw_l2_mom5_rel_mkt_rank_accel5": [],
    "sw_l2_mom20_rel_mkt_rank_accel5": [],
    # 1-minute intraday features, built beforehand by build_minute_features_v5b.py
    "first_1m_ret": LAGS_3D,
    "first_1m_range": LAGS_3D,
    "first_1m_amount_ratio": LAGS_3D,
    "first_1m_vwap_ratio": LAGS_3D,
    "first_5m_ret": LAGS_3D,
    "first_5m_range": LAGS_3D,
    "first_5m_amount_ratio": LAGS_3D,
    "first_5m_vwap_ratio": LAGS_3D,
    "first_10m_ret": LAGS_3D,
    "first_10m_range": LAGS_3D,
    "first_10m_amount_ratio": LAGS_3D,
    "first_10m_vwap_ratio": LAGS_3D,
    "first_15m_ret": LAGS_3D,
    "first_15m_range": LAGS_3D,
    "first_15m_amount_ratio": LAGS_3D,
    "first_15m_vwap_ratio": LAGS_3D,
    "first_30m_ret": LAGS_3D,
    "first_60m_ret": LAGS_3D,
    "morning_ret": LAGS_3D,
    "afternoon_ret": LAGS_3D,
    "last_60m_ret": LAGS_3D,
    "last_30m_ret": LAGS_3D,
    "first_30m_range": LAGS_3D,
    "first_60m_range": LAGS_3D,
    "morning_range": LAGS_3D,
    "afternoon_range": LAGS_3D,
    "last_60m_range": LAGS_3D,
    "last_30m_range": LAGS_3D,
    "first_30m_amount_ratio": LAGS_3D,
    "first_60m_amount_ratio": LAGS_3D,
    "morning_amount_ratio": LAGS_3D,
    "afternoon_amount_ratio": LAGS_3D,
    "last_60m_amount_ratio": LAGS_3D,
    "last_30m_amount_ratio": LAGS_3D,
    "first_30m_vwap_ratio": LAGS_3D,
    "first_60m_vwap_ratio": LAGS_3D,
    "morning_vwap_ratio": LAGS_3D,
    "afternoon_vwap_ratio": LAGS_3D,
    "last_60m_vwap_ratio": LAGS_3D,
    "last_30m_vwap_ratio": LAGS_3D,
    "last_15m_ret": LAGS_3D,
    "last_15m_range": LAGS_3D,
    "last_15m_amount_ratio": LAGS_3D,
    "last_15m_vwap_ratio": LAGS_3D,
    "last_10m_ret": LAGS_3D,
    "last_10m_range": LAGS_3D,
    "last_10m_amount_ratio": LAGS_3D,
    "last_10m_vwap_ratio": LAGS_3D,
    "last_5m_ret": LAGS_3D,
    "last_5m_range": LAGS_3D,
    "last_5m_amount_ratio": LAGS_3D,
    "last_5m_vwap_ratio": LAGS_3D,
    "max_10m_ret": LAGS_SLOW,
    "min_10m_ret": LAGS_SLOW,
    "max_15m_ret": LAGS_SLOW,
    "min_15m_ret": LAGS_SLOW,
    "max_15m_range": LAGS_SLOW,
    "mean_15m_range": LAGS_SLOW,
    "realized_vol_5m": LAGS_SLOW,
    "late_realized_vol_5m": LAGS_SLOW,
    "trend_efficiency": LAGS_SLOW,
    "intraday_sign_changes": LAGS_SLOW,
    "intraday_return_skew": LAGS_SLOW,
    "up_bar_ratio": LAGS_3D,
    "down_bar_ratio": LAGS_3D,
    "late_up_bar_ratio": LAGS_3D,
    "pct_bars_above_vwap": LAGS_SLOW,
    "vwap_cross_count": LAGS_SLOW,
    "mean_vwap_distance": LAGS_SLOW,
    "max_5m_ret": LAGS_SLOW,
    "min_5m_ret": LAGS_SLOW,
    "max_5m_range": LAGS_SLOW,
    "mean_5m_range": LAGS_SLOW,
    "max_consecutive_up_bars": LAGS_SLOW,
    "max_consecutive_down_bars": LAGS_SLOW,
    "amount_concentration_top3": LAGS_SLOW,
    "intraday_max_drawdown": LAGS_SLOW,
    "drawdown_duration": LAGS_SLOW,
    "minute_of_high": LAGS_SLOW,
    "minute_of_low": LAGS_SLOW,
    "minute_of_high_raw": LAGS_SLOW,
    "minute_of_low_raw": LAGS_SLOW,
    "pct_time_above_vwap": LAGS_SLOW,
    "max_vwap_distance": LAGS_SLOW,
    "std_vwap_distance": LAGS_SLOW,
    "last30_vwap_distance": LAGS_SLOW,
    "last60_vwap_distance": LAGS_SLOW,
    "close_vwap_distance": LAGS_SLOW,
    "vwap_recovery_ratio": LAGS_SLOW,
    "morning_efficiency": LAGS_SLOW,
    "afternoon_efficiency": LAGS_SLOW,
    "morning_afternoon_corr": LAGS_SLOW,
    "realized_kurtosis": LAGS_SLOW,
    "buy_volume_ratio": LAGS_3D,
    "sell_volume_ratio": LAGS_3D,
    "buy_pressure": LAGS_3D,
    "afternoon_volume_share": LAGS_SLOW,
    "volume_curve_skew": LAGS_SLOW,
    "variance_top1_share": LAGS_SLOW,
    "variance_top3_share": LAGS_SLOW,
    "variance_top5_share": LAGS_SLOW,
    "variance_top10_share": LAGS_SLOW,
    "variance_top20_share": LAGS_SLOW,
    "gap_fill_ratio": LAGS_3D,
    "gap_persistence": LAGS_3D,
    "gap_same_side_ratio": LAGS_3D,
    "gap_fill_time": LAGS_3D,
    "gap_fill_time_raw": LAGS_3D,
    "first_reversal_minute": LAGS_3D,
    "realized_vol_1m": LAGS_SLOW,
    "late_realized_vol_1m": LAGS_SLOW,
    "realized_up_semivar": LAGS_SLOW,
    "realized_down_semivar": LAGS_SLOW,
    "realized_semivar_imbalance": LAGS_SLOW,
    "bipower_variation": LAGS_SLOW,
    "jump_variation": LAGS_SLOW,
    "jump_variation_share": LAGS_SLOW,
    "first30_vol_share": LAGS_SLOW,
    "morning_vol_share": LAGS_SLOW,
    "afternoon_vol_share": LAGS_SLOW,
    "last30_vol_share": LAGS_SLOW,
    "last60_vol_share": LAGS_SLOW,
    "return_autocorr_1": LAGS_SLOW,
    "return_autocorr_5": LAGS_SLOW,
    "return_autocorr_10": LAGS_SLOW,
    "bar_entropy": LAGS_SLOW,
    "hurst_intraday": LAGS_SLOW,
    "max_1m_ret": LAGS_SLOW,
    "minute_of_max_1m_ret": LAGS_SLOW,
    "minute_of_max_1m_ret_raw": LAGS_SLOW,
    "max_1m_amount": LAGS_SLOW,
    "minute_of_max_amount": LAGS_SLOW,
    "minute_of_max_amount_raw": LAGS_SLOW,
    "max_1m_volume": LAGS_SLOW,
    "minute_of_max_volume": LAGS_SLOW,
    "minute_of_max_volume_raw": LAGS_SLOW,
    "max_1m_range": LAGS_SLOW,
    "minute_of_max_range": LAGS_SLOW,
    "minute_of_max_range_raw": LAGS_SLOW,
    "start_minute_longest_up_run": LAGS_SLOW,
    "start_minute_longest_up_run_raw": LAGS_SLOW,
    "start_minute_longest_down_run": LAGS_SLOW,
    "start_minute_longest_down_run_raw": LAGS_SLOW,
    "late_max_5m_drop": LAGS_SLOW,
    "late_max_5m_range": LAGS_SLOW,
    "afternoon_minus_morning_ret": LAGS_3D,
    "last60_minus_first60_ret": LAGS_3D,
}

CROSS_SECTIONAL_LAGS = {
    "ret_1_cs_rank": LAGS_FAST,
    "ret_1_prank_1500": LAGS_FAST,
    "ret_2_prank_1500": LAGS_TREND,
    "ret_3_prank_1500": LAGS_TREND,
    "ret_5_prank_1500": LAGS_FAST,
    "ret_10_prank_1500": LAGS_TREND,
    "vol_prank_1500": LAGS_FAST,
    "oc_ret_cs_rank": LAGS_FAST,
    "amount_cs_rank": LAGS_FAST,
    "amount_prank_1500": LAGS_FAST,
    "turnover_f_cs_rank": LAGS_FAST,
    "turnover_prank_1500": LAGS_FAST,
    "mf_inst_rate_cs_rank": LAGS_SHOCK,
    "mf_inst_rate_prank_1500": LAGS_SHOCK,
    "vwap_close_ratio_prank_1500": LAGS_FAST,
    "mf_price_align_prank_1500": LAGS_SHOCK,
    "close_to_high_cs_rank": LAGS_FAST,
    "winner_rate_cs_rank": LAGS_MED,
    "circ_mv_cs_rank": LAGS_MED,
    "mv_prank_1500": LAGS_MED,
    "sw_l2_ret_1_ind_rank": LAGS_SLOW,
    "sw_l2_mom_5_ind_rank": LAGS_SLOW,
    "sw_l2_mom_20_ind_rank": LAGS_SLOW,
    "sw_l2_amount_z_ind_rank": LAGS_SLOW,
    "sw_l2_turnover_z_ind_rank": LAGS_SLOW,
    "sw_l2_ret_std_20_ind_rank": LAGS_SLOW,
}


def read_csv_any(path: Path, **kwargs) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)


def safe_div(a, b, eps: float = EPS):
    return a / (pd.Series(b).replace(0, np.nan) + eps)


def safe_scalar_div(num: float, den: float, eps: float = EPS) -> float:
    if pd.isna(num) or pd.isna(den) or abs(float(den)) <= eps:
        return np.nan
    return float(num) / float(den)


def zscore_safe(s: pd.Series, window: int, eps: float = EPS) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    return (s - mu) / (sd + eps)


def rolling_prank_past(s: pd.Series, window: int) -> pd.Series:
    values = s.to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    for i in range(window, len(values)):
        cur = values[i]
        if np.isnan(cur):
            continue
        hist = values[i - window : i]
        hist = hist[~np.isnan(hist)]
        if hist.size:
            out[i] = np.mean(hist <= cur)
    return pd.Series(out, index=s.index)


def parse_time_to_minutes_since_open(s: pd.Series, fill_value: float = 999.0) -> pd.Series:
    text = s.astype("string").str.strip()
    extracted = text.str.extract(r"(?P<h>\d{1,2}):(?P<m>\d{2})")
    h = pd.to_numeric(extracted["h"], errors="coerce")
    m = pd.to_numeric(extracted["m"], errors="coerce")

    numeric_text = text.str.replace(r"\.0+$", "", regex=True).str.replace(r"\D", "", regex=True)
    numeric_h = pd.to_numeric(numeric_text.str[:-4], errors="coerce")
    numeric_m = pd.to_numeric(numeric_text.str[-4:-2], errors="coerce")
    hhmm_h = pd.to_numeric(numeric_text.str[:-2], errors="coerce")
    hhmm_m = pd.to_numeric(numeric_text.str[-2:], errors="coerce")
    use_hhmmss = numeric_text.str.len() >= 5
    h = h.fillna(numeric_h.where(use_hhmmss, hhmm_h))
    m = m.fillna(numeric_m.where(use_hhmmss, hhmm_m))
    minutes = (h - 9) * 60 + (m - 30)
    return minutes.fillna(fill_value)


def stock_code_from_path(path: Path) -> str:
    return path.name.replace(".all.csv", "")


def list_candidate_stock_files() -> list[Path]:
    return sorted(MERGED_DIR.glob("*.all.csv"))


def load_missing_margin_stocks_from_config() -> tuple[set[str], int | None]:
    try:
        from delfile_v5b import MIN_MARGIN_DETAIL_ROWS, MISSING_MARGIN_DETAIL_STOCKS
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing delfile_v5b.py. Run build_margin_detail_filter_v5b.py first, "
            "then rerun prepare_training_v5b.py."
        ) from exc
    return set(MISSING_MARGIN_DETAIL_STOCKS), MIN_MARGIN_DETAIL_ROWS


def preflight_stock_filter(max_stocks: int | None = None) -> list[Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = list_candidate_stock_files()
    candidate_codes = [stock_code_from_path(p) for p in candidates]
    missing_config, min_margin_rows = load_missing_margin_stocks_from_config()
    missing = sorted(set(candidate_codes) & missing_config)
    stale_missing = sorted(missing_config - set(candidate_codes))
    selected = [p for p in candidates if stock_code_from_path(p) not in set(missing)]
    if max_stocks is not None:
        selected = selected[:max_stocks]

    pd.DataFrame({"ts_code": missing}).to_csv(
        REPORT_DIR / "missing_margin_detail_stocks.csv", index=False, encoding="utf_8_sig"
    )
    pd.DataFrame([{
        "candidate_stock_count": len(candidate_codes),
        "margin_detail_stock_count": len(candidate_codes) - len(missing),
        "missing_margin_detail_count": len(missing),
        "stale_missing_config_count": len(stale_missing),
        "processed_stock_count": len(selected),
        "min_margin_rows": min_margin_rows,
        "max_stocks": "" if max_stocks is None else max_stocks,
    }]).to_csv(REPORT_DIR / "stock_universe_filter_summary.csv", index=False, encoding="utf_8_sig")
    return selected


def prepare_csi1500_features() -> pd.DataFrame:
    idx = read_csv_any(CSI1500_INDEX_PATH, dtype={"trade_date": str})
    num_cols = [c for c in idx.columns if c != "trade_date"]
    idx[num_cols] = idx[num_cols].apply(pd.to_numeric, errors="coerce")
    idx = idx.sort_values("trade_date").reset_index(drop=True)
    idx = filter_date_pandas(idx, G_SOURCE_START_DATE, G_SOURCE_END_DATE)

    def idx_num_col(name: str) -> pd.Series:
        if name in idx.columns:
            return pd.to_numeric(idx[name], errors="coerce")
        return pd.Series(np.nan, index=idx.index, dtype="float64")

    idx["csi1500_ew_ret"] = idx["csi1500_ew_close_ret"]
    idx["csi1500_mcap_ret"] = idx["csi1500_mcap_close_ret"]
    idx["csi1500_ew_oc_ret"] = safe_div(idx["csi1500_ew_close"], idx["csi1500_ew_open"]) - 1
    idx["csi1500_mcap_oc_ret"] = safe_div(idx["csi1500_mcap_close"], idx["csi1500_mcap_open"]) - 1
    idx["csi1500_ew_mom_2"] = np.log(idx["csi1500_ew_close"]).diff(2)
    idx["csi1500_ew_mom_3"] = np.log(idx["csi1500_ew_close"]).diff(3)
    idx["csi1500_ew_mom_5"] = np.log(idx["csi1500_ew_close"]).diff(5)
    idx["csi1500_ew_mom_10"] = np.log(idx["csi1500_ew_close"]).diff(10)
    idx["csi1500_ew_mom_20"] = np.log(idx["csi1500_ew_close"]).diff(20)
    idx["csi1500_mcap_mom_2"] = np.log(idx["csi1500_mcap_close"]).diff(2)
    idx["csi1500_mcap_mom_3"] = np.log(idx["csi1500_mcap_close"]).diff(3)
    idx["csi1500_mcap_mom_5"] = np.log(idx["csi1500_mcap_close"]).diff(5)
    idx["csi1500_mcap_mom_10"] = np.log(idx["csi1500_mcap_close"]).diff(10)
    idx["csi1500_mcap_mom_20"] = np.log(idx["csi1500_mcap_close"]).diff(20)
    idx["csi1500_ew_ret_std_20"] = idx["csi1500_ew_ret"].rolling(20).std(ddof=0)
    idx["csi1500_mcap_ret_std_20"] = idx["csi1500_mcap_ret"].rolling(20).std(ddof=0)
    idx["csi1500_ew_up_ratio"] = idx["csi1500_up_ratio"]
    idx["csi1500_ew_down_ratio"] = idx["csi1500_down_ratio"]
    idx["csi1500_ew_gt_2pct_ratio"] = idx["csi1500_gt_2pct_ratio"]
    idx["csi1500_ew_lt_minus_2pct_ratio"] = idx["csi1500_lt_minus_2pct_ratio"]
    idx["mkt_pos_rate"] = idx["csi1500_up_ratio"]
    idx["mkt_gt3_rate"] = idx_num_col("csi1500_gt_3pct_ratio")
    idx["mkt_gt5_rate"] = idx_num_col("csi1500_gt_5pct_ratio")
    idx["mkt_limit_up_count"] = idx_num_col("csi1500_limit_up_count")
    idx["mkt_limit_down_count"] = idx_num_col("csi1500_limit_down_count")

    keep = [
        "trade_date",
        "csi1500_ew_ret", "csi1500_ew_oc_ret",
        "csi1500_ew_mom_2", "csi1500_ew_mom_3", "csi1500_ew_mom_5",
        "csi1500_ew_mom_10", "csi1500_ew_mom_20",
        "csi1500_ew_ret_std_20", "csi1500_ew_up_ratio", "csi1500_ew_down_ratio",
        "csi1500_ew_gt_2pct_ratio", "csi1500_ew_lt_minus_2pct_ratio", "csi1500_coverage_ratio",
        "mkt_pos_rate", "mkt_gt3_rate", "mkt_gt5_rate", "mkt_limit_up_count", "mkt_limit_down_count",
        "csi1500_mcap_ret", "csi1500_mcap_oc_ret",
        "csi1500_mcap_mom_2", "csi1500_mcap_mom_3", "csi1500_mcap_mom_5",
        "csi1500_mcap_mom_10", "csi1500_mcap_mom_20",
        "csi1500_mcap_ret_std_20", "csi1500_mcap_max_weight", "csi1500_mcap_top10_weight",
        "csi1500_mcap_coverage_ratio",
    ]
    return idx[keep].copy()


def prepare_market_features() -> pd.DataFrame:
    m = read_csv_any(MARKET_PANEL_PATH, dtype={"trade_date": str})
    for c in m.columns:
        if c != "trade_date":
            m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.sort_values("trade_date").reset_index(drop=True)
    m = filter_date_pandas(m, G_SOURCE_START_DATE, G_SOURCE_END_DATE)

    m["mkt_ret_sh"] = m.get("mkt_sh_ret") / 100.0
    m["mkt_ret_sz"] = m.get("mkt_sz_ret") / 100.0
    m["mkt_ret_spread"] = m["mkt_ret_sh"] - m["mkt_ret_sz"]
    m["mkt_mf_inst_rate"] = m.get("mkt_net_amount_rate") * 0.01
    m["mkt_mf_inst_ma_3"] = m["mkt_mf_inst_rate"].rolling(3).mean()
    m["mkt_mf_inst_ma_5"] = m["mkt_mf_inst_rate"].rolling(5).mean()
    m["mkt_mf_inst_ma_10"] = m["mkt_mf_inst_rate"].rolling(10).mean()
    m["mkt_flow_concentration"] = (
        m.get("mkt_buy_elg_amount_rate").abs() + m.get("mkt_buy_lg_amount_rate").abs()
    ) * 0.01
    m["mkt_mf_retail_rate"] = m.get("mkt_buy_sm_amount_rate") * 0.01
    m["mkt_mf_inst_z_20"] = zscore_safe(m["mkt_mf_inst_rate"], 20)
    m["mkt_mf_available"] = m["mkt_mf_inst_rate"].notna().astype("int8")

    m["hsgt_available"] = m.get("hsgt_north_money").notna().astype("int8")
    m["hsgt_hgt_minus_sgt"] = m.get("hsgt_hgt") - m.get("hsgt_sgt")
    m["hsgt_north_money_z_20"] = zscore_safe(m.get("hsgt_north_money"), 20)
    m["hsgt_north_money_ma_3"] = m.get("hsgt_north_money").rolling(3).mean()
    m["hsgt_north_money_ma_5"] = m.get("hsgt_north_money").rolling(5).mean()
    m["hsgt_north_money_accel_3"] = m["hsgt_north_money_ma_3"] - m["hsgt_north_money_ma_3"].shift(3)
    m["hsgt_south_money_z_20"] = zscore_safe(m.get("hsgt_south_money"), 20)
    m["hsgt_south_money_ma_3"] = m.get("hsgt_south_money").rolling(3).mean()
    m["hsgt_south_money_ma_5"] = m.get("hsgt_south_money").rolling(5).mean()

    for prefix in ["hsi", "hktech", "xin9", "spx", "ixic", "n225"]:
        pct = f"{prefix}_pct_chg"
        if pct in m.columns:
            m[f"{prefix}_ret"] = m[pct] / 100.0
    for prefix in [ "spx", "ixic"]:
        vol = f"{prefix}_vol"
        if vol in m.columns:
            m[f"{prefix}_vol_z"] = zscore_safe(np.log(m[vol].replace(0, np.nan)), 60)

    keep = [
        "trade_date", "mkt_mf_available", "mkt_ret_sh", "mkt_ret_sz", "mkt_ret_spread",
        "mkt_mf_inst_rate", "mkt_mf_inst_ma_3", "mkt_mf_inst_ma_5", "mkt_mf_inst_ma_10",
         "mkt_flow_concentration", "mkt_mf_retail_rate", "mkt_mf_inst_z_20",
        "hsgt_available", "hsgt_hgt", "hsgt_sgt", "hsgt_north_money", "hsgt_south_money",
        "hsgt_hgt_minus_sgt", "hsgt_north_money_z_20", "hsgt_north_money_ma_3",
        "hsgt_north_money_ma_5", "hsgt_north_money_accel_3", "hsgt_south_money_z_20",
        "hsgt_south_money_ma_3", "hsgt_south_money_ma_5",
        "hsi_ret", "hsi_swing", "hktech_ret", "hktech_swing", "xin9_ret",
        "xin9_swing", "spx_ret", "spx_swing", "spx_vol_z", "ixic_ret", "ixic_swing", "ixic_vol_z", 
        "n225_ret", "n225_swing", 
    ]
    keep = [c for c in keep if c in m.columns]
    return m[keep].copy()


def prepare_sw_l2_features(csi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sw = read_csv_any(SW_L2_DAILY_PATH, dtype={"ts_code": str, "trade_date": str})
    sw = sw.rename(columns={"ts_code": "sw_l2_index_code", "name": "sw_l2_index_name"})
    for c in ["open", "low", "high", "close", "change", "pct_change", "vol", "amount", "float_mv", "total_mv"]:
        sw[c] = pd.to_numeric(sw[c], errors="coerce")
    sw = sw.sort_values(["sw_l2_index_code", "trade_date"]).reset_index(drop=True)
    sw = filter_date_pandas(sw, G_SOURCE_START_DATE, G_SOURCE_END_DATE)

    # Attach SW L2 code for stable categorical join/ranks.
    mapping = read_csv_any(SW_L2_MAPPING_PATH, dtype=str)
    map_l2 = mapping[["sw_l2_code", "sw_l2_index_code"]].dropna().drop_duplicates("sw_l2_index_code")
    sw = sw.merge(map_l2, on="sw_l2_index_code", how="left")

    g = sw.groupby("sw_l2_index_code", group_keys=False)
    sw["sw_l2_ret_1"] = sw["pct_change"] / 100.0
    sw["sw_l2_oc_ret"] = safe_div(sw["close"], sw["open"]) - 1
    sw["sw_l2_mom_2"] = g["close"].transform(lambda s: np.log(s).diff(2))
    sw["sw_l2_mom_3"] = g["close"].transform(lambda s: np.log(s).diff(3))
    sw["sw_l2_mom_5"] = g["close"].transform(lambda s: np.log(s).diff(5))
    sw["sw_l2_mom_10"] = g["close"].transform(lambda s: np.log(s).diff(10))
    sw["sw_l2_mom_20"] = g["close"].transform(lambda s: np.log(s).diff(20))
    sw["sw_l2_ret_std_5"] = g["sw_l2_ret_1"].transform(lambda s: s.rolling(5).std(ddof=0))
    sw["sw_l2_ret_std_20"] = g["sw_l2_ret_1"].transform(lambda s: s.rolling(20).std(ddof=0))
    sw["sw_l2_range_pct"] = safe_div(sw["high"] - sw["low"], sw["close"])
    sw["sw_l2_turnover_proxy"] = safe_div(sw["amount"], sw["float_mv"])
    sw["sw_l2_range_z_20"] = g["sw_l2_range_pct"].transform(lambda s: zscore_safe(s, 20))
    sw["sw_l2_amount_z_20"] = g["amount"].transform(lambda s: zscore_safe(np.log(s.replace(0, np.nan)), 20))
    sw["sw_l2_vol_z_20"] = g["vol"].transform(lambda s: zscore_safe(np.log(s.replace(0, np.nan)), 20))
    sw["sw_l2_turnover_z_20"] = g["sw_l2_turnover_proxy"].transform(lambda s: zscore_safe(s, 20))

    sw = sw.merge(csi, on="trade_date", how="left")
    sw["sw_l2_ret_rel_csi1500_ew"] = sw["sw_l2_ret_1"] - sw["csi1500_ew_ret"]
    sw["sw_l2_ret_rel_csi1500_mcap"] = sw["sw_l2_ret_1"] - sw["csi1500_mcap_ret"]
    sw["sw_l2_mom5_rel_csi1500_ew"] = sw["sw_l2_mom_5"] - sw["csi1500_ew_mom_5"]
    sw["sw_l2_mom5_rel_csi1500_mcap"] = sw["sw_l2_mom_5"] - sw["csi1500_mcap_mom_5"]
    sw["sw_l2_mom20_rel_csi1500_ew"] = sw["sw_l2_mom_20"] - sw["csi1500_ew_mom_20"]
    sw["sw_l2_mom20_rel_csi1500_mcap"] = sw["sw_l2_mom_20"] - sw["csi1500_mcap_mom_20"]

    sw = sw.sort_values(["sw_l2_index_code", "trade_date"]).reset_index(drop=True)
    g = sw.groupby("sw_l2_index_code", group_keys=False)
    sw["sw_l2_market_beta_ew_20d"] = g.apply(
        lambda x: x["sw_l2_ret_1"].rolling(20).cov(x["csi1500_ew_ret"]) /
        (x["csi1500_ew_ret"].rolling(20).var() + EPS),
        include_groups=False,
    ).reset_index(level=0, drop=True)
    sw["sw_l2_market_beta_mcap_20d"] = g.apply(
        lambda x: x["sw_l2_ret_1"].rolling(20).cov(x["csi1500_mcap_ret"]) /
        (x["csi1500_mcap_ret"].rolling(20).var() + EPS),
        include_groups=False,
    ).reset_index(level=0, drop=True)
    sw["sw_l2_market_alpha_ew_5"] = sw["sw_l2_mom_5"] - sw["sw_l2_market_beta_ew_20d"] * sw["csi1500_ew_mom_5"]
    sw["sw_l2_market_alpha_mcap_5"] = sw["sw_l2_mom_5"] - sw["sw_l2_market_beta_mcap_20d"] * sw["csi1500_mcap_mom_5"]

    rank_cols = {
        "sw_l2_ret_1": "sw_l2_ret_1_ind_rank",
        "sw_l2_mom_5": "sw_l2_mom_5_ind_rank",
        "sw_l2_mom_20": "sw_l2_mom_20_ind_rank",
        "sw_l2_amount_z_20": "sw_l2_amount_z_ind_rank",
        "sw_l2_turnover_z_20": "sw_l2_turnover_z_ind_rank",
        "sw_l2_ret_std_20": "sw_l2_ret_std_20_ind_rank",
    }
    for src, dst in rank_cols.items():
        sw[dst] = sw.groupby("trade_date")[src].rank(pct=True)

    # --------------------------------------------------
    # Cross-sectional SW L2 industry strength
    # --------------------------------------------------

    strength_cols = {
        "sw_l2_ret_1": "sw_l2_ret_1_rank",
        "sw_l2_mom_2": "sw_l2_mom_2_rank",
        "sw_l2_mom_3": "sw_l2_mom_3_rank",
        "sw_l2_mom_5": "sw_l2_mom_5_rank",
        "sw_l2_mom_10": "sw_l2_mom_10_rank",
        "sw_l2_mom_20": "sw_l2_mom_20_rank",
        "sw_l2_ret_rel_csi1500_ew": "sw_l2_ret_rel_mkt_rank",
        "sw_l2_mom5_rel_csi1500_ew": "sw_l2_mom5_rel_mkt_rank",
        "sw_l2_mom20_rel_csi1500_ew": "sw_l2_mom20_rel_mkt_rank",
    }

    for src, dst in strength_cols.items():
        sw[dst] = sw.groupby("trade_date")[src].rank(pct=True)

    # Industry leadership rotation:
    # positive means industry rank improved versus its recent average
    g = sw.groupby("sw_l2_index_code", group_keys=False)

    for rank_col in strength_cols.values():
        sw[f"{rank_col}_ma5"] = g[rank_col].transform(lambda x: x.rolling(5, min_periods=3).mean())
        sw[f"{rank_col}_accel5"] = sw[rank_col] - sw[f"{rank_col}_ma5"]
    
    feature_cols = [
        "sw_l2_code", "sw_l2_index_code", "trade_date",
        "sw_l2_ret_1", "sw_l2_oc_ret", "sw_l2_mom_2", "sw_l2_mom_3", "sw_l2_mom_5", "sw_l2_mom_10",
        "sw_l2_mom_20", "sw_l2_ret_std_5", "sw_l2_ret_std_20", "sw_l2_range_pct",
        "sw_l2_range_z_20", "sw_l2_amount_z_20", "sw_l2_vol_z_20", "sw_l2_turnover_proxy",
        "sw_l2_turnover_z_20", "sw_l2_ret_rel_csi1500_ew", "sw_l2_ret_rel_csi1500_mcap",
        "sw_l2_mom5_rel_csi1500_ew", "sw_l2_mom5_rel_csi1500_mcap",
        "sw_l2_mom20_rel_csi1500_ew", "sw_l2_mom20_rel_csi1500_mcap",
        "sw_l2_market_beta_ew_20d", "sw_l2_market_beta_mcap_20d",
        "sw_l2_market_alpha_ew_5", "sw_l2_market_alpha_mcap_5",
    ]

    industry_strength_feature_cols = (
        list(strength_cols.values())
        + [f"{c}_ma5" for c in strength_cols.values()]
        + [f"{c}_accel5" for c in strength_cols.values()]
    )

    feature_cols += industry_strength_feature_cols
    
    rank_feature_cols = (["sw_l2_code", "trade_date"] + list(rank_cols.values()))
    
    return sw[feature_cols].copy(), sw[rank_feature_cols].drop_duplicates(["sw_l2_code", "trade_date"]).copy()


def init_worker(
    csi: pd.DataFrame,
    market: pd.DataFrame,
    sw_feat: pd.DataFrame,
    sw_mapping: pd.DataFrame,
    minute_feature_source: str | None = None,
    source_start_date: str | None = None,
    source_end_date: str | None = None,
    output_start_date: str | None = None,
    output_end_date: str | None = None,
) -> None:
    global G_CSI, G_MARKET, G_SW_FEAT, G_SW_MAPPING, G_MINUTE_FEATURE_SOURCE, G_MINUTE_FEATURE_BY_STOCK
    global G_SOURCE_START_DATE, G_SOURCE_END_DATE, G_OUTPUT_START_DATE, G_OUTPUT_END_DATE
    G_CSI = csi
    G_MARKET = market
    G_SW_FEAT = sw_feat
    G_SW_MAPPING = sw_mapping
    G_SOURCE_START_DATE = source_start_date
    G_SOURCE_END_DATE = source_end_date
    G_OUTPUT_START_DATE = output_start_date
    G_OUTPUT_END_DATE = output_end_date
    G_MINUTE_FEATURE_SOURCE = minute_feature_source
    G_MINUTE_FEATURE_BY_STOCK = load_minute_feature_cache(minute_feature_source)


def clean_5min_amount(day: pd.DataFrame) -> pd.DataFrame:
    """Standardize 5-minute vol/amount before VWAP/amount features.

    Most raw files use vol in hands. Some update sources use vol in shares
    for part of 2026, and a few bars have undercounted vol or inflated amount.
    Normalize toward: vol = hands, amount = yuan.
    """
    out = day.copy()
    for col in ["open", "close", "high", "low", "vol", "amount"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    vol = out["vol"]
    close = out["close"]
    amount = out["amount"]
    valid = vol.gt(0) & close.gt(0) & amount.notna()
    ratio_hand = amount / (vol.replace(0, np.nan) * close * 100.0)
    ratio_share = amount / (vol.replace(0, np.nan) * close)

    # Source vol is shares, not hands. Convert vol to hands.
    share_vol = valid & ratio_hand.between(0.005, 0.02) & ratio_share.between(0.5, 1.5)
    out.loc[share_vol, "vol"] = out.loc[share_vol, "vol"] / 100.0

    vol = out["vol"]
    amount = out["amount"]
    ratio_hand = amount / (vol.replace(0, np.nan) * close * 100.0)
    valid = vol.gt(0) & close.gt(0) & amount.notna()

    # Source vol is severely undercounted; amount looks more plausible.
    undercounted_vol = valid & ratio_hand.gt(50.0)
    out.loc[undercounted_vol, "vol"] = out.loc[undercounted_vol, "amount"] / (
        out.loc[undercounted_vol, "close"] * 100.0
    )

    vol = out["vol"]
    amount = out["amount"]
    ratio_hand = amount / (vol.replace(0, np.nan) * close * 100.0)
    valid = vol.gt(0) & close.gt(0) & amount.notna()

    # Remaining moderate mismatches are usually bad amount rows.
    bad_amount = valid & (ratio_hand.lt(0.5) | ratio_hand.gt(1.5))
    out.loc[bad_amount, "amount"] = out.loc[bad_amount, "close"] * out.loc[bad_amount, "vol"] * 100.0
    return out


def build_5min_features(ts_code: str) -> pd.DataFrame:
    path = FIVE_MIN_DIR / f"{ts_code}.parquet"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame({"ts_code": [], "trade_date": []})
    df = pl.read_parquet(path)
    if df.is_empty():
        return pd.DataFrame({"ts_code": [], "trade_date": []})
    pdf = df.to_pandas()
    pdf["datetime"] = pd.to_datetime(pdf["datetime"])
    pdf["trade_date"] = pdf["datetime"].dt.strftime("%Y%m%d")
    pdf = pdf.sort_values(["trade_date", "datetime"])

    rows = []
    for trade_date, day in pdf.groupby("trade_date", sort=True):
        day = day.reset_index(drop=True)
        if len(day) < 10:
            continue
        day = clean_5min_amount(day)
        total_amount = day["amount"].sum()
        total_amount = np.nan if total_amount == 0 else total_amount
        # data/raw stores vol in hands and amount in yuan, so price VWAP needs shares.
        daily_vwap = safe_scalar_div(day["amount"].sum(), day["vol"].sum() * 100.0)
        bar_ret = day["close"].pct_change()
        path_ret = pd.concat([
            pd.Series([day["close"].iloc[0] / day["open"].iloc[0] - 1], index=[day.index[0]]),
            bar_ret.iloc[1:],
        ])
        bar_range = safe_div(day["high"] - day["low"], day["open"])
        cum_amt = day["amount"].cumsum()
        cum_vol = day["vol"].replace(0, np.nan).cumsum()
        intraday_vwap = cum_amt / (cum_vol * 100.0)

        def window_stats(prefix: str, w: pd.DataFrame) -> dict[str, float]:
            if w.empty:
                return {}
            vwap = safe_scalar_div(w["amount"].sum(), w["vol"].sum() * 100.0)
            return {
                f"{prefix}_ret": w["close"].iloc[-1] / w["open"].iloc[0] - 1,
                f"{prefix}_range": (w["high"].max() - w["low"].min()) / w["open"].iloc[0],
                f"{prefix}_amount_ratio": safe_scalar_div(w["amount"].sum(), total_amount),
                f"{prefix}_vwap_ratio": safe_scalar_div(vwap, daily_vwap) - 1,
            }

        row = {"ts_code": ts_code, "trade_date": trade_date}
        row.update(window_stats("first_30m", day.iloc[:6]))
        row.update(window_stats("first_60m", day.iloc[:12]))
        row.update(window_stats("morning", day.iloc[:24]))
        row.update(window_stats("afternoon", day.iloc[24:]))
        row.update(window_stats("last_60m", day.iloc[-12:]))
        row.update(window_stats("last_30m", day.iloc[-6:]))
        row.update(window_stats("last_15m", day.iloc[-3:]))

        roll3_ret = day["close"].pct_change(3)
        roll3_range = day["high"].rolling(3).max().sub(day["low"].rolling(3).min()) / day["open"]
        roll3_amount = day["amount"].rolling(3).sum() / total_amount

        path_abs_sum = path_ret.abs().sum()

        trend_efficiency = (
            abs(day["close"].iloc[-1] / day["open"].iloc[0] - 1) / path_abs_sum
            if pd.notna(path_abs_sum) and path_abs_sum > EPS
            else 0.0
        )
        
        row.update({
            "max_15m_ret": roll3_ret.max(),
            "min_15m_ret": roll3_ret.min(),
            "max_15m_range": roll3_range.max(),
            "mean_15m_range": roll3_range.mean(),
            "max_15m_amount_ratio": roll3_amount.max(),
            "realized_vol_5m": bar_ret.std(ddof=0),
            
            
            "trend_efficiency": trend_efficiency,
        
            "up_bar_ratio": (bar_ret > 0).mean(),
            "down_bar_ratio": (bar_ret < 0).mean(),
            "pct_bars_above_vwap": (day["close"] > intraday_vwap).mean(),
            "vwap_cross_count": np.sign(day["close"] - intraday_vwap).diff().abs().fillna(0).gt(0).sum(),
            "max_5m_ret": bar_ret.max(),
            "min_5m_ret": bar_ret.min(),
            "max_5m_range": bar_range.max(),
            "mean_5m_range": bar_range.mean(),
            "amount_concentration_top3": day["amount"].nlargest(3).sum() / total_amount,
      #      "large_amount_bar_count": (day["amount"] > day["amount"].quantile(0.9)).sum(),
        })
        up = (bar_ret > 0).fillna(False).to_numpy()
        down = (bar_ret < 0).fillna(False).to_numpy()
        row["max_consecutive_up_bars"] = max_consecutive_true(up)
        row["max_consecutive_down_bars"] = max_consecutive_true(down)
        late = day.iloc[-6:]
        late_ret = late["close"].pct_change()
        late_range = safe_div(late["high"] - late["low"], late["open"])
        row["late_up_bar_ratio"] = (late_ret > 0).mean()
        row["late_realized_vol_5m"] = late_ret.std(ddof=0)
        row["late_max_5m_drop"] = late_ret.min()
        row["late_max_5m_range"] = late_range.max()
        rows.append(row)
    return pd.DataFrame(rows)


def clean_minute_feature_frame(df: pl.DataFrame) -> pd.DataFrame:
    if df.is_empty():
        return pd.DataFrame({"ts_code": [], "trade_date": []})
    pdf = df.to_pandas()
    pdf["ts_code"] = pdf["ts_code"].astype(str)
    pdf["trade_date"] = pdf["trade_date"].astype(str)
    drop_cols = [
        c
        for c in ["minute_bar_count", "minute_total_amount", "minute_total_vol", "minute_daily_vwap"]
        if c in pdf.columns and c not in BASE_FEATURE_LAGS
    ]
    if drop_cols:
        pdf = pdf.drop(columns=drop_cols)
    return pdf.sort_values("trade_date").drop_duplicates(["ts_code", "trade_date"], keep="last")


def clean_minute_feature_pandas(pdf: pd.DataFrame) -> pd.DataFrame:
    if pdf.empty:
        return pd.DataFrame({"ts_code": [], "trade_date": []})
    pdf = pdf.copy()
    pdf["ts_code"] = pdf["ts_code"].astype(str)
    pdf["trade_date"] = pdf["trade_date"].astype(str)
    drop_cols = [
        c
        for c in ["minute_bar_count", "minute_total_amount", "minute_total_vol", "minute_daily_vwap"]
        if c in pdf.columns and c not in BASE_FEATURE_LAGS
    ]
    if drop_cols:
        pdf = pdf.drop(columns=drop_cols)
    return pdf.sort_values("trade_date").drop_duplicates(["ts_code", "trade_date"], keep="last")


def load_minute_feature_cache(minute_feature_source: str | Path | None) -> dict[str, pd.DataFrame] | None:
    if not minute_feature_source:
        return None
    source = Path(minute_feature_source)
    if not source.is_file() or source.stat().st_size == 0:
        return None
    df = pl.read_parquet(source)
    if df.is_empty():
        return {}
    df = filter_date_polars(df, G_SOURCE_START_DATE, G_SOURCE_END_DATE)
    pdf = clean_minute_feature_pandas(df.to_pandas())
    cache = {str(ts_code): g.copy() for ts_code, g in pdf.groupby("ts_code", sort=False)}
    print(
        f"[minute-cache] loaded panel stocks={len(cache)} rows={len(pdf)} source={source}",
        flush=True,
    )
    return cache


def read_minute_features(ts_code: str, minute_feature_source: str | Path) -> pd.DataFrame:
    if G_MINUTE_FEATURE_BY_STOCK is not None:
        cached = G_MINUTE_FEATURE_BY_STOCK.get(str(ts_code))
        if cached is None:
            return pd.DataFrame({"ts_code": [], "trade_date": []})
        return cached
    source = Path(minute_feature_source)
    path = source / f"{ts_code}.parquet" if source.is_dir() else source
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame({"ts_code": [], "trade_date": []})
    if source.is_file():
        scan = pl.scan_parquet(path).filter(pl.col("ts_code") == ts_code)
        if G_SOURCE_START_DATE:
            scan = scan.filter(pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", "") >= G_SOURCE_START_DATE)
        if G_SOURCE_END_DATE:
            scan = scan.filter(pl.col("trade_date").cast(pl.Utf8).str.replace_all("-", "") <= G_SOURCE_END_DATE)
        df = scan.collect()
    else:
        df = pl.read_parquet(path)
        df = filter_date_polars(df, G_SOURCE_START_DATE, G_SOURCE_END_DATE)
    return clean_minute_feature_frame(df)


def max_consecutive_true(values: np.ndarray) -> int:
    best = cur = 0
    for v in values:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def add_limit_features_old(s: pd.DataFrame, limit_source_available: bool) -> pd.DataFrame:
    s["first_time_min"] = parse_time_to_minutes_since_open(s.get("first_time", pd.Series(index=s.index)), np.nan)
    limit_flag = s.get("limit", pd.Series(index=s.index)).astype("string").fillna("")
    is_limit_up = (limit_flag == "U").astype("int8")
    is_broken = (limit_flag == "Z").astype("int8")
    s["limit_data_available"] = np.int8(1 if limit_source_available else 0)
    s["is_limit_up"] = is_limit_up
    s["is_broken_board"] = is_broken

    fd_amount = s["fd_amount"] if "fd_amount" in s.columns else pd.Series(0.0, index=s.index)
    fd_amount = pd.to_numeric(fd_amount, errors="coerce").fillna(0.0)
    if "float_mv" in s.columns:
        denom = pd.to_numeric(s["float_mv"], errors="coerce")
        if "total_mv" in s.columns:
            denom = denom.fillna(pd.to_numeric(s["total_mv"], errors="coerce"))
    else:
        denom = pd.to_numeric(s.get("total_mv", pd.Series(np.nan, index=s.index)), errors="coerce")

    s["fd_ratio"] = safe_div(fd_amount, denom)
    s.loc[s["fd_ratio"] < EPS, "fd_ratio"] = 0.0
    s["fd_ratio"] = s["fd_ratio"].fillna(0.0)
    s["fd_ratio_log"] = np.log1p(s["fd_ratio"] * 1000.0).fillna(0.0)
#    s["fd_ratio_log_delta_1"] = (s["fd_ratio_log"] - s["fd_ratio_log"].shift(1)).fillna(0.0)
#    s["fd_ratio_log_delta_2"] = (s["fd_ratio_log"] - s["fd_ratio_log"].shift(2)).fillna(0.0)

    open_times = s["open_times"] if "open_times" in s.columns else pd.Series(0.0, index=s.index)
    open_times = pd.to_numeric(open_times, errors="coerce").fillna(0.0)
    s["failure_pressure"] = (s.get("turnover_rate_f", 0.0) * np.log1p(open_times)).fillna(0.0)
    limit_up_cnt_3 = is_limit_up.rolling(3, min_periods=1).sum()
    broken_cnt_3 = is_broken.rolling(3, min_periods=1).sum()
    s["board_accel_3"] = (limit_up_cnt_3 - broken_cnt_3).fillna(0.0)
    s["up_crowding_3"] = ((limit_up_cnt_3 ** 2) / 3.0).fillna(0.0)
    s["board_success_ratio_3"] = (limit_up_cnt_3 / (limit_up_cnt_3 + broken_cnt_3 + EPS)).fillna(0.0)
    limit_rate = pd.to_numeric(s["limit_rate"], errors="coerce") if "limit_rate" in s.columns else pd.Series(np.nan, index=s.index)
    ts = s["ts_code"].astype("string") if "ts_code" in s.columns else pd.Series("", index=s.index, dtype="string")
    prefix_20cm = ts.str.startswith(("300", "301", "688", "689")).fillna(False)
    rate_20cm = (limit_rate >= 0.19) | (limit_rate >= 19.0)
    s["is_20cm"] = (prefix_20cm | rate_20cm.fillna(False)).astype("int8")

    if not limit_source_available:
        unknown_cols = [
            "fd_ratio", "fd_ratio_log", # "fd_ratio_log_delta_1", "fd_ratio_log_delta_2",
            "failure_pressure", "board_accel_3", "up_crowding_3", "board_success_ratio_3",
            "first_time_min", "is_limit_up", "is_broken_board", "is_20cm",
        ]
        for col in unknown_cols:
            s[col] = np.nan
    return s

def capped_days_since_event(event: pd.Series, cap: int = 60) -> pd.Series:
    """
    0 = event today
    1 = event yesterday
    ...
    cap = no event in last cap trading days
    """
    event_arr = event.fillna(False).astype(bool).to_numpy()
    out = np.full(len(event_arr), cap, dtype="float32")

    last_seen = None
    for i, flag in enumerate(event_arr):
        if flag:
            last_seen = i
            out[i] = 0.0
        elif last_seen is not None:
            out[i] = min(i - last_seen, cap)
        else:
            out[i] = cap

    return pd.Series(out, index=event.index)


def add_limit_features(s: pd.DataFrame, limit_source_available: bool) -> pd.DataFrame:
    # Raw categorical limit:
    # U = limit-up, D = limit-down, Z = broken board, NA = no limit event
    limit_flag = s.get("limit", pd.Series(index=s.index)).astype("string")

    is_limit_up = limit_flag.eq("U").fillna(False)
    is_limit_down = limit_flag.eq("D").fillna(False)
    is_broken = limit_flag.eq("Z").fillna(False)
    is_limit_event = is_limit_up | is_limit_down | is_broken

    open_times_raw = s["open_times"] if "open_times" in s.columns else pd.Series(np.nan, index=s.index)
    open_times = pd.to_numeric(open_times_raw, errors="coerce")

    # Daily event count features: no event means 0, not NA.
    s["limit_up_day_cnt_1"] = is_limit_up.astype("float32")
    s["broken_board_day_cnt_1"] = is_broken.astype("float32")
    s["limit_down_day_cnt_1"] = is_limit_down.astype("float32")

    # Sealed board: true limit-up and no reopen.
    # Non-limit days contribute 0.
    sealed_board = is_limit_up & open_times.eq(0).fillna(False)
    s["sealed_board_day_cnt_1"] = sealed_board.astype("float32")

    # Rolling day counts, not limit_times sums.
    for w in [2, 3, 5, 10]:
        s[f"limit_up_day_cnt_{w}"] = (
            is_limit_up.astype(float).rolling(w, min_periods=1).sum().astype("float32")
        )
        s[f"broken_board_day_cnt_{w}"] = (
            is_broken.astype(float).rolling(w, min_periods=1).sum().astype("float32")
        )
        s[f"limit_down_day_cnt_{w}"] = (
            is_limit_down.astype(float).rolling(w, min_periods=1).sum().astype("float32")
        )
        s[f"sealed_board_day_cnt_{w}"] = (
            sealed_board.astype(float).rolling(w, min_periods=1).sum().astype("float32")
        )

    # Board success ratio:
    # no limit-up history in the window => 0.
    for w in [1, 2, 3, 5, 10]:
        up_cnt = s[f"limit_up_day_cnt_{w}"]
        sealed_cnt = s[f"sealed_board_day_cnt_{w}"]
        s[f"board_success_ratio_{w}"] = (
            sealed_cnt / up_cnt.replace(0, np.nan)
        ).fillna(0.0).astype("float32")

    # Capped recency features.
    s["days_since_last_limit_event_60"] = capped_days_since_event(is_limit_event, cap=60)
    s["days_since_last_limit_up_60"] = capped_days_since_event(is_limit_up, cap=60)
    s["days_since_last_broken_board_60"] = capped_days_since_event(is_broken, cap=60)
    s["days_since_last_limit_down_60"] = capped_days_since_event(is_limit_down, cap=60)

    # First limit-up time: only meaningful when a limit event exists.
    s["first_time_min"] = parse_time_to_minutes_since_open(
        s.get("first_time", pd.Series(index=s.index)),
        np.nan,
    )

    # fd_ratio: sealing strength, only meaningful for limit-up related rows.
    fd_amount = s["fd_amount"] if "fd_amount" in s.columns else pd.Series(np.nan, index=s.index)
    fd_amount = pd.to_numeric(fd_amount, errors="coerce")

    if "float_mv" in s.columns:
        denom = pd.to_numeric(s["float_mv"], errors="coerce")
        if "total_mv" in s.columns:
            denom = denom.fillna(pd.to_numeric(s["total_mv"], errors="coerce"))
    else:
        denom = pd.to_numeric(s.get("total_mv", pd.Series(np.nan, index=s.index)), errors="coerce")

    s["fd_ratio"] = safe_div(fd_amount, denom)
    s["fd_ratio_log"] = np.log1p(s["fd_ratio"] * 1000.0)

    # failure pressure: only meaningful when open_times exists.
    turnover_f = pd.to_numeric(
        s.get("turnover_rate_f", pd.Series(np.nan, index=s.index)),
        errors="coerce",
    )
    s["failure_pressure"] = turnover_f * np.log1p(open_times)

    # 20cm board flag: stock-level property, not conditional on limit event.
    limit_rate = (
        pd.to_numeric(s["limit_rate"], errors="coerce")
        if "limit_rate" in s.columns
        else pd.Series(np.nan, index=s.index)
    )
    ts = (
        s["ts_code"].astype("string")
        if "ts_code" in s.columns
        else pd.Series("", index=s.index, dtype="string")
    )
    prefix_20cm = ts.str.startswith(("300", "301", "688", "689")).fillna(False)
    rate_20cm = (limit_rate >= 0.19) | (limit_rate >= 19.0)
    s["is_20cm"] = (prefix_20cm | rate_20cm.fillna(False)).astype("int8")

    if not limit_source_available:
        unknown_cols = [
            "fd_ratio", "fd_ratio_log",
            "failure_pressure", "first_time_min", "is_20cm",
            "limit_up_day_cnt_1", "limit_up_day_cnt_2", "limit_up_day_cnt_3", "limit_up_day_cnt_5", "limit_up_day_cnt_10",
            "broken_board_day_cnt_1", "broken_board_day_cnt_2", "broken_board_day_cnt_3", "broken_board_day_cnt_5", "broken_board_day_cnt_10",
            "limit_down_day_cnt_1", "limit_down_day_cnt_2", "limit_down_day_cnt_3", "limit_down_day_cnt_5", "limit_down_day_cnt_10",
            "sealed_board_day_cnt_1", "sealed_board_day_cnt_2", "sealed_board_day_cnt_3", "sealed_board_day_cnt_5", "sealed_board_day_cnt_10",
            "board_success_ratio_1", "board_success_ratio_2", "board_success_ratio_3", "board_success_ratio_5", "board_success_ratio_10",
            "days_since_last_limit_event_60", "days_since_last_limit_up_60",
            "days_since_last_broken_board_60", "days_since_last_limit_down_60",
        ]
        for col in unknown_cols:
            s[col] = np.nan

    return s

def time_aware_sw_mapping(ts_code: str, trade_dates: pd.Series, mapping: pd.DataFrame) -> pd.DataFrame:
    m = mapping[mapping["ts_code"] == ts_code].copy()
    if m.empty:
        return pd.DataFrame({"trade_date": trade_dates, "sw_l2_code": pd.NA, "sw_l2_index_code": pd.NA})
    m["effective_start_date"] = m["effective_start_date"].astype(str)
    m["effective_end_date"] = m["effective_end_date"].astype(str)
    rows = pd.DataFrame({"trade_date": trade_dates.astype(str)})
    pieces = []
    for _, r in m.iterrows():
        mask = (rows["trade_date"] >= r["effective_start_date"]) & (rows["trade_date"] <= r["effective_end_date"])
        tmp = rows.loc[mask].copy()
        tmp["sw_l2_code"] = r["sw_l2_code"]
        tmp["sw_l2_index_code"] = r["sw_l2_index_code"]
        pieces.append(tmp)
    if not pieces:
        rows["sw_l2_code"] = pd.NA
        rows["sw_l2_index_code"] = pd.NA
        return rows
    return pd.concat(pieces, ignore_index=True).drop_duplicates("trade_date", keep="last")


def compact_stock_frame(df: pd.DataFrame, downcast_float32: bool) -> pd.DataFrame:
    """Reduce IPC/memory size before returning or writing one-stock features."""
    out = df.copy()
    for col in out.columns:
        if col in ("ts_code", "trade_date", "sw_l2_code", "sw_l2_index_code"):
            continue
        if pd.api.types.is_float_dtype(out[col]):
            if downcast_float32:
                out[col] = out[col].astype("float32")
        elif pd.api.types.is_integer_dtype(out[col]):
            cmin = out[col].min(skipna=True)
            cmax = out[col].max(skipna=True)
            if pd.isna(cmin) or pd.isna(cmax):
                continue
            if cmin >= 0 and cmax <= np.iinfo(np.uint8).max:
                out[col] = out[col].astype("uint8")
            elif cmin >= np.iinfo(np.int8).min and cmax <= np.iinfo(np.int8).max:
                out[col] = out[col].astype("int8")
            elif cmin >= np.iinfo(np.int16).min and cmax <= np.iinfo(np.int16).max:
                out[col] = out[col].astype("int16")
            elif cmin >= np.iinfo(np.int32).min and cmax <= np.iinfo(np.int32).max:
                out[col] = out[col].astype("int32")
    return out


def process_one_stock(
    stock_path: str,
    save_single_stock: bool,
    return_data: bool,
    downcast_float32: bool,
    out_dir: str,
    minute_feature_dir: str,
) -> dict:
    path = Path(stock_path)
    ts_code = stock_code_from_path(path)
    try:
        if G_CSI is None or G_MARKET is None or G_SW_FEAT is None or G_SW_MAPPING is None:
            raise RuntimeError("Worker shared feature tables are not initialized.")
        csi = G_CSI
        market = G_MARKET
        sw_feat = G_SW_FEAT
        sw_mapping = G_SW_MAPPING

        s = read_csv_any(path, dtype={"ts_code": str, "trade_date": str})
        s = s.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)
        s = filter_date_pandas(s, G_SOURCE_START_DATE, G_SOURCE_END_DATE)
        for c in s.columns:
            if c not in ("ts_code", "trade_date", "name", "first_time", "last_time", "limit"):
                s[c] = pd.to_numeric(s[c], errors="coerce")
        s = s[s["invalid4train"].fillna(1).astype(int) == 0].copy()
        required = ["open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount", "adj_factor"]
        s = s.dropna(subset=[c for c in required if c in s.columns]).copy()
        if len(s) < 80:
            return {"ts_code": ts_code, "status": "too_few_rows", "rows": len(s)}

        adj = s["adj_factor"].astype(float)
        s["open_adj"] = s["open"] * adj
        s["high_adj"] = s["high"] * adj
        s["low_adj"] = s["low"] * adj
        s["close_adj"] = s["close"] * adj
        if "vwap" in s.columns:
            s["vwap_adj"] = s["vwap"] * adj
        else:
            s["vwap_adj"] = safe_div(s["amount"], s["vol"]) * adj

        close = s["close_adj"]
        high = s["high_adj"]
        low = s["low_adj"]
        open_ = s["open_adj"]
        vwap = s["vwap_adj"]

        s["ret_1"] = s["pct_chg"] / 100.0
        close_log = np.log(close.replace(0, np.nan))
        for window in [2, 3, 5, 10, 20]:
            s[f"ret_{window}"] = close_log.diff(window)
        s["oc_ret"] = safe_div(close - open_, open_)
        s["close_to_high"] = safe_div(high - close, close)
        s["close_to_low"] = safe_div(close - low, close)
        s["vwap_close_ratio"] = safe_div(close - vwap, vwap)
        s["vwap_high_ratio"] = safe_div(high - vwap, vwap)
        s["vwap_low_ratio"] = np.minimum(safe_div(low - vwap, vwap), 0.0)
        vwap_log = np.log(vwap.replace(0, np.nan))
        s["vwap_log_ret_2"] = vwap_log.diff(2)
        s["vwap_log_ret_3"] = vwap_log.diff(3)
        s["vwap_log_ret_5"] = vwap_log.diff(5)
        s["vwap_log_ret_10"] = vwap_log.diff(10)
        s["vwap_log_ret_20"] = vwap_log.diff(20)
        s["ret_3_minus_ret_10"] = s["ret_3"] - s["ret_10"]
        s["ret_5_minus_ret_20"] = s["ret_5"] - s["ret_20"]
        s["vwap_log_ret_3_minus_10"] = s["vwap_log_ret_3"] - s["vwap_log_ret_10"]
        s["ret_1_minus_ret_5_avg"] = s["ret_1"] -  (s["ret_5"] / 5.0)
        s["vwap_z"] = zscore_safe(vwap, 60)
        s["vwap_close_z"] = zscore_safe(s["vwap_close_ratio"], 20)

        s["true_range_pct"] = (high - low).abs() / close
        s["true_range_pct_z"] = zscore_safe(s["true_range_pct"], 20)
        s["true_range_mean_5"] = s["true_range_pct"].rolling(5).mean()
        s["true_range_mean_10"] = s["true_range_pct"].rolling(10).mean()
        s["true_range_mean_20"] = s["true_range_pct"].rolling(20).mean()
        s["true_range_mean_5_z"] = zscore_safe(s["true_range_mean_5"], 60)
        s["true_range_mean_10_z"] = zscore_safe(s["true_range_mean_10"], 60)
        s["true_range_mean_20_z"] = zscore_safe(s["true_range_mean_20"], 60)
        s["ret_std_5"] = s["ret_1"].rolling(5).std(ddof=0)
        s["ret_std_10"] = s["ret_1"].rolling(10).std(ddof=0)
        s["ret_std_20"] = s["ret_1"].rolling(20).std(ddof=0)
        s["ret_std_5_z"] = zscore_safe(s["ret_std_5"], 60)
        s["ret_std_10_z"] = zscore_safe(s["ret_std_10"], 60)
        s["ret_std_20_z"] = zscore_safe(s["ret_std_20"], 60)
        s["vwap_imbalance"] = safe_div((close - vwap) / close, s["true_range_mean_20"])

        s["vol_z"] = zscore_safe(s["vol"], 60)
        s["amount_z"] = zscore_safe(s["amount"], 60)
        log_vol = np.log(s["vol"].replace(0, np.nan))
        log_amt = np.log(s["amount"].replace(0, np.nan))
        s["vol_rv_5_z"] = zscore_safe(log_vol.rolling(5).std(ddof=0), 60)
        s["vol_rv_20_z"] = zscore_safe(log_vol.rolling(20).std(ddof=0), 60)
        s["amt_rv_5_z"] = zscore_safe(log_amt.rolling(5).std(ddof=0), 60)
        s["amt_rv_20_z"] = zscore_safe(log_amt.rolling(20).std(ddof=0), 60)

        t_log = np.log1p(s["turnover_rate_f"].clip(lower=0.0))
        s["turnover_f_log"] = t_log
        s["turnover_z_20"] = (t_log - t_log.rolling(20).mean().shift(1)) / (t_log.rolling(20).std(ddof=0).shift(1) + EPS)
        s["turnover_surprise_10"] = t_log - t_log.rolling(10).mean().shift(1)
        s["turnover_prank_60"] = rolling_prank_past(t_log, 60)
        trend = t_log.rolling(5).mean().shift(1) - t_log.rolling(20).mean().shift(1)
        s["turnover_trend_5_20"] = trend
        s["turnover_accel_5_20"] = trend.diff(1)
        s["vol_comp_z"] = s["ret_std_5_z"] - s["ret_std_20_z"]
        s["turnover_in_compression"] = s["turnover_surprise_10"] * (-s["vol_comp_z"])
        s["pressure_build"] = s["turnover_trend_5_20"] * (-s["vol_comp_z"])
        s["exhaustion_risk"] = s["turnover_z_20"] * np.maximum(s["vol_comp_z"], 0.0)
        s["turnover_range_pressure"] = s["turnover_trend_5_20"] * (-(s["true_range_mean_5_z"] - s["true_range_mean_20_z"]))

        s["mf_elg_rate"] = s.get("buy_elg_amount_rate") * 0.01
        s["mf_lg_rate"] = s.get("buy_lg_amount_rate") * 0.01
        s["mf_md_rate"] = s.get("buy_md_amount_rate") * 0.01
        s["mf_sm_rate"] = s.get("buy_sm_amount_rate") * 0.01
        s["mf_inst_rate"] = s["mf_elg_rate"] + s["mf_lg_rate"]
        for col in ["mf_inst_rate", "mf_elg_rate", "mf_lg_rate", "mf_md_rate", "mf_sm_rate"]:
            s[f"{col}_z"] = zscore_safe(s[col], 20)
        s["mf_price_align_z_20"] = zscore_safe(s["mf_inst_rate"] * s["ret_1"], 20)
        s["mf_available"] = s["mf_inst_rate"].notna().astype("int8")

        auc_o_open = pd.to_numeric(s.get("open_auction_o", pd.Series(np.nan, index=s.index)), errors="coerce") * adj
        auc_o_close = pd.to_numeric(s.get("close_auction_o", pd.Series(np.nan, index=s.index)), errors="coerce") * adj
        auc_o_high = pd.to_numeric(s.get("high_auction_o", pd.Series(np.nan, index=s.index)), errors="coerce") * adj
        auc_o_low = pd.to_numeric(s.get("low_auction_o", pd.Series(np.nan, index=s.index)), errors="coerce") * adj
        auc_o_vwap = pd.to_numeric(s.get("vwap_auction_o", pd.Series(np.nan, index=s.index)), errors="coerce") * adj
        auc_o_amount = pd.to_numeric(s.get("amount_auction_o", pd.Series(np.nan, index=s.index)), errors="coerce")
        auc_o_vol = pd.to_numeric(s.get("vol_auction_o", pd.Series(np.nan, index=s.index)), errors="coerce")

        auc_c_open = pd.to_numeric(s.get("open_auction_c", pd.Series(np.nan, index=s.index)), errors="coerce") * adj
        auc_c_close = pd.to_numeric(s.get("close_auction_c", pd.Series(np.nan, index=s.index)), errors="coerce") * adj
        auc_c_high = pd.to_numeric(s.get("high_auction_c", pd.Series(np.nan, index=s.index)), errors="coerce") * adj
        auc_c_low = pd.to_numeric(s.get("low_auction_c", pd.Series(np.nan, index=s.index)), errors="coerce") * adj
        auc_c_vwap = pd.to_numeric(s.get("vwap_auction_c", pd.Series(np.nan, index=s.index)), errors="coerce") * adj
        auc_c_amount = pd.to_numeric(s.get("amount_auction_c", pd.Series(np.nan, index=s.index)), errors="coerce")
        auc_c_vol = pd.to_numeric(s.get("vol_auction_c", pd.Series(np.nan, index=s.index)), errors="coerce")

        auc_o_range_abs = auc_o_high - auc_o_low
        auc_c_range_abs = auc_c_high - auc_c_low
        s["auc_o_ret"] = safe_div(auc_o_close - auc_o_open, auc_o_open)
        s["auc_o_range"] = safe_div(auc_o_range_abs, auc_o_open)
        s["auc_o_vwap_ratio"] = safe_div(auc_o_vwap - vwap, vwap)
        s["auc_o_amt_ratio"] = safe_div(auc_o_amount, s["amount"])
        s["auc_o_vol_ratio"] = safe_div(auc_o_vol, s["vol"])
        s["auc_o_close_position"] = safe_div(auc_o_close - auc_o_low, auc_o_range_abs)
        s["auc_o_efficiency"] = safe_div((auc_o_close - auc_o_open).abs(), auc_o_range_abs)
        s["auc_o_vs_open"] = safe_div(auc_o_close - open_, open_)

        s["auc_c_ret"] = safe_div(auc_c_close - auc_c_open, auc_c_open)
        s["auc_c_range"] = safe_div(auc_c_range_abs, auc_c_open)
        s["auc_c_vwap_ratio"] = safe_div(auc_c_vwap - vwap, vwap)
        s["auc_c_amt_ratio"] = safe_div(auc_c_amount, s["amount"])
        s["auc_c_vol_ratio"] = safe_div(auc_c_vol, s["vol"])
        s["auc_c_close_position"] = safe_div(auc_c_close - auc_c_low, auc_c_range_abs)
        s["auc_c_efficiency"] = safe_div((auc_c_close - auc_c_open).abs(), auc_c_range_abs)
        s["auc_c_vs_close"] = safe_div(auc_c_close - close, close)
        s["auction_reversal"] = -(s["auc_o_ret"] * s["auc_c_ret"])
        s["total_mv_z"] = zscore_safe(s["total_mv"], 120)
        s["circ_mv_z"] = zscore_safe(s["circ_mv"], 120)
# replace chip data
#        for col in ["cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", "cost_95pct", "weight_avg"]:
#            s[f"{col}_z"] = zscore_safe(s[col], 120)
#        s["chip_density"] = safe_div(s["cost_95pct"] - s["cost_5pct"], s["cost_50pct"])
#        s["chip_pressure"] = safe_div(s["cost_50pct"] * adj - s["close_adj"], s["close_adj"])
#        s["crowdedness"] = (s["winner_rate"] / 100.0 - 0.5) * s["chip_density"]
# --------------------------------------------------
# Chip distribution
# --------------------------------------------------

        cyq_raw_cols = [
            "winner_rate",
            "cost_5pct",
            "cost_15pct",
            "cost_50pct",
            "cost_85pct",
            "cost_95pct",
            "weight_avg",
        ]

        for col in cyq_raw_cols:
            if col in s.columns:
                s[col] = pd.to_numeric(s[col], errors="coerce")

        for col in ["cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", "cost_95pct", "weight_avg"]:
            s[f"{col}_z"] = zscore_safe(s[col], 120)

        cost_50_safe = s["cost_50pct"].replace(0, np.nan)
        close_adj_safe = s["close_adj"].replace(0, np.nan)

        s["chip_density"] = (
            (s["cost_95pct"] - s["cost_5pct"])
            / (cost_50_safe + EPS)
        )

        s["chip_pressure"] = (
            (s["cost_50pct"] * adj - s["close_adj"])
            / (close_adj_safe + EPS)
        )

        s["crowdedness"] = (
            (s["winner_rate"] / 100.0 - 0.5)
            * s["chip_density"]
        )
# chip data code replaced. 
        
        for col in ["rzye", "rqye", "rzmre", "rqyl", "rzche", "rqchl", "rqmcl", "rzrqye"]:
            s[f"{col}_z"] = zscore_safe(s[col], 60)
        s["margin_pressure"] = safe_div(s["rzmre"], s["float_share"])
        s["short_pressure"] = safe_div(s["rqyl"], s["float_share"])
        s["net_margin_flow"] = safe_div(s["rzmre"] - s["rzche"], s["float_share"])
        limit_path = LIMIT_DIR / f"{ts_code}.limit.csv"
        limit_source_available = limit_path.exists() and limit_path.stat().st_size > 0
        s = add_limit_features(s, limit_source_available)

        s = s.merge(csi, on="trade_date", how="left")
        s = s.merge(market, on="trade_date", how="left")
        sw_map = time_aware_sw_mapping(ts_code, s["trade_date"], sw_mapping)
        s = s.merge(sw_map, on="trade_date", how="left", suffixes=("", "_map"))
        s = s.merge(sw_feat, on=["sw_l2_index_code", "trade_date"], how="left", suffixes=("", "_sw"))

        minute = read_minute_features(ts_code, minute_feature_dir)
        if not minute.empty:
            s = s.merge(minute, on=["ts_code", "trade_date"], how="left")

        s["ret_1_rel_csi1500_ew"] = s["ret_1"] - s["csi1500_ew_ret"]
        s["ret_1_rel_csi1500_mcap"] = s["ret_1"] - s["csi1500_mcap_ret"]
        s["ret_2_rel_csi1500_ew"] = s["ret_2"] - s["csi1500_ew_mom_2"]
        s["ret_2_rel_csi1500_mcap"] = s["ret_2"] - s["csi1500_mcap_mom_2"]
        s["ret_3_rel_csi1500_ew"] = s["ret_3"] - s["csi1500_ew_mom_3"]
        s["ret_3_rel_csi1500_mcap"] = s["ret_3"] - s["csi1500_mcap_mom_3"]
        s["ret_10_rel_csi1500_ew"] = s["ret_10"] - s["csi1500_ew_mom_10"]
        s["ret_10_rel_csi1500_mcap"] = s["ret_10"] - s["csi1500_mcap_mom_10"]
        s["oc_ret_rel_csi1500_ew"] = s["oc_ret"] - s["csi1500_ew_oc_ret"]
        s["oc_ret_rel_csi1500_mcap"] = s["oc_ret"] - s["csi1500_mcap_oc_ret"]
        s["ret_5_rel_csi1500_ew"] = s["ret_5"] - s["csi1500_ew_mom_5"]
        s["ret_5_rel_csi1500_mcap"] = s["ret_5"] - s["csi1500_mcap_mom_5"]
        s["ret_20_rel_csi1500_ew"] = s["ret_20"] - s["csi1500_ew_mom_20"]
        s["ret_20_rel_csi1500_mcap"] = s["ret_20"] - s["csi1500_mcap_mom_20"]
        s["beta_to_csi1500_ew_20d"] = s["ret_1"].rolling(20).cov(s["csi1500_ew_ret"]) / (s["csi1500_ew_ret"].rolling(20).var() + EPS)
        s["beta_to_csi1500_mcap_20d"] = s["ret_1"].rolling(20).cov(s["csi1500_mcap_ret"]) / (s["csi1500_mcap_ret"].rolling(20).var() + EPS)
  #      s["vol_over_csi1500_ew_vol"] = safe_div(s["ret_std_20"], s["csi1500_ew_ret_std_20"])
  #      s["vol_over_csi1500_mcap_vol"] = safe_div(s["ret_std_20"], s["csi1500_mcap_ret_std_20"])

        # Relative volatility ratios
        # Make sure column exists even if no legacy 5-minute data was merged.
        if "realized_vol_5m" not in s.columns:
            s["realized_vol_5m"] = np.nan

        vol_over_csi1500_ew = safe_div(s["realized_vol_5m"], s["csi1500_ew_ret_std_20"],)

        vol_over_csi1500_mcap = safe_div(s["realized_vol_5m"], s["csi1500_mcap_ret_std_20"],)

        vol_over_sw_l2 = safe_div(s["realized_vol_5m"],s["sw_l2_ret_std_20"],)

        # Log-transform to reduce right skew.
        # Values are strictly positive, so log() is safe after clipping.
        rv5 = pd.to_numeric(s["realized_vol_5m"], errors="coerce")

        s["vol_over_csi1500_ew_vol"] = (
            np.log1p(rv5) - np.log1p(pd.to_numeric(s["csi1500_ew_ret_std_20"], errors="coerce"))
        )

        s["vol_over_csi1500_mcap_vol"] = (
            np.log1p(rv5) - np.log1p(pd.to_numeric(s["csi1500_mcap_ret_std_20"], errors="coerce"))
        )

        s["vol_over_sw_l2_vol"] = (
            np.log1p(rv5) - np.log1p(pd.to_numeric(s["sw_l2_ret_std_20"], errors="coerce"))
        )
        
        s["ret_1_over_mkt_gt5_rate"] = safe_div(s["ret_1"], s["mkt_gt5_rate"])
        # new
        mkt_gt5 = pd.to_numeric(s["mkt_gt5_rate"], errors="coerce").clip(0.0, 1.0)
        s["ret_1_in_weak_mkt"] = (pd.to_numeric(s["ret_1"], errors="coerce") * (1.0 - mkt_gt5))

        s["ret_vol_confirm"] = s["ret_1"] * s["vol_z"]
        s["vol_expand_ret"] = s["ret_1"] * s["volume_ratio"].clip(lower=0, upper=20)
        s["mf_inst_rate_rel_mkt_z20"] = zscore_safe(s["mf_inst_rate"] - s["mkt_mf_inst_rate"], 20)

        s["ret_1_rel_sw_l2"] = s["ret_1"] - s["sw_l2_ret_1"]
        s["ret_2_rel_sw_l2"] = s["ret_2"] - s["sw_l2_mom_2"]
        s["ret_3_rel_sw_l2"] = s["ret_3"] - s["sw_l2_mom_3"]
        s["ret_10_rel_sw_l2"] = s["ret_10"] - s["sw_l2_mom_10"]
        s["oc_ret_rel_sw_l2"] = s["oc_ret"] - s["sw_l2_oc_ret"]
        s["ret_5_rel_sw_l2"] = s["ret_5"] - s["sw_l2_mom_5"]
        s["ret_20_rel_sw_l2"] = s["ret_20"] - s["sw_l2_mom_20"]
        s["beta_to_sw_l2_20d"] = s["ret_1"].rolling(20).cov(s["sw_l2_ret_1"]) / (s["sw_l2_ret_1"].rolling(20).var() + EPS)
  #      s["vol_over_sw_l2_vol"] = safe_div(s["ret_std_20"], s["sw_l2_ret_std_20"])
  #      s["stock_ind_strength_5"] = s["ret_5_rel_sw_l2"]
  #      s["stock_ind_strength_20"] = s["ret_20_rel_sw_l2"]

        lagged_cols = {}
        for col, lags in BASE_FEATURE_LAGS.items():
            if col in s.columns:
                for lag in lags:
                    lagged_cols[f"{col}_lag{lag}"] = s[col].shift(lag)
        if lagged_cols:
            s = pd.concat([s, pd.DataFrame(lagged_cols, index=s.index)], axis=1).copy()

        keep_non_numeric = ["ts_code", "trade_date", "sw_l2_code", "sw_l2_index_code"]
        feature_cols = [c for c in s.columns if c in set(BASE_FEATURE_LAGS) or any(c.startswith(f"{b}_lag") for b in BASE_FEATURE_LAGS)]
        raw_for_cs = [
            "ret_1", "oc_ret", "vwap_log_ret_5", "vol", "amount", "turnover_f_log",
            "mf_inst_rate", "mf_price_align_z_20", "vwap_close_ratio", "close_to_high",
            "winner_rate", "circ_mv",
        ]
        keep = list(dict.fromkeys(keep_non_numeric + raw_for_cs + feature_cols))
        out = compact_stock_frame(s[[c for c in keep if c in s.columns]].copy(), downcast_float32)
        out = filter_date_pandas(out, G_OUTPUT_START_DATE, G_OUTPUT_END_DATE)
        if out.empty:
            return {"ts_code": ts_code, "status": "no_output_rows", "rows": 0, "output_file": ""}
        output_file = ""
        if save_single_stock:
            out_path = Path(out_dir) / f"{ts_code}.parquet"
            out.to_parquet(out_path, index=False)
            output_file = str(out_path)
        result = {"ts_code": ts_code, "status": "saved", "rows": len(out), "output_file": output_file}
        if return_data:
            result["data"] = out
        return result
    except Exception as exc:
        return {"ts_code": ts_code, "status": "error", "rows": 0, "error": repr(exc)}


def process_stock_chunk(
    chunk_id: int,
    stock_paths: list[str],
    save_single_stock: bool,
    return_data: bool,
    downcast_float32: bool,
    single_out_dir: str,
    chunk_out_dir: str,
    minute_feature_dir: str,
) -> dict:
    results = []
    frames = []
    for i, stock_path in enumerate(stock_paths, 1):
        res = process_one_stock(
            stock_path,
            save_single_stock,
            return_data,
            downcast_float32,
            single_out_dir,
            minute_feature_dir,
        )
        data = res.pop("data", None)
        if data is not None:
            frames.append(pl.from_pandas(data))
        results.append(res)
        err = res.get("error")
        err_msg = f" error={err}" if err else ""
        print(
            f"[chunk {chunk_id} {i}/{len(stock_paths)}] "
            f"{res.get('ts_code')} {res.get('status')} rows={res.get('rows')}{err_msg}",
            flush=True,
        )

    chunk_file = ""
    if return_data and frames:
        chunk_path = Path(chunk_out_dir) / f"chunk_{chunk_id:03d}.parquet"
        pl.concat(frames, how="diagonal_relaxed").write_parquet(chunk_path)
        chunk_file = str(chunk_path)
        frames.clear()

    saved = sum(1 for r in results if r.get("status") == "saved")
    return {
        "results": results,
        "chunk_file": chunk_file,
        "stock_count": len(stock_paths),
        "saved_count": saved,
        "row_count": sum(int(r.get("rows", 0) or 0) for r in results),
    }


def split_evenly(items: list[Path], chunks: int) -> list[list[Path]]:
    chunks = max(1, min(chunks, len(items)))
    out = [[] for _ in range(chunks)]
    for i, item in enumerate(items):
        out[i % chunks].append(item)
    return [x for x in out if x]


def add_cross_sectional_features(panel: pl.DataFrame, sw_rank: pl.DataFrame) -> pl.DataFrame:
    rank_specs = [
        ("ret_1", "ret_1_cs_rank"),
        ("ret_1", "ret_1_prank_1500"),
        ("ret_2", "ret_2_prank_1500"),
        ("ret_3", "ret_3_prank_1500"),
        ("ret_5", "ret_5_prank_1500"),
        ("ret_10", "ret_10_prank_1500"),
        ("vol", "vol_prank_1500"),
        ("oc_ret", "oc_ret_cs_rank"),
        ("amount", "amount_cs_rank"),
        ("amount", "amount_prank_1500"),
        ("turnover_f_log", "turnover_f_cs_rank"),
        ("turnover_f_log", "turnover_prank_1500"),
        ("mf_inst_rate", "mf_inst_rate_cs_rank"),
        ("mf_inst_rate", "mf_inst_rate_prank_1500"),
        ("vwap_close_ratio", "vwap_close_ratio_prank_1500"),
        ("mf_price_align_z_20", "mf_price_align_prank_1500"),
        ("close_to_high", "close_to_high_cs_rank"),
        ("winner_rate", "winner_rate_cs_rank"),
        ("circ_mv", "circ_mv_cs_rank"),
        ("circ_mv", "mv_prank_1500"),
    ]
    exprs = []
    for src, dst in rank_specs:
        if src in panel.columns:
            exprs.append((pl.col(src).rank(method="average").over("trade_date") / pl.len().over("trade_date")).alias(dst))
    rel_specs = {
        "vol": "vol_rel_1500",
        "amount": "amount_rel_1500",
        "turnover_f_log": "turnover_rel_1500",
    }
    for src, dst in rel_specs.items():
        if src in panel.columns:
            denom = pl.col(src).mean().over("trade_date")
            exprs.append(
                pl.when(denom.abs() > EPS)
                .then(pl.col(src) / denom)
                .otherwise(None)
                .alias(dst)
            )
    if "mf_inst_rate" in panel.columns:
        exprs.append((pl.col("mf_inst_rate") - pl.col("mf_inst_rate").mean().over("trade_date")).alias("mf_inst_rate_rel_1500"))
        exprs.append(
            (pl.col("mf_inst_rate") - pl.col("mf_inst_rate").mean().over(["trade_date", "sw_l2_code"]))
            .alias("mf_inst_rate_rel_sw_l2")
        )
    if "turnover_f_log" in panel.columns:
        exprs.append(
            (pl.col("turnover_f_log") - pl.col("turnover_f_log").mean().over(["trade_date", "sw_l2_code"]))
            .alias("turnover_rel_sw_l2")
        )
    if exprs:
        panel = panel.with_columns(exprs)

    panel = panel.join(sw_rank, on=["sw_l2_code", "trade_date"], how="left")

    panel = panel.sort(["ts_code", "trade_date"])
    lag_exprs = []
    for col, lags in CROSS_SECTIONAL_LAGS.items():
        if col in panel.columns:
            for lag in lags:
                lag_exprs.append(pl.col(col).shift(lag).over("ts_code").alias(f"{col}_lag{lag}"))
    if lag_exprs:
        panel = panel.with_columns(lag_exprs)
    return panel


def save_feature_dictionary(path: Path, columns: Iterable[str]) -> None:
    rows = []
    non_features = {
        "ts_code", "trade_date", "sw_l2_index_code",
    }
    for c in columns:
        if c in non_features:
            role = "non_feature"
        elif c in {"sw_l2_code", "is_20cm"}:
            role = "categorical_feature"
        else:
            role = "numeric_feature"
        rows.append({"column": c, "role": role})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf_8_sig")


def main() -> None:
    global OUT_DIR, SINGLE_DIR, CHUNK_DIR, REPORT_DIR, FINAL_PATH, PANEL_NO_CS_PATH, FEATURE_DICT_PATH
    global G_SOURCE_START_DATE, G_SOURCE_END_DATE, G_OUTPUT_START_DATE, G_OUTPUT_END_DATE
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stocks", type=int, default=None, help="Process only the first N eligible stocks.")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--clean", action="store_true", help="Remove existing processed/train_v5b before running.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory. Defaults to processed/train_v5b.",
    )
    parser.add_argument(
        "--output-name",
        default="train_v5b.parquet",
        help="Final parquet file name inside --output-dir.",
    )
    parser.add_argument(
        "--prediction-date",
        default=None,
        help="YYYYMMDD shorthand for prediction build. Sets source end and output date unless explicitly overridden.",
    )
    parser.add_argument("--source-start-date", default=None, help="YYYYMMDD inclusive warmup/source start date.")
    parser.add_argument("--source-end-date", default=None, help="YYYYMMDD inclusive warmup/source end date.")
    parser.add_argument("--output-start-date", default=None, help="YYYYMMDD inclusive final output start date.")
    parser.add_argument("--output-end-date", default=None, help="YYYYMMDD inclusive final output end date.")
    parser.add_argument(
        "--merge-mode",
        choices=["memory", "disk", "chunked"],
        default="memory",
        help=(
            "memory returns each stock frame; disk writes per-stock parquet and scans them; "
            "chunked lets each worker merge its assigned stock frames and write one chunk parquet."
        ),
    )
    parser.add_argument(
        "--downcast-float32",
        action="store_true",
        help="Optionally downcast float columns to float32 before worker return/write. Default keeps float64.",
    )
    parser.add_argument(
        "--save-single-stock",
        action="store_true",
        help="Also save per-stock feature parquet files for debugging/recovery.",
    )
    parser.add_argument(
        "--minute-feature-dir",
        type=Path,
        default=MINUTE_FEATURE_DIR,
        help="Directory containing prebuilt per-stock 1-minute feature parquet files, or a panel parquet file.",
    )
    parser.add_argument(
        "--minute-feature-panel",
        type=Path,
        default=None,
        help="Optional single panel parquet from build_minute_features_v5b.py --output-mode chunked/memory.",
    )
    args = parser.parse_args()

    OUT_DIR = args.output_dir
    SINGLE_DIR = OUT_DIR / "single_stock_features"
    CHUNK_DIR = OUT_DIR / "chunk_features"
    REPORT_DIR = OUT_DIR / "report"
    FINAL_PATH = OUT_DIR / args.output_name
    PANEL_NO_CS_PATH = OUT_DIR / "panel_no_cs.parquet"
    FEATURE_DICT_PATH = OUT_DIR / "v5b_feature_dictionary.csv"

    prediction_date = normalize_date_arg(args.prediction_date)
    source_start_date = normalize_date_arg(args.source_start_date)
    source_end_date = normalize_date_arg(args.source_end_date) or prediction_date
    output_start_date = normalize_date_arg(args.output_start_date) or prediction_date
    output_end_date = normalize_date_arg(args.output_end_date) or output_start_date
    G_SOURCE_START_DATE = source_start_date
    G_SOURCE_END_DATE = source_end_date
    G_OUTPUT_START_DATE = output_start_date
    G_OUTPUT_END_DATE = output_end_date

    minute_feature_source = args.minute_feature_panel or args.minute_feature_dir

    if args.clean and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    effective_save_single = args.save_single_stock or args.merge_mode == "disk"
    return_stock_data = args.merge_mode in ("memory", "chunked")
    if effective_save_single:
        SINGLE_DIR.mkdir(parents=True, exist_ok=True)
    if args.merge_mode == "chunked":
        CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    stock_files = preflight_stock_filter(args.max_stocks)
    print(f"[preflight] processing stocks: {len(stock_files)}", flush=True)

    csi = prepare_csi1500_features()
    market = prepare_market_features()
    sw_feat, sw_rank = prepare_sw_l2_features(csi)
    csi = filter_date_pandas(csi, source_start_date, source_end_date)
    market = filter_date_pandas(market, source_start_date, source_end_date)
    sw_feat = filter_date_pandas(sw_feat, source_start_date, source_end_date)
    sw_rank = filter_date_pandas(sw_rank, source_start_date, source_end_date)
    sw_mapping = read_csv_any(SW_L2_MAPPING_PATH, dtype=str)
    print(
        "[date-window] "
        f"source_start={source_start_date or 'min'} source_end={source_end_date or 'max'} "
        f"output_start={output_start_date or 'min'} output_end={output_end_date or 'max'}",
        flush=True,
    )

    results = []
    stock_frames = []
    chunk_files = []
    if args.merge_mode == "chunked" and args.workers > 1:
        chunks = split_evenly(stock_files, args.workers)
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(
                csi,
                market,
                sw_feat,
                sw_mapping,
                str(minute_feature_source),
                source_start_date,
                source_end_date,
                output_start_date,
                output_end_date,
            ),
        ) as ex:
            futs = [
                ex.submit(
                    process_stock_chunk,
                    chunk_id,
                    [str(path) for path in chunk],
                    effective_save_single,
                    return_stock_data,
                    args.downcast_float32,
                    str(SINGLE_DIR),
                    str(CHUNK_DIR),
                    str(minute_feature_source),
                )
                for chunk_id, chunk in enumerate(chunks, 1)
            ]
            done_stocks = 0
            for i, fut in enumerate(as_completed(futs), 1):
                chunk_res = fut.result()
                chunk_file = chunk_res.get("chunk_file")
                if chunk_file:
                    chunk_files.append(Path(chunk_file))
                results.extend(chunk_res["results"])
                done_stocks += chunk_res["stock_count"]
                print(
                    f"[chunk {i}/{len(futs)}] stocks={chunk_res['stock_count']} "
                    f"saved={chunk_res['saved_count']} rows={chunk_res['row_count']} "
                    f"done_stocks={done_stocks}/{len(stock_files)}",
                    flush=True,
                )
    elif args.workers <= 1:
        init_worker(
            csi,
            market,
            sw_feat,
            sw_mapping,
            str(minute_feature_source),
            source_start_date,
            source_end_date,
            output_start_date,
            output_end_date,
        )
        for i, path in enumerate(stock_files, 1):
            res = process_one_stock(
                str(path),
                effective_save_single,
                return_stock_data,
                args.downcast_float32,
                str(SINGLE_DIR),
                str(minute_feature_source),
            )
            data = res.pop("data", None)
            if data is not None:
                stock_frames.append(pl.from_pandas(data))
            results.append(res)
            err = res.get("error")
            err_msg = f" error={err}" if err else ""
            print(
                f"[{i}/{len(stock_files)}] {res.get('ts_code')} {res.get('status')} rows={res.get('rows')}{err_msg}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(
                csi,
                market,
                sw_feat,
                sw_mapping,
                str(minute_feature_source),
                source_start_date,
                source_end_date,
                output_start_date,
                output_end_date,
            ),
        ) as ex:
            futs = [
                ex.submit(
                    process_one_stock,
                    str(path),
                    effective_save_single,
                    return_stock_data,
                    args.downcast_float32,
                    str(SINGLE_DIR),
                    str(minute_feature_source),
                )
                for path in stock_files
            ]
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                data = res.pop("data", None)
                if data is not None:
                    stock_frames.append(pl.from_pandas(data))
                results.append(res)
                err = res.get("error")
                err_msg = f" error={err}" if err else ""
                print(
                    f"[{i}/{len(futs)}] {res.get('ts_code')} {res.get('status')} rows={res.get('rows')}{err_msg}",
                    flush=True,
                )

    pd.DataFrame(results).to_csv(REPORT_DIR / "single_stock_processing_summary.csv", index=False, encoding="utf_8_sig")
    saved_count = sum(1 for r in results if r.get("status") == "saved")
    if args.merge_mode == "disk":
        saved_files = [Path(r["output_file"]) for r in results if r.get("status") == "saved" and r.get("output_file")]
        if not saved_files:
            raise RuntimeError("No single-stock parquet files were produced.")
        panel = pl.concat([pl.scan_parquet(p) for p in saved_files], how="diagonal_relaxed").collect()
    elif args.merge_mode == "chunked" and args.workers > 1:
        if not chunk_files:
            raise RuntimeError("No chunk parquet files were produced.")
        panel = pl.concat([pl.scan_parquet(p) for p in chunk_files], how="diagonal_relaxed").collect()
    else:
        if not stock_frames:
            raise RuntimeError("No single-stock feature frames were produced.")
        panel = pl.concat(stock_frames, how="diagonal_relaxed")
    panel_rows_before_cs = panel.height
    panel.write_parquet(PANEL_NO_CS_PATH)
    # Stable final order + permanent integer key
    panel = (
        panel
        .sort(["trade_date", "ts_code"])
        .with_row_index(name="sample_id", offset=0)
    )
    panel = add_cross_sectional_features(panel, pl.from_pandas(sw_rank))
    panel.write_parquet(FINAL_PATH)
    save_feature_dictionary(FEATURE_DICT_PATH, panel.columns)

    elapsed_sec = time.perf_counter() - t0
    summary = {
        "processed_stock_count": len(stock_files),
        "single_stock_success_count": saved_count,
        "single_stock_failure_count": len(results) - saved_count,
        "panel_rows_before_cs": panel_rows_before_cs,
        "final_rows": panel.height,
        "date_min": panel.select(pl.col("trade_date").min()).item(),
        "date_max": panel.select(pl.col("trade_date").max()).item(),
        "column_count": len(panel.columns),
        "minute_feature_source": str(minute_feature_source),
        "save_single_stock": effective_save_single,
        "merge_mode": args.merge_mode,
        "downcast_float32": args.downcast_float32,
        "source_start_date": source_start_date,
        "source_end_date": source_end_date,
        "output_start_date": output_start_date,
        "output_end_date": output_end_date,
        "elapsed_seconds": round(elapsed_sec, 3),
        "elapsed_minutes": round(elapsed_sec / 60.0, 3),
        "final_path": str(FINAL_PATH),
    }
    pd.DataFrame([summary]).to_csv(REPORT_DIR / "v5b_build_summary.csv", index=False, encoding="utf_8_sig")
    print("[done]", flush=True)
    print(f"[time] elapsed_seconds={elapsed_sec:.3f} elapsed_minutes={elapsed_sec / 60.0:.3f}", flush=True)
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
