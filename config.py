import tushare as ts
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path

proj_root = Path(__file__).resolve().parent
config_dir = proj_root / "config"

TUSHARE_TOKEN = '6d0701013b3c0d193064f94c0d71d0a115171bc34462d012ea6c002b'

dev_dir = Path(__file__).parent

STOCK_DATA_DIR       = Path(__file__).parent / "processed/daily"
STOCK905_DATA_DIR       = Path(__file__).parent / "processed/daily/905"
STOCK_INDEX_DIR  = Path(__file__).parent / "processed/index"
# Input CSV paths
CSI985_CSV = STOCK_INDEX_DIR / "csi985.csv"     # edit as needed
CSI300_CSV = STOCK_INDEX_DIR / "csi300.csv"     # optional
MKTDC_CSV = STOCK_INDEX_DIR / "mktdc.csv" 
CSI905_CSV = STOCK_INDEX_DIR / "csi905.csv"

# Output processed paths
INDEX_DIR    = Path(__file__).parent / "processed/index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CSI300_UNIVERSE_PATH = Path(__file__).parent / "CSI300cons.csv"
CSI985_UNIVERSE_PATH = Path(__file__).parent / "CSI985cons.csv"
CSI985_OUT = INDEX_DIR / "csi985_out.csv"
CSI300_OUT = INDEX_DIR / "csi300_out.csv"   # optional
CSI905_UNIVERSE_PATH = Path(__file__).parent / "CSI905cons.csv"
CSI905_OUT = INDEX_DIR / "csi905_out.csv"

MARKET_PANEL_CSV = INDEX_DIR / "market_panel.csv"

# MARKET_CTX = OUT_INDEX_DIR / "csi985+300_merged.csv"  # optional merged context
#INDEX_MERGED    = Path("./processed/index/csi985+300_merged.csv")   # merged 985+300 raw
INDEX_CONTEXT    = Path(__file__).parent /"processed/index/csi300+985_context.csv"  # optional save
OUT_TRAIN_PATH       = Path(__file__).parent / "processed/csi300_train_lag20.csv"

HIGH_RET_THRESHOLD = 0.08      # e.g. 8% future max-high over 5 days
HORIZON_DAYS      = 5
MAX_LAG           = 19         # 20-day history
# Label threshold for next-day strength
STRONG_THR = 0.02   # or 0.03, 0.015, etc.
THRESHOLD_REL = 0.05  # +2% outperformance vs market over T+1→T+2
EPS = 1e-8            # for safety in divisions

# Stock data directory: each file is one stock with Tushare daily + adj_factor + moneyflow
STOCK_DIR = STOCK_DATA_DIR
STOCK905_DIR = STOCK905_DATA_DIR
MERGED_DIR  = STOCK_DIR / "merged"              # where your per-stock merged CSVs are

ALPHA_THRESHOLD = 0.08   # 8%
eps = 1e-8

# RAW index / market files (adjust to your actual paths!)
CSI985_RAW_PATH = CSI985_CSV    # TODO: set your real path
CSI300_RAW_PATH = CSI300_CSV    # TODO: e.g. ./source/index/csi300.csv
MKTDC_PATH      = MKTDC_CSV    # TODO: the uploaded mktdc.csv path
CSI905_RAW_PATH = CSI905_CSV

# Output directory for per-stock lagged panels
OUT_PANEL_DIR = Path(__file__).parent / "processed/panels_v3"

signal_dir = proj_root / "processed/daily/train"
signal_file = signal_dir / "lgbm_signals.csv"
