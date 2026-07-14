# 1 define scope
define constituent stocks: csi500+csi1000
define training dataset date
define inference dataset date, start date is different than training start date

- generate csi1500con.csv
download_stock_scope_v2.py

- update csi1500con.csv, remove non-margin_details, then add 4 stocks manually
update_csi1500_universe_v5b.py

- update config_date.py， create historical start_date and refresh start_date


# 2 download market data
download_ashare_indexes.py
download_global_indexes.py, be careful the start_date is hard coded
download_hsgt_moneyflow.py
build_market_panel.py

# 2.1 sw_l2 data
StockClassifyUse_stock.xls get from offical website for stock-l3 mapping
sw_l2_si.csv, download l2-si code mapping from tushare
build_stock_sw_l2_mapping.py, generate stock_sw_l2_mapping_since_2023.csv in index folder, move file to project root directory
download_sw_l2_daily.py, download allday l2 index daily in one file, sw_l2_daily.csv in index folder

# 3 download stock data
prepare stock history intraday data
refresh stock intraday data
download stock history interday data
refresh stock intreday data
build csi1500 index 

download_csi1500_daily.py
download_csi1500_margin_detail.py
merge_csi1500_daily_polars.py



## Production daily interday upday

Run before market open to refresh recent non-margin stock daily source files.
The default window is the latest 10 open trade dates, inclusive of `config_date.end_date`.
This refreshes:

- stkfactor
- moneyflow
- cyq_perf
- auction_o
- auction_c
- limit

It downloads date-first from Tushare, keeps the downloaded data in memory, filters to
`csi1500con.csv`, then updates the per-stock CSV files under `processed/daily`.
Files are skipped when the downloaded rows match existing rows. Overlapping rows are
overwritten when changed. New rows are appended. For `limit`, stale rows inside the
refresh window are pruned when a stock no longer appears in the latest limit data.

```bash
python download_csi1500_daily_upday.py --workers 2
```

Manual window:

```bash
python download_csi1500_daily_upday.py --start-upday 20260625 --end-upday 20260701
```

Dry run:

```bash
python download_csi1500_daily_upday.py --dry-run
```

Margin detail is intentionally refreshed separately because it is published later.

## Production margin_detail upday

Run later in the day after Tushare publishes margin_detail. The default window is
also the latest 10 open trade dates, inclusive of `config_date.end_date`.

It downloads margin_detail date-first, keeps the downloaded data in memory, filters
to `csi1500con.csv`, then updates per-stock CSV files under
`processed/daily/margin_detail`. Files are skipped when the downloaded rows match
existing rows. Overlapping rows are overwritten when changed. New rows are appended.

```bash
python download_csi1500_margin_detail_upday.py
```

Manual window:

```bash
    python download_csi1500_margin_detail_upday.py --start-upday 20260625 --end-upday 20260701
```

Dry run:

```bash
python download_csi1500_margin_detail_upday.py --dry-run
```

# 3.1 csi1500 index
must do it after upday
python build_csi1500_custom_index.py



# 3.2 build min stock data 
download_1min.ipynb, audit_1min.ipynb is onetime downloader, work folder ../min_data/
python build_minute_features_v5b.py   --start_date 20250801 --output-mode by_stock   --workers 8   --overwrite 

below is not actual building panel. it convert per stock min data to per day min data. 
python build_minute_day_panel.py \
  --source-dir data/raw \
  --out-dir data/raw_minute_by_date \
  --con-file csi1500con.csv \
  --mode chunked \
  --stock-chunk-size 50 \
  --overwrite
   python download_1min_upday.py \
  --lookback-trade-days 30 \
  - workers 4

python audit_minute_day_panel.py \
  --input-dir data/raw_minute_by_date \
  --start-date 20260601 \
  --end-date 20260701 \
  --report-dir data/report/minute_day_panel_audit

python update_minute_features_panel_v5b.py \
  --end-upday \
  --lookback-trade-days 20 \
  --source-start-date 20250801 \
  --download-workers 4 \
  --feature-workers 8 \
  --feature-task-chunk-size 30 \
  --overwrite
  
  
1. build_minute_features_v5b.py
   input: data/raw per-stock 1min files
   output: processed/minute_features_v5b/by_stock/*.parquet

3. prepare_training_v5b.py
   input: daily/market/SW/margin/custom CSI1500 + per-stock minute feature files
   output: one prediction feature parquet for date T

4. production/h3m55d15_dual/generate_signals2.py
   input: prediction feature parquet for date T
   output: signals for execution on T+1

python build_minute_features_v5b.py \
  --start-date 20250801 \
  --output-mode by_stock \
  --workers 8 \
  --overwrite

python prepare_training_v5b.py \
  --merge-mode chunked \
  --workers 8 \
  --prediction-date 20260625 \
  --source-start-date 20250802 \
  --output-dir processed/predict_v5b \
  --output-name predict_features_20260625.parquet \
  --minute-feature-dir processed/minute_features_v5b/by_stock \
  --clean

python prepare_training_v5b.py \
  --merge-mode memory \
  --workers 8 \
  --output-dir processed/predict_v5b \
  --minute-feature-dir processed/minute_features_v5b/by_stock \
  --clean

python generate_signals_range.py \
  --input processed/predict_v5b/train_v5b.parquet \
  --start-date 20260708 \
  --end-date 20260710 \
  --out-dir signals/
  --save-ranked

build_minute_features_v5b.py
| Switch | Default | Meaning |
|---|---:|---|
| `--raw-dir` | `data/raw` | Input folder containing per-stock 1-minute parquet files, e.g. `data/raw/600004.SH.parquet`. |
| `--out-dir` | `processed/minute_features_v5b` | Root output folder. |
| `--stocks` | empty list | Optional stock list. If empty, process all parquet files in `--raw-dir`. Accepts `600004.SH` or `600004`. |
| `--workers` | `1` | Number of worker processes. |
| `--start-date` | `None` | Inclusive start date, `YYYYMMDD`. If omitted, use earliest data available. |
| `--end-date` | `None` | Inclusive end date, `YYYYMMDD`. If omitted, use latest data available. |
| `--save-panel` | `False` | With `by_stock`, also save a combined panel parquet. |
| `--output-mode` | `by_stock` | Output mode: `by_stock`, `chunked`, or `memory`. |
| `--stock-chunk-size` | `150` | Number of stocks per chunk when `--output-mode chunked`. |
| `--panel-name` | `minute_features_panel.parquet` | Panel parquet filename for `chunked`, `memory`, or `--save-panel`. |
| `--overwrite` | `False` | Overwrite existing per-stock output files. Without this, existing stock files may be skipped. |
| `--min-continuous-bars` | `180` | Minimum valid intraday bars required for a day to build features. |
| `--report-name` | `minute_feature_build_summary.csv` | Summary report filename under `--out-dir`. |

for training only: 
python prepare_training_v5b.py --merge-mode chunked --workers 8

python generate_targets.py

python prepare_dataset_v5b.py \
  --target-spec h3m55d15 \
  --oos-days 60 \
  --train-ratio 0.70 \
  --valid-ratio 0.15 \
  --test-ratio 0.15
# skip 4 prepare training dataset

# out-of-date 3.2 build 5min stock data 


  
# 4 prepare training dataset
generate training dataset
prepare_training_v5b.py --workers 8 --merge-mode chunked --clean
--merge-mode: chunked, memory, disk

low resource version, not useful. try above less worker with disk model
prepare_training_v5b_low_resource.py --workers 2 --clean
--skip-existing
--stage stock-ts, cs-by-date, stock-final, final-merge
--final-path

check training data quality
python.exe .\audit_v5b_feature_distribution.py --input processed\train_v5b\train_v5b.parquet --out-dir processed\train_v5b\report

prepare_dataset_v5b.py

python train_lgbm_v5b_binary.py --device cuda --num-boost-round 3000 --early-stopping-rounds 150  --learning-rate 0.03 --num-leaves 63   --min-data-in-leaf 200  --max-bin 63   --train-float32


# 5 train model
python train_lgbm_v5b_binary.py --device cuda --num-boost-round 3000 --early-stopping-rounds 150  --learning-rate 0.03 --num-leaves 63   --min-data-in-leaf 200  --max-bin 63   --train-float32


# 6 prepare 

## to be verified: 
# updated min upday and prepare training

one time assign minute buckets. ts_code xxx00-xxx04 to bucket0, xxx05-xxx09 to bucket1, etc. 
python assign_minute_buckets.py \
  --input csi1500con.csv \
  --inplace

one-time to build minute buckets
python build_minute_raw_buckets_from_stock_files.py \
  --source-dir data/raw \
  --out-dir data/minute_raw_buckets \
  --report-dir processed/minute_raw_buckets_report \
  --start-date 20260101 \
  --end-date 20260710 \
  --overwrite


python 01_upday_minute_buckets_and_features.py \
  --end-upday 20260710 \
  --lookback-trade-days 30 \
  --source-start-date 20260101 \
  --download-workers 4


Prefers new minute feature bucket files by default:processed/minute_feature_buckets/bucket_*.parquet
Uses csi1500con.csv / minute_bucket_file to find each stock’s bucket.
Loads minute feature buckets lazily inside each worker, then caches the bucket for reuse.

python prepare_training_v5b.py \
  --prediction-date 20260710 \
  --source-start-date 20260201 \
  --merge-mode chunked \
  --workers 8