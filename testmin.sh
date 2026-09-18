python assign_minute_buckets.py \
  --input csi1500con.csv \
  --inplace

python build_minute_raw_buckets_from_stock_files.py \
  --source-dir data/raw \
  --out-dir data/minute_raw_buckets \
  --report-dir processed/minute_raw_buckets_report \
  --start-date 20250801 \
  --end-date 20260713 \
  --overwrite

python 01_upday_minute_buckets_and_features.py \
  --end-upday 20260713 \
  --lookback-trade-days 30 \
  --source-start-date 20250801 \
  --download-workers 4

python prepare_training_v5b_0714.py \
  --prediction-date 20260713 \
  --source-start-date 20250801 \
  --merge-mode chunked \
  --workers 8

