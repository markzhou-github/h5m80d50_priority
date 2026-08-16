#!/usr/bin/env bash

TODAY=$(date +%Y%m%d)

LOG_DIR="processed/train_v5b/logs"
LOG_FILE="${LOG_DIR}/run_${TODAY}.log"
ERROR_LOG="${LOG_DIR}/error_${TODAY}.log"

mkdir -p "$LOG_DIR"

run_step() {
    step_name="$1"
    shift

    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" | tee -a "$LOG_FILE"
    echo "$step_name" | tee -a "$LOG_FILE"
    echo "Started at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" | tee -a "$LOG_FILE"

    start_time=$(date +%s)

    "$@" >> "$LOG_FILE" 2>> "$ERROR_LOG"

    status=$?
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    echo "Finished at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    echo "Elapsed time: ${elapsed} seconds" | tee -a "$LOG_FILE"

    if [ $status -ne 0 ]; then
        echo "ERROR: $step_name failed with status $status" | tee -a "$ERROR_LOG"
        exit $status
    fi

    echo "" | tee -a "$LOG_FILE"
}

echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" | tee -a "$LOG_FILE"
echo "Daily process" | tee -a "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" | tee -a "$LOG_FILE"

run_step "download_ashare_indexes.py" \
    python download_ashare_indexes.py

run_step "download_global_indexes.py" \
    python download_global_indexes.py

run_step "download_hsgt_moneyflow.py" \
    python download_hsgt_moneyflow.py

run_step "build_market_panel.py" \
    python build_market_panel.py

run_step "download_sw_l2_daily.py" \
    python download_sw_l2_daily.py

run_step "download_csi1500_daily_upday.py" \
    python download_csi1500_daily_upday.py --workers 2 --lookback-trade-days 90

run_step "download_csi1500_margin_detail_upday.py" \
    python download_csi1500_margin_detail_upday.py --start-upday 20260410

run_step "download_1min_upday_all.py" \
    python download_1min_upday_all.py --workers 4 

run_step "build_minute_features_v5b.py" \
    python build_minute_features_v5b.py  --start-date 20250801 --output-mode by_stock --workers 8 --overwrite

run_step "merge_csi1500_daily_polars.py, incomplete only for custom index" \
    python merge_csi1500_daily_polars.py

run_step "build_csi1500_custom_index.py" \
    python build_csi1500_custom_index.py
    
echo python 01_upday_minute_buckets_and_features.py
python 01_upday_minute_buckets_and_features.py \
  --lookback-trade-days 30 \
  --source-start-date 20250801 \
  --download-workers 4
