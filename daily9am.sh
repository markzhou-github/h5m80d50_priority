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

run_step "download_csi1500_margin_detail_upday.py" \
    python download_csi1500_margin_detail_upday.py

run_step "merge_csi1500_daily_polars.py" \
    python merge_csi1500_daily_polars.py

PRED_DATA="pred_${TODAY}.parquet"
PRED_DATA_PATH="processed/predict_v5b/${PRED_DATA}"

run_step "prepare_training_v5b.py" \
    python prepare_training_v5b.py   --merge-mode memory   --workers 8   --output-dir processed/predict_v5b  \
    --minute-feature-dir processed/minute_features_v5b/by_stock   --clean

run_step "h5m80d50 signals" \
    python h5m80d50_priority/signal_h5m80d50.py   --input processed/predict_v5b/train_v5b.parquet    --out-dir signals

run_step "h2m80d50 signals" \
    python h2m80d50_dual/signal_h2m80d50.py   --input processed/predict_v5b/train_v5b.parquet    --out-dir signals

python notify_daily.py