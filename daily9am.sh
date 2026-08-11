#!/usr/bin/env bash

TODAY=$(date +%Y%m%d)

LOG_DIR="processed/train_v5b/logs"
LOG_FILE="${LOG_DIR}/run_${TODAY}.log"
ERROR_LOG="${LOG_DIR}/error_${TODAY}.log"

mkdir -p "$LOG_DIR"

echo "Script started. Waiting until 9:00 AM..."

# Loop until the current time matches 09:00
#while [[ "$(date +%H:%M)" != "09:00" ]]; do
#    sleep 1
#done

echo "It is 9:00 AM. Executing command..."

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

run_step "download_global_indexes.py" \
    python download_global_indexes.py

run_step "build_market_panel.py" \
    python build_market_panel.py

run_step "download_csi1500_margin_detail_upday.py" \
    python download_csi1500_margin_detail_upday.py

run_step "merge_csi1500_daily_margin_detail.py" \
    python merge_csi1500_daily_margin_detail.py

PRED_DATA="pred_${TODAY}.parquet"
PRED_DATA_PATH="processed/predict_v5b/${PRED_DATA}"

echo python prepare_training_v5b_0714.py 
python prepare_training_v5b.py \
  --source-start-date 20250801 \
  --merge-mode chunked \
  --workers 8
  
run_step "h5m80d50 signals" \
    python h5m80d50_priority/signal_h5m80d50.py   --input processed/train_v5b/train_v5b.parquet    --out-dir signals_h5priority

run_step "h2m80d50 signals" \
    python h2m80d50_dual/signal_h2m80d50.py   --input processed/train_v5b/train_v5b.parquet    --out-dir signals_h2dual

run_step "h2m80d50 signals" \
    python h5m80d50_ensemble/signal_h5ensemble.py   --input processed/train_v5b/train_v5b.parquet    --out-dir signals_h5ensemble

python h5m80d50_neural_top5/generate_signals.py \
  --input processed/train_v5b/train_v5b.parquet \
  --out-dir signals_h5neural \
  --save-ranked

python notify_daily.py


run_step "prepare_training_v5b.py" \
    python prepare_training_v5b.py   --merge-mode memory   --workers 8   --output-dir processed/train_v5b_0715  \
    --minute-feature-dir processed/minute_features_v5b/by_stock   --clean

run_step "h5m80d50 signals" \
    python h5m80d50_priority/signal_h5m80d50.py   --input processed/train_v5b_0715/train_v5b.parquet    --out-dir signals_0715

run_step "h2m80d50 signals" \
    python h2m80d50_dual/signal_h2m80d50.py   --input processed/train_v5b_0715/train_v5b.parquet    --out-dir signals_0715




