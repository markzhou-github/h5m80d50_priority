# Model Details: h3m55d15 Dual Production Package

This folder is a portable production package for the `target_h3m55d15` alpha model.

The package is designed to be moved as one directory to another machine or server. It can either:

1. generate signals from an already prepared prediction dataset, or
2. update minute data, rebuild prediction features, then generate signals.

## 1. Model Objective

The model predicts high-quality short-term China A-share opportunities.

Current target:

- Target name: `target_h3m55d15`
- Horizon: 3 trading days
- Profit threshold used in target design: `+5.5%`
- Stop threshold used in target design: `-1.5%`
- Prediction focus: top daily candidates, especially top 1 / top 3 / top 5

The production package does not directly trade every positive prediction. It creates ranked signals and applies confidence gates.

## 2. Production Signal Layers

The production generator creates several signal layers:

- `strict_signal`: robust full walk-forward / family-consensus style signal.
- `strong_signal`: high-confidence fixed-window model using day-regime probability and seed-model stability.
- `layer2_signal`: broader opportunity layer based on two additional mid-frequency rules.

Final signal labels:

- `overlap_core`: selected by both strict and strong.
- `strong_only`: selected only by strong.
- `strict_only`: selected only by strict.
- `layer2_m1_m3_overlap`: selected by both layer2 rules.
- `layer2_m1_only`: selected only by layer2 M1.
- `layer2_m3_only`: selected only by layer2 M3.
- `none`: no production signal.

Recommended priority:

1. `overlap_core`
2. `strong_only`
3. `strict_only`
4. `layer2_m1_m3_overlap`
5. `layer2_m1_only`
6. `layer2_m3_only`

## 3. Model Components

The model package uses LightGBM model files saved under `models/`.

Required model folders:

```text
models/
  family/
    production/
    no_auction_chip/
    no_sw_l2/
  seed/
    ff80_seed20260629/
    ff80_seed20260630/
    ff80_seed20260701/
    ff80_seed20260702/
    ff80_seed20260703/
  day_regime/
    label_broad_top5_tp_ge_3/
```

Each model folder must contain:

```text
model.txt
model_meta.pkl
```

Some folders may also include `feature_importance.csv`; that is useful for inspection but not required for prediction.

## 4. Decision Rules

The thresholds are stored in `config.json`.

High-confidence market filter:

```text
high_confidence =
  ixic_swing <= 1.185989
  AND csi1500_mcap_oc_ret <= 0.006831
```

Strict signal:

```text
strict_signal =
  avg_rank <= 7
  AND high_confidence
  AND family_pred_cv <= 0.074348
```

Strong signal:

```text
strong_signal =
  avg_rank <= 7
  AND high_confidence
  AND prob_good_day >= 0.20
  AND seed_pred_cv <= 0.06
```

Layer2 M1:

```text
M1 =
  avg_rank <= 2
  AND prob_good_day >= 0.10
  AND family_pred_cv <= 0.10
  AND ixic_swing <= 1.50
```

Layer2 M3:

```text
M3 =
  avg_rank <= 5
  AND prob_good_day >= 0.20
  AND seed_pred_cv <= 0.06
  AND ixic_swing <= 1.50
```

Layer2 signal:

```text
layer2_signal = M1 OR M3
```

Definitions:

- `avg_rank`: average rank from the three family-model predictions.
- `family_pred_cv`: standard deviation of family-model predictions divided by absolute mean prediction.
- `seed_pred_cv`: standard deviation of seed-model predictions divided by absolute mean prediction.
- `prob_good_day`: output of the day-regime model.

## 5. Current Performance Summary

Latest confirmed high-confidence candidate:

```text
trades          196
signal days      54
precision      60.20%
avg trade ret    3.23%
profit factor    7.49
max drawdown    -6.48%
```

Broader opportunity candidate:

```text
trades          268
signal days      57
precision      58.58%
avg trade ret    2.94%
profit factor    6.52
max drawdown    -6.67%
```

Earlier baseline final gate:

```text
trades          291
signal days      97
precision      53.61%
profit factor    5.71
max drawdown    -6.33%
```

Interpretation:

- The strict high-confidence layer has the best quality.
- The broader opportunity layer gives more trades while keeping strong precision.
- For production sizing, use the strict/overlap layer as the highest-confidence layer and the broader layer for additional opportunity.

## 6. Exit Rule Used in Reports

Current raw-price exit rule:

```text
take_profit = +8%
stop_loss = -5%
max_hold_days = 2
```

Execution assumptions:

1. Signal is produced on day `T`.
2. Buy at day `T+1` open.
3. China A-share rule: no same-day sell on the entry day.
4. If entry-day close is below stop price, sell at the next trading day's open.
5. From the second holding day onward:
   - if open >= take-profit price, sell at open;
   - else if high >= take-profit price, sell at take-profit price;
   - else if close <= stop price, sell at close;
   - else if max hold days is reached, sell at close.

Reference report:

```text
reports/raw_exit_sweep_fullpanel_loose_dd/raw_exit_sweep_all.csv
```

## 7. Minimal Files Needed for Signal Generation

If the new server already has a prepared prediction dataset, only these are required:

```text
config.json
generate_signals.py
models/
```

Recommended optional files:

```text
README.md
MODEL_DETAILS.md
reports/
evaluate_prediction_window_raw.py
evaluate_signal_exit_thresholds.py
```

Example minimal portable folder:

```text
h3m55d15_dual/
  config.json
  generate_signals.py
  MODEL_DETAILS.md
  README.md
  models/
  signals/
```

Run signal generation:

```bash
conda activate m1deepl

python generate_signals.py \
  --input /path/to/prediction_dataset.parquet \
  --out-dir signals \
  --save-ranked
```

The prediction dataset must contain:

- `trade_date`
- `ts_code`
- all model feature columns required by `model_meta.pkl`
- filter columns, especially:
  - `ixic_swing`
  - `csi1500_mcap_oc_ret`

## 8. Files Needed to Build Prediction Dataset

If the new server must also build the prediction dataset, copy the full production folder:

```text
h3m55d15_dual/
  csi1500con.csv
  config.json
  prepare_training_v5b.py
  update_minute_features_panel_v5b.py
  generate_signals.py
  models/
  data/
  processed/
```

Important data paths inside the portable production folder:

```text
data/raw_minute_by_date/
processed/minute_features_v5b/minute_features_panel.parquet
processed/train_v5b/train_v5b.parquet
```

The minute updater now treats the script folder as the project root, so if the folder is moved to another server, default paths stay inside the moved folder.

## 9. Minute Data Update

The current minute architecture uses daily all-stock 1-minute parquet files:

```text
data/raw_minute_by_date/trade_date=YYYYMMDD.parquet
```

The updater downloads recent 1-minute data by stock, merges it in memory, updates daily raw files, and builds one no-lag minute feature panel:

```text
processed/minute_features_v5b/minute_features_panel.parquet
```

Recommended command:

```bash
python update_minute_features_panel_v5b.py \
  --end-upday 20260710 \
  --lookback-trade-days 20 \
  --source-start-date 20260612 \
  --download-workers 4 \
  --feature-executor serial \
  --overwrite
```

Notes:

- `--end-upday` is the latest trading day to download.
- `--lookback-trade-days` controls the download window if `--start-upday` is not provided.
- `--source-start-date` controls the feature-building source window, not the download window.
- `--save-raw-day-files` is default true.
- `--no-save-raw-day-files` builds features from memory without updating daily raw files.
- `--feature-executor serial` is currently the safest feature-generation mode.

## 10. Prediction Dataset Build

After minute feature panel is ready, build the final prediction dataset:

```bash
python prepare_training_v5b.py \
  --merge-mode chunked \
  --workers 8 \
  --lag-mode panel \
  --prediction-date 20260710 \
  --source-start-date 20260101 \
  --output-start-date 20260710 \
  --output-end-date 20260710 \
  --overwrite
```

The script now prefers:

```text
processed/minute_features_v5b/minute_features_panel.parquet
```

over legacy per-stock minute feature files.

Minute features are stored without lags. Lags are created later by `prepare_training_v5b.py`, especially with:

```text
--lag-mode panel
```

## 11. Full Daily Production Flow

Typical daily flow:

```bash
conda activate m1deepl
cd /path/to/h3m55d15_dual
```

Update minute data and minute features:

```bash
python update_minute_features_panel_v5b.py \
  --end-upday YYYYMMDD \
  --lookback-trade-days 20 \
  --source-start-date YYYYMMDD \
  --download-workers 4 \
  --feature-executor serial \
  --overwrite
```

Build prediction dataset:

```bash
python prepare_training_v5b.py \
  --merge-mode chunked \
  --workers 8 \
  --lag-mode panel \
  --prediction-date YYYYMMDD \
  --source-start-date YYYYMMDD \
  --output-start-date YYYYMMDD \
  --output-end-date YYYYMMDD \
  --overwrite
```

Generate signals:

```bash
python generate_signals.py \
  --input processed/train_v5b/train_v5b.parquet \
  --trade-date YYYYMMDD \
  --out-dir signals \
  --save-ranked
```

## 12. Environment

Expected environment:

```text
conda env: m1deepl
Python 3.11 preferred
```

Main packages:

```text
polars
pandas
numpy
lightgbm
tushare
scikit-learn
pyarrow
```

LightGBM prediction does not require CUDA. CUDA was used for training experiments, but production inference can run on CPU.

## 13. Moving Checklist

Before moving to a new server:

- Copy the whole production folder.
- Confirm `config.json` is present.
- Confirm `models/` contains all family, seed, and day-regime models.
- Confirm `csi1500con.csv` exists if the server will rebuild features.
- Confirm `data/raw_minute_by_date/` exists if the server will update/build minute data.
- Confirm `processed/minute_features_v5b/minute_features_panel.parquet` exists if skipping minute rebuild.
- Confirm the prediction dataset exists if only generating signals.
- Run a dry signal-generation test on one known `trade_date`.

Useful sanity checks:

```bash
python -m py_compile generate_signals.py
python -m py_compile prepare_training_v5b.py
python -m py_compile update_minute_features_panel_v5b.py
```

## 14. Known Operational Notes

- `update_minute_features_panel_v5b.py` should use `--feature-executor serial` for reliability.
- Earlier parallel feature execution with in-memory Polars frames could stall with no completed batches.
- If feature generation seems slow, prefer correctness first; optimize after a reliable serial benchmark.
- Keep `MODEL_DETAILS.md`, `README.md`, and `config.json` with the model folder so the package remains self-describing.
