# h3m55d15_dual Production Package

This folder is a portable production package for `target_h3m55d15`.

It produces three signal layers:

- `strict_signal`: robust full walk-forward family ensemble.
- `strong_signal`: high-confidence fixed-window model using day-regime probability and seed-model stability.
- `layer2_signal`: mid-frequency opportunity layer using `M1 OR M3`.

Final signal labels:

- `overlap_core`: selected by both strict and strong.
- `strong_only`: selected only by strong.
- `strict_only`: selected only by strict.
- `layer2_m1_m3_overlap`: selected by both layer2 rules.
- `layer2_m1_only`: selected only by layer2 M1.
- `layer2_m3_only`: selected only by layer2 M3.
- `none`: no production signal.

Recommended priority is `overlap_core`, then `strong_only`, then `strict_only`, then layer2.

Current raw-price exit rule:

- Take profit: `+8%`
- Stop loss: `-5%`
- Max hold: `2` trading days after entry
- Buy: next trading day's open after the signal date
- China A-share rule: no same-day sell on the entry day

## Files

- `generate_signals.py`: production signal generator.
- `config.json`: thresholds and model names.
- `models/family/*`: three family models used by strict and strong base ranking.
- `models/day_regime/*`: good-day classifier used by strong.
- `models/seed/*`: seed models used to calculate `seed_pred_cv`.
- `reports/*`: backtest reference snapshots.

## Usage

Use the `m1deepl` conda environment:

```bash
conda activate m1deepl
```

Run from project root or from this directory:

```bash
python production/h3m55d15_dual/generate_signals.py \
  --input processed/train_v5b/train_v5b.parquet \
  --trade-date 20260625 \
  --out-dir production/h3m55d15_dual/signals \
  --save-ranked
```

For live use, `--input` should be the prepared feature panel for the prediction date.
It must contain `trade_date`, `ts_code`, all model feature columns, and the two high-confidence filter columns:

- `ixic_swing`
- `csi1500_mcap_oc_ret`

By default the script fails if any model feature is missing. Use `--allow-missing-features`
only for debugging.

## Current Decision Rules

```text
high_confidence =
  ixic_swing <= 1.185989
  AND csi1500_mcap_oc_ret <= 0.006831

strict_signal =
  avg_rank <= 7
  AND high_confidence
  AND family_pred_cv <= 0.074348

strong_signal =
  avg_rank <= 7
  AND high_confidence
  AND prob_good_day >= 0.20
  AND seed_pred_cv <= 0.06

layer2_signal = M1 OR M3

M1 =
  avg_rank <= 2
  AND prob_good_day >= 0.10
  AND family_pred_cv <= 0.10
  AND ixic_swing <= 1.50

M3 =
  avg_rank <= 5
  AND prob_good_day >= 0.20
  AND seed_pred_cv <= 0.06
  AND ixic_swing <= 1.50
```

`avg_rank` is calculated from the average rank of the three family-model predictions.
`family_pred_cv` and `seed_pred_cv` are prediction standard deviation divided by absolute mean prediction.

## Current Exit Rule

```text
take_profit = +8%
stop_loss = -5%
max_hold_days = 2

Execution:
1. Buy at T+1 open after the signal date.
2. Because this is China A-share, do not sell on entry day.
3. If entry-day close <= stop price, sell at the next trading day's open.
4. From the second holding day onward:
   - if open >= take-profit price, sell at open;
   - else if high >= take-profit price, sell at take-profit price;
   - else if close <= stop price, sell at close;
   - else if max_hold_days is reached, sell at close.
```

This rule was selected from `reports/raw_exit_sweep_fullpanel_loose_dd/raw_exit_sweep_all.csv`.
