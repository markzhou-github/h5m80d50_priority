# H3M55D15 Consensus Production System

## Purpose

This self-contained bundle ranks CSI1500 stocks after day T closes and produces
candidates for execution from T+1. It combines two feature-count experts, three seeds
per expert, and a causal China-market day-risk gate.

The frozen target is `h3m55d15`: buy at T+1 open, seek a +5.5% adjusted first-hit,
use a -1.5% adjusted-close stop under the target rules, and expire at T+3 close.
Precision is the proportion of selected stocks whose target label is positive.

## Architecture

### Stock experts

1. `top1500_lr20_l2_12`: 1,500 stable features, learning rate 0.02, L2 12.
2. `top1600_leaf600`: 1,600 stable features, minimum 600 rows per leaf.

Each expert contains seeds `20260801`, `20260811`, and `20260821`. All six models rank
the daily universe. A stock receives a vote whenever it is in a model's Top3. The
standard layer selects three stocks by descending vote count, ascending pooled average
rank, and stock code for deterministic tie-breaking.

### Day-risk gate

The logistic gate uses ensemble confidence plus CSI1500 breadth, momentum, limit, and
domestic flow state. `keep50` accepts a day when its risk score is at or below the
causal median of available historical scores. It does not depend on global-index gates.

## Signal Priorities

| Priority | Output tag | Meaning |
|---|---|---|
| P1 | `P1_HIGH_CONF_ACCEPT` | Both experts agree on Top1 and keep50 accepts the day. |
| P2 | `P2_STANDARD_ACCEPT` | Pooled Top3 candidate on a keep50-accepted day. |
| Watch | `WATCH_HIGH_CONF_DAY_REJECT` | Shared Top1, but the day gate rejects exposure. |
| Watch | `WATCH_STANDARD_DAY_REJECT` | Standard Top3 on a rejected day. |

Rejected candidates remain visible for diagnosis. Production trading should normally
use P1 and P2 only. `signal_tag` records stock confidence independently of `day_gate`.

## Frozen Performance

### Stock ranking

| Layer | Test precision | Locked-OOS precision | OOS coverage |
|---|---:|---:|---:|
| Pooled seed-vote Top3 | 44.67% | 44.44% | 60/60 days |
| Pooled seed-vote Top1 | 45.00% | 48.33% | 60/60 days |
| Strict shared Top1 | 49.74% | 51.61% | 31/60 days |

### Keep50 gated Top3

| Metric | Walk-forward test | Locked OOS |
|---|---:|---:|
| Signal days | 93/180 (51.7%) | 29/60 (48.3%) |
| Trades | 279 | 87 |
| Precision | 46.59% | 49.43% |
| Average raw trade return | 1.57% | 1.46% |
| Day profit factor | 2.29 | 1.95 |
| Day maximum drawdown | -13.15% | -12.42% |
| Annualized day Sharpe | 3.73 | 2.91 |

Returns are historical target-path results before live slippage, fees, capacity limits,
and execution failures. Drawdown is portfolio-level compounded drawdown from the daily
equal-weight average of selected trades, with no-signal days counted as zero.

## Contents

- `models/`: six frozen LightGBM models.
- `features/`: exact ordered feature list for each expert.
- `risk/`: logistic gate, ordered features, and causal score history.
- `reports/`: locked-OOS model and gate reports.
- `generate_signals.py`: latest-date or single-date generation.
- `generate_signals_range.py`: sequential date-range generation.
- `build_day_risk_artifacts.py`: recreates the deployable logistic gate from reports.
- `manifest.json` and `verify_bundle.py`: deployment integrity checks.

## Assemble And Verify

```bash
cd /home/mark/dev/csi1500
bash production/h3m55d15_consensus/run_package_wsl.sh
python production/h3m55d15_consensus/verify_bundle.py
```

Copy the complete `production/h3m55d15_consensus` directory to the server. The intended
environment is conda `m1deepl`; dependencies are listed in `requirements.txt` and the
trained LightGBM version is 4.6.0.

## Input

Input is the V5B feature parquet produced after T close. It must contain `trade_date`,
`ts_code`, both expert feature lists, and raw fields required by
`risk/day_risk_features.txt`. Targets and future returns are not production inputs.
Missing required columns stop execution explicitly.

## Generate One Date

`--trade-date` defaults to the latest input date.

```bash
conda activate m1deepl
python production/h3m55d15_consensus/generate_signals.py \
  --input processed/train_v5b/train_v5b.parquet \
  --trade-date 20260901 \
  --out-dir production/h3m55d15_consensus/signals
```

Outputs are `signals_YYYYMMDD.csv`, `signals_latest.csv`, and a daily diagnostic file.

## Generate A Date Range

Dates are inclusive. Models load once, dates run in trading-date order, and each risk
score enters the in-memory history before the next date, preserving causal threshold
evolution within the range.

```bash
python production/h3m55d15_consensus/generate_signals_range.py \
  --input processed/train_v5b/train_v5b.parquet \
  --start-date 20260801 \
  --end-date 20260901 \
  --out-dir production/h3m55d15_consensus/signals_range
```

The range runner saves one file per date, one combined file, and daily diagnostics.
The bundled history reflects information available when assembled. Reconstructing dates
before that cutoff requires the corresponding as-of history; otherwise stock ranks are
valid but the gate is not a strict historical replay.

## Output Columns

- `trade_date`, `ts_code`: signal identity.
- `priority`, `priority_name`: production priority or watchlist class.
- `signal_tag`: strict shared Top1 or standard pooled Top3.
- `day_gate`: `KEEP50_ACCEPT` or `KEEP50_REJECT`.
- `vote3`: number of six models placing the stock in Top3.
- `pooled_avg_rank`: average daily rank across six models; lower is stronger.
- `bad_day_risk`, `bad_day_threshold`: gate score and causal threshold.

## Operating Rules

- A row dated T uses information through T close and is intended for T+1 execution.
- Do not silently trade watchlist rows as primary signals.
- Do not change features, votes, Top3 size, gate model, or keep rate without a new
  test-only selection and locked-OOS validation cycle.
- Archive daily signals and diagnostics. Monitor coverage, votes, risk scores, realized
  precision, raw return, and drawdown for drift.
- The generator does not mutate `risk_score_history.csv`. Append each completed day's
  risk score through controlled production state, or periodically rebuild history from
  archived diagnostics.
