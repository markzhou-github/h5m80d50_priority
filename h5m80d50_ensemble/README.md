# h5m80d50 ensemble

Production package for the validated F220 Top1 signal with a cross-sectional
late/early intraday gate. Frozen F150 and C185 models remain packaged for diagnostics.

## Production signal

The production rule is:

```text
F220 five-seed average rank = 1
AND late_early_cs_pct <= 0.75791696
```

`late_early_cs_pct` is the same-day cross-sectional percentile rank of
`last60_minus_first60_ret_lag1`. Null values fail the gate. Passing rows are tagged
`F220_TOP1_LATE_EARLY_GATE`.

F150/C185 scores, family selections, family agreement, and seed dispersion are retained
in all candidate output. The F220 gate determines `production_signal`; it does not remove
the broader family-agreement candidates from `signals_*.csv`.

## Target and model structure

`h5m80d50` uses a five-trading-day horizon, an executable +8% profit target,
and a -5% SW L2-relative close stop. A signal is generated after feature date T,
with entry at T+1 open subject to the A-share T+1 selling restriction.

The package contains 15 frozen LightGBM models: five seeds for each family.

| Family | Features | Primary selection | Main role |
|---|---:|---:|---|
| F150 | 150 | Top-1 | Highest-conviction compact model |
| C185 | 185 | Top-3 | Balanced main model |
| F220 | 220 | Top-5 | Broad and seed-stable opportunity model |

## Locked-OOS benchmark

The locked OOS window contains 60 dates. Five latest dates had pending outcomes,
leaving 55 resolved dates for return and precision statistics.

### Family results

| Family | Selection | Trades | Precision | Avg trade return | Trade PF | Day max DD | Day Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| F150 | Top-1 | 55 | 61.82% | 3.48% | 2.38 | -35.01% | 6.62 |
| C185 | Top-3 | 165 | 56.36% | 2.74% | 1.97 | -17.13% | 7.34 |
| F220 | Top-5 | 275 | 56.00% | 2.19% | 1.67 | -19.42% | 6.22 |

### Family-agreement layers

The following statistics use each family's primary selection and count each
stock-date once.

| Layer | Resolved trades | Signal dates | Precision | Avg trade return |
|---|---:|---:|---:|---:|
| `CONSENSUS_3` | 40 | 40 | 60.00% | 3.35% |
| `CONSENSUS_2` | 98 | 54 | 56.12% | 2.21% |
| `FAMILY_ONLY` | 179 | 55 | 55.31% | 2.29% |
| All unique signals | 317 | 55 | 56.15% | 2.40% |

Derived equal-weight daily risk statistics:

| Layer | Trade PF | Day max DD | Day Sharpe |
|---|---:|---:|---:|
| `CONSENSUS_3` | 2.30 | -25.86% | 6.40 |
| `CONSENSUS_2` | 1.71 | -30.38% | 4.46 |
| `FAMILY_ONLY` | 1.69 | -19.78% | 7.54 |
| All unique signals | 1.76 | -12.70% | 7.98 |

The layer-level risk table is derived from the saved resolved signals and is not
a separate frozen training report. It assumes equal weighting within each day.

### Seed stability

| Family | Mean precision | Seed range | Std |
|---|---:|---:|---:|
| F150 Top-1 | 57.45% | 56.36%-60.00% | 1.63 pp |
| C185 Top-3 | 55.03% | 49.70%-60.00% | 3.75 pp |
| F220 Top-5 | 55.93% | 55.27%-56.73% | 0.54 pp |

F150 has the strongest precision and average return. C185 offers the best
balance of opportunity count, return, and drawdown. F220 is the most stable
across seeds. `CONSENSUS_3` is the highest-quality agreement layer, although
false-positive days remain strongly correlated across families.

## Install and verify

Run from the project root in the `m1deepl` environment:

```bash
python production/h5m80d50_ensemble/install_frozen_models.py
python production/h5m80d50_ensemble/check_package.py
```

The installer copies only `model.txt` and feature lists from the immutable
`retrain_robust_v5b/work/final_oos` results. `model_meta.pkl` is deliberately not used,
which keeps this package portable between Windows and WSL.

## Generate signals

```bash
python production/h5m80d50_ensemble/generate_signals.py \
  --input processed/train_v5b/train_v5b.parquet \
  --trade-date 20260708 \
  --save-ranked
```

The input must contain `last60_minus_first60_ret_lag1` for the full daily universe because
the gate is calculated cross-sectionally. Omit `--trade-date` to use the latest feature date. The date is feature date T; execution
starts at the next trading day's open under the h5m80d50 target definition.

Outputs are written to `production/h5m80d50_ensemble/signals/`.

- `signals_YYYYMMDD.csv` and `signals_latest.csv` contain the union of
  `CONSENSUS_3`, `CONSENSUS_2`, and `FAMILY_ONLY` candidates. Gate-passing rows are
  additionally tagged `F220_TOP1_LATE_EARLY_GATE` and have `production_signal=true`.
- `production_signals_YYYYMMDD.csv` and `production_signals_latest.csv` contain only
  the F220 Top-1 candidates that pass the production gate.
- `ranked_YYYYMMDD.parquet`, when `--save-ranked` is used, contains the full universe.

Use `production_signal` for execution. Use `legacy_signal_tag`, `family_tags`, and
`family_count` to inspect or separately size the broader ensemble layers.

## Model lifecycle

These 15 models are frozen and must remain unchanged. Production selection uses only the
five F220 models; F150 and C185 remain diagnostic. A later live
retraining job should write to a separate versioned folder and must not overwrite them.
Collect 40-60 completed forward dates before deciding whether a Stage-2 filter adds value.

## Failure overlap audit

To check whether false positives cluster on the same dates across families:

```bash
python production/h5m80d50_ensemble/analyze_failure_overlap.py
```

The audit distinguishes days containing any false positive from days where every selected
stock failed. Pending OOS outcomes are excluded using `ret_raw_h5m80d50`.
