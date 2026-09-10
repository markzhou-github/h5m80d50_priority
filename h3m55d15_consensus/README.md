# H3M55D15 Consensus: Frozen P1/P2 Release

## Scope and Release Status

Model name remains **h3m55d15_consensus**. This revision replaces the earlier
single keep50 output with the approved P1/P2 rules. Stock models and the frozen
market gate are not retuned. Run assembly on WSL where the model artifacts live.
The Windows code-only folder is NOT deployable until assembly and verification
succeed. After assembly the entire folder is portable; inference needs no
research imports, training data, targets or API credentials. It still needs a
compatible feature parquet prepared by the upstream feature pipeline.

## Exact Priorities

For each date define stock sets:

- B3 = pooled six-model seed-vote Top3.
- B1 = pooled six-model seed-vote Top1.
- A50 = B3 if the earlier logistic keep50 gate accepts, otherwise empty.
- A60 = B3 if the earlier logistic keep60 gate accepts, otherwise empty.
- C = (B3 minus B1) if the new market raw-loss gate accepts, otherwise empty.

**P1 = A50 intersect C**

**P2 = (A60 union C) minus P1**

P2 uses OR, not AND. Priorities are exclusive per stock/date, but both priorities
can trade on the same day. Do not add their signal-day counts. There are at most
three unique signals per day. No rejected slots are refilled. No P3 or watchlist
is promoted to a trade.

The intended subtraction is Top3 minus Top1, not the reverse. Top1 and Top3 use
different vote counts, so B1 is not guaranteed to be inside B3. C can therefore
have two or three names before gating.

| Signal tag | Interpretation |
| --- | --- |
| KEEP50_AND_CURRENT | P1 |
| KEEP60_AND_CURRENT | P2 accepted by both A60 and C, excluding P1 |
| CURRENT_ONLY | P2 accepted by C only |
| KEEP60_ONLY | P2 accepted by A60 only |
| REJECTED | Candidate only, priority 0, not a trade |

## Stock Models

Two experts, each trained with seeds 20260801, 20260811 and 20260821:

| Expert | Feature set | Main recipe |
| --- | --- | --- |
| top1500_lr20_l2_12 | top1500_stable | learning rate .02, min leaf 800, L2 12 |
| top1600_leaf600 | top1600_stable | learning rate .025, min leaf 600, L2 8 |

Both use 31 leaves, max_bin 31, feature fraction .8 and bagging fraction .9.
Saved model texts are authoritative for exact parameters and tree counts.
Feature files preserve exact order and are checked against every model.

For pooled TopK, rank each model's scores across the full daily stock universe,
highest first, with average ranks for ties. Count models ranking each stock <=K.
Sort vote count descending, six-model average rank ascending, ts_code ascending;
take K. This is rank/vote consensus, not an average-probability model.

The stock target is h3m55d15: three-day horizon with +5.5% adjusted profit
criterion and -1.5% adjusted-close stop. Target labels and raw trade returns
are different measurements. Target generation is external to inference.

## Two Gates

### Earlier keep50 and keep60

Logistic regression estimates a zero-TP Top3 day. It uses 41 features: 30 market
fields plus 11 score/consensus summaries. Pipeline: median imputation with
missing indicators, StandardScaler, LogisticRegression(C=.1,
class_weight='balanced', max_iter=3000, random_state=20260719).

Packaging reconstructs this model from the saved 300-date development table.
Its original calibration convention is preserved: initial development scores
plus subsequently observed daily risk scores. On T, accept risk <= the 50th
or 60th percentile of scores dated strictly BEFORE T. keep50 is nested in
keep60. These names do not guarantee exactly 50%/60% future coverage.

### Current market raw-loss gate

The separately frozen logistic model predicts whether mean raw return of the
selected Top3 is negative. It uses the 30 fields in `risk/market_protocol.json`,
not the score summaries. C=.01, no class weighting. The complete fitted pipeline
is copied from research, not retrained during packaging.

Training ends 20260324; calibration runs 20260401 through 20260601; development
ends 20260608. Fixed keep60 threshold: **0.4247131293461601**. Accept risk <=
threshold. The JSON protocol is authoritative. No OOS labels update these
models during inference. The earlier gate's score history updates without labels;
the new market threshold remains fixed.

## Recorded Performance

Source: `reports/priority_benchmark.csv`. Test contains 120 repeatedly inspected
development-evaluation dates. OOS contains 57 dates with complete outcomes.
20260828, 20260831 and 20260901 were excluded because outcomes were incomplete,
not because of unfavorable results.

| Split / layer | Signal days | Trades | Precision | Avg raw trade return | Trade win rate | Trade PF | Synthetic max DD | Synthetic Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Test P1 | 20/120 | 40 | 65.00% | 3.755% | 72.50% | 3.57 | -10.99% | 4.29 |
| Test P2 exclusive | 85/120 | 185 | 45.95% | 1.718% | 52.43% | 1.69 | -21.05% | 3.57 |
| Test combined | 85/120 | 225 | 49.33% | 2.080% | 56.00% | 1.91 | -18.03% | 4.79 |
| OOS P1 | 21/57 | 44 | 54.55% | 1.881% | 61.36% | 1.80 | -11.86% | 2.92 |
| OOS P2 exclusive | 38/57 | 70 | 41.43% | 0.146% | 45.71% | 1.05 | -32.45% | 0.49 |
| OOS combined | 40/57 | 114 | 46.49% | 0.816% | 51.75% | 1.30 | -24.20% | 2.17 |

Precision is target-positive fraction, not raw-return win rate. Average trade
return is the arithmetic mean of completed raw returns. Trade PF is positive
raw-return sum divided by absolute negative-return sum. No total-return metric
is presented.

Synthetic DD/Sharpe place the mean full-horizon trade return on its signal date,
use zero on rejected dates and compound from initial equity one. Sharpe uses
252-day annualization. They are NOT portfolio mark-to-market statistics: no
overlapping-position capital allocation, fees, slippage or funding is modeled.

P1 has a small sample. P2 has marginal OOS return and material drawdown. Freezing
this package is not proof of profitability or stability. Test was reused for
selection and OOS has been inspected. Test metrics come from historical evaluation
models, not final frozen models replayed on their training period.

## Assemble on WSL

```bash
conda activate m1deepl
cd /home/mark/dev/csi1500
bash production/h3m55d15_consensus/run_package_wsl.sh
```

Required assembly inputs:

- `complete_features_h3m55d15/work/top1500_top1600_locked_oos/models/`: six stock models.
- Same work folder's `reports/`: consensus summary and day-quality reports.
- `complete_features_h3m55d15/work/top1500_top1600_multiseed_sweep/catalog/feature_sets.json`.
- `complete_features_h3m55d15/work/day_risk_filter/date_regime_dataset.parquet` and `regime_features.txt`.
- `complete_features_h3m55d15/work/day_risk_locked_oos/locked_oos_day_risk_decisions.csv` and summary.
- `complete_features_h3m55d15/raw_loss_filter/frozen_oos/market_loss_logit.joblib` and `frozen_protocol.json`.
- `complete_features_h3m55d15/raw_loss_filter/priority_benchmark/summary.csv`.

Assembly reconstructs only the earlier logistic artifact, copies the other
models, records installed environment versions and verifies file hashes/model
feature order. Bark minuet notification is sent on success or error. Packaging
is a release operation, not a daily prediction task.

## Portable Layout

```text
h3m55d15_consensus/
  generate_signals.py
  generate_signals_range.py
  build_day_risk_artifacts.py  # assembly only
  package_production.py       # assembly only
  run_package_wsl.sh          # assembly only
  verify_bundle.py
  test_signal_rules.py
  README.md
  requirements.txt
  environment_freeze.txt
  manifest.json
  models/<expert>/seed_<seed>/model.txt
  features/<expert>.txt
  risk/day_risk_logit.joblib
  risk/day_risk_features.txt
  risk/dated_history.csv
  risk/market_loss_logit.joblib
  risk/market_protocol.json
  reports/
  state/risk_history.csv      # mutable, created by inference
  signals/                   # mutable outputs
  signals_range/             # mutable outputs
```

Copy the WHOLE assembled folder to the server, including state if continuing
an existing deployment. Match the packaging Python/library environment,
especially sklearn/joblib. `environment_freeze.txt` records installed packages;
`requirements.txt` is only the minimal dependency list. Load only trusted joblib
artifacts. Training datasets and the project repository are not needed for inference.

```bash
python /server/path/h3m55d15_consensus/verify_bundle.py
python /server/path/h3m55d15_consensus/test_signal_rules.py
```

## Input Contract

Input is a daily feature parquet with `trade_date` as YYYYMMDD string/integer,
`ts_code`, all stock feature-list columns and all market-gate protocol columns.
Extra columns are ignored. Targets/future returns are unnecessary. Feature
definitions, normalization, lags, adjustments and timestamp alignment must match
training. Upstream feature preparation is not bundled here.

Supply the full intended stock universe per date, not a watchlist. Changing
coverage changes ranks and gate score summaries. The program rejects duplicate
or null stock keys, missing feature columns and fewer than ten stocks; ten is
only a technical minimum, not a substitute for the training universe.

T is the feature/signal date. Inputs must be available at the decision time
before T+1 entry. Preserve the established China/global timestamp alignment.
Frozen prediction refuses dates on/before 20260608; it cannot reproduce rolling
historical test metrics by running final models on their training period.

## Single-Date Prediction

```bash
python production/h3m55d15_consensus/generate_signals.py \
  --input processed/train_v5b/train_v5b.parquet \
  --trade-date 20260902
```

Omit `--trade-date` for the latest input date. Optional `--out-dir` changes the
output folder. On the server, use absolute script/input paths; default artifact
and output paths resolve relative to the package, not the working directory.

## Date-Range Prediction

```bash
python production/h3m55d15_consensus/generate_signals_range.py \
  --input processed/train_v5b/train_v5b.parquet \
  --start-date 20260609 --end-date 20260901 \
  --out-dir production/h3m55d15_consensus/signals_range
```

Bounds are inclusive; only dates present in the input are processed, chronologically.
`--combined-out PATH` changes the combined CSV location. Both prediction scripts
accept `--history-file PATH`, defaulting to `state/risk_history.csv` in the package.

## History and Resume

Immutable seed history is dated. Mutable history is merged with it. Only dates
before T calibrate old thresholds, so future saved OOS scores cannot enter an
earlier replay. The new market threshold stays fixed.

Process new dates chronologically, including no-signal dates. Skipping dates
changes the earlier-gate distribution; catch up over the full missing range.
There is no automatic exchange-calendar check for gaps in the supplied panel.

State is atomically saved after each date. Rerunning a range regenerates outputs
without adding duplicate history. Conflicting scores on an existing date fail
instead of mixing model/data versions. Investigate revisions rather than deleting
history to silence errors. Do not run concurrent writers on the same history file.
Keep immutable model bundles and mutable histories versioned together.

## Outputs and Troubleshooting

- `signals_DATE.csv`: accepted P1/P2 only; a rejected day has a header-only file.
- `signals_latest.csv`: latest single-date output.
- `candidates_DATE.csv`: all three candidates, including priority 0/reason REJECTED.
- `diagnostic_DATE.csv`: daily mode gate scores, thresholds and counts.
- Range mode: combined signal CSV, `signals_latest_range.csv`, combined diagnostics.

Each candidate includes priority/reason, pooled Top1 membership, old gate and
current-selection flags, vote counts, average rank and both gate scores/thresholds.
No signals is a valid result; inspect diagnostics before assuming an error.

Missing artifacts mean assembly is incomplete. Missing feature/order mismatch
means wrong data or inconsistent bundle: do not fill missing columns with zero.
Risk-history mismatch means a data/model/library version changed. Hash mismatch
means an immutable file changed; restore the release or explicitly rebuild it.
Unexpected ranks warrant checking universe coverage and source revisions.

## Execution and Validation Limits

This package does not generate orders or position sizes, guarantee fills, or
handle suspended/limit-locked stocks. China T+1 restrictions remain an execution
responsibility. Research target stock-row horizons and daily benchmark adjustments
need separate execution validation, particularly around suspensions and benchmark
information unavailable intraday. Reported raw returns are not guaranteed realizable.

Before deploying, replay known OOS dates using the SAME feature panel and compare
to saved signals. Unit tests cover priority set algebra and history replay; hashes
and feature-order checks do not replace end-to-end output verification.
