# h5m80d50 Priority Production Candidate

This package generates the h5m80d50 priority signals.

Priority definitions:

- P1A: current consensus signal, selected by all 4 kept signal models.
- P1B: low-turnover exception, `avg_pred_rank == 1` and `turnover_prank_1500 <= 0.20042194092827004`, excluding P1A.
- P2: current broader signal, selected by at least one of the 4 kept signal models, excluding P1A/P1B.
- P3: lower-confidence global+China expansion, excluding P1A/P1B/P2:
  `vote3_top3` and one of:
  `spx_swing <= 0.576`,
  `hktech_swing <= 2.0 and n225_swing <= 2.0`,
  `ret_5_rel_csi1500_ew <= -0.03941249250602667`.

Validated test+OOS summary:

| layer | trades | precision | avg raw return |
|---|---:|---:|---:|
| P1A test | 22 | 77.27% | 7.43% |
| P1A OOS | 6 | 83.33% | 9.49% |
| P1B incremental test | 12 | 75.00% | 4.57% |
| P1B incremental OOS | 5 | 100.00% | 8.44% |
| P1A+P1B+P2 test | 71 | 71.83% | 5.48% |
| P1A+P1B+P2 OOS | 22 | 81.82% | 6.67% |
| P3 incremental test | 120 | 46.67% | 1.91% |
| P3 incremental OOS | 44 | 47.73% | 1.63% |

Execution note:

- P1A and P1B are both high-confidence tiers.
- P2 is the broader normal-confidence tier.
- P3 is a lower-confidence opportunity expansion layer; size it separately from P1/P2.
- Do not rebuy the same `ts_code` if it is already held from a prior signal.

## Required Model Files

Copy each seed model into:

```text
models/seed20260710/model.txt
models/seed20260710/model_meta.pkl
models/seed20260711/model.txt
models/seed20260711/model_meta.pkl
models/seed20260712/model.txt
models/seed20260712/model_meta.pkl
models/seed20260713/model.txt
models/seed20260713/model_meta.pkl
models/seed20260714/model.txt
models/seed20260714/model_meta.pkl
```

Source hint:

```text
processed/train_v5b/h5m80d50_top3_a11_refined_seed_confirm/models/oos60_701515/drop_pressure_keep_net_l2_8_seed{seed}_l2_8_l95_m500_ff70_bf80/
```

## Check Package

```bash
python production/h5m80d50_priority/check_package.py
```

## Generate Signals

For latest date in a prediction feature file:

```bash
python production/h5m80d50_priority/generate_signals.py \
  --input processed/train_v5b/train_v5b.parquet \
  --out-dir production/h5m80d50_priority/signals \
  --save-ranked
```

For a specific date:

```bash
python production/h5m80d50_priority/generate_signals.py \
  --input processed/train_v5b/train_v5b.parquet \
  --trade-date 20260710 \
  --out-dir production/h5m80d50_priority/signals \
  --save-ranked
```

Output:

- `signals_latest.csv`
- `signals_h5m80d50_priority_YYYYMMDD.csv`
- `watchlist_h5m80d50_priority_YYYYMMDD.csv`
- `diagnostic_summary_YYYYMMDD.csv`
- `diagnostic_topN_YYYYMMDD.csv`
- `signal_summary_YYYYMMDD.csv`
- optional ranked full table
