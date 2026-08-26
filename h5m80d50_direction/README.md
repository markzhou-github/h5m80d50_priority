# H5m80d50 Competing-Direction Production Model

Frozen policy:

1. Build the union of F150 Top1, C185 Top3, and F220 Top5 candidates.
2. Average five seeds for each competing head: up, down, intensity, direction.
3. Rank candidates by `P(intensity) * P(up | extreme)`.
4. Keep conditional Top3.
5. Apply the Platt-calibrated direction-ratio threshold `>= 0.45` without filling rejected slots.

Install frozen WSL models once:

```bash
python production/h5m80d50_competing_direction/install_frozen_package.py
python production/h5m80d50_competing_direction/check_package.py
```

Generate signals from feature date T:

```bash
python production/h5m80d50_competing_direction/generate_signals.py \
  --input processed/train_v5b/train_v5b.parquet \
  --trade-date YYYYMMDD \
  --save-ranked
```

Locked-OOS result: 59.15% precision, 2.74% average trade return, 2.94 daily profit factor, -13.93% maximum daily drawdown, and 7.24 daily Sharpe across 55 evaluable dates.
