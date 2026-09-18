# Generate 20220104–20260901 signals without loading the full panel

Use `generate_signals_range_memory_safe.py`, placed beside the existing `generate_signals.py` and `generate_signals_range.py`:

```text
/home/mark/dev/csi1500/production/h5m80d50_priority/h5m80d50_priority/
```

It imports the original model loader, prediction logic, signal rules and daily CSV writer. Keep both original files, `config.json`, and the model directories in place. The existing WSL files have not been edited. The runner requires pandas, Polars, PyArrow and LightGBM; psutil is optional for memory logging.

Replace `/path/to/new_dataset.parquet` with the actual newly generated dataset path, then run inside WSL:

```bash
cd /home/mark/dev/csi1500/production/h5m80d50_priority/h5m80d50_priority
set -o pipefail
python -u generate_signals_range_memory_safe.py \
  --input /path/to/new_dataset.parquet \
  --start-date 20220104 \
  --end-date 20260901 \
  --out-dir signals_20220104_20260901_memory_safe \
  --read-batch-rows 2048 \
  2>&1 | tee signals_memory_safe.log
```

This writes daily signal, watchlist, summary and diagnostic CSVs using the original naming conventions. `signals_latest.csv` represents the latest completed date in this run. Add `--save-ranked` only if you need the original optional ranked CSVs. A small range summary is updated after each completed day. Output dates are actual dates present in the requested inclusive interval.

## Why the original has an OOM risk

The inspected `generate_signals_range.py` loads the entire requested range at line 41:

```python
out = lf.collect().to_pandas()
```

With almost the entire 23 GB parquet selected, the lazy scan materializes the whole wide dataset and converts it to pandas. File compression makes the parquet size a poor estimate of this allocation. Calling `collect(engine="streaming")` alone still returns a complete in-memory DataFrame and does not fix the subsequent pandas allocation.

The imported `generate_signals.py` adds further allocations:

| Source location | Operation and consequence |
|---|---|
| `normalize_keys`, line 118 | Copies the full input DataFrame. |
| `add_predictions`, lines 138–156 | Copies the full input, then builds feature matrices for the five configured seed models, one model at a time. |
| `ensure_features`, lines 124–135 | Converts each feature, concatenates the columns, replaces infinities, and casts the matrix to float32. Multiple intermediate buffers can coexist. On the next model iteration, the previous `x` remains alive while the new right-hand side is evaluated. |
| `add_base_selections`, lines 159 onward | Copies the full wide frame, groups all dates, and merges selected stock keys back into the wide frame. |
| `add_signal_models` and `add_watchlist`, lines 226 and 280 onward | Each copies the wide frame again. These are successive peaks, not necessarily all copies retained simultaneously. |
| `write_one_date` in the range runner | Daily writing begins only after all dates have been predicted and processed. Writing one CSV per day does not make the preceding steps memory-bounded. |

This is a code-level risk diagnosis, not a measurement of an actual failed run of this signal script. At 1,500 rows per day and 2,800 float64 features, one day's numeric inputs occupy approximately 32 MiB. A 2,048-row read batch at the same width is about 44 MiB. Models, conversion temporaries, decoder buffers, strings, and prediction matrices are additional. Processing a day instead of years removes the dominant scaling factor; the 23 GB compressed dataset itself never needs to fit in RAM.

## What the replacement does

1. Reads just `trade_date` in bounded batches to find the requested dates and verify physical date order.
2. Loads the configured models once. Reads the union of their feature lists plus the rule, output and diagnostic columns. Missing model inputs still raise by default, as in the original. The existing `--allow-missing-features` option is preserved, but is not a memory optimization.
3. For date-sorted parquet, reads selected feature columns in one sequential pass with PyArrow. Your earlier memory-safe dataset builder writes dates in order; this runner verifies the property instead of assuming it. It carries partial days across batch and row-group boundaries.
4. For unsorted parquet, uses a Polars filter to load one complete date at a time. `--read-mode auto` chooses this fallback when needed. It is bounded at the resulting daily frame, but repeated scans can be substantially slower and parquet decoding may require additional buffers.
5. Uses the original prediction function on a complete day. All models run sequentially. The original rank tie-break order (`rank(method="first")`) is preserved by keeping each day's source row order.
6. Drops model-only features after prediction, before the downstream functions copy their input. Retains all columns needed by the inspected rule configuration and output selectors.
7. Writes that day's results, releases references, and continues. Only small summary records accumulate across dates.

There is no time-series feature calculation in the inspected signal runner: those features and lags are already in the dataset. Model inference is row-wise, and the ranking, voting and watchlist selections group by `trade_date`. This is why complete-day processing preserves the intended behavior. Arbitrary read batches must not be ranked independently: a day's highest-ranked stocks could otherwise be split across batches.

The runner rejects duplicate stock/date keys to avoid ambiguous rankings and selection-join expansion. It also defaults to a 20,000-row daily limit as a guard against unexpected input. Raise `--max-day-rows` only if the input legitimately contains more rows per day. It accepts one parquet file, with non-null YYYYMMDD integer/string dates or YYYY-MM-DD strings. It does not implement the old CSV/text input path.

## Speed and memory tuning

- Start with 2,048 read rows. If memory is stable, benchmark 4,096 or 8,192. This adjusts I/O batches, not the complete-day ranking boundary. If necessary, reduce it to 512.
- Avoid running several date workers initially. Each would load its own models and decoder buffers, competing for RAM and disk bandwidth.
- Keep model-only columns out of the voting and output stages. This reduction is already included.
- CUDA is not necessary to fix this failure mode. The existing functions use LightGBM's ordinary prediction API; enabling CUDA elsewhere does not remove full-panel copies. Benchmark acceleration only after the daily pipeline runs reliably.
- The sorted reader avoids scanning all feature columns once for every date. It performs a narrow date-only inspection plus one feature pass. When requesting a small subrange it still walks the sorted file until the end of that range; skipping earlier row groups is a possible future optimization.

The bound is one day's selected data plus a read batch, model memory and library buffers. It is not a fixed-byte RAM guarantee. Run a short date interval in a separate output directory first, then the full range. While it runs, `free -h` and `vmstat 1` in another WSL terminal show memory and swap pressure. The runner logs RSS after each date when psutil is installed, but these checkpoints may miss transient peaks.

Completed daily files and the range summary survive later failures. There is no automatic resume/skip logic: resume explicitly at the first unfinished date, and retain the earlier summary if needed. Run one instance per output directory. Daily writes reuse the original writer and can leave an incomplete day's files after an interruption; regenerate that day.

## Verification

Tested with pandas 3.0.2, Polars 1.41.2, PyArrow 23.0.1, and LightGBM 4.6.0. Five small real LightGBM models were used with the inspected signal-rule structure. Sorted inputs at two read-batch sizes and shuffled input through the fallback matched the original full-range pipeline exactly for predictions, ranks, signals and all generated daily CSV bytes. Fixtures included ties, infinities, nulls, mixed dashed/plain date strings, model feature-order differences, and dates spanning both read batches and parquet row groups. Missing-feature behavior, duplicate-key rejection, daily row caps, and empty date ranges were also checked.

The production model files were not loaded and the 23 GB input was not supplied by path in this request. Therefore actual full-data throughput, peak RSS, and production-model equivalence remain unmeasured. No signals were generated in your production folder.

Implementation references: [PyArrow ParquetFile.iter_batches](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html) provides column-projected bounded read batches; [LightGBM Booster.predict](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html) documents the prediction API used by your original functions.
