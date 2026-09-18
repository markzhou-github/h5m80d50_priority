import polars as pl

file_path = "processed/predict_v5b/train_v5b.parquet"

filtered_df = (
    pl.scan_parquet(file_path)
    .filter(pl.col("trade_date") == "20260709")
    .collect()
)

filtered_df.write_csv("filtered_20260709.csv")