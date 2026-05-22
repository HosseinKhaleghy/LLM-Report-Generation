"""
Build the demo's sample input file from real WP3 dairy-farm data.

A customer uploads a real hourly electricity export: a timestamp column and an
aggregate-consumption column. This script extracts exactly that from the real
WP3 farm series (`WP3/hourly.csv`) and writes it as `sample_farm.csv`.

The data is used as-is — no anomalies are injected. Whatever irregularities the
MOMENT detector finds are genuine features of the real recording.

Run:  python make_sample_data.py
"""

import pandas as pd

SOURCE = "WP3/hourly.csv"

df = pd.read_csv(SOURCE, index_col=0)
# Plain, timezone-free hourly timestamps so the file uploads cleanly anywhere.
ts = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_localize(None)
sample = pd.DataFrame({
    "timestamp": ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
    "consumption": df["Aggregate"].round(2),
})
sample.to_csv("sample_farm.csv", index=False)
print(f"Wrote sample_farm.csv: {len(sample)} hourly rows of real farm data "
      f"({sample['timestamp'].iloc[0]} -> {sample['timestamp'].iloc[-1]})")
