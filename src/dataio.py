"""
Loading and cleaning a real farm's hourly electricity data.

A customer uploads a CSV with a timestamp column and an hourly aggregate
electricity-consumption column. Real exports are messy — irregular sampling,
missing hours — so this module normalises any such file into a clean, gap-free
hourly series that the rest of the pipeline can rely on:

    DataFrame[timestamp, consumption]   (hourly, sorted, no gaps)

Missing hours are filled with the average consumption for that hour-of-day,
which preserves the daily load shape.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

# Column-name candidates, matched case-insensitively.
_TIME_CANDIDATES = ("timestamp", "datetime", "date", "time", "ds")
_CONS_CANDIDATES = ("aggregate", "consumption", "kwh", "wh", "power",
                    "load", "energy", "value")


def _find_column(columns, candidates) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    # Substring match as a fallback (e.g. "Aggregate (kW)").
    for cand in candidates:
        for low, orig in lower.items():
            if cand in low:
                return orig
    return None


def load_farm_series(source, timestamp_col: str | None = None,
                     consumption_col: str | None = None) -> pd.DataFrame:
    """Load and clean one farm's hourly series from a CSV.

    Parameters
    ----------
    source : path, file-like object, bytes, or an already-loaded DataFrame.
    timestamp_col, consumption_col : optional explicit column names;
        auto-detected when omitted.

    Returns
    -------
    DataFrame with columns ``timestamp`` (hourly, gap-free) and
    ``consumption`` (float).
    """
    df = source if isinstance(source, pd.DataFrame) else _read_any(source)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    tcol = timestamp_col or _find_column(df.columns, _TIME_CANDIDATES)
    ccol = consumption_col or _find_column(df.columns, _CONS_CANDIDATES)
    if tcol is None:
        raise ValueError("No timestamp column found (expected one of: "
                          f"{', '.join(_TIME_CANDIDATES)}).")
    if ccol is None:
        # Last resort: the single numeric column that is not the timestamp.
        numeric = [c for c in df.columns
                   if c != tcol and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric) == 1:
            ccol = numeric[0]
        else:
            raise ValueError("No consumption column found (expected one of: "
                              f"{', '.join(_CONS_CANDIDATES)}).")

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[tcol], errors="coerce", utc=True),
        "consumption": pd.to_numeric(df[ccol], errors="coerce"),
    }).dropna(subset=["timestamp"])
    if out.empty:
        raise ValueError("No valid timestamped rows after parsing.")

    # Drop any timezone so downstream date arithmetic is naive and consistent.
    out["timestamp"] = out["timestamp"].dt.tz_convert(None)
    out = out.sort_values("timestamp")

    # Collapse to hourly means, then reindex onto a complete hourly grid.
    hourly = (out.set_index("timestamp")["consumption"]
                 .resample("1h").mean())
    full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="1h")
    hourly = hourly.reindex(full_index)

    hourly = _fill_missing_by_hour_of_day(hourly)
    return pd.DataFrame({"timestamp": hourly.index,
                         "consumption": hourly.to_numpy()}).reset_index(drop=True)


def _fill_missing_by_hour_of_day(series: pd.Series) -> pd.Series:
    """Fill NaN / non-positive readings with that hour-of-day's mean."""
    s = series.copy()
    s[s <= 0] = np.nan
    hod = s.index.hour
    hod_mean = s.groupby(hod).transform("mean")
    s = s.fillna(hod_mean)
    # If a whole hour-of-day was missing, fall back to the global mean.
    return s.fillna(s.mean())


def _read_any(source) -> pd.DataFrame:
    """Read a CSV from a path, bytes, or file-like object."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    df = pd.read_csv(source)
    # WP3-style exports carry an unnamed integer index column — drop it.
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(columns=df.columns[0])
    return df


def data_quality(series: pd.DataFrame, raw_rows: int | None = None) -> dict:
    """Summarise completeness of a loaded series for the report."""
    n = len(series)
    return {
        "hours_analysed": int(n),
        "first_timestamp": str(series["timestamp"].iloc[0]),
        "last_timestamp": str(series["timestamp"].iloc[-1]),
        "span_days": round(n / 24.0, 1),
    }
