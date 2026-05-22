"""
Feature engineering for herd-size prediction.

A farm is described to the model by collapsing its year-long hourly
electricity series into one fixed-length feature vector. Only
*consumption-derived* features are used (statistics, load-shape descriptors and
the average daily profile) so the same vector can be built from the synthetic
training data and from a real customer upload alike — no metadata such as
milking-machine count or water-heating flag is required.

The herd size is encoded in the data mainly through total consumption and the
magnitude of the morning/evening milking peaks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Column names in the synthetic training CSV.
COL_FARM = "FarmID"
COL_HOUR = "Hour"
COL_CONS = "Consumption"
COL_COWS = "#Cows"

HOURS_PER_DAY = 24


def extract_features(consumption: np.ndarray, hour_of_day: np.ndarray) -> dict:
    """Collapse one farm's hourly series into a feature dictionary.

    Parameters
    ----------
    consumption : hourly consumption values.
    hour_of_day : matching hour-of-day (0-23) for each reading.
    """
    c = np.asarray(consumption, dtype=float)
    hod = np.asarray(hour_of_day, dtype=int)
    mask = np.isfinite(c)
    c, hod = c[mask], hod[mask]
    if c.size == 0:
        raise ValueError("Farm has no valid consumption readings.")

    profile = np.array([c[hod == h].mean() if np.any(hod == h) else 0.0
                        for h in range(HOURS_PER_DAY)])
    baseline = np.percentile(c, 5)
    peak, mean = c.max(), c.mean()

    feats: dict[str, float] = {
        "cons_mean": mean,
        "cons_std": c.std(),
        "cons_min": c.min(),
        "cons_max": peak,
        "cons_median": np.median(c),
        "cons_total_kwh": c.sum() / 1000.0,
        "cons_p10": np.percentile(c, 10),
        "cons_p25": np.percentile(c, 25),
        "cons_p75": np.percentile(c, 75),
        "cons_p90": np.percentile(c, 90),
        "cons_p95": np.percentile(c, 95),
        "cons_p99": np.percentile(c, 99),
        "cons_iqr": np.percentile(c, 75) - np.percentile(c, 25),
        "cons_cv": c.std() / mean if mean else 0.0,
        "cons_skew": _safe_skew(c),
        "baseline_wh": baseline,
        "baseline_ratio": baseline / mean if mean else 0.0,
        "peak_to_mean": peak / mean if mean else 0.0,
        "peak_to_median": peak / np.median(c) if np.median(c) else 0.0,
        "load_factor": mean / peak if peak else 0.0,
        "frac_above_2x_mean": float(np.mean(c > 2 * mean)),
        "frac_near_baseline": float(np.mean(c < 1.5 * baseline)),
        "profile_max": profile.max(),
        "profile_min": profile.min(),
        "profile_range": profile.max() - profile.min(),
        "profile_std": profile.std(),
        "active_hours": float(np.sum(profile > 3 * max(baseline, 1.0))),
        "milk_peak_energy": float(np.sort(profile)[-4:].sum()),
    }
    for h in range(HOURS_PER_DAY):
        feats[f"hod_{h:02d}"] = profile[h]
    return feats


def _safe_skew(x: np.ndarray) -> float:
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - x.mean()) / s) ** 3))


def features_from_series(series: pd.DataFrame) -> dict:
    """Build the feature dict from a loaded real farm series (dataio output)."""
    hod = pd.DatetimeIndex(series["timestamp"]).hour.to_numpy()
    return extract_features(series["consumption"].to_numpy(), hod)


def build_training_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build (features, herd-size target) from the synthetic multi-farm CSV."""
    rows, targets, index = [], [], []
    for farm_id, farm_df in df.groupby(COL_FARM):
        hod = (farm_df[COL_HOUR].to_numpy() % HOURS_PER_DAY)
        rows.append(extract_features(farm_df[COL_CONS].to_numpy(), hod))
        index.append(farm_id)
        targets.append(float(farm_df[COL_COWS].iloc[0]))
    X = pd.DataFrame(rows, index=pd.Index(index, name=COL_FARM))
    y = pd.Series(targets, index=X.index, name=COL_COWS)
    return X, y


def feature_names(X: pd.DataFrame) -> list[str]:
    """Stable, sorted feature ordering so train and inference always align."""
    return sorted(X.columns)
