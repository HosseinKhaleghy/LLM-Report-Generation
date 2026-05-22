"""
Herd-size prediction from a trained model bundle.

Loads the joblib bundle produced by `train.py` and turns one farm's hourly
consumption series into an estimated number of cows, with a confidence margin
derived from the model's cross-validated error.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features import features_from_series

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "herd_model.joblib"

_BUNDLE_CACHE: dict | None = None

# Consumption-based estimation constants (KB benchmarks, kWh/cow/year).
# Used to anchor the herd-size estimate when the ML model extrapolates.
_KWH_PER_COW_LO  = 300.0   # efficient large dairy (KB lower bound)
_KWH_PER_COW_MID = 375.0   # typical Northern European dairy (KB midpoint)
_KWH_PER_COW_HI  = 500.0   # above-average consumption (KB upper bound)
_BLEND_ML_WEIGHT = 0.40    # weight given to the ML model when blending
_BLEND_CB_WEIGHT = 0.60    # weight given to the consumption-based estimate


def load_bundle(path: Path = MODEL_PATH) -> dict:
    """Load (and cache) the trained model bundle."""
    global _BUNDLE_CACHE
    if _BUNDLE_CACHE is None:
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Run `python -m src.train` first.")
        _BUNDLE_CACHE = joblib.load(path)
    return _BUNDLE_CACHE


def _consumption_based(total_kwh: float,
                       fleet_stats: dict | None) -> tuple[float, float, float]:
    """Return (estimate, plausible_lo, plausible_hi) from energy benchmarks.

    Uses the fleet median kWh/cow when available (computed from training data),
    falling back to the KB midpoint.  The range spans the KB 300–500 kWh/cow
    bounds: fewer cows when each cow uses more, more cows when each uses less.
    """
    if fleet_stats and "kwh_per_cow" in fleet_stats:
        ref = float(fleet_stats["kwh_per_cow"]["median"])
    else:
        ref = _KWH_PER_COW_MID
    est = total_kwh / ref
    lo  = total_kwh / _KWH_PER_COW_HI   # fewest cows (inefficient per cow)
    hi  = total_kwh / _KWH_PER_COW_LO   # most cows (efficient per cow)
    return est, lo, hi


def predict_herd_size(series: pd.DataFrame, bundle: dict | None = None,
                      fleet_stats: dict | None = None) -> dict:
    """Estimate the herd size for one farm's loaded hourly series.

    `series` is the (timestamp, consumption) DataFrame from `dataio`.

    The primary model (GBRT) is accurate *within* the training range but cannot
    extrapolate. When the farm's annual consumption exceeds every training
    farm, the estimate is taken from a linear model instead and flagged as an
    extrapolation.
    """
    bundle = bundle or load_bundle()
    feats = features_from_series(series)
    row = [[feats[name] for name in bundle["feature_names"]]]

    total_kwh = feats["cons_total_kwh"]
    kwh_max = bundle.get("train_total_kwh_max", float("inf"))
    kwh_min = bundle.get("train_total_kwh_min", 0.0)
    cows_lo, cows_hi = bundle.get("train_cows_range", [1, 999])

    in_range = kwh_min <= total_kwh <= kwh_max

    if in_range or "linear_model" not in bundle:
        # Within the trained range — use the accurate primary model.
        raw = float(bundle["model"].predict(row)[0])
        cv = bundle["cv_metrics"][bundle["model_name"]]
        model_used, note = bundle["model_name"], None
        estimate = max(1, int(round(raw)))
        margin = max(cv.get("MAE", 0.0), 1.0)
        plausible_lo = max(1, int(round(raw - margin)))
        plausible_hi = int(round(raw + margin))
    else:
        # Beyond the trained range — blend the linear fit with a
        # consumption-based estimate anchored to fleet kWh/cow benchmarks.
        linear_raw = float(bundle["linear_model"].predict([[total_kwh]])[0])
        cb_est, cb_lo, cb_hi = _consumption_based(total_kwh, fleet_stats)
        raw = _BLEND_ML_WEIGHT * linear_raw + _BLEND_CB_WEIGHT * cb_est
        cv = bundle.get("linear_cv_metrics", {"MAE": 0.0, "R2": 0.0})
        model_used = "energy-benchmarked estimate"
        side = "larger" if total_kwh > kwh_max else "smaller"
        note = (f"This farm's annual consumption ({total_kwh:,.0f} kWh) is "
                f"{side} than farms used for model training "
                f"({kwh_min:,.0f}–{kwh_max:,.0f} kWh). The estimate blends the "
                f"regression model with a consumption-benchmarked estimate "
                f"({cb_est:.0f} cows at the fleet median of "
                f"{total_kwh / cb_est:.0f} kWh/cow/year) and should be "
                f"treated as indicative.")
        estimate = max(1, int(round(raw)))
        # Plausible range spans the KB kWh/cow bounds (300–500 kWh/cow/year).
        plausible_lo = max(1, int(round(cb_lo)))
        plausible_hi = int(round(cb_hi))

    result = {
        "predicted_cows": estimate,
        "raw_prediction": round(raw, 2),
        "plausible_range": [plausible_lo, plausible_hi],
        "model": model_used,
        "model_cv_mae": round(cv.get("MAE", 0.0), 3),
        "model_cv_r2": round(cv.get("R2", 0.0), 4),
        "in_training_range": bool(in_range),
    }
    if note:
        result["extrapolation_note"] = note
    return result


if __name__ == "__main__":
    import sys

    from src.dataio import load_farm_series

    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <farm_csv>")
        raise SystemExit(1)
    series = load_farm_series(sys.argv[1])
    for k, v in predict_herd_size(series).items():
        print(f"  {k}: {v}")
