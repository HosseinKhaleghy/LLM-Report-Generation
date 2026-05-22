"""
Anomaly detection from the MOMENT foundation model's reconstruction gap.

MOMENT (`AutonLab/MOMENT-1-large`) reconstructs the farm's hourly consumption
signal zero-shot — no training. An hour is flagged as anomalous when the
**absolute gap** between MOMENT's reconstruction (its prediction of the normal
value) and the actual consumption is large — measured in the consumption unit
itself, so a genuinely big departure is caught regardless of how unusual it
looks in relative terms.

Pipeline (ported from `WP3/farm_report.py :: moment_hourly_scores`)
-------------------------------------------------------------------
1. The hourly series is normalised by the farm's own mean and std.
2. 512-hour windows are slid across it (24-hour stride); MOMENT reconstructs
   each window.
3. The per-hour reconstruction is the mean over the windows covering it,
   then de-normalised back to the consumption unit.
4. residual = actual - reconstruction. An hour is flagged when |residual|
   exceeds `abs_threshold`; left unset it defaults to the 99.5th percentile
   of the farm's own absolute residuals.

The MOMENT weights (~1.4 GB) download from the Hugging Face Hub on first use.
"""

from __future__ import annotations

import os

# Force the PyTorch backend before momentfm imports transformers, so a slow or
# broken TensorFlow install cannot hang the import.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import pandas as pd
import torch

MOMENT_MODEL = "AutonLab/MOMENT-1-large"
SEQ_LEN = 512           # MOMENT's fixed input length
STRIDE = 24             # one day
BATCH_SIZE = 8
HOURS_PER_DAY = 24
_DEFAULT_PCT = 99.5     # auto threshold = this percentile of |residual|

_MODEL_CACHE = None


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def load_moment():
    """Load (and cache, once per process) the pretrained MOMENT pipeline."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        from momentfm import MOMENTPipeline
        model = MOMENTPipeline.from_pretrained(
            MOMENT_MODEL, task_name="anomaly_detection")
        model.init()
        model.eval()
        _MODEL_CACHE = model
    return _MODEL_CACHE


# --------------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------------
def moment_reconstruction(consumption: np.ndarray,
                          batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Per-hour MOMENT reconstruction of the consumption series.

    Returns an array the same length as `consumption`, in the original
    consumption unit — MOMENT's prediction of the "normal" value for each hour.
    """
    series = np.asarray(consumption, dtype=np.float32)
    n = len(series)
    if n < SEQ_LEN:
        raise ValueError(f"Need at least {SEQ_LEN} hourly readings for MOMENT "
                         f"(got {n}).")

    mu = float(series.mean())
    sigma = float(series.std()) + 1e-8
    normed = ((series - mu) / sigma).astype(np.float32)

    starts = list(range(0, n - SEQ_LEN + 1, STRIDE))
    recon_sum = np.zeros(n, dtype=np.float64)
    recon_cnt = np.zeros(n, dtype=np.int64)

    model = load_moment()
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            batch_s = starts[i:i + batch_size]
            bx = torch.tensor(
                np.array([normed[s:s + SEQ_LEN] for s in batch_s]),
                dtype=torch.float32).unsqueeze(1)            # (B, 1, 512)
            mask = torch.ones(bx.shape[0], SEQ_LEN, dtype=torch.long)
            out = model(x_enc=bx, input_mask=mask)
            rec = out.reconstruction.squeeze(1).numpy()      # (B, 512), normed
            for j, s in enumerate(batch_s):
                recon_sum[s:s + SEQ_LEN] += rec[j]
                recon_cnt[s:s + SEQ_LEN] += 1

    recon_cnt[recon_cnt == 0] = 1
    recon_normed = recon_sum / recon_cnt
    return recon_normed * sigma + mu                         # de-normalised


# --------------------------------------------------------------------------
# Report section
# --------------------------------------------------------------------------
def detect(consumption: np.ndarray, timestamps: pd.DatetimeIndex,
           abs_threshold: float | None = None) -> dict:
    """Flag hours where actual consumption departs sharply from MOMENT's
    reconstruction. Returns the `anomaly_detection` section of the report.
    """
    consumption = np.asarray(consumption, dtype=float)
    ts = pd.DatetimeIndex(timestamps)
    hod = ts.hour.to_numpy()

    reconstruction = moment_reconstruction(consumption)
    residual = consumption - reconstruction          # actual minus prediction
    abs_res = np.abs(residual)
    direction = np.where(residual >= 0, "high", "low")

    auto = abs_threshold is None or abs_threshold <= 0
    if auto:
        abs_threshold = float(np.percentile(abs_res, _DEFAULT_PCT))
    flagged = np.where(abs_res > abs_threshold)[0]

    n_high = int(np.sum(direction[flagged] == "high"))
    n_low = int(np.sum(direction[flagged] == "low"))

    def _record(i: int) -> dict:
        return {
            "timestamp": str(ts[i]),
            "hour_of_day": int(hod[i]),
            "consumption": round(float(consumption[i]), 1),
            "expected": round(float(reconstruction[i]), 1),   # MOMENT prediction
            "deviation": round(float(residual[i]), 1),
            "abs_deviation": round(float(abs_res[i]), 1),
            "direction": str(direction[i]),
        }

    top = sorted(flagged, key=lambda i: abs_res[i], reverse=True)[:8]
    top_anomalies = [_record(i) for i in top]
    first_anomaly = _record(int(flagged[0])) if len(flagged) else None

    rate = round(100.0 * len(flagged) / len(consumption), 2) if len(consumption) else 0.0
    return {
        "method": "MOMENT-1-large reconstruction gap (absolute residual)",
        "model": MOMENT_MODEL,
        "description": ("an hour is flagged when the absolute gap between "
                        "MOMENT's reconstruction and the actual consumption "
                        "exceeds the threshold"),
        "abs_deviation_threshold": round(float(abs_threshold), 1),
        "abs_threshold_auto": auto,
        "anomalous_hours": int(len(flagged)),
        "anomaly_rate_pct": rate,
        "high_consumption_events": n_high,
        "low_consumption_events": n_low,
        "largest_deviation": round(float(abs_res.max()), 1) if abs_res.size else 0.0,
        "first_anomaly": first_anomaly,
        "top_anomalies": top_anomalies,
        "assessment": _assess(rate, n_high, n_low),
    }


def _assess(rate: float, n_high: int, n_low: int) -> str:
    if rate == 0:
        return ("No anomalous hours detected; actual consumption stays close "
                "to MOMENT's reconstruction of normal behaviour.")
    sev = "minor" if rate < 0.5 else "moderate" if rate < 2.0 else "significant"
    lean = ("predominantly high-consumption spikes" if n_high > 2 * n_low
            else "predominantly low-consumption dips" if n_low > 2 * n_high
            else "a mix of high and low deviations")
    return (f"{sev.capitalize()} irregularity ({rate}% of hours), {lean}. "
            f"The flagged hours are large absolute departures from what MOMENT "
            f"reconstructs as normal and warrant a maintenance review.")
