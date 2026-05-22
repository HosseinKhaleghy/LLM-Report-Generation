"""
Electricity-consumption report for a single real farm.

Given a cleaned hourly series (from `dataio.load_farm_series`), this module
produces a structured, JSON-serialisable report covering:

  * an energy summary (annual total, peak, baseline/phantom load, cost);
  * the average daily load profile;
  * anomaly detection via the pre-trained WP3 LSTM autoencoder (`src/anomaly`);
  * optional herd-efficiency and fleet-benchmark sections.

The report is compact and self-describing so it can be handed straight to a
language model for narration (see `llm_report.py`).
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from src import anomaly
from src.dataio import data_quality

DEFAULT_TARIFF = 0.30           # EUR per kWh
HOURS_PER_DAY = 24


def build_energy_report(series: pd.DataFrame, *,
                        herd_size: int | None = None,
                        fleet_stats: dict | None = None,
                        tariff: float = DEFAULT_TARIFF,
                        abs_threshold: float | None = None) -> dict:
    """Return a JSON-serialisable energy + anomaly report for one farm.

    `series` is the (timestamp, consumption) DataFrame from `dataio`.
    `abs_threshold` sets the magnitude trigger for anomaly detection (see
    `anomaly.detect`); when omitted it is derived from the data.
    """
    cons = series["consumption"].to_numpy(dtype=float)
    ts = pd.DatetimeIndex(series["timestamp"])
    hod = ts.hour.to_numpy()

    report: dict = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "data_quality": data_quality(series),
        "energy_summary": _energy_summary(cons, tariff),
        "hourly_analysis": _daily_profile(cons, hod),
        "monthly_analysis": _monthly_analysis(cons, ts, tariff),
        "seasonal_analysis": _seasonal_analysis(cons, ts),
        "anomaly_detection": _anomaly_section(cons, ts, abs_threshold),
    }
    if herd_size:
        report["herd_efficiency"] = _herd_efficiency(cons, herd_size, fleet_stats)
    if fleet_stats:
        report["fleet_benchmark"] = _fleet_benchmark(cons, fleet_stats)
    return report


def _anomaly_section(cons: np.ndarray, ts: pd.DatetimeIndex,
                     abs_threshold: float | None = None) -> dict:
    """Run the anomaly detector; degrade gracefully on failure."""
    try:
        return anomaly.detect(cons, ts, abs_threshold=abs_threshold)
    except Exception as exc:  # noqa: BLE001 - never let detection break the report
        return {"method": "hour-of-day magnitude deviation",
                "error": f"detector unavailable: {exc}",
                "anomalous_hours": 0, "anomaly_rate_pct": 0.0,
                "high_consumption_events": 0, "low_consumption_events": 0,
                "top_anomalies": [],
                "assessment": "Anomaly detection could not be run."}


def _energy_summary(cons: np.ndarray, tariff: float) -> dict:
    valid = cons[np.isfinite(cons)]
    total_kwh = float(valid.sum()) / 1000.0
    peak, mean = float(valid.max()), float(valid.mean())
    baseline = float(np.percentile(valid, 5))
    baseline_annual_cost = round(baseline / 1000.0 * 8760 * tariff, 2)
    return {
        "total_consumption_kwh": round(total_kwh, 1),
        "mean_hourly_wh": round(mean, 1),
        "peak_hourly_wh": round(peak, 1),
        "baseline_load_wh": round(baseline, 1),
        "baseline_annual_cost_eur": baseline_annual_cost,
        "load_factor": round(mean / peak, 3) if peak else 0.0,
        "baseline_share_pct": round(100.0 * baseline / mean, 1) if mean else 0.0,
        "estimated_annual_cost_eur": round(total_kwh * tariff, 2),
        "tariff_eur_per_kwh": tariff,
    }


def _daily_profile(cons: np.ndarray, hod: np.ndarray) -> dict:
    """Hourly analysis: the average load at each hour of day (0-23)."""
    profile = np.array([
        cons[(hod == h) & np.isfinite(cons)].mean()
        if np.any((hod == h) & np.isfinite(cons)) else 0.0
        for h in range(HOURS_PER_DAY)
    ])
    night = profile[[22, 23, 0, 1, 2, 3, 4, 5]].mean()
    day = profile[[8, 9, 10, 11, 12, 13, 14, 15, 16, 17]].mean()
    return {
        "hourly_avg_wh": [round(float(v), 1) for v in profile],
        "peak_hour": int(np.argmax(profile)),
        "off_peak_hour": int(np.argmin(profile)),
        "peak_hour_avg_wh": round(float(profile.max()), 1),
        "off_peak_hour_avg_wh": round(float(profile.min()), 1),
        "night_avg_wh": round(float(night), 1),
        "daytime_avg_wh": round(float(day), 1),
    }


_SEASON = {12: "Winter", 1: "Winter", 2: "Winter",
           3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer",
           9: "Autumn", 10: "Autumn", 11: "Autumn"}


def _monthly_analysis(cons: np.ndarray, ts: pd.DatetimeIndex,
                      tariff: float) -> dict:
    """Per calendar-month totals, costs and average load."""
    s = pd.Series(cons, index=ts)
    rows = []
    for period, grp in s.groupby(s.index.to_period("M")):
        valid = grp[np.isfinite(grp)]
        total_kwh = float(valid.sum()) / 1000.0
        rows.append({
            "month": str(period),
            "total_kwh": round(total_kwh, 1),
            "estimated_cost_eur": round(total_kwh * tariff, 2),
            "mean_hourly_wh": round(float(valid.mean()), 1) if len(valid) else 0.0,
            "peak_hourly_wh": round(float(valid.max()), 1) if len(valid) else 0.0,
            "hours": int(len(grp)),
        })
    # Rank by average load, and only over reasonably complete months (>= 600 h)
    # so a partial first/last month cannot distort the highest/lowest pick.
    rate = {r["month"]: r["mean_hourly_wh"] for r in rows if r["hours"] >= 600}
    if not rate:
        rate = {r["month"]: r["mean_hourly_wh"] for r in rows}
    hi_month = max(rate, key=rate.get) if rate else None
    lo_month = min(rate, key=rate.get) if rate else None
    hi_row = next((r for r in rows if r["month"] == hi_month), None)
    lo_row = next((r for r in rows if r["month"] == lo_month), None)
    return {
        "by_month": rows,
        "highest_month": hi_month,
        "lowest_month": lo_month,
        "swing_kwh": round(hi_row["total_kwh"] - lo_row["total_kwh"], 1)
                     if hi_row and lo_row else None,
        "swing_mean_wh": round(hi_row["mean_hourly_wh"] - lo_row["mean_hourly_wh"], 1)
                         if hi_row and lo_row else None,
    }


def _seasonal_analysis(cons: np.ndarray, ts: pd.DatetimeIndex) -> dict:
    """Consumption grouped into the four meteorological seasons."""
    s = pd.Series(cons, index=ts)
    season = np.array([_SEASON[m] for m in ts.month])
    rows = []
    for name in ("Winter", "Spring", "Summer", "Autumn"):
        grp = s[season == name]
        valid = grp[np.isfinite(grp)]
        if len(valid) == 0:
            continue
        rows.append({
            "season": name,
            "total_kwh": round(float(valid.sum()) / 1000.0, 1),
            "mean_hourly_wh": round(float(valid.mean()), 1),
            "hours": int(len(grp)),
        })
    # Rank by total consumption — each season spans ~3 months, so totals are
    # the natural, unambiguous "which season uses most".
    total = {r["season"]: r["total_kwh"] for r in rows}
    hi_season = max(total, key=total.get) if total else None
    lo_season = min(total, key=total.get) if total else None
    hi_row = next((r for r in rows if r["season"] == hi_season), None)
    lo_row = next((r for r in rows if r["season"] == lo_season), None)
    return {
        "by_season": rows,
        "highest_season": hi_season,
        "lowest_season": lo_season,
        "swing_kwh": round(hi_row["total_kwh"] - lo_row["total_kwh"], 1)
                     if hi_row and lo_row else None,
        "swing_mean_wh": round(hi_row["mean_hourly_wh"] - lo_row["mean_hourly_wh"], 1)
                         if hi_row and lo_row else None,
    }


def _herd_efficiency(cons: np.ndarray, herd_size: int,
                     fleet_stats: dict | None) -> dict:
    total_kwh = float(cons[np.isfinite(cons)].sum()) / 1000.0
    per_cow = total_kwh / herd_size
    out = {"herd_size_used": int(herd_size),
           "kwh_per_cow_year": round(per_cow, 1)}
    if fleet_stats and "kwh_per_cow" in fleet_stats:
        ref = fleet_stats["kwh_per_cow"]["median"]
        out["fleet_median_kwh_per_cow"] = round(ref, 1)
        out["vs_fleet_pct"] = round(100.0 * (per_cow - ref) / ref, 1)
        out["rating"] = ("efficient" if per_cow < ref * 0.95
                         else "typical" if per_cow <= ref * 1.05
                         else "above-average consumption")
    return out


def _fleet_benchmark(cons: np.ndarray, fleet_stats: dict) -> dict:
    valid = cons[np.isfinite(cons)]
    total_kwh = float(valid.sum()) / 1000.0
    baseline = float(np.percentile(valid, 5))
    ref_base = fleet_stats.get("baseline_wh", {})
    return {
        "total_kwh_percentile": _percentile_of(total_kwh,
                                               fleet_stats.get("total_kwh", {})),
        "baseline_vs_fleet": ("elevated phantom load"
                              if baseline > ref_base.get("p75", 1e18)
                              else "normal idle load"),
        "fleet_size": fleet_stats.get("n_farms"),
    }


def _percentile_of(value: float, ref: dict) -> str:
    if not ref:
        return "unknown"
    if value <= ref.get("p25", float("-inf")):
        return "bottom 25% (low consumer)"
    if value <= ref.get("median", float("-inf")):
        return "lower-middle"
    if value <= ref.get("p75", float("inf")):
        return "upper-middle"
    return "top 25% (high consumer)"
