"""
Train and compare two herd-size regressors on the synthetic farm dataset.

The synthetic dataset (`all_farms_hourly_data.csv`) exists solely to *train*
the herd-size model: it provides hourly consumption together with the known
number of cows, which real smart-meter data does not. Once trained, the model
is applied to a real farm's consumption series.

Two models are evaluated under identical 5-fold cross-validation:

  * GBRT  - a gradient-boosted regression-tree ensemble (scikit-learn).
  * MLP   - a PyTorch multilayer perceptron (see `models.TorchMLPRegressor`).

The better model is saved as a joblib bundle, along with fleet-level statistics
used to benchmark a single farm against the rest.

Run:  python -m src.train      (from the project root)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.features import build_training_table, feature_names
from src.models import TorchMLPRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "all_farms_hourly_data.csv"
MODEL_DIR = ROOT / "models"
N_SPLITS = 5
RANDOM_STATE = 42


def _metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _make_gbrt() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.9, random_state=RANDOM_STATE,
    )


def cross_validate(make_model, X: np.ndarray, y: np.ndarray) -> dict:
    """Collect out-of-fold predictions and return aggregate metrics."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        model = make_model()
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    return _metrics(y, oof)


def main() -> None:
    print(f"Loading training dataset: {DATA_CSV}")
    raw = pd.read_csv(DATA_CSV)
    X_df, y_ser = build_training_table(raw)

    feats = feature_names(X_df)
    X = X_df[feats].to_numpy(dtype=float)
    y = y_ser.to_numpy(dtype=float)
    print(f"Built {X.shape[0]} farm samples x {X.shape[1]} consumption features.\n")

    # --- cross-validated comparison -------------------------------------
    print(f"Running {N_SPLITS}-fold cross-validation...")
    gbrt_cv = cross_validate(_make_gbrt, X, y)
    mlp_cv = cross_validate(
        lambda: TorchMLPRegressor(random_state=RANDOM_STATE), X, y)

    print("\n=== Model comparison (cross-validated) ===")
    print(f"{'Model':<8}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    for name, m in (("GBRT", gbrt_cv), ("MLP", mlp_cv)):
        print(f"{name:<8}{m['MAE']:>10.3f}{m['RMSE']:>10.3f}{m['R2']:>10.4f}")

    best_name = "GBRT" if gbrt_cv["MAE"] <= mlp_cv["MAE"] else "MLP"
    print(f"\nBest model by MAE: {best_name}")

    # --- fit the chosen model on all farms ------------------------------
    if best_name == "GBRT":
        best_model = _make_gbrt()
    else:
        best_model = TorchMLPRegressor(random_state=RANDOM_STATE)
    best_model.fit(X, y)
    importances = None
    if best_name == "GBRT":
        importances = dict(sorted(
            zip(feats, best_model.feature_importances_.tolist()),
            key=lambda kv: kv[1], reverse=True))

    # --- linear extrapolation model -------------------------------------
    # GBRT and the MLP cannot predict beyond the training range (tree
    # ensembles clamp to the boundary; a high-dimensional linear fit amplifies
    # noise). A *one-dimensional* fit of herd size on total consumption — the
    # most robust single proxy — extrapolates sanely for farms larger than any
    # in the training set.
    kwh_col = X_df["cons_total_kwh"].to_numpy().reshape(-1, 1)
    linear_model = LinearRegression().fit(kwh_col, y)
    linear_cv = cross_validate(LinearRegression, kwh_col, y)
    print(f"Linear extrapolation model (total kWh -> cows): "
          f"CV MAE {linear_cv['MAE']:.2f}, R2 {linear_cv['R2']:.3f}")

    # --- persist the model bundle ---------------------------------------
    MODEL_DIR.mkdir(exist_ok=True)
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "linear_model": linear_model,
        "linear_cv_metrics": linear_cv,
        "feature_names": feats,
        "cv_metrics": {"GBRT": gbrt_cv, "MLP": mlp_cv},
        "feature_importances": importances,
        "n_train_farms": int(X.shape[0]),
        # Training-range bounds, used to detect out-of-range farms.
        "train_cows_range": [float(y.min()), float(y.max())],
        "train_total_kwh_max": float(X_df["cons_total_kwh"].max()),
        "train_total_kwh_min": float(X_df["cons_total_kwh"].min()),
    }
    joblib.dump(bundle, MODEL_DIR / "herd_model.joblib")
    print(f"Saved model bundle -> {MODEL_DIR / 'herd_model.joblib'}")

    # --- fleet statistics for later benchmarking ------------------------
    fleet = _fleet_stats(X_df, y_ser)
    with open(MODEL_DIR / "fleet_stats.json", "w") as fh:
        json.dump(fleet, fh, indent=2)
    print(f"Saved fleet statistics -> {MODEL_DIR / 'fleet_stats.json'}")

    with open(MODEL_DIR / "model_comparison.json", "w") as fh:
        json.dump({"GBRT": gbrt_cv, "MLP": mlp_cv,
                   "best_model": best_name}, fh, indent=2)
    print(f"Saved comparison report -> {MODEL_DIR / 'model_comparison.json'}")


def _fleet_stats(X_df: pd.DataFrame, y_ser: pd.Series) -> dict:
    """Summarise the fleet so a single farm can be benchmarked against it."""
    total_kwh = X_df["cons_total_kwh"]
    per_cow = total_kwh / y_ser
    return {
        "n_farms": int(len(X_df)),
        "total_kwh": _dist(total_kwh),
        "kwh_per_cow": _dist(per_cow),
        "baseline_wh": _dist(X_df["baseline_wh"]),
        "load_factor": _dist(X_df["load_factor"]),
        "mean_hourly_wh": _dist(X_df["cons_mean"]),
    }


def _dist(s: pd.Series) -> dict:
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


if __name__ == "__main__":
    main()
