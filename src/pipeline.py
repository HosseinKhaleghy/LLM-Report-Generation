"""
End-to-end analysis pipeline for one uploaded farm.

The flow a customer experiences:

    real hourly CSV  ->  clean & gap-fill  ->  herd-size estimate
                     ->  energy + anomaly report  ->  LLM narrative
                     ->  single JSON result

The Streamlit app and the CLI both call into here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.dataio import load_farm_series
from src.energy_report import DEFAULT_TARIFF, build_energy_report
from src.llm_report import generate_llm_report
from src.predict import load_bundle, predict_herd_size

ROOT = Path(__file__).resolve().parents[1]
FLEET_STATS_PATH = ROOT / "models" / "fleet_stats.json"
REPORTS_DIR = ROOT / "reports"


def load_fleet_stats() -> dict | None:
    """Load fleet benchmark statistics if training produced them."""
    if FLEET_STATS_PATH.exists():
        return json.loads(FLEET_STATS_PATH.read_text())
    return None


def analyse(series: pd.DataFrame, *,
            farm_name: str = "uploaded farm",
            bundle: dict | None = None,
            fleet_stats: dict | None = None,
            llm_backend: str = "auto",
            tariff: float = DEFAULT_TARIFF,
            abs_threshold: float | None = None,
            with_llm: bool = True) -> dict:
    """Run the full pipeline on one cleaned farm series (from `dataio`).

    `abs_threshold` sets the magnitude trigger for anomaly detection; omit it
    (or pass 0) to derive it from the data.
    """
    bundle = bundle or load_bundle()
    fleet_stats = fleet_stats if fleet_stats is not None else load_fleet_stats()

    # 1. Herd-size estimate (fleet_stats anchors the consumption-based blend).
    herd = predict_herd_size(series, bundle, fleet_stats=fleet_stats)

    # 2. Energy + anomaly report.
    report = build_energy_report(series, herd_size=herd["predicted_cows"],
                                 fleet_stats=fleet_stats, tariff=tariff,
                                 abs_threshold=abs_threshold)
    report["farm_name"] = farm_name
    report["herd_size_prediction"] = herd

    # 3. Natural-language narrative.
    if with_llm:
        report["llm_narrative"] = generate_llm_report(report, herd,
                                                      backend=llm_backend)
    return report


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Analyse one farm's electricity CSV.")
    ap.add_argument("csv", help="Path to the hourly-data CSV "
                                "(timestamp + aggregate consumption).")
    ap.add_argument("--llm", default="auto",
                    choices=["auto", "ollama", "transformers", "template"],
                    help="LLM backend for the narrative.")
    ap.add_argument("--tariff", type=float, default=DEFAULT_TARIFF,
                    help="Electricity tariff in EUR/kWh.")
    ap.add_argument("--abs-threshold", type=float, default=0.0,
                    help="Flag any hour deviating from the hour-of-day norm by "
                         "more than this absolute amount (0 = auto from data).")
    args = ap.parse_args()

    series = load_farm_series(args.csv)
    report = analyse(series, farm_name=Path(args.csv).stem,
                     llm_backend=args.llm, tariff=args.tariff,
                     abs_threshold=args.abs_threshold)

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"{Path(args.csv).stem}_report.json"
    out.write_text(json.dumps(report, indent=2))

    herd = report["herd_size_prediction"]
    an = report["anomaly_detection"]
    print(f"\nFarm: {report['farm_name']}")
    print(f"  Predicted herd size : {herd['predicted_cows']} cows "
          f"(range {herd['plausible_range']}, model {herd['model']})")
    print(f"  Annual consumption  : "
          f"{report['energy_summary']['total_consumption_kwh']} kWh")
    print(f"  Anomalous hours     : {an['anomalous_hours']} "
          f"({an['anomaly_rate_pct']}%)")
    print(f"\n--- LLM narrative ({report['llm_narrative']['backend']}) ---")
    print(report["llm_narrative"]["text"])
    print(f"\nFull JSON report -> {out}")


if __name__ == "__main__":
    _cli()
