"""
Batch pipeline + LLM-as-judge for multiple farms.

For each farm number supplied:
  1. Read Aggregate_N.csv (second-level, three-phase kW)
  2. Resample to clean hourly Wh series
  3. Run full analysis pipeline (herd-size, energy, anomaly, LLM narrative)
  4. Run LLM-as-judge on the generated report
  5. Print a summary table across all farms

Usage:
    python -m src.batch_eval 2 4 5 6 7
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

# ── paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
AGGREGATE   = Path(r"C:\Users\22238233\Downloads\Aggregate\Aggregate")
HOURLY_DIR  = ROOT / "data" / "farms"
REPORTS_DIR = ROOT / "reports"

OLLAMA_HOST  = "http://localhost:11434"
JUDGE_MODEL  = "qwen2.5:14b"
JUDGE_TIMEOUT = 300

# ── judge system prompt (same as llm_judge.py) ────────────────────────────
SYSTEM_PROMPT = """\
You are an expert evaluator checking whether an AI-generated dairy farm energy \
advisory report is factually grounded, hallucination-free, and physically correct.

Score the section on THREE dimensions, each 1-5:

DIMENSION 1 – FACTUAL GROUNDING
Every numeric value (kWh, Wh, EUR, %, cow count, hours, dates) cited in the \
section must appear verbatim in FARM DATA. Deduct for: values that differ \
from FARM DATA; values that appear to have been computed; values not present \
anywhere in FARM DATA.
5 = every value traceable  |  1 = multiple invented or computed values

DIMENSION 2 – BENCHMARK INTEGRITY
All benchmarks, thresholds, and percentage ranges cited (e.g. "10-15% healthy \
baseline", "25-35% water heating share", "50-60% plate cooler saving") must be \
attributable to the KB TOPICS listed.
5 = all benchmarks KB-grounded  |  1 = benchmarks are invented

DIMENSION 3 – PHYSICAL SOUNDNESS
Causal reasoning must match Irish dairy farm reality:
- Cold months (e.g. Feb) -> higher water-heating demand -> HIGHER consumption
- Warm months (e.g. Aug) -> lower heating, higher milk cooling/ventilation
- Low-consumption anomalies = faults or outages, NOT scheduling gaps
5 = all reasoning correct  |  1 = wrong direction or misclassified faults

Reply in this EXACT format, nothing else:
D1: <1-5>
D1_note: <one sentence>
D2: <1-5>
D2_note: <one sentence>
D3: <1-5>
D3_note: <one sentence>
OVERALL: <1-5>
VERDICT: <15 words or fewer>
"""


# ── preprocessing ──────────────────────────────────────────────────────────
def resample_to_hourly(farm_num: int) -> Path:
    """Read Aggregate_N.csv in chunks, resample to hourly Wh, save CSV."""
    src = AGGREGATE / f"Aggregate_{farm_num}.csv"
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    dst = HOURLY_DIR / f"farm_{farm_num}_hourly.csv"

    if dst.exists():
        print(f"  [farm {farm_num}] Hourly CSV already exists, skipping resample.")
        return dst

    print(f"  [farm {farm_num}] Resampling {src.name} "
          f"({src.stat().st_size / 1e9:.2f} GB) ...", flush=True)

    hour_sum:   defaultdict[pd.Timestamp, float] = defaultdict(float)
    hour_count: defaultdict[pd.Timestamp, int]   = defaultdict(int)
    chunk_n = 0

    for chunk in pd.read_csv(
        src, chunksize=500_000,
        parse_dates=["Time"],
        dtype={"phase1": "float32", "phase2": "float32", "phase3": "float32"},
        low_memory=False,
    ):
        chunk_n += 1
        chunk["total_kw"] = (
            chunk["phase1"].fillna(0)
            + chunk["phase2"].fillna(0)
            + chunk["phase3"].fillna(0)
        )
        chunk["hour"] = chunk["Time"].dt.floor("h")
        for hour, grp in chunk.groupby("hour", sort=False)["total_kw"]:
            hour_sum[hour]   += float(grp.sum())
            hour_count[hour] += len(grp)
        if chunk_n % 10 == 0:
            print(f"    ... {chunk_n * 500_000:,} rows processed", flush=True)

    hours = sorted(hour_sum.keys())
    wh    = [hour_sum[h] / hour_count[h] for h in hours]   # W mean per hour = Wh

    result = pd.DataFrame({"timestamp": hours, "consumption": wh})
    result.to_csv(dst, index=False)
    print(f"  [farm {farm_num}] Saved {len(result)} hourly rows -> {dst.name}")
    return dst


# ── pipeline ───────────────────────────────────────────────────────────────
def run_pipeline(farm_num: int, hourly_csv: Path) -> dict:
    from src.dataio import load_farm_series
    from src.pipeline import analyse, load_fleet_stats
    from src.predict import load_bundle

    print(f"  [farm {farm_num}] Running analysis pipeline ...", flush=True)
    bundle      = load_bundle()
    fleet_stats = load_fleet_stats()
    series      = load_farm_series(hourly_csv)
    report      = analyse(series, farm_name=f"farm_{farm_num}",
                          bundle=bundle, fleet_stats=fleet_stats,
                          llm_backend="ollama")

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"farm_{farm_num}_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  [farm {farm_num}] Report saved -> {out.name}")
    return report


# ── judge ──────────────────────────────────────────────────────────────────
def _compact_farm_data(report: dict) -> dict:
    es = report.get("energy_summary", {})
    ha = report.get("hourly_analysis", {})
    ma = report.get("monthly_analysis", {})
    sa = report.get("seasonal_analysis", {})
    an = report.get("anomaly_detection", {})
    hs = report.get("herd_size_prediction", {})
    return {
        "hours_analysed": report.get("data_quality", {}).get("hours_analysed"),
        "herd": {
            "predicted_cows": hs.get("predicted_cows"),
            "plausible_range": hs.get("plausible_range"),
            "in_training_range": hs.get("in_training_range"),
        },
        "energy": {
            "total_kwh": es.get("total_consumption_kwh"),
            "mean_hourly_wh": es.get("mean_hourly_wh"),
            "peak_hourly_wh": es.get("peak_hourly_wh"),
            "baseline_wh": es.get("baseline_load_wh"),
            "baseline_share_pct": es.get("baseline_share_pct"),
            "baseline_annual_cost_eur": es.get("baseline_annual_cost_eur"),
            "annual_cost_eur": es.get("estimated_annual_cost_eur"),
            "tariff": es.get("tariff_eur_per_kwh"),
        },
        "hourly": {
            "peak_hour": ha.get("peak_hour"),
            "off_peak_hour": ha.get("off_peak_hour"),
            "peak_avg_wh": ha.get("peak_hour_avg_wh"),
            "off_peak_avg_wh": ha.get("off_peak_hour_avg_wh"),
            "daytime_avg_wh": ha.get("daytime_avg_wh"),
            "night_avg_wh": ha.get("night_avg_wh"),
        },
        "monthly": {
            "highest_month": ma.get("highest_month"),
            "lowest_month": ma.get("lowest_month"),
            "swing_kwh": ma.get("swing_kwh"),
            "swing_mean_wh": ma.get("swing_mean_wh"),
            "by_month": {
                m["month"]: {
                    "total_kwh": m["total_kwh"],
                    "cost_eur": m["estimated_cost_eur"],
                    "mean_hourly_wh": m["mean_hourly_wh"],
                    "peak_hourly_wh": m.get("peak_hourly_wh"),
                }
                for m in ma.get("by_month", [])
            },
        },
        "seasonal": {
            "highest": sa.get("highest_season"),
            "lowest": sa.get("lowest_season"),
            "swing_kwh": sa.get("swing_kwh"),
            "by_season": {
                s["season"]: {
                    "total_kwh": s["total_kwh"],
                    "hours": s["hours"],
                    "mean_hourly_wh": s.get("mean_hourly_wh"),
                }
                for s in sa.get("by_season", [])
            },
        },
        "anomalies": {
            "total": an.get("anomalous_hours"),
            "rate_pct": an.get("anomaly_rate_pct"),
            "high_events": an.get("high_consumption_events"),
            "low_events": an.get("low_consumption_events"),
            "first": an.get("first_anomaly"),
            "top_5": an.get("top_anomalies", [])[:5],
        },
    }


def _parse_scores(text: str) -> dict:
    result: dict = {}
    for key in ("D1", "D2", "D3", "OVERALL"):
        m = re.search(rf"^{key}:\s*([1-5])", text, re.MULTILINE)
        result[key] = int(m.group(1)) if m else None
    for key in ("D1_note", "D2_note", "D3_note", "VERDICT"):
        m = re.search(rf"^{key}:\s*(.+)", text, re.MULTILINE)
        result[key] = m.group(1).strip() if m else ""
    return result


def _judge_section(name: str, body: str, farm_data: dict,
                   kb_topics: list[str]) -> dict:
    user_msg = (
        f"SECTION: {name}\n\n"
        f"FARM DATA (JSON):\n{json.dumps(farm_data, indent=2)}\n\n"
        f"KB TOPICS retrieved:\n"
        + "\n".join(f"- {t}" for t in kb_topics)
        + f"\n\nSECTION TEXT TO EVALUATE:\n{body}"
    )
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": JUDGE_MODEL,
            "system": SYSTEM_PROMPT,
            "prompt": user_msg,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300},
        },
        timeout=JUDGE_TIMEOUT,
    )
    resp.raise_for_status()
    result = _parse_scores(resp.json().get("response", ""))
    return result


def judge_report(farm_num: int, report: dict) -> dict[str, float]:
    """Score all sections; return mean scores."""
    narrative = report.get("llm_narrative", {})
    text      = narrative.get("text", "")
    kb_topics = [s["topic"] for s in narrative.get("rag_sources", [])]
    farm_data = _compact_farm_data(report)

    sections: dict[str, str] = {}
    for part in re.split(r"^## ", text, flags=re.MULTILINE):
        part = part.strip()
        if not part:
            continue
        nl      = part.index("\n") if "\n" in part else len(part)
        heading = part[:nl].strip()
        body    = part[nl:].strip().replace("_Energy Advisor_", "").strip()
        if heading and body:
            sections[heading] = body

    totals: dict[str, list[int]] = {"D1": [], "D2": [], "D3": [], "OVERALL": []}
    print(f"\n  [farm {farm_num}] Judging {len(sections)} sections ...", flush=True)

    for name, body in sections.items():
        result = _judge_section(name, body, farm_data, kb_topics)
        for k in ("D1", "D2", "D3", "OVERALL"):
            v = result.get(k)
            if isinstance(v, int):
                totals[k].append(v)
        verdict = (result.get("VERDICT") or "")[:40]
        d1, d2, d3, ov = result.get("D1"), result.get("D2"), result.get("D3"), result.get("OVERALL")
        print(f"    {name:<26} D1={d1} D2={d2} D3={d3} OVR={ov}  {verdict}")

    def avg(lst: list[int]) -> float | None:
        return round(sum(lst) / len(lst), 2) if lst else None

    return {
        "D1_mean": avg(totals["D1"]),
        "D2_mean": avg(totals["D2"]),
        "D3_mean": avg(totals["D3"]),
        "OVR_mean": avg(totals["OVERALL"]),
        "n_sections": len(sections),
    }


# ── main ───────────────────────────────────────────────────────────────────
def main(farm_nums: list[int]) -> None:
    summary: list[dict] = []

    for n in farm_nums:
        print(f"\n{'='*60}")
        print(f"  FARM {n}")
        print(f"{'='*60}")
        hourly_csv = resample_to_hourly(n)
        report     = run_pipeline(n, hourly_csv)
        scores     = judge_report(n, report)
        summary.append({"farm": n, **scores})

    # ── summary table ──────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  MULTI-FARM JUDGE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Farm':<8} {'D1':>6} {'D2':>6} {'D3':>6} {'OVR':>6}  "
          f"(D1=Factual  D2=Benchmark  D3=Physical)")
    print(f"  {'-'*52}")
    d1s, d2s, d3s, ovrs = [], [], [], []
    for row in summary:
        print(f"  Farm {row['farm']:<3}  {str(row['D1_mean']):>6} "
              f"{str(row['D2_mean']):>6} {str(row['D3_mean']):>6} "
              f"{str(row['OVR_mean']):>6}")
        if row["D1_mean"]: d1s.append(row["D1_mean"])
        if row["D2_mean"]: d2s.append(row["D2_mean"])
        if row["D3_mean"]: d3s.append(row["D3_mean"])
        if row["OVR_mean"]: ovrs.append(row["OVR_mean"])
    print(f"  {'-'*52}")
    def avg(lst): return f"{sum(lst)/len(lst):.2f}" if lst else "n/a"
    print(f"  {'MEAN':<8}  {avg(d1s):>6} {avg(d2s):>6} {avg(d3s):>6} {avg(ovrs):>6}")
    print()

    out = REPORTS_DIR / "batch_judge_summary.json"
    REPORTS_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Summary saved -> {out}\n")


if __name__ == "__main__":
    nums = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [2, 4, 5, 6, 7]
    main(nums)
