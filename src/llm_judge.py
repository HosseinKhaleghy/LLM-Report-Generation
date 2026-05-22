"""
LLM-as-judge evaluation for the dairy farm energy advisory.

Scores each section on three dimensions (1-5):
  D1  Factual Grounding   — every cited number traceable to farm data
  D2  Benchmark Integrity — every threshold/range grounded in the KB
  D3  Physical Soundness  — causal reasoning correct for Irish dairy farms

Usage:
    python -m src.llm_judge reports/hourly_report.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import requests

OLLAMA_HOST = "http://localhost:11434"
JUDGE_MODEL = "qwen2.5:14b"
TIMEOUT = 300

SYSTEM_PROMPT = """\
You are an expert evaluator checking whether an AI-generated dairy farm energy \
advisory report is factually grounded, hallucination-free, and physically correct.

Score the section on THREE dimensions, each 1-5:

DIMENSION 1 – FACTUAL GROUNDING
Every numeric value (kWh, Wh, EUR, %, cow count, hours, dates) cited in the \
section must appear verbatim in FARM DATA. Deduct for: values that differ \
from FARM DATA; values that appear to have been computed (e.g. subtracting \
two fields to get a new figure); values not present anywhere in FARM DATA.
5 = every value traceable  |  1 = multiple invented or computed values

DIMENSION 2 – BENCHMARK INTEGRITY
All benchmarks, thresholds, and percentage ranges cited (e.g. "10-15% healthy \
baseline", "25-35% water heating share", "50-60% plate cooler saving") must be \
attributable to the KB TOPICS listed. Deduct for any benchmark that has no \
plausible KB source.
5 = all benchmarks KB-grounded  |  1 = benchmarks are invented

DIMENSION 3 – PHYSICAL SOUNDNESS
Causal reasoning must match Irish dairy farm reality:
- Cold months (e.g. Feb) → higher water-heating demand → HIGHER consumption
- Warm months (e.g. Aug) → lower heating, but higher milk cooling/ventilation
- Low-consumption anomalies = equipment faults or outages, NOT scheduling gaps
- High-consumption anomalies = faults or peak-load events
5 = all reasoning correct  |  1 = wrong direction or misclassified faults

Reply in this EXACT format, nothing else:
D1: <1-5>
D1_note: <one sentence: which values were or were not traceable>
D2: <1-5>
D2_note: <one sentence: which benchmarks matched or did not match KB>
D3: <1-5>
D3_note: <one sentence: physical reasoning quality>
OVERALL: <1-5>
VERDICT: <15 words or fewer>
"""


def _call_judge(section_name: str, section_text: str,
                farm_data: dict, kb_topics: list[str]) -> dict:
    user_msg = (
        f"SECTION: {section_name}\n\n"
        f"FARM DATA (JSON):\n{json.dumps(farm_data, indent=2)}\n\n"
        f"KB TOPICS retrieved:\n"
        + "\n".join(f"- {t}" for t in kb_topics)
        + f"\n\nSECTION TEXT TO EVALUATE:\n{section_text}"
    )
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": JUDGE_MODEL,
            "system": SYSTEM_PROMPT,
            "prompt": user_msg,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 350},
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.json().get("response", "")
    result: dict = {"raw": raw}
    for key in ("D1", "D2", "D3", "OVERALL"):
        m = re.search(rf"^{key}:\s*([1-5])", raw, re.MULTILINE)
        result[key] = int(m.group(1)) if m else None
    for key in ("D1_note", "D2_note", "D3_note", "VERDICT"):
        m = re.search(rf"^{key}:\s*(.+)", raw, re.MULTILINE)
        result[key] = m.group(1).strip() if m else ""
    return result


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


def evaluate(report_path: str) -> None:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    narrative = report.get("llm_narrative", {})
    text = narrative.get("text", "")
    kb_topics = [s["topic"] for s in narrative.get("rag_sources", [])]
    farm_data = _compact_farm_data(report)

    # Split narrative into sections on ## headings
    sections: dict[str, str] = {}
    for part in re.split(r"^## ", text, flags=re.MULTILINE):
        part = part.strip()
        if not part:
            continue
        nl = part.index("\n") if "\n" in part else len(part)
        heading = part[:nl].strip()
        body = part[nl:].strip().replace("_Energy Advisor_", "").strip()
        if heading and body:
            sections[heading] = body

    print(f"\n{'='*72}")
    print(f"  LLM-AS-JUDGE EVALUATION  |  judge: {JUDGE_MODEL}")
    print(f"  Report : {report_path}")
    print(f"{'='*72}\n")
    print(f"  {'Section':<26} {'D1':>4} {'D2':>4} {'D3':>4} {'OVR':>4}  Verdict")
    print(f"  {'-'*68}")

    totals: dict[str, list[int]] = {"D1": [], "D2": [], "D3": [], "OVERALL": []}
    details: list[dict] = []

    for name, body in sections.items():
        result = _call_judge(name, body, farm_data, kb_topics)
        d1 = result.get("D1")
        d2 = result.get("D2")
        d3 = result.get("D3")
        ov = result.get("OVERALL")
        verdict = (result.get("VERDICT") or "")[:44]
        print(f"  {name:<26} {str(d1):>4} {str(d2):>4} {str(d3):>4} {str(ov):>4}  {verdict}")
        for k, v in [("D1", d1), ("D2", d2), ("D3", d3), ("OVERALL", ov)]:
            if isinstance(v, int):
                totals[k].append(v)
        details.append({"section": name, **result})

    def avg(lst: list[int]) -> str:
        return f"{sum(lst)/len(lst):.2f}" if lst else "n/a"

    print(f"  {'-'*68}")
    print(f"  {'MEAN':<26} {avg(totals['D1']):>4} {avg(totals['D2']):>4} "
          f"{avg(totals['D3']):>4} {avg(totals['OVERALL']):>4}")
    print(f"\n  D1=Factual Grounding  D2=Benchmark Integrity  "
          f"D3=Physical Soundness  OVR=Overall\n")

    # Detail notes
    print(f"{'='*72}")
    print("  SECTION NOTES")
    print(f"{'='*72}")
    for d in details:
        print(f"\n  [{d['section']}]")
        if d.get("D1_note"):
            print(f"    D1: {d['D1_note']}")
        if d.get("D2_note"):
            print(f"    D2: {d['D2_note']}")
        if d.get("D3_note"):
            print(f"    D3: {d['D3_note']}")

    # Save full results
    out_path = Path(report_path).with_name(
        Path(report_path).stem + "_judge.json"
    )
    out_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(f"\n  Full results saved to {out_path}\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "reports/hourly_report.json"
    evaluate(path)
