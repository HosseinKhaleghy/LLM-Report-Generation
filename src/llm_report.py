"""
Natural-language report generation from the structured farm report.

The structured JSON report is precise but not friendly to a non-technical
dairy farmer.  This module turns it into a short advisory narrative.

It is deliberately *pluggable* across four backends:

  * ``transformers`` - loads a small instruction-tuned LLM locally with Hugging
                       Face Transformers (default ``meta-llama/Llama-3.2-3B-
                       Instruct``).  Fully offline after the first download;
                       the most reliable option for a live demonstration.
  * ``ollama``       - calls a local Ollama server (default ``qwen2.5:7b``).
  * ``template``     - a deterministic, rule-based narrative used as a fallback
                       so the pipeline never breaks.
  * ``auto``         - uses Ollama when reachable, otherwise the template.
                       (It never auto-loads the Transformers model, since that
                       triggers a multi-GB download; select it explicitly.)

The Llama 3.2 weights are gated on the Hugging Face Hub: before first use run
``huggingface-cli login`` and accept the model licence, or switch to a
non-gated model such as ``Qwen/Qwen2.5-3B-Instruct``.
"""

from __future__ import annotations

import json
import os

# Tell Transformers to use the PyTorch backend only. Without this it probes for
# TensorFlow/Flax and imports them if installed; a slow or broken TensorFlow
# install can hang the import for minutes. These must be set before any
# `transformers` import, so they live at module top level.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import requests

DEFAULT_OLLAMA_MODEL = "qwen2.5:14b"
DEFAULT_HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_HOST = "http://localhost:11434"
_TIMEOUT_PROBE = 2      # seconds, for the availability check
_TIMEOUT_GEN = 600      # seconds, for generation (a 7B model on CPU is slow)
_SECTION_TOKENS = 480   # generation cap per report section
_REC_TOKENS     = 600   # recommendations need more room (5 numbered items)


# The report is generated one section at a time — a separate, focused LLM call
# per section. A small model wraps up early when asked for one long report;
# giving each section its own call reliably produces a deep, fully developed
# treatment of every part.
_ANALYST_ROLE = (
    "You are a senior energy analyst preparing an advisory report on ONE dairy "
    "farm for the farmer and their energy advisor, based only on the JSON data "
    "provided. Be specific, quantitative and analytical: interpret every figure "
    "you cite, and explain the likely cause and the practical consequence.\n\n"
    "STRICT RULES — follow all of these without exception:\n"
    "- Quote only numbers that already appear in the FARM DATA below (or, in "
    "the Recommendations section, in the REFERENCE GUIDANCE).\n"
    "- Do NOT compute, derive or estimate new figures — no euro saving "
    "projections, no percentage changes between figures, no annual totals "
    "from hourly values. Quote, never calculate.\n"
    "- Keep the unit exactly as given in the data (Wh stays Wh, kWh stays "
    "kWh); do not relabel one as the other.\n"
    "- Cite benchmarks and thresholds ONLY from the REFERENCE GUIDANCE — "
    "never invent a percentage, range or rule-of-thumb of your own.\n"
    "- 'baseline_share_pct' is the idle load as a share of MEAN hourly "
    "consumption, not of peak consumption — do not confuse these two figures.\n"
    "- Never parenthetically convert a unit: write '4,679 Wh', not "
    "'4,679 Wh (or 4.679 kWh)' or any similar parenthetical. "
    "One unit per value, exactly as given — no exceptions.\n"
    "- Do NOT state or invent a threshold for the anomaly rate (e.g. '1% "
    "threshold'). No such threshold exists in the data or guidance — quote "
    "only the 'assessment' text verbatim.\n"
    "- 'baseline_annual_cost_eur' in the data is the only pre-computed cost "
    "figure for the idle load — quote it directly; never compute your own "
    "phantom or baseline cost figure.\n"
    "- Anomaly hours flagged as low-consumption are fault events (tripped "
    "breaker, equipment failure), NOT off-peak scheduling opportunities — "
    "never suggest that a low-consumption anomaly indicates an opportunity "
    "to schedule loads.\n"
    "- If 'low_consumption_events' is greater than zero, the Anomalies "
    "section MUST mention the count of low-consumption events alongside "
    "the high-consumption count and explain them as likely outages, "
    "tripped breakers or equipment failures. Do not silently ignore them.\n"
    "- Always preserve the sign of a deviation: a negative deviation (low "
    "consumption) must be written as a negative number, e.g. -4,337.1 Wh, "
    "not 4,337.1 Wh.\n"
    "- The 'farm_name' field is a raw file identifier, not a display name. "
    "Never use it in prose. Refer to the farm simply as 'the farm' or "
    "'this farm' throughout the report.\n"
    "- The baseline-vs-healthy-range comparison is pre-computed in the "
    "'baseline_assessment' field. QUOTE 'baseline_assessment.description' "
    "verbatim once when discussing the idle load. NEVER write your own "
    "comparison such as 'above the healthy range', 'within the healthy "
    "range' or 'below the healthy range' — the verdict in the data is the "
    "only correct one. The healthy range of 10-15% may NOT be mentioned "
    "unless it appears in 'baseline_assessment.description'.\n"
    "- The monthly and seasonal driver explanations are pre-computed in "
    "'monthly_drivers' and 'seasonal_drivers'. QUOTE these strings verbatim "
    "when explaining why the highest or lowest month/season is what it is. "
    "Never invent your own water-heating-versus-cooling reasoning, because "
    "the correct driver depends on which month/season is actually highest "
    "and that varies farm to farm.")

# (heading, focus) for each section, generated by its own LLM call.
_SECTIONS = [
    ("Overview",
     "the farm, the period analysed, the headline findings across herd size, "
     "consumption and anomalies, and the single most important action"),
    ("Herd size",
     "the herd-size estimate and plausible range and how confident it is; if "
     "'in_training_range' is false, explain it is an extrapolation beyond the "
     "model's tested range and only indicative, and why"),
    ("Electricity consumption",
     "total annual kWh and cost, average and peak hourly load, and the "
     "baseline/idle load — quote 'baseline_assessment.description' VERBATIM "
     "as the verdict; do NOT write 'above', 'within' or 'below the healthy "
     "range' in any other words; quote 'baseline_annual_cost_eur' for the "
     "idle-load annual cost, do not compute it"),
    ("Hourly analysis",
     "the daily load shape — the peak and off-peak hours, daytime vs overnight "
     "averages — and what it reveals about the milking, water-heating and "
     "milk-cooling routine"),
    ("Monthly analysis",
     "the highest and lowest months (use 'highest_month' and 'lowest_month' "
     "fields — do NOT pick months yourself from any list), their consumption "
     "and cost from 'highest_month_data' and 'lowest_month_data', and the "
     "swing between them — quote 'swing_kwh' and 'swing_mean_wh' directly, "
     "never compute them. For the driver explanation, QUOTE "
     "'monthly_drivers' verbatim — never write your own reasoning about "
     "which month is hot or cold or what drives its consumption"),
    ("Seasonal analysis",
     "a comparison of the four seasons — which draws the most and least power "
     "and by how much — quote 'swing_kwh' from the data directly, never "
     "compute or derive it. For the driver explanation, QUOTE "
     "'seasonal_drivers' verbatim — never write your own reasoning about "
     "which season is hot or cold or what drives its consumption"),
    ("Anomalies",
     "how many hours were flagged and the rate, the FIRST anomaly event in "
     "detail (its date, actual consumption vs MOMENT's reconstruction, the size "
     "of the gap), the most significant events, any pattern among them, and the "
     "most likely physical causes"),
    ("Recommendations",
     "five prioritised actions, ordered by likely financial impact; each one "
     "must name a specific measure from the reference guidance and be tied to "
     "the data figure that motivates it, with the expected benefit"),
]

_GUIDANCE_HEADER = (
    "REFERENCE GUIDANCE — retrieved from the dairy-farm energy knowledge base. "
    "Base the recommendations on these measures and benchmarks:\n\n")


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def generate_llm_report(report: dict, herd_prediction: dict,
                        backend: str = "auto",
                        ollama_model: str = DEFAULT_OLLAMA_MODEL,
                        hf_model: str = DEFAULT_HF_MODEL,
                        host: str = DEFAULT_HOST) -> dict:
    """Produce a narrative report. Returns dict with `backend`, `model`, `text`
    and `rag_sources` (the knowledge-base chunks retrieved to ground it)."""
    payload = _compact_payload(report, herd_prediction)
    guidance, sources = _retrieve_guidance(report)

    # Explicit local-LLM backend (Hugging Face Transformers).
    if backend == "transformers":
        try:
            text = _call_transformers(payload, hf_model, guidance)
            return {"backend": "transformers", "model": hf_model, "text": text,
                    "rag_sources": sources}
        except Exception as exc:  # noqa: BLE001
            return {"backend": "template", "model": "rule-based",
                    "text": _template_report(report, herd_prediction),
                    "error": f"Transformers backend failed: {exc}"}

    # Ollama backend (explicit, or via 'auto' when a server is reachable).
    if backend in ("auto", "ollama") and _ollama_available(host):
        try:
            text = _call_ollama(payload, ollama_model, host, guidance)
            return {"backend": "ollama", "model": ollama_model, "text": text,
                    "rag_sources": sources}
        except Exception as exc:  # noqa: BLE001 - fall back on any failure
            if backend == "ollama":
                return {"backend": "template", "model": "rule-based",
                        "text": _template_report(report, herd_prediction),
                        "error": f"Ollama backend failed: {exc}"}

    # Deterministic fallback.
    return {"backend": "template", "model": "rule-based",
            "text": _template_report(report, herd_prediction)}


def _retrieve_guidance(report: dict) -> tuple[str, list[dict]]:
    """Retrieve KB chunks relevant to this farm.

    Returns the guidance text block to inject into the prompt and a list of
    `{topic, score}` sources for transparency. RAG is optional — any failure
    (model download, missing KB) degrades silently to no guidance.
    """
    try:
        from src import rag
        chunks = rag.retrieve_for_report(report, k=6)
    except Exception:  # noqa: BLE001 - RAG is optional
        return "", []
    if not chunks:
        return "", []
    block = "\n\n".join(f"- {c['topic']}: {c['text']}" for c in chunks)
    sources = [{"topic": c["topic"], "score": c["score"]} for c in chunks]
    return block, sources


# --------------------------------------------------------------------------
# Hugging Face Transformers backend (local inference)
# --------------------------------------------------------------------------
_HF_CACHE: dict = {}    # model_name -> (model, tokenizer); loaded once per process


def _from_pretrained(loader, model_name: str, **kwargs):
    """Load offline from the HF cache first, downloading only if not cached.

    `from_pretrained` normally makes a network metadata check before using the
    cache; on a flaky connection that check can hang even though every file is
    already downloaded. Trying `local_files_only=True` first avoids the hang.
    """
    try:
        return loader.from_pretrained(model_name, local_files_only=True, **kwargs)
    except Exception:  # noqa: BLE001 - not cached; fall back to a real download
        return loader.from_pretrained(model_name, **kwargs)


def _load_hf(model_name: str):
    """Load and cache a Transformers causal-LM model + tokenizer."""
    if model_name in _HF_CACHE:
        return _HF_CACHE[model_name]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Pick the best available device (Apple Silicon MPS, CUDA, or CPU).
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    tokenizer = _from_pretrained(AutoTokenizer, model_name)
    model = _from_pretrained(AutoModelForCausalLM, model_name,
                             torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device).eval()

    _HF_CACHE[model_name] = (model, tokenizer)
    return model, tokenizer


def _section_payload(heading: str, full: dict) -> dict:
    """Slice the full compact payload to only what this section needs.

    Each section sees a focused subset so the model is not tempted to mix up
    figures (e.g. citing a yearly peak as a monthly peak, or using a fleet
    benchmark in a seasonal paragraph).
    """
    base = {"farm_name": full.get("farm_name"),
            "hours_analysed": full.get("hours_analysed")}
    es = full.get("energy_summary", {})
    if heading == "Overview":
        return {**base,
                "herd_size_estimate": full.get("herd_size_estimate"),
                "energy_summary": {k: es.get(k) for k in (
                    "total_consumption_kwh", "baseline_share_pct",
                    "baseline_annual_cost_eur",
                    "estimated_annual_cost_eur", "tariff_eur_per_kwh")},
                "baseline_assessment": full.get("baseline_assessment"),
                "anomaly_detection": {k: full.get("anomaly_detection", {}).get(k)
                                      for k in ("anomalous_hours",
                                                "anomaly_rate_pct")}}
    if heading == "Herd size":
        return {**base, "herd_size_estimate": full.get("herd_size_estimate"),
                "herd_efficiency": full.get("herd_efficiency")}
    if heading == "Electricity consumption":
        return {**base, "energy_summary": es,
                "baseline_assessment": full.get("baseline_assessment"),
                "herd_efficiency": full.get("herd_efficiency"),
                "fleet_benchmark": full.get("fleet_benchmark")}
    if heading == "Hourly analysis":
        return {**base, "hourly_analysis": full.get("hourly_analysis")}
    if heading == "Monthly analysis":
        ma = full.get("monthly_analysis", {})
        return {**base, "monthly_analysis": {
            "highest_month": ma.get("highest_month"),
            "lowest_month":  ma.get("lowest_month"),
            "swing_kwh":     ma.get("swing_kwh"),
            "swing_mean_wh": ma.get("swing_mean_wh"),
            "highest_month_data": next(
                (r for r in ma.get("by_month", [])
                 if r["month"] == ma.get("highest_month")), None),
            "lowest_month_data": next(
                (r for r in ma.get("by_month", [])
                 if r["month"] == ma.get("lowest_month")), None),
            "num_months": len(ma.get("by_month", [])),
        }, "monthly_drivers": full.get("monthly_drivers")}
    if heading == "Seasonal analysis":
        sa = full.get("seasonal_analysis", {})
        # Strip mean_hourly_wh from by_season rows — seasonal ranking is by
        # total kWh only; mean hourly can be opposite direction (confuses model).
        by_season_totals = [
            {"season": r["season"], "total_kwh": r["total_kwh"],
             "hours": r["hours"]}
            for r in sa.get("by_season", [])
        ]
        return {**base, "seasonal_analysis": {
            "highest_season": sa.get("highest_season"),
            "lowest_season":  sa.get("lowest_season"),
            "swing_kwh":      sa.get("swing_kwh"),
            "by_season":      by_season_totals,
            "note": ("Seasons are ranked by TOTAL kWh. Do not compute or "
                     "compare mean hourly values across seasons."),
        }, "seasonal_drivers": full.get("seasonal_drivers")}
    if heading == "Anomalies":
        return {**base, "anomaly_detection": full.get("anomaly_detection"),
                "most_significant_anomalies":
                    full.get("most_significant_anomalies")}
    if heading == "Recommendations":
        return {**base,
                "energy_summary": {k: es.get(k) for k in (
                    "total_consumption_kwh", "baseline_share_pct",
                    "baseline_load_wh", "baseline_annual_cost_eur",
                    "estimated_annual_cost_eur", "tariff_eur_per_kwh")},
                "baseline_assessment": full.get("baseline_assessment"),
                "hourly_analysis": {k: full.get("hourly_analysis", {}).get(k)
                                    for k in ("peak_hour", "off_peak_hour")},
                "anomaly_detection": {k: full.get("anomaly_detection", {}).get(k)
                                      for k in ("high_consumption_events",
                                                "low_consumption_events")},
                "herd_efficiency": full.get("herd_efficiency")}
    return full


def _section_prompt(payload: dict, heading: str, focus: str,
                    guidance: str) -> str:
    """Build the per-section user prompt: trimmed data + (guidance) + instr."""
    slim = _section_payload(heading, payload)
    body = f"FARM DATA (JSON):\n{json.dumps(slim, indent=2)}\n\n"
    if heading == "Recommendations" and guidance:
        body += _GUIDANCE_HEADER + guidance + "\n\n"
    if heading == "Recommendations":
        instr = (f"Write ONLY the '{heading}' section of the advisory report. "
                 f"Cover: {focus}. Give five numbered recommendations; each "
                 f"must name a specific measure from the reference guidance "
                 f"above (plate cooler, heat recovery, variable-speed drive, "
                 f"hot-water-tank insulation and off-peak scheduling, LED "
                 f"lighting, ventilation control). Tie each to a figure that "
                 f"already appears in the FARM DATA above, and state the "
                 f"expected benefit using a phrase from the reference "
                 f"guidance (e.g. 'around 50-60%' for a plate cooler). Do NOT "
                 f"invent euro saving estimates or compute new figures. "
                 f"CRITICAL: low_consumption_events are equipment faults or "
                 f"supply interruptions — NEVER mention them in the off-peak "
                 f"scheduling recommendation and NEVER use them as "
                 f"justification for scheduling recommendations. Off-peak "
                 f"scheduling is justified solely by the tariff and the "
                 f"off_peak_hour value in the data. "
                 f"Do NOT add a markdown heading; do NOT write any other section.")
    else:
        instr = (f"Write ONLY the '{heading}' section of the advisory report. "
                 f"Cover: {focus}. Write exactly two focused paragraphs — "
                 f"every sentence must carry information; do not repeat or "
                 f"pad. Quote only numbers from the FARM DATA above, keep "
                 f"their units exactly as given, and cite benchmarks only "
                 f"from the REFERENCE GUIDANCE — never invent thresholds. "
                 f"Do NOT add a markdown heading; do NOT write any other "
                 f"section.")
    return body + instr


def _call_transformers(payload: dict, model_name: str, guidance: str = "") -> str:
    """Generate the report section by section with a local Hugging Face model."""
    import torch

    model, tokenizer = _load_hf(model_name)
    sections = []
    for heading, focus in _SECTIONS:
        messages = [
            {"role": "system", "content": _ANALYST_ROLE},
            {"role": "user",
             "content": _section_prompt(payload, heading, focus, guidance)},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                inputs, max_new_tokens=_SECTION_TOKENS, do_sample=True,
                temperature=0.35, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inputs.shape[-1]:],
                                skip_special_tokens=True).strip()
        sections.append(f"## {heading}\n\n{text}")
    return "\n\n".join(sections) + "\n\n_Energy Advisor_"


# --------------------------------------------------------------------------
# Ollama backend
# --------------------------------------------------------------------------
def _ollama_available(host: str) -> bool:
    try:
        r = requests.get(f"{host}/api/tags", timeout=_TIMEOUT_PROBE)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _ollama_generate(prompt: str, model: str, host: str,
                     max_tokens: int = _SECTION_TOKENS) -> str:
    """One Ollama completion call."""
    resp = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.35, "num_predict": max_tokens}},
        timeout=_TIMEOUT_GEN,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _call_ollama(payload: dict, model: str, host: str, guidance: str = "") -> str:
    """Generate the report section by section against an Ollama server."""
    sections = []
    for heading, focus in _SECTIONS:
        prompt = (f"{_ANALYST_ROLE}\n\n"
                  f"{_section_prompt(payload, heading, focus, guidance)}\n\n"
                  f"Section text:")
        tokens = _REC_TOKENS if heading == "Recommendations" else _SECTION_TOKENS
        text = _ollama_generate(prompt, model, host, max_tokens=tokens)
        sections.append(f"## {heading}\n\n{text.strip()}")
    return "\n\n".join(sections) + "\n\n_Energy Advisor_"


_COLD_MONTHS = {11, 12, 1, 2}
_WARM_MONTHS = {6, 7, 8}


def _baseline_assessment(share_pct: float | None) -> dict:
    """Pre-classify the baseline share against the healthy 10-15% range.

    The LLM otherwise template-fills 'above the healthy range' for every farm,
    regardless of whether the value is actually below, within or above. We do
    the comparison here so the prompt can demand a verbatim quote.
    """
    if share_pct is None:
        return {}
    low, high = 10.0, 15.0
    if share_pct < low:
        verdict = "below"
        description = (
            f"below the healthy baseline range of {low:.0f}-{high:.0f}%. A "
            f"very low idle load means the farm draws almost no power when "
            f"not actively milking — overnight equipment is essentially off, "
            f"which is efficient but worth a quick check that overnight "
            f"refrigeration is genuinely running on demand and not failing.")
    elif share_pct > high:
        verdict = "above"
        description = (
            f"above the healthy baseline range of {low:.0f}-{high:.0f}%. An "
            f"elevated idle load points to always-on equipment (refrigeration, "
            f"ventilation, water heating) drawing power continuously, even "
            f"when the farm is not actively milking.")
    else:
        verdict = "within"
        description = (
            f"within the healthy baseline range of {low:.0f}-{high:.0f}%. The "
            f"idle load is normal for a working dairy farm and does not "
            f"indicate a phantom-load problem.")
    return {"share_pct": share_pct, "healthy_range_pct": [low, high],
            "verdict": verdict, "description": description}


def _month_temp_class(month_str: str | None) -> str:
    """Classify a 'YYYY-MM' month as 'cold', 'warm' or 'shoulder'."""
    if not month_str or "-" not in month_str:
        return "shoulder"
    try:
        n = int(month_str.split("-")[1])
    except (ValueError, IndexError):
        return "shoulder"
    if n in _COLD_MONTHS:
        return "cold"
    if n in _WARM_MONTHS:
        return "warm"
    return "shoulder"


def _monthly_drivers(highest_month: str | None,
                     lowest_month: str | None) -> str:
    """Compose a physical-driver sentence based on the actual hi/lo months.

    The driver depends on which month is highest: a cold month is driven by
    water heating; a warm month by milk cooling and ventilation. Doing this
    once here avoids the LLM forcing a 'cold = high' template on every farm.
    """
    if not highest_month or not lowest_month:
        return ""
    hi_class = _month_temp_class(highest_month)
    lo_class = _month_temp_class(lowest_month)
    hi_driver = {
        "cold": "higher water-heating demand for udder preparation and parlour "
                "washing during the colder months",
        "warm": "higher milk-cooling and ventilation demand during the warmer "
                "months",
        "shoulder": "shoulder-season demand where both heating and cooling "
                    "are partially active",
    }[hi_class]
    lo_driver = {
        "cold": "reduced cooling and ventilation demand during the colder "
                "months",
        "warm": "reduced water-heating demand during the warmer months",
        "shoulder": "shoulder-season demand where neither heating nor cooling "
                    "is at its peak",
    }[lo_class]
    return (f"The higher consumption in {highest_month} reflects "
            f"{hi_driver}, while the lower consumption in {lowest_month} "
            f"reflects {lo_driver}.")


def _seasonal_drivers(highest_season: str | None,
                      lowest_season: str | None) -> str:
    """Compose a physical-driver sentence based on the actual hi/lo seasons."""
    if not highest_season or not lowest_season:
        return ""
    season_driver = {
        "Winter": ("water-heating demand for udder preparation and parlour "
                   "washing", "reduced cooling and ventilation demand"),
        "Summer": ("milk-cooling and ventilation demand as ambient "
                   "temperatures rise", "reduced water-heating demand"),
        "Spring": ("transition-period demand where heating is tapering and "
                   "cooling is ramping up",
                   "moderate, balanced heating and cooling demand"),
        "Autumn": ("transition-period demand where cooling is tapering and "
                   "heating is ramping up",
                   "moderate, balanced heating and cooling demand"),
    }
    hi_driver = season_driver.get(highest_season, ("seasonal demand", ""))[0]
    lo_driver = season_driver.get(lowest_season, ("", "seasonal demand"))[1]
    return (f"{highest_season} draws the most power, driven by {hi_driver}, "
            f"while {lowest_season} draws the least, reflecting {lo_driver}.")


def _compact_payload(report: dict, herd_prediction: dict) -> dict:
    """Trim the report to the fields the LLM actually needs."""
    es = report.get("energy_summary", {}) or {}
    ma = report.get("monthly_analysis", {}) or {}
    sa = report.get("seasonal_analysis", {}) or {}
    return {
        "farm_name": report.get("farm_name"),
        "hours_analysed": report.get("data_quality", {}).get("hours_analysed"),
        "herd_size_estimate": {
            "predicted_cows": herd_prediction.get("predicted_cows"),
            "plausible_range": herd_prediction.get("plausible_range"),
            "model": herd_prediction.get("model"),
            "in_training_range": herd_prediction.get("in_training_range"),
        },
        "energy_summary": report.get("energy_summary"),
        "baseline_assessment": _baseline_assessment(es.get("baseline_share_pct")),
        "hourly_analysis": {k: report.get("hourly_analysis", {}).get(k)
                            for k in ("hourly_avg_wh", "peak_hour",
                                      "off_peak_hour", "peak_hour_avg_wh",
                                      "off_peak_hour_avg_wh", "night_avg_wh",
                                      "daytime_avg_wh")},
        "monthly_analysis": report.get("monthly_analysis"),
        "monthly_drivers": _monthly_drivers(ma.get("highest_month"),
                                            ma.get("lowest_month")),
        "seasonal_analysis": report.get("seasonal_analysis"),
        "seasonal_drivers": _seasonal_drivers(sa.get("highest_season"),
                                              sa.get("lowest_season")),
        "anomaly_detection": {k: report.get("anomaly_detection", {}).get(k)
                              for k in ("method", "anomalous_hours",
                                        "anomaly_rate_pct",
                                        "abs_deviation_threshold",
                                        "largest_deviation",
                                        "high_consumption_events",
                                        "low_consumption_events",
                                        "first_anomaly", "assessment")},
        "most_significant_anomalies":
            report.get("anomaly_detection", {}).get("top_anomalies", [])[:5],
        "herd_efficiency": report.get("herd_efficiency"),
        "fleet_benchmark": report.get("fleet_benchmark"),
    }


# --------------------------------------------------------------------------
# Deterministic fallback
# --------------------------------------------------------------------------
def _template_report(report: dict, herd: dict) -> str:
    """A longer, structured rule-based report — the offline fallback."""
    es = report.get("energy_summary", {})
    an = report.get("anomaly_detection", {})
    dp = report.get("hourly_analysis", {})
    ma = report.get("monthly_analysis", {})
    sa = report.get("seasonal_analysis", {})
    eff = report.get("herd_efficiency", {})
    dq = report.get("data_quality", {})
    farm = report.get("farm_name", "the farm")
    parts: list[str] = []

    # --- Overview --------------------------------------------------------
    parts.append(
        f"## Overview\n\nThis report covers {dq.get('hours_analysed', '?')} "
        f"hours of hourly electricity data for {farm}. Over the period the "
        f"farm used about {es.get('total_consumption_kwh')} kWh, and "
        f"{an.get('anomalous_hours', 0)} hour(s) stood out as anomalous. The "
        f"sections below break down the herd-size estimate, the energy "
        f"profile, the anomalies found, and what to act on.")

    # --- Herd size -------------------------------------------------------
    rng = herd.get("plausible_range", [None, None])
    hs = (f"## Herd size\n\nThe consumption pattern indicates approximately "
          f"**{herd.get('predicted_cows')} cows** (plausible range "
          f"{rng[0]}-{rng[1]}), estimated with the {herd.get('model')} model.")
    if herd.get("in_training_range") is False:
        hs += (" This farm is larger than any farm the model was trained on, "
               "so the figure is an extrapolation beyond the model's tested "
               "range and should be treated as indicative only.")
    parts.append(hs)

    # --- Electricity consumption ----------------------------------------
    share = es.get("baseline_share_pct", 0) or 0
    base_comment = ("unusually high — a large share of power is drawn even when "
                    "the farm is idle, which points to always-on equipment"
                    if share > 25 else
                    "moderate" if share > 12 else
                    "low, which is healthy")
    parts.append(
        f"## Electricity consumption\n\nTotal use was about "
        f"{es.get('total_consumption_kwh')} kWh, costing roughly "
        f"EUR {es.get('estimated_annual_cost_eur')} at "
        f"{es.get('tariff_eur_per_kwh')} EUR/kWh. Hourly consumption averaged "
        f"{es.get('mean_hourly_wh')} and peaked at {es.get('peak_hourly_wh')}. "
        f"The baseline (idle) load is {es.get('baseline_load_wh')}, about "
        f"{share}% of the average hourly use — {base_comment}.")

    # --- Hourly analysis ------------------------------------------------
    parts.append(
        f"## Hourly analysis\n\nAcross the day, consumption peaks around "
        f"{dp.get('peak_hour')}:00 and is lowest around {dp.get('off_peak_hour')}:00. "
        f"Daytime hours average {dp.get('daytime_avg_wh')} and overnight hours "
        f"{dp.get('night_avg_wh')}. The peaks line up with the morning and "
        f"evening milking sessions and the water heating and milk cooling they "
        f"drive; a high overnight figure points to refrigeration and other "
        f"equipment running continuously through the night.")

    # --- Monthly analysis -----------------------------------------------
    if ma.get("by_month"):
        months = ma["by_month"]
        hi = next((m for m in months if m["month"] == ma.get("highest_month")), None)
        lo = next((m for m in months if m["month"] == ma.get("lowest_month")), None)
        txt = (f"## Monthly analysis\n\nConsumption was recorded across "
               f"{len(months)} calendar months.")
        if hi and lo:
            swing = abs(hi["mean_hourly_wh"] - lo["mean_hourly_wh"])
            txt += (f" The highest average load fell in {hi['month']} "
                    f"({hi['mean_hourly_wh']}/h, {hi['total_kwh']} kWh) and the "
                    f"lowest in {lo['month']} ({lo['mean_hourly_wh']}/h, "
                    f"{lo['total_kwh']} kWh) — a swing of about {swing:.0f} per "
                    f"hour between the busiest and quietest months, which tracks "
                    f"changing water-heating and cooling demand through the year.")
        parts.append(txt)

    # --- Seasonal analysis ----------------------------------------------
    if sa.get("by_season"):
        per = "; ".join(f"{r['season']} {r['total_kwh']:,.0f} kWh "
                        f"({r['mean_hourly_wh']:.0f}/h)" for r in sa["by_season"])
        parts.append(
            f"## Seasonal analysis\n\nBy season: {per}. "
            f"{sa.get('highest_season')} draws the most power and "
            f"{sa.get('lowest_season')} the least. The seasonal swing reflects "
            f"cold-season water heating for udder preparation and parlour "
            f"washing set against warm-season milk cooling and shed ventilation.")

    # --- Anomalies -------------------------------------------------------
    an_text = f"## Anomalies\n\n{an.get('assessment', 'Not assessed.')}"
    fa = an.get("first_anomaly")
    if fa:
        cause = ("equipment switched on outside the normal routine, such as a "
                 "heater or pump left running"
                 if fa.get("direction") == "high"
                 else "a supply interruption or a tripped circuit")
        an_text += (
            f"\n\nThe first flagged hour was **{fa.get('timestamp')}**: "
            f"consumption was {fa.get('consumption')} against MOMENT's "
            f"reconstruction of ~{fa.get('expected')} — a {fa.get('direction')} "
            f"gap of {fa.get('deviation')}. A jump of this size typically "
            f"points to {cause}.")
    parts.append(an_text)

    # --- Efficiency ------------------------------------------------------
    if eff:
        parts.append(
            f"## Efficiency\n\nPer-cow consumption is about "
            f"{eff.get('kwh_per_cow_year')} kWh per cow per year "
            f"({eff.get('rating', 'n/a')} compared with the fleet).")

    # --- Recommendations (dairy-farm specific) --------------------------
    recs: list[str] = []
    if share > 20:
        recs.append("Cut the idle load — the biggest opportunity here. "
                    "Insulate the hot-water tank and pipework and fit a timer "
                    "so water heating runs only ahead of milking, and check "
                    "the bulk-tank refrigeration and vacuum pump are not "
                    "cycling needlessly overnight.")
    recs.append("Install a plate (pre-)cooler on the milk line so the bulk "
                "tank's refrigeration does less work, and recover heat from "
                "the refrigeration compressor to pre-heat parlour wash water.")
    if an.get("high_consumption_events", 0) > 0:
        recs.append("Investigate the flagged high-consumption hours — inspect "
                    "the milk cooling unit and vacuum pump running at those "
                    "times for a fault or short-cycling.")
    if an.get("low_consumption_events", 0) > 0:
        recs.append("Review the flagged low-consumption hours for a tripped "
                    "breaker or milking/cooling equipment that failed to start.")
    recs.append("Fit variable-speed drives on the vacuum and milk pumps, and "
                "move water heating to off-peak hours, to cut both energy use "
                "and tariff cost.")
    if eff.get("rating") == "above-average consumption":
        recs.append("Per-cow consumption is above the fleet median — review "
                    "milking-parlour and cooling efficiency and parlour/shed "
                    "lighting.")
    parts.append("## Recommendations\n\n"
                 + "\n".join(f"{i}. {r}" for i, r in enumerate(recs, 1)))

    return "\n\n".join(parts) + "\n\n_Energy Advisor_"


if __name__ == "__main__":
    import sys
    from pathlib import Path

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    backend = sys.argv[2] if len(sys.argv) > 2 else "auto"
    if path and path.exists():
        rep = json.loads(path.read_text())
        pred = rep.get("herd_size_prediction", {"predicted_cows": "?",
                                                "plausible_range": ["?", "?"],
                                                "model": "?"})
        out = generate_llm_report(rep, pred, backend=backend)
        print(f"[backend: {out['backend']} | model: {out['model']}]")
        if "error" in out:
            print(f"[note: {out['error']}]")
        print()
        print(out["text"])
    else:
        print("Usage: python -m src.llm_report <report.json> "
              "[auto|ollama|transformers|template]")
