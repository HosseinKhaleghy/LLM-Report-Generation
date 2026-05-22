"""
Streamlit demonstration interface — Dairy Farm Energy Analyser.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dataio import (_CONS_CANDIDATES, _TIME_CANDIDATES, _find_column,
                        _read_any, load_farm_series)
from src.energy_report import DEFAULT_TARIFF
from src.energy_report import build_energy_report
from src.llm_report import (_baseline_assessment, _monthly_drivers,
                            _seasonal_drivers, generate_llm_report)
from src.pipeline import load_bundle, load_fleet_stats
from src.predict import predict_herd_size

ROOT = Path(__file__).resolve().parent

# Preloaded demo: if the cached hourly.csv outputs are on disk, the app
# opens straight to that dashboard so it can be shown without running
# the full pipeline. Any uploaded CSV overrides this.
DEMO_REPORT_PATH = ROOT / "reports" / "hourly_report.json"
DEMO_SERIES_PATH = ROOT / "data" / "hourly.csv"
DEMO_AVAILABLE   = DEMO_REPORT_PATH.exists() and DEMO_SERIES_PATH.exists()

# ── colour palette ────────────────────────────────────────────────────────────
_BLUE  = "#2563EB"
_NAVY  = "#1E3A5F"
_RED   = "#DC2626"
_GREEN = "#16A34A"
_AMBER = "#D97706"
_SLATE = "#64748B"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dairy Farm Energy Analyser",
                   page_icon="🐄", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
#MainMenu, footer { visibility: hidden; }

/* hero */
.hero {
    background: linear-gradient(135deg, #1E3A5F 0%, #1d4ed8 100%);
    border-radius: 14px;
    padding: 32px 40px;
    margin-bottom: 28px;
}
.hero h1 {
    font-size: 1.85rem; font-weight: 800; color: white;
    margin: 0 0 8px; letter-spacing: -0.02em;
}
.hero p { font-size: 0.97rem; color: rgba(255,255,255,0.93); margin: 0; line-height: 1.55; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.7rem; font-weight: 600;
    color: rgba(255,255,255,0.9);
    letter-spacing: 0.07em; text-transform: uppercase;
    margin-bottom: 12px;
}

/* KPI cards — dark with white numbers */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
    border: 1px solid #1E3A5F;
    border-top: 4px solid #60A5FA;
    border-radius: 10px;
    padding: 18px 22px 14px;
    box-shadow: 0 2px 10px rgba(30, 58, 95, 0.18);
}
[data-testid="metric-container"] > label,
[data-testid="metric-container"] > label * {
    font-size: 0.72rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
    color: rgba(255, 255, 255, 0.85) !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] * {
    font-size: 2rem !important; font-weight: 900 !important;
    color: #FFFFFF !important; letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * {
    font-size: 0.85rem !important; font-weight: 700 !important;
    color: rgba(255, 255, 255, 0.92) !important;
}
[data-testid="stMetricDelta"] svg { fill: rgba(255, 255, 255, 0.92) !important; }

/* tabs — pill style */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #F1F5F9; padding: 5px;
    border-radius: 10px; border: none; margin-bottom: 8px;
}
.stTabs [data-baseweb="tab"] {
    height: 38px; padding: 0 22px; background: transparent;
    border-radius: 7px; border: none;
    font-size: 13.5px; font-weight: 600; color: #334155;
    white-space: nowrap; transition: background 0.15s, color 0.15s;
}
.stTabs [data-baseweb="tab"]:hover { background: white; color: #1E3A5F; }
.stTabs [aria-selected="true"] {
    background: white !important; color: #2563EB !important;
    box-shadow: 0 1px 5px rgba(0,0,0,0.11);
}

/* section headings */
.sec-head {
    font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.09em;
    color: #475569; margin: 28px 0 10px;
    padding-bottom: 8px; border-bottom: 1px solid #E5E7EB;
}

/* upload zone */
[data-testid="stFileUploader"] > section {
    border: 2px dashed #CBD5E1 !important;
    border-radius: 10px !important;
    background: #F8FAFC !important;
    transition: background 0.15s, border-color 0.15s !important;
}
[data-testid="stFileUploader"] > section:hover {
    background: #EFF6FF !important;
    border-color: #2563EB !important;
}
/* keep the inner text dark on hover (Streamlit otherwise washes it out) */
[data-testid="stFileUploader"] > section *,
[data-testid="stFileUploader"] > section:hover * {
    color: #1E293B !important;
}
[data-testid="stFileUploader"] > section small,
[data-testid="stFileUploader"] > section:hover small {
    color: #475569 !important;
}
/* file uploader's Browse button — keep it legible in both states */
[data-testid="stFileUploader"] button {
    background: #2563EB !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #1D4ED8 !important;
    color: white !important;
}

/* sidebar */
[data-testid="stSidebar"] > div:first-child {
    background: #F8FAFC;
    border-right: 1px solid #E5E7EB;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label {
    font-size: 0.72rem !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 0.06em; color: #334155 !important;
}

/* primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #1E3A5F, #2563EB);
    border: none; border-radius: 8px;
    font-weight: 700; font-size: 15px;
    padding: 10px 36px; letter-spacing: 0.02em; transition: opacity 0.2s;
}
.stButton > button[kind="primary"]:hover { opacity: 0.88; }

/* tables */
[data-testid="stTable"] table { font-size: 0.87rem; }
[data-testid="stTable"] thead tr th {
    background: #F1F5F9 !important; font-weight: 700;
    color: #374151; font-size: 0.75rem;
    text-transform: uppercase; letter-spacing: 0.05em;
}
[data-testid="stTable"] tbody tr:nth-child(even) td { background: #F8FAFC; }

[data-testid="stAlert"] { border-radius: 8px !important; }

/* baseline verdict card */
.verdict-card {
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0 14px;
    border: 1px solid;
    font-size: 0.92rem;
    line-height: 1.5;
}
.verdict-card .vtitle {
    font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin-bottom: 4px;
}
.verdict-above   { background: #FEF2F2; border-color: #FCA5A5; color: #7F1D1D; }
.verdict-above   .vtitle { color: #B91C1C; }
.verdict-within  { background: #ECFDF5; border-color: #6EE7B7; color: #064E3B; }
.verdict-within  .vtitle { color: #047857; }
.verdict-below   { background: #FFFBEB; border-color: #FCD34D; color: #78350F; }
.verdict-below   .vtitle { color: #B45309; }
</style>
""", unsafe_allow_html=True)


# ── cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _bundle():
    return load_bundle()


@st.cache_data(show_spinner=False)
def _read_raw(data: bytes) -> pd.DataFrame:
    return _read_any(data)


# ── chart helpers ─────────────────────────────────────────────────────────────
_AXIS_TITLE = dict(size=13, color="#1E293B", family="Inter, sans-serif")
_TICK_FONT  = dict(size=12, color="#334155", family="Inter, sans-serif")


def _base_layout(title: str, xaxis_title: str, yaxis_title: str,
                 height: int = 330) -> dict:
    return dict(
        template="plotly_white",
        title=dict(text=title,
                   font=dict(size=15, color=_NAVY, family="Inter, sans-serif"),
                   x=0, xanchor="left"),
        xaxis=dict(title=dict(text=xaxis_title, font=_AXIS_TITLE),
                   showgrid=False, tickfont=_TICK_FONT),
        yaxis=dict(title=dict(text=yaxis_title, font=_AXIS_TITLE),
                   gridcolor="#E2E8F0", tickfont=_TICK_FONT),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#1E293B"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=48, b=44, l=56, r=12),
        height=height,
        legend=dict(font=dict(size=12, color="#1E293B")),
    )


def _profile_chart(report: dict) -> go.Figure:
    profile = report["hourly_analysis"]["hourly_avg_wh"]
    peak = report["hourly_analysis"]["peak_hour"]
    colors = [_RED if h == peak else _BLUE for h in range(24)]
    fig = go.Figure(go.Bar(
        x=list(range(24)), y=profile,
        marker_color=colors, marker_line_width=0,
        hovertemplate="Hour %{x}:00<br>%{y:,.0f} Wh<extra></extra>",
    ))
    fig.update_layout(**_base_layout("Average daily load profile",
                                     "Hour of day", "Avg consumption (Wh)"))
    return fig


def _series_chart(series: pd.DataFrame, report: dict) -> go.Figure:
    fa = report["anomaly_detection"].get("first_anomaly")
    fig = go.Figure()
    if fa:
        t0 = pd.Timestamp(fa["timestamp"])
        win = series[(series["timestamp"] >= t0 - pd.Timedelta(hours=5)) &
                     (series["timestamp"] <= t0 + pd.Timedelta(hours=14))]
        fig.add_trace(go.Scatter(
            x=win["timestamp"], y=win["consumption"],
            mode="lines+markers", name="Consumption",
            line=dict(color=_BLUE, width=2), marker=dict(size=4),
            hovertemplate="%{x|%H:%M} · %{y:,.0f} Wh<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[t0], y=[fa["consumption"]], mode="markers", name="First anomaly",
            marker=dict(color=_RED, size=14, symbol="x",
                        line=dict(width=2, color=_RED)),
            hovertemplate=f"{t0:%Y-%m-%d %H:%M}<br>{fa['consumption']:,.0f} Wh (anomaly)<extra></extra>",
        ))
        title = f"First anomaly · {t0:%Y-%m-%d %H:%M}"
    else:
        win = series.head(24)
        fig.add_trace(go.Scatter(
            x=win["timestamp"], y=win["consumption"],
            mode="lines+markers", name="Consumption",
            line=dict(color=_BLUE, width=2),
            hovertemplate="%{x|%H:%M} · %{y:,.0f} Wh<extra></extra>",
        ))
        title = "Consumption (first 24 h) — no anomalies detected"
    fig.update_layout(**_base_layout(title, "Time", "Consumption (Wh)"))
    return fig


def _monthly_chart(report: dict) -> go.Figure:
    rows = report["monthly_analysis"]["by_month"]
    hi = report["monthly_analysis"].get("highest_month")
    colors = [_NAVY if r["month"] == hi else _BLUE for r in rows]
    fig = go.Figure(go.Bar(
        x=[r["month"] for r in rows],
        y=[r["total_kwh"] for r in rows],
        marker_color=colors, marker_line_width=0,
        hovertemplate="%{x}<br>%{y:,.1f} kWh<extra></extra>",
    ))
    fig.update_layout(**_base_layout("Monthly consumption", "Month", "Total (kWh)"))
    fig.update_xaxes(tickangle=-40)
    return fig


def _seasonal_chart(report: dict) -> go.Figure:
    rows = report["seasonal_analysis"]["by_season"]
    palette = {"Winter": _BLUE, "Spring": _GREEN,
               "Summer": _AMBER, "Autumn": "#B45309"}
    fig = go.Figure(go.Bar(
        x=[r["season"] for r in rows],
        y=[r["total_kwh"] for r in rows],
        marker_color=[palette.get(r["season"], _BLUE) for r in rows],
        marker_line_width=0,
        hovertemplate="%{x}<br>%{y:,.1f} kWh<extra></extra>",
    ))
    fig.update_layout(**_base_layout("Consumption by season",
                                     "Season", "Total (kWh)"))
    return fig


# ── table helpers ─────────────────────────────────────────────────────────────
_ENERGY_LABELS = {
    "total_consumption_kwh":     "Total consumption (kWh)",
    "mean_hourly_wh":            "Mean hourly consumption (Wh)",
    "peak_hourly_wh":            "Peak hourly consumption (Wh)",
    "baseline_load_wh":          "Baseline / idle load (Wh)",
    "baseline_annual_cost_eur":  "Baseline annual cost (EUR)",
    "load_factor":               "Load factor",
    "baseline_share_pct":        "Baseline share of mean (%)",
    "estimated_annual_cost_eur": "Estimated annual cost (EUR)",
    "tariff_eur_per_kwh":        "Electricity tariff (EUR/kWh)",
}
_EFF_LABELS = {
    "herd_size_used":           "Herd size used",
    "kwh_per_cow_year":         "kWh per cow / year",
    "fleet_median_kwh_per_cow": "Fleet median (kWh/cow)",
    "vs_fleet_pct":             "vs fleet (%)",
    "rating":                   "Efficiency rating",
}


def _kv_table(d: dict, labels: dict) -> pd.DataFrame:
    rows = []
    for k, v in d.items():
        label = labels.get(k, k.replace("_", " ").capitalize())
        if isinstance(v, bool):
            v = "Yes" if v else "No"
        elif isinstance(v, float):
            v = f"{v:,.2f}"
        elif isinstance(v, int):
            v = f"{v:,}"
        rows.append({"Metric": label, "Value": str(v)})
    return pd.DataFrame(rows)


def _sec(text: str) -> None:
    st.markdown(f'<p class="sec-head">{text}</p>', unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🐄 Farm Energy Analyser")
    st.caption("PAAMS 2026 demonstration")
    st.divider()
    llm_backend = st.selectbox(
        "LLM backend", ["auto", "ollama", "transformers", "template"],
        help="'ollama' calls a local Ollama server. 'auto' uses it when "
             "reachable, otherwise falls back to a rule-based narrative.")
    tariff = st.number_input("Electricity tariff (EUR/kWh)",
                             0.05, 2.0, DEFAULT_TARIFF, 0.01)
    abs_threshold = st.number_input(
        "Anomaly gap threshold", min_value=0.0, value=0.0, step=100.0,
        help="Flag any hour where |actual − MOMENT reconstruction| exceeds "
             "this value. 0 = derive automatically from the data.")
    st.divider()
    st.caption("MOMENT-1-large · GBRT herd model · all-MiniLM-L6-v2 RAG")

try:
    bundle = _bundle()
except FileNotFoundError:
    st.sidebar.error("No trained model. Run `python -m src.train` first.")
    st.stop()


# ── hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="badge">PAAMS 2026 · Demonstration</div>
  <h1>🐄 Dairy Farm Energy Analyser</h1>
  <p>Upload a farm's hourly electricity export with a <strong>timestamp</strong>
     column and an <strong>aggregate consumption</strong> column to estimate the
     herd size and receive an energy report with anomaly detection and a
     plain-language advisory note.</p>
</div>
""", unsafe_allow_html=True)


# ── upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload hourly-data CSV", type="csv",
                            label_visibility="collapsed")

series, farm_name = None, "uploaded farm"
if uploaded is not None:
    raw_df = _read_raw(uploaded.getvalue())
    cols = list(raw_df.columns)
    st.caption(f"Columns detected: {', '.join(map(str, cols))}")
    guess_t = _find_column(cols, _TIME_CANDIDATES)
    guess_c = _find_column(cols, _CONS_CANDIDATES)
    cc1, cc2 = st.columns(2)
    tcol = cc1.selectbox("Timestamp column", cols,
                         index=cols.index(guess_t) if guess_t in cols else 0)
    ccol = cc2.selectbox("Consumption column", cols,
                         index=cols.index(guess_c) if guess_c in cols
                         else min(1, len(cols) - 1))
    try:
        series = load_farm_series(raw_df, timestamp_col=tcol, consumption_col=ccol)
        farm_name = uploaded.name.rsplit(".", 1)[0]
    except Exception as exc:
        st.error(f"Could not load the file with those columns: {exc}")
        st.stop()

# Default path: no upload, but the pre-computed farm-6 demo is on disk.
# Skip the pipeline and render the cached report straight away.
use_demo = (uploaded is None) and DEMO_AVAILABLE

if series is None and not use_demo:
    st.info("Upload a CSV file above to begin the analysis.")
    st.stop()

if use_demo:
    st.info("Showing the preloaded demonstration farm. Upload a CSV above "
            "to run the full pipeline on your own data.", icon="🐄")
    series = load_farm_series(DEMO_SERIES_PATH)
    report = json.loads(DEMO_REPORT_PATH.read_text(encoding="utf-8"))
    farm_name = report.get("farm_name", "demo farm")
    st.caption(
        f"Loaded **{len(series):,}** hourly readings · "
        f"{series['timestamp'].iloc[0]:%Y-%m-%d} → "
        f"{series['timestamp'].iloc[-1]:%Y-%m-%d}"
    )
else:
    st.caption(
        f"Loaded **{len(series):,}** hourly readings · "
        f"{series['timestamp'].iloc[0]:%Y-%m-%d} → "
        f"{series['timestamp'].iloc[-1]:%Y-%m-%d}"
    )

    if not st.button("Analyse farm", type="primary"):
        st.stop()

    # ── run pipeline ──────────────────────────────────────────────────────────
    fleet_stats = load_fleet_stats()
    bar = st.progress(0, text="Starting analysis...")

    # Herd-size estimation (fast, ~1 s)
    bar.progress(5, text="Estimating herd size...")
    herd = predict_herd_size(series, bundle, fleet_stats=fleet_stats)
    bar.progress(18, text="Herd size estimated")

    # Energy analysis + anomaly detection (slow, ~60-120 s) — animate bar in bg
    _box: dict = {}

    def _run_energy() -> None:
        _box["report"] = build_energy_report(
            series, herd_size=herd["predicted_cows"],
            fleet_stats=fleet_stats, tariff=tariff,
            abs_threshold=abs_threshold,
        )

    _t = threading.Thread(target=_run_energy, daemon=True)
    _t.start()
    _p = 18
    while _t.is_alive():
        time.sleep(2.5)
        _p = min(_p + 1, 57)
        bar.progress(_p, text="Detecting anomalies...")
    _t.join()

    report = _box["report"]
    report["farm_name"]           = farm_name
    report["herd_size_prediction"] = herd
    bar.progress(60, text="Anomaly detection complete")

    # LLM narrative + RAG retrieval (slow, ~3-5 min) — animate bar in bg
    def _run_llm() -> None:
        _box["narrative"] = generate_llm_report(report, herd, backend=llm_backend)

    _t = threading.Thread(target=_run_llm, daemon=True)
    _t.start()
    _p = 60
    while _t.is_alive():
        time.sleep(5)
        _p = min(_p + 1, 97)
        bar.progress(_p, text="Generating advisory report...")
    _t.join()

    report["llm_narrative"] = _box["narrative"]
    bar.progress(100, text="Analysis complete!")
    time.sleep(0.4)
    bar.empty()

herd = report["herd_size_prediction"]
es   = report["energy_summary"]
an   = report["anomaly_detection"]

if not use_demo:
    st.success(f"Analysis complete for **{farm_name}**", icon="✅")
st.markdown("<br>", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Estimated herd size", f"{herd['predicted_cows']} cows",
          help=f"Plausible range {herd['plausible_range']} · model: {herd['model']}")
k2.metric("Annual consumption", f"{es['total_consumption_kwh']:,.0f} kWh")
k3.metric("Estimated annual cost", f"€ {es['estimated_annual_cost_eur']:,.0f}")
k4.metric("Anomalous hours", str(an["anomalous_hours"]),
          f"{an['anomaly_rate_pct']} % of hours")

st.markdown("<br>", unsafe_allow_html=True)

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Energy & Load Profile",
    "📅  Monthly & Seasonal",
    "⚠️  Anomalies",
    "📝  Advisory Report",
])

# tab 1 ── Energy & Load Profile
with tab1:
    c1, c2 = st.columns(2)
    c1.plotly_chart(_profile_chart(report), use_container_width=True)
    c2.plotly_chart(_series_chart(series, report), use_container_width=True)
    t1, t2 = st.columns(2)
    with t1:
        _sec("Energy summary")
        st.table(_kv_table(es, _ENERGY_LABELS))
    if "herd_efficiency" in report:
        with t2:
            _sec("Herd efficiency")
            st.table(_kv_table(report["herd_efficiency"], _EFF_LABELS))

# tab 2 ── Monthly & Seasonal
with tab2:
    ma = report.get("monthly_analysis", {})
    sa = report.get("seasonal_analysis", {})

    if ma.get("by_month"):
        _sec("Monthly consumption")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Peak month", ma.get("highest_month") or "—")
        col_b.metric("Lowest month", ma.get("lowest_month") or "—")
        if ma.get("swing_kwh") is not None:
            col_c.metric("Month-to-month swing", f"{ma['swing_kwh']:,.1f} kWh")
        m_drivers = _monthly_drivers(ma.get("highest_month"), ma.get("lowest_month"))
        if m_drivers:
            st.info(m_drivers, icon="🌡️")
        st.plotly_chart(_monthly_chart(report), use_container_width=True)
        st.dataframe(pd.DataFrame(ma["by_month"]),
                     use_container_width=True, hide_index=True)

    if sa.get("by_season"):
        _sec("Seasonal consumption")
        s1, s2, s3 = st.columns(3)
        s1.metric("Highest season", sa.get("highest_season") or "—")
        s2.metric("Lowest season", sa.get("lowest_season") or "—")
        if sa.get("swing_kwh") is not None:
            s3.metric("Seasonal swing", f"{sa['swing_kwh']:,.1f} kWh")
        s_drivers = _seasonal_drivers(sa.get("highest_season"), sa.get("lowest_season"))
        if s_drivers:
            st.info(s_drivers, icon="🌡️")
        sc1, sc2 = st.columns([2, 3])
        sc1.plotly_chart(_seasonal_chart(report), use_container_width=True)
        sc2.dataframe(pd.DataFrame(sa["by_season"]),
                      use_container_width=True, hide_index=True)

# tab 3 ── Anomalies
with tab3:
    a1, a2, a3 = st.columns(3)
    a1.metric("Anomalous hours", an["anomalous_hours"])
    a2.metric("High-consumption events", an["high_consumption_events"])
    a3.metric("Low-consumption events", an["low_consumption_events"])
    st.markdown("<br>", unsafe_allow_html=True)

    st.info(an["assessment"], icon="ℹ️")

    if an["top_anomalies"]:
        _sec("Most significant anomalies")
        st.dataframe(pd.DataFrame(an["top_anomalies"]),
                     use_container_width=True, hide_index=True)
    else:
        st.success("No anomalies detected.", icon="✅")

# tab 4 ── Advisory Report
with tab4:
    nar = report["llm_narrative"]
    col_meta, col_dl = st.columns([4, 1])
    col_meta.caption(
        f"Generated by **{nar['backend']}** · model: `{nar['model']}`"
    )
    col_dl.download_button(
        "⬇️ Download JSON",
        data=json.dumps(report, indent=2),
        file_name=f"{farm_name}_report.json",
        mime="application/json",
    )
    if "error" in nar:
        st.warning(
            f"Requested backend unavailable — used fallback. {nar['error']}",
            icon="⚠️"
        )
    st.divider()
    st.markdown(nar["text"])

    sources = nar.get("rag_sources")
    if sources:
        with st.expander(
            f"📚 Knowledge-base sources used to ground this report "
            f"({len(sources)} chunks)"
        ):
            for s in sources:
                st.markdown(
                    f"- **{s['topic']}** &nbsp;·&nbsp; "
                    f"relevance score {round(float(s['score']), 3)}"
                )
