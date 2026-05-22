# Dairy Farm Energy Analyser

**A demonstration system for herd-size estimation and LLM-narrated energy reporting from smart-meter data.**

Submitted as a demonstration for [PAAMS 2026](https://paams.net/open-calls/call-demonstrations).

---

## What it does

A customer uploads a real dairy farm's hourly electricity export — just a
**timestamp** column and an **aggregate consumption** column. From that series
alone, the system:

1. **Estimates the herd size** (number of cows) with a machine-learning model.
2. **Builds a structured energy report** — annual consumption and cost,
   idle/phantom load, and hourly, monthly and seasonal breakdowns.
3. **Detects anomalies** with the **MOMENT** foundation model — flagging hours
   whose consumption departs sharply, in absolute terms, from MOMENT's
   reconstruction of the normal signal.
4. **Narrates the report** as a detailed analytical advisory note with a local
   LLM (`qwen2.5:7b` via [Ollama](https://ollama.com)), its recommendations
   **grounded by RAG** in a curated dairy-energy knowledge base.
5. Presents everything through a **Streamlit web interface**.

The pipeline runs **offline on a laptop** — no cloud APIs, no keys.

## Why it is interesting

Herd size is normally self-reported and rarely audited. Recovering it from the
electricity signal alone enables independent verification and lets an energy
advisor profile a farm it has never visited. The milking parlour dominates the
load curve, so herd size is encoded in total consumption and the morning/evening
milking peaks.

---

## Pipeline

```
 real hourly CSV ──▶ load & gap-fill ──▶ herd-size model (GBRT / MLP)
       │                                          │
       │                                          ▼
       └──▶ energy report + MOMENT anomaly detection ──▶ JSON report
                                                  │
                                                  ▼
                                     local LLM (qwen2.5:7b)
                                                  │
                                                  ▼
                                    plain-language advisory note
```

## Two data worlds

- **Synthetic data** (`all_farms_hourly_data.csv`) — 96 simulated farms with
  hourly consumption *and the known number of cows*. Used **only** to train the
  herd-size model, because real smart-meter data has no cow-count labels.
- **Real data** — the customer's actual hourly export. Everything at demo time
  runs on this. `sample_farm.csv` is real WP3 dairy-farm data, provided as a
  sample input and used **as-is** (no anomalies are injected).

## Herd-size model

Each farm is reduced to a **52-dimensional, consumption-only feature vector**
(statistics, percentiles, load-shape descriptors, 24-value hour-of-day profile).
Two regressors are compared under 5-fold cross-validation:

| Model | MAE (cows) | RMSE | R² |
|-------|-----------:|-----:|---:|
| **GBRT** (gradient-boosted trees) | **0.016** | 0.071 | 1.0000 |
| **MLP** (PyTorch, 128-64-32)      | 1.077 | 1.532 | 0.9955 |

GBRT is selected automatically. Both score near-perfectly because the *training*
data is synthetic and near-deterministic.

Tree ensembles cannot extrapolate, so for a farm whose annual consumption
exceeds every training farm the system falls back to a **one-dimensional linear
fit** (herd size vs. total consumption) and flags the estimate as an
extrapolation in the report.

## Anomaly detection — MOMENT reconstruction gap

The detector uses **MOMENT-1-large** (`AutonLab/MOMENT-1-large`), a time-series
foundation model, **zero-shot** — no training (`src/anomaly.py`):

- MOMENT reconstructs the farm's hourly consumption (512-hour windows, 24-hour
  stride); the per-hour reconstruction is de-normalised back to the consumption
  unit — MOMENT's prediction of the *normal* value for that hour.
- An hour is flagged when the **absolute gap** between the actual consumption
  and MOMENT's reconstruction exceeds the threshold — a genuinely large
  departure, not just a relative oddity.
- The threshold is configurable (sidebar / `--abs-threshold`); left at 0 it
  defaults to the 99.5th percentile of the farm's own absolute residuals.

The MOMENT weights (~1.4 GB) download from the Hugging Face Hub on first use.

## Grounded recommendations (RAG)

The numeric analysis is computed, but the *advice* would otherwise rely on the
LLM's hazy memory of dairy-farm energy practice. Instead, the advisory is
**retrieval-augmented** (`src/rag.py`):

- `knowledge/dairy_energy_kb.md` is a curated knowledge base of dairy-energy
  measures, benchmarks and fault signatures (plate coolers, heat recovery,
  vacuum-pump VSDs, water-heating practice, typical kWh/cow, etc.).
- For each farm, a query is built from its situation (high baseline, anomaly
  types, per-cow benchmark) and the most relevant KB chunks are retrieved with
  a sentence-embedding model (`all-MiniLM-L6-v2`, ~80 MB, CPU).
- Those chunks are injected into the prompt; the LLM grounds its
  recommendations in them. The report records which sources were used.

Editing or extending `knowledge/dairy_energy_kb.md` is enough to improve the
advice — the retriever re-embeds the file automatically.

---

## Installation

The project uses a dedicated virtual environment (`.venv`, Python 3.11):

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

For **LLM-generated** narratives, install Ollama and pull the model
(`qwen2.5:7b`, 4-bit quantised, ~4.7 GB — fits an 8 GB laptop):

```bash
brew install ollama          # or download from https://ollama.com
brew services start ollama
ollama pull qwen2.5:7b
```

Any other Ollama model can be used by setting `DEFAULT_OLLAMA_MODEL` in
`src/llm_report.py` (e.g. a smaller one if RAM is tight).

## Usage

All commands use the virtual-environment interpreter (`.venv/bin/python`).

**1. Train the herd-size model:**

```bash
.venv/bin/python -m src.train
```

**2. Build the sample input file** (real WP3 farm data):

```bash
.venv/bin/python make_sample_data.py
```

**3. Launch the demo interface:**

```bash
.venv/bin/python -m streamlit run app.py
```

Upload a farm CSV (or pick the bundled sample) and click **Analyse farm**.

**4. Command line:**

```bash
.venv/bin/python -m src.pipeline sample_farm.csv --llm ollama
```

## LLM backend

The narrative generator is **pluggable** (`src/llm_report.py`):

- `ollama` *(recommended)* — a running Ollama server (`qwen2.5:7b`).
- `template` — deterministic rule-based narrative, fully offline.
- `auto` *(default)* — Ollama if reachable, otherwise the template.

(A `transformers` Llama backend also exists in the code; Ollama is the
recommended default.)

## Project layout

```
all_farms_hourly_data.csv   Synthetic dataset — trains the herd-size model only
sample_farm.csv             Real WP3 farm data — demo input (generated)
make_sample_data.py         Builds sample_farm.csv from WP3/hourly.csv
knowledge/
  dairy_energy_kb.md     Curated dairy-energy knowledge base (RAG source)
src/
  dataio.py          Loads/cleans a real hourly CSV (gap-fills missing hours)
  features.py        Consumption-only herd-size feature engineering
  models.py          PyTorch MLP regressor (sklearn-style wrapper)
  anomaly.py         MOMENT anomaly detector (reconstruction-gap magnitude)
  rag.py             Retriever — embeds the KB, grounds the advisory
  train.py           Train + 5-fold cross-validation, GBRT vs MLP
  predict.py         Herd-size prediction from the saved model
  energy_report.py   Energy summary + anomaly detection -> JSON
  llm_report.py      RAG-grounded LLM narrative (Ollama / template)
  pipeline.py        End-to-end orchestration + CLI
app.py               Streamlit web interface
.venv/               Python 3.11 virtual environment
models/              herd_model.joblib, fleet_stats.json (generated)
reports/             Generated JSON reports
```

## Notes & limitations

- **Herd-size on real data:** the model is trained on synthetic farms (20-90
  cows). A real farm larger than that is outside the trained range; the system
  detects this, switches to the linear extrapolation, and flags the estimate as
  *indicative only*. For accurate large-farm estimates, regenerate the synthetic
  training data spanning the real farms' size range.
- **Anomaly detection** runs on real data as-is; the rate reflects genuine
  irregularities in the recording, not injected faults.
