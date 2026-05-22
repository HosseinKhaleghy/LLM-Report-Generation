# Dairy Farm Energy Analyser

**A demonstration system for herd size estimation and locally narrated energy reporting from smart meter data.**

Submitted as a demonstration for [PAAMS 2026](https://paams.net/open-calls/call-demonstrations).

## What it does

A customer uploads a real dairy farm's hourly electricity export with just a **timestamp** column and an **aggregate consumption** column. From that series alone, the system:

1. **Estimates the herd size** (number of cows) with a machine learning model.
2. **Builds a structured energy report** covering annual consumption and cost, idle or phantom load, and hourly, monthly and seasonal breakdowns.
3. **Detects anomalies** with the **MOMENT** foundation model, flagging hours whose consumption departs sharply, in absolute terms, from MOMENT's reconstruction of the normal signal.
4. **Narrates the report** as a detailed analytical advisory note with a local LLM (`qwen2.5:14b` via [Ollama](https://ollama.com)), with recommendations **grounded by RAG** in a curated dairy energy knowledge base.
5. Presents everything through a **Streamlit web interface**.

The pipeline runs **offline**. No cloud APIs, no keys.

## Why it is interesting

Herd size is normally self reported and rarely audited. Recovering it from the electricity signal alone enables independent verification and lets an energy advisor profile a farm it has never visited. The milking parlour dominates the load curve, so herd size is encoded in total consumption and the morning and evening milking peaks.

## Pipeline

```
 real hourly CSV ──▶ load and gap fill ──▶ herd size model (GBRT or MLP)
       │                                          │
       │                                          ▼
       └──▶ energy report + MOMENT anomaly detection ──▶ JSON report
                                                  │
                                                  ▼
                                     local LLM (qwen2.5:14b)
                                                  │
                                                  ▼
                                    plain language advisory note
```

## Two data worlds

- **Synthetic data** (`all_farms_hourly_data.csv`): 96 simulated farms with hourly consumption *and the known number of cows*. Used **only** to train the herd size model, because real smart meter data has no cow count labels.
- **Real data**: the customer's actual hourly export. Everything at demo time runs on this. `sample_farm.csv` is real WP3 dairy farm data, provided as a sample input and used unchanged (no anomalies are injected).

## Herd size model

Each farm is reduced to a **52 dimensional consumption only feature vector** (statistics, percentiles, load shape descriptors, and a 24 value hour of day profile). Two regressors are compared under 5 fold cross validation:

| Model | MAE (cows) | RMSE | R² |
|-------|-----------:|-----:|---:|
| **GBRT** (gradient boosted trees) | **0.016** | 0.071 | 1.0000 |
| **MLP** (PyTorch, 128, 64, 32)    | 1.077     | 1.532 | 0.9955 |

GBRT is selected automatically. Both score almost perfectly because the *training* data is synthetic and almost deterministic.

Tree ensembles cannot extrapolate, so for a farm whose annual consumption exceeds every training farm the system falls back to a **one dimensional linear fit** (herd size against total consumption) and flags the estimate as an extrapolation in the report.

## Anomaly detection with MOMENT reconstruction gap

The detector uses **MOMENT-1-large** (`AutonLab/MOMENT-1-large`), a time series foundation model, in **zero shot** mode with no training (`src/anomaly.py`):

- MOMENT reconstructs the farm's hourly consumption using 512 hour windows with a 24 hour stride. The per hour reconstruction is denormalised back to the consumption unit, giving MOMENT's prediction of the *normal* value for that hour.
- An hour is flagged when the **absolute gap** between the actual consumption and MOMENT's reconstruction exceeds the threshold. A genuinely large departure, not just a relative oddity.
- The threshold is configurable (sidebar or `--abs-threshold`). Left at 0 it defaults to the 99.5th percentile of the farm's own absolute residuals.

The MOMENT weights (around 1.4 GB) download from the Hugging Face Hub on first use.

## Grounded recommendations with RAG

The numeric analysis is computed, but the *advice* would otherwise rely on the LLM's hazy memory of dairy farm energy practice. Instead, the advisory is **retrieval augmented** (`src/rag.py`):

- `knowledge/dairy_energy_kb.md` is a curated knowledge base of dairy energy measures, benchmarks and fault signatures (plate coolers, heat recovery, vacuum pump VSDs, water heating practice, typical kWh per cow, and so on).
- For each farm, a query is built from its situation (high baseline, anomaly types, per cow benchmark) and the most relevant KB chunks are retrieved with a sentence embedding model (`all-MiniLM-L6-v2`, around 80 MB, CPU).
- Those chunks are injected into the prompt; the LLM grounds its recommendations in them. The report records which sources were used.

Editing or extending `knowledge/dairy_energy_kb.md` is enough to improve the advice. The retriever embeds the file again automatically.

## Installation

Tested on Python 3.11. You also need around 15 GB of free disk space for the Ollama model and the MOMENT weights, and at least 12 GB of RAM to run `qwen2.5:14b` comfortably.

### 1. Clone the repository

```bash
git clone https://github.com/HosseinKhaleghy/LLM-Report-Generation.git
cd LLM-Report-Generation
```

### 2. Create the Python environment

**Linux (Ubuntu or Debian based):**

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv git
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**macOS:**

```bash
brew install python@3.11 git
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**

Install Python 3.11 from https://www.python.org/downloads/ (tick "Add python.exe to PATH" during setup) and Git from https://git-scm.com/download/win. Then open PowerShell in the project folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script, allow user scripts once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Alternatively use Command Prompt instead of PowerShell and activate with `.venv\Scripts\activate.bat`.

### 3. Install Ollama and pull the model

The narrative generator calls a local LLM through Ollama. The default model is `qwen2.5:14b` (4 bit quantised, around 9 GB on disk).

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:14b
```

The install script registers Ollama as a system service that starts on boot.

**macOS:**

```bash
brew install ollama
brew services start ollama
ollama pull qwen2.5:14b
```

**Windows:**

Download and run the installer from https://ollama.com/download/windows. Ollama runs as a background service after install. Then, in any terminal:

```powershell
ollama pull qwen2.5:14b
```

If RAM is tight, set `DEFAULT_OLLAMA_MODEL` in `src/llm_report.py` to a smaller model such as `qwen2.5:7b` (around 4.7 GB) and run `ollama pull qwen2.5:7b`.

## Usage

Activate the virtual environment first:

- Linux or macOS: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Windows Command Prompt: `.venv\Scripts\activate.bat`

Then run any of the following with plain `python`:

**1. Train the herd size model:**

```bash
python -m src.train
```

**2. Build the sample input file** (real WP3 farm data):

```bash
python make_sample_data.py
```

**3. Launch the demo interface:**

```bash
python -m streamlit run app.py
```

The app opens at http://localhost:8501. The pre cached `hourly.csv` farm loads instantly so you can browse the dashboard without waiting for the pipeline. Upload your own CSV to override the demo and run the full pipeline.

**4. Command line:**

```bash
python -m src.pipeline sample_farm.csv --llm ollama
```

## LLM backend

The narrative generator is **pluggable** (`src/llm_report.py`):

- `ollama` (recommended): a running Ollama server with `qwen2.5:14b`.
- `template`: deterministic rule based narrative, fully offline.
- `auto` (default): Ollama if reachable, otherwise the template.

A `transformers` Llama backend also exists in the code. Ollama is the recommended default.

## Project layout

```
all_farms_hourly_data.csv   Synthetic dataset, trains the herd size model only
sample_farm.csv             Real WP3 farm data, demo input (generated)
make_sample_data.py         Builds sample_farm.csv from WP3/hourly.csv
knowledge/
  dairy_energy_kb.md     Curated dairy energy knowledge base (RAG source)
src/
  dataio.py          Loads and cleans a real hourly CSV (gap fills missing hours)
  features.py        Consumption only herd size feature engineering
  models.py          PyTorch MLP regressor (sklearn style wrapper)
  anomaly.py         MOMENT anomaly detector (reconstruction gap magnitude)
  rag.py             Retriever, embeds the KB and grounds the advisory
  train.py           Train and 5 fold cross validation, GBRT vs MLP
  predict.py         Herd size prediction from the saved model
  energy_report.py   Energy summary plus anomaly detection to JSON
  llm_report.py      RAG grounded LLM narrative (Ollama or template)
  pipeline.py        End to end orchestration plus CLI
app.py               Streamlit web interface
.venv/               Python 3.11 virtual environment
models/              herd_model.joblib, fleet_stats.json (generated)
reports/             Generated JSON reports
```

## Notes and limitations

- **Herd size on real data:** the model is trained on synthetic farms (20 to 90 cows). A real farm larger than that is outside the trained range; the system detects this, switches to the linear extrapolation, and flags the estimate as *indicative only*. For accurate large farm estimates, regenerate the synthetic training data spanning the real farms' size range.
- **Anomaly detection** runs on real data unchanged; the rate reflects genuine irregularities in the recording, not injected faults.
