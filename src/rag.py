"""
Retrieval-augmented generation: ground the advisory in a knowledge base.

The numeric analysis (herd size, energy, anomalies) is computed and needs no
retrieval. What RAG improves is the *advice*: instead of relying on the LLM's
vague memory of dairy-farm energy practice, relevant guidance is retrieved from
a curated knowledge base (`knowledge/dairy_energy_kb.md`) and injected into the
prompt, so the recommendations are grounded in referenced domain knowledge.

Retrieval uses a small sentence-embedding model (`all-MiniLM-L6-v2`, ~80 MB,
CPU) loaded through Hugging Face Transformers with mean pooling — no extra
dependency beyond what the project already installs. The model downloads from
the Hugging Face Hub on first use.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Force the PyTorch backend before importing transformers (see llm_report.py).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "knowledge" / "dairy_energy_kb.md"

_EMBEDDER = None        # cached (tokenizer, model)
_KB = None              # cached list of {topic, text}
_KB_EMB = None          # cached chunk embeddings (n_chunks, dim)


# --------------------------------------------------------------------------
# Knowledge base
# --------------------------------------------------------------------------
def load_kb(path: Path = KB_PATH) -> list[dict]:
    """Parse the markdown KB into chunks — one per `## ` heading."""
    if not path.exists():
        return []
    text = path.read_text()
    chunks = []
    for part in re.split(r"^## ", text, flags=re.M)[1:]:   # skip the preamble
        head, _, body = part.partition("\n")
        body = body.strip()
        if body:
            chunks.append({"topic": head.strip(), "text": body})
    return chunks


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------
def _load_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
        model = AutoModel.from_pretrained(EMBED_MODEL).eval()
        _EMBEDDER = (tok, model)
    return _EMBEDDER


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts into L2-normalised sentence vectors (mean-pooled MiniLM)."""
    import torch

    tok, model = _load_embedder()
    enc = tok(texts, padding=True, truncation=True, max_length=256,
              return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    summed = (out.last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    emb = summed / counts
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.numpy()


def _kb_embeddings() -> tuple[list[dict], np.ndarray]:
    """Load and embed the KB once, then cache."""
    global _KB, _KB_EMB
    if _KB is None:
        _KB = load_kb()
        if _KB:
            _KB_EMB = embed([f"{c['topic']}. {c['text']}" for c in _KB])
        else:
            _KB_EMB = np.zeros((0, 384), dtype=np.float32)
    return _KB, _KB_EMB


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def retrieve(query: str, k: int = 3) -> list[dict]:
    """Return the top-k KB chunks most similar to `query`."""
    kb, kb_emb = _kb_embeddings()
    if not kb:
        return []
    sims = embed([query])[0] @ kb_emb.T          # cosine (vectors normalised)
    order = np.argsort(sims)[::-1][:k]
    return [{**kb[i], "score": round(float(sims[i]), 3)} for i in order]


def _report_queries(report: dict) -> list[str]:
    """Build retrieval queries from a farm's situation."""
    es = report.get("energy_summary", {})
    an = report.get("anomaly_detection", {})
    eff = report.get("herd_efficiency", {})

    queries = ["dairy farm electricity efficiency milk cooling water heating",
               "seasonal and monthly dairy consumption water heating cooling"]
    if es.get("baseline_share_pct", 0) > 18:
        queries.append("high baseline idle phantom load water heater insulation "
                        "equipment left running")
    if an.get("high_consumption_events", 0) > 0:
        queries.append("high consumption spike refrigeration short-cycling "
                        "vacuum pump fault")
    if an.get("low_consumption_events", 0) > 0:
        queries.append("low consumption drop outage tripped breaker equipment "
                        "failed to start")
    if eff.get("rating") == "above-average consumption":
        queries.append("high electricity use per cow benchmark milking "
                        "efficiency plate cooler heat recovery")
    return queries


def retrieve_for_report(report: dict, k: int = 6) -> list[dict]:
    """Retrieve the KB chunks most relevant to a farm report (de-duplicated)."""
    seen: dict[str, dict] = {}
    for query in _report_queries(report):
        for chunk in retrieve(query, k=2):
            prev = seen.get(chunk["topic"])
            if prev is None or chunk["score"] > prev["score"]:
                seen[chunk["topic"]] = chunk
    ranked = sorted(seen.values(), key=lambda c: c["score"], reverse=True)
    return ranked[:k]
