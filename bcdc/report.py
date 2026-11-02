"""Retrieval-grounded explanation reports.

Every background sentence in a report is a passage from ``knowledge/kb.json``, quoted verbatim
with a numbered citation. Passages are retrieved with TF-IDF similarity to a query built from
the prediction (the classes in the prediction set). No text is generated freely, so the report
cannot invent medical claims. It is framed as background and questions to discuss with a
doctor, never as advice or a diagnosis.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .inference import DISCLAIMER, Prediction

KB_PATH = Path(__file__).parent / "knowledge" / "kb.json"
CLASS_QUERY = {
    "brain_glioma": "glioma glial tumour diagnosis grade molecular IDH treatment",
    "brain_menin": "meningioma meninges grade imaging monitoring treatment",
    "brain_tumor": "brain tumour type unspecified label",
}


@lru_cache(maxsize=1)
def load_kb(path: str = str(KB_PATH)) -> dict:
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=1)
def _index(path: str = str(KB_PATH)):
    from sklearn.feature_extraction.text import TfidfVectorizer
    kb = load_kb(path)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform([p["text"] for p in kb["passages"]])
    return vec, mat


def retrieve(query: str, classes=(), k: int = 4) -> list[dict]:
    """Top-k passages by TF-IDF cosine similarity, restricted to passages tagged with one of
    ``classes`` or with "*" (general)."""
    kb = load_kb()
    vec, mat = _index()
    sims = (mat @ vec.transform([query]).T).toarray().ravel()
    allowed = np.array([bool(set(p["classes"]) & (set(classes) | {"*"})) for p in kb["passages"]])
    sims = np.where(allowed, sims, -1.0)
    order = [i for i in np.argsort(-sims, kind="stable") if sims[i] >= 0][:k]
    return [{**kb["passages"][i], "score": float(sims[i])} for i in order]


def build_report(pred: Prediction, display_names: dict | None = None, k_per_class: int = 3) -> dict:
    """Returns {"markdown": str, "citations": [ {n, citation, url} ], "passages": [ids]}."""
    kb = load_kb()
    name = (display_names or {}).get
    lines = ["## About this result", ""]
    if pred.rejected:
        lines += [f"**No prediction was made.** {pred.reason}", "",
                  "This can happen with photos, scans of other body parts, other imaging types (such as CT), "
                  "or MRI slices that look very different from the training data.", "", f"_{DISCLAIMER}_"]
        return {"markdown": "\n".join(lines), "citations": [], "passages": []}

    pset = pred.prediction_set or [pred.top_class]
    pct = int(round(100 * (pred.coverage_target or 0.9)))
    labels = [name(c, c) or c for c in pset]
    if len(pset) == 1:
        lines.append(f"The software's output is most consistent with **{labels[0]}** "
                     f"({100 * pred.top_probability:.0f}% calibrated probability). Its {pct}% prediction set "
                     f"contains only this label.")
    else:
        lines.append(f"The software **could not narrow the result down to one label**. Its {pct}% prediction set "
                     f"contains: **{', '.join(labels)}**. The highest probability is "
                     f"{name(pred.top_class, pred.top_class)} at {100 * pred.top_probability:.0f}%.")
    if pred.epistemic_uncertainty is not None and pred.epistemic_uncertainty > 0.1:
        lines.append("The models in the ensemble disagree noticeably on this image, which suggests extra caution.")
    for w in pred.warnings:
        lines.append(f"- ⚠️ {w}")
    lines += ["", "This is the output of a research model trained on 2D MRI slices from one public dataset. "
                  "It is **not a diagnosis**.", ""]

    chosen, seen = [], set()
    kb_order = {p["id"]: i for i, p in enumerate(kb["passages"])}
    for cls in pset:
        hits = retrieve(CLASS_QUERY.get(cls, cls), classes=[cls], k=k_per_class)
        for p in sorted(hits, key=lambda h: kb_order[h["id"]]):     # what it is -> diagnosis -> management
            if p["id"] not in seen:
                chosen.append(p)
                seen.add(p["id"])
    for p in retrieve("brain tumour type diagnosis tissue care team", classes=[], k=2):
        if p["id"] not in seen:
            chosen.append(p)
            seen.add(p["id"])

    cite_no, citations = {}, []
    lines += ["## Background", ""]
    for p in chosen:
        src = p["source"]
        if src not in cite_no:
            cite_no[src] = len(cite_no) + 1
            citations.append({"n": cite_no[src], **kb["sources"][src]})
        lines.append(f"- {p['text']} [{cite_no[src]}]")
    lines += ["", "## Questions you could ask your doctor", ""]
    lines += [f"- {q}" for q in kb["questions"]]
    lines += ["", "## Sources", ""]
    lines += [f"{c['n']}. {c['citation']} {c['url']}" for c in citations]
    lines += ["", f"_{DISCLAIMER}_"]
    return {"markdown": "\n".join(lines), "citations": citations, "passages": [p["id"] for p in chosen]}
