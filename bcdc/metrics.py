"""Classification metrics and calibration."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def expected_calibration_error(y, probs, n_bins=15) -> float:
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def apply_temperature(probs, T):
    """Temperature scaling applied to probabilities (log-probs are logits up to a constant)."""
    logp = np.log(np.clip(probs, 1e-12, 1.0)) / T
    logp = logp - logp.max(axis=-1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=-1, keepdims=True)


def nll(y, probs) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def fit_temperature(y, probs) -> float:
    temps = np.exp(np.linspace(np.log(0.01), np.log(20.0), 600))
    losses = [nll(y, apply_temperature(probs, T)) for T in temps]
    return float(temps[int(np.argmin(losses))])


def evaluate(y, probs, class_names) -> tuple[dict, np.ndarray]:
    """Accuracy, balanced accuracy, macro-F1, ECE, NLL and per-class sensitivity/specificity/
    precision/AUROC. Returns (metrics dict, confusion matrix)."""
    k_all = len(class_names)
    pred = probs.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=range(k_all))
    total = cm.sum()
    out = {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=range(k_all), zero_division=0)),
        "ece": expected_calibration_error(y, probs),
        "nll": nll(y, probs),
    }
    aurocs = []
    for k, cls in enumerate(class_names):
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - tp - fn - fp
        out[f"sensitivity[{cls}]"] = float(tp / max(tp + fn, 1))
        out[f"specificity[{cls}]"] = float(tn / max(tn + fp, 1))
        out[f"precision[{cls}]"] = float(tp / max(tp + fp, 1))
        yk = (y == k).astype(int)
        auc = roc_auc_score(yk, probs[:, k]) if 0 < yk.sum() < len(yk) else float("nan")
        out[f"auroc[{cls}]"] = float(auc)
        aurocs.append(auc)
    out["macro_auroc"] = float(np.nanmean(aurocs)) if not np.all(np.isnan(aurocs)) else float("nan")
    return out, cm


def risk_coverage(correct, confidence):
    """Selective classification: keep the most confident fraction of cases.

    Returns (coverage, risk) arrays and the area under the risk-coverage curve (AURC, lower is better).
    """
    order = np.argsort(-np.asarray(confidence), kind="stable")
    err = 1.0 - np.asarray(correct, dtype=float)[order]
    n = len(err)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    return coverage, risk, float(risk.mean())
