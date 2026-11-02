"""Split-conformal prediction sets.

Instead of one label, the model outputs a *set* of labels that contains the true class with
probability >= 1 - alpha (on average over new cases from the same distribution as the
calibration data). Implemented scores:

* ``lac`` - least ambiguous classifier: score = 1 - p(y). Gives the smallest sets on average.
* ``aps`` - adaptive prediction sets (Romano et al., 2020): score = probability mass of all
  labels ranked at or above y. Sets grow when the model is unsure.

``class_conditional=True`` (Mondrian conformal) calibrates one threshold per class, so the
coverage guarantee holds *for each tumour type separately*, not just on average.
"""
from __future__ import annotations

import math

import numpy as np


def score_matrix(probs, method="lac", u=None):
    """Non-conformity score of every candidate label: shape (N, K). Lower = more plausible."""
    probs = np.asarray(probs, dtype=float)
    if method == "lac":
        return 1.0 - probs
    if method == "aps":
        n, k = probs.shape
        order = np.argsort(-probs, axis=1, kind="stable")
        sorted_p = np.take_along_axis(probs, order, axis=1)
        cums_sorted = np.cumsum(sorted_p, axis=1)
        cums = np.empty_like(cums_sorted)
        np.put_along_axis(cums, order, cums_sorted, axis=1)  # cumulative mass up to and incl. each label
        u = np.ones((n, 1)) if u is None else np.asarray(u, dtype=float).reshape(n, 1)
        return cums - u * probs                               # u ~ U(0,1) randomises; u = 1 is deterministic
    raise ValueError(f"unknown method {method!r}")


def conformal_quantile(scores, alpha) -> float:
    """The k-th smallest calibration score, k = ceil((n + 1)(1 - alpha))."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or k > n:
        return float("inf")          # too few calibration points for this alpha: include everything
    return float(scores[k - 1])


class ConformalClassifier:
    def __init__(self, alpha=0.1, method="lac", class_conditional=False, randomized=False, seed=0):
        self.alpha, self.method = alpha, method
        self.class_conditional, self.randomized = class_conditional, randomized
        self.rng = np.random.default_rng(seed)
        self.qhat = None

    def _u(self, n):
        return self.rng.uniform(size=n) if (self.randomized and self.method == "aps") else None

    def fit(self, probs, y):
        y = np.asarray(y)
        s = score_matrix(probs, self.method, self._u(len(y)))[np.arange(len(y)), y]
        k = np.asarray(probs).shape[1]
        if self.class_conditional:
            self.qhat = np.array([conformal_quantile(s[y == c], self.alpha) for c in range(k)])
        else:
            self.qhat = np.full(k, conformal_quantile(s, self.alpha))
        return self

    def predict_sets(self, probs) -> np.ndarray:
        """Boolean array (N, K): True where the label is in the prediction set."""
        probs = np.asarray(probs)
        return score_matrix(probs, self.method, self._u(len(probs))) <= self.qhat[None, :]

    def to_dict(self):
        return {"alpha": self.alpha, "method": self.method, "class_conditional": self.class_conditional,
                "randomized": self.randomized, "qhat": [float(q) for q in self.qhat]}


def set_metrics(sets, y, class_names=None) -> dict:
    """Coverage, average set size, share of singleton/empty sets and per-class coverage."""
    sets = np.asarray(sets, dtype=bool)
    y = np.asarray(y)
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(axis=1)
    out = {
        "coverage": float(covered.mean()),
        "avg_set_size": float(size.mean()),
        "singleton_rate": float((size == 1).mean()),
        "empty_rate": float((size == 0).mean()),
    }
    names = class_names or [str(c) for c in range(sets.shape[1])]
    for c, name in enumerate(names):
        m = y == c
        out[f"coverage[{name}]"] = float(covered[m].mean()) if m.any() else float("nan")
    return out
