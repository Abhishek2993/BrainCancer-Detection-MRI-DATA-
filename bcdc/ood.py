"""Out-of-distribution (OOD) scores: higher score = looks more like the training data."""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score


def msp_score(probs):
    """Maximum softmax probability (Hendrycks & Gimpel, 2017)."""
    return np.asarray(probs).max(axis=1)


def energy_score(logits, T=1.0):
    """Negative free energy, T * logsumexp(logits / T) (Liu et al., 2020)."""
    z = np.asarray(logits, dtype=float) / T
    m = z.max(axis=1, keepdims=True)
    return T * (m[:, 0] + np.log(np.exp(z - m).sum(axis=1)))


class Mahalanobis:
    """Class-conditional Gaussians with a shared covariance over penultimate-layer features
    (Lee et al., 2018). Score = minus the smallest squared Mahalanobis distance to a class mean."""

    def fit(self, feats, y):
        feats, y = np.asarray(feats, dtype=np.float64), np.asarray(y)
        self.classes_ = np.unique(y)
        self.means_ = np.stack([feats[y == c].mean(axis=0) for c in self.classes_])
        centred = feats - self.means_[np.searchsorted(self.classes_, y)]
        self.precision_ = LedoitWolf().fit(centred).precision_
        return self

    def score(self, feats):
        feats = np.asarray(feats, dtype=np.float64)
        d = []
        for mu in self.means_:
            diff = feats - mu
            d.append(np.einsum("ij,jk,ik->i", diff, self.precision_, diff))
        return -np.min(np.stack(d, axis=1), axis=1)

    def save(self, path):
        np.savez_compressed(path, classes=self.classes_, means=self.means_, precision=self.precision_)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        obj = cls()
        obj.classes_, obj.means_, obj.precision_ = z["classes"], z["means"], z["precision"]
        return obj


def threshold_at_retention(id_scores, retain=0.95) -> float:
    """Score threshold that keeps ``retain`` of in-distribution cases (fit on validation data)."""
    return float(np.quantile(np.asarray(id_scores), 1 - retain))


def ood_metrics(id_scores, ood_scores, threshold=None) -> dict:
    """AUROC (ID = positive) and FPR at 95% TPR. With ``threshold``, also the share of OOD
    inputs that would be accepted and of ID inputs that would be rejected."""
    id_scores, ood_scores = np.asarray(id_scores), np.asarray(ood_scores)
    y = np.r_[np.ones(len(id_scores)), np.zeros(len(ood_scores))]
    s = np.r_[id_scores, ood_scores]
    thr95 = np.quantile(id_scores, 0.05)
    out = {"auroc": float(roc_auc_score(y, s)), "fpr_at_95tpr": float((ood_scores >= thr95).mean())}
    if threshold is not None:
        out["ood_accepted"] = float((ood_scores >= threshold).mean())
        out["id_rejected"] = float((id_scores < threshold).mean())
    return out
