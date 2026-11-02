"""Deep ensembles: average the members' probabilities and split uncertainty into parts."""
from __future__ import annotations

import numpy as np


def entropy(p, axis=-1):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def ensemble_probs(member_probs):
    """member_probs: (M, N, K) -> mean probabilities (N, K)."""
    return np.asarray(member_probs).mean(axis=0)


def uncertainty(member_probs) -> dict:
    """Predictive-entropy decomposition (in nats).

    * total     = H[mean_m p_m]              (all uncertainty)
    * aleatoric = mean_m H[p_m]              (ambiguity the members agree on)
    * epistemic = total - aleatoric          (mutual information: members disagree)
    """
    mp = np.asarray(member_probs)
    total = entropy(mp.mean(axis=0))
    aleatoric = entropy(mp).mean(axis=0)
    return {"total": total, "aleatoric": aleatoric, "epistemic": np.maximum(total - aleatoric, 0.0)}
