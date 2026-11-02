import numpy as np
import pytest

from bcdc.conformal import ConformalClassifier, conformal_quantile, score_matrix, set_metrics


def synthetic(n, k=3, seed=0):
    """Probabilities that are calibrated by construction: y is sampled from p."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(k) * 0.7, size=n)
    y = np.array([rng.choice(k, p=row) for row in p])
    return p, y


@pytest.mark.parametrize("method,randomized,cc", [
    ("lac", False, False), ("aps", True, False), ("aps", False, False), ("lac", False, True),
])
def test_marginal_coverage(method, randomized, cc):
    alpha = 0.1
    p_cal, y_cal = synthetic(3000, seed=1)
    p_test, y_test = synthetic(20000, seed=2)
    cp = ConformalClassifier(alpha, method, class_conditional=cc, randomized=randomized).fit(p_cal, y_cal)
    m = set_metrics(cp.predict_sets(p_test), y_test)
    assert m["coverage"] >= 1 - alpha - 0.01
    if randomized or method == "lac":
        assert m["coverage"] <= 1 - alpha + 0.03   # not trivially conservative


def test_class_conditional_coverage_per_class():
    rng = np.random.default_rng(3)
    # Class 2 is systematically under-confident, so a single threshold under-covers it.
    def make(n):
        y = rng.integers(0, 3, n)
        logits = rng.normal(0, 1, (n, 3))
        logits[np.arange(n), y] += np.where(y == 2, 0.3, 3.0)
        p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        return p, y
    p_cal, y_cal = make(6000)
    p_test, y_test = make(30000)
    cp = ConformalClassifier(0.1, "lac", class_conditional=True).fit(p_cal, y_cal)
    m = set_metrics(cp.predict_sets(p_test), y_test, ["a", "b", "c"])
    for c in "abc":
        assert m[f"coverage[{c}]"] >= 0.88


def test_aps_deterministic_never_empty():
    p, y = synthetic(500)
    cp = ConformalClassifier(0.3, "aps").fit(p, y)
    assert cp.predict_sets(p).sum(axis=1).min() >= 1


def test_aps_scores():
    p = np.array([[0.6, 0.3, 0.1]])
    s = score_matrix(p, "aps")                 # u = 1: mass ranked strictly above each label
    np.testing.assert_allclose(s, [[0.0, 0.6, 0.9]])
    s = score_matrix(p, "aps", u=np.array([0.0]))
    np.testing.assert_allclose(s, [[0.6, 0.9, 1.0]])


def test_quantile_edge_cases():
    assert conformal_quantile([0.1, 0.2], alpha=0.1) == float("inf")   # too few points
    assert conformal_quantile([], alpha=0.1) == float("inf")
    scores = np.arange(1, 101) / 100
    assert conformal_quantile(scores, 0.1) == pytest.approx(0.91)       # ceil(101*0.9)=91st value
