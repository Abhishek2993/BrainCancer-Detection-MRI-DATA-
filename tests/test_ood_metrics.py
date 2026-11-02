import numpy as np
import pytest

from bcdc.ensemble import uncertainty
from bcdc.metrics import apply_temperature, evaluate, expected_calibration_error, fit_temperature, risk_coverage
from bcdc.ood import Mahalanobis, energy_score, msp_score, ood_metrics, threshold_at_retention


def test_mahalanobis_separates_far_cluster(tmp_path):
    rng = np.random.default_rng(0)
    means = rng.normal(0, 3, (3, 16))
    y = rng.integers(0, 3, 900)
    x = means[y] + rng.normal(0, 1, (900, 16))
    maha = Mahalanobis().fit(x, y)
    id_test = means[rng.integers(0, 3, 300)] + rng.normal(0, 1, (300, 16))
    ood = rng.normal(8, 1, (300, 16))
    m = ood_metrics(maha.score(id_test), maha.score(ood))
    assert m["auroc"] > 0.99 and m["fpr_at_95tpr"] < 0.05
    maha.save(tmp_path / "m.npz")
    np.testing.assert_allclose(Mahalanobis.load(tmp_path / "m.npz").score(ood), maha.score(ood))


def test_energy_and_msp():
    logits = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(energy_score(logits), np.log(np.exp(logits).sum(1)))
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    assert msp_score(p)[0] > msp_score(p)[1]


def test_threshold_and_rates():
    id_s = np.linspace(0, 1, 1001)
    thr = threshold_at_retention(id_s, 0.95)
    assert thr == pytest.approx(0.05)
    m = ood_metrics(id_s, np.full(10, -1.0), threshold=thr)
    assert m["ood_accepted"] == 0 and m["id_rejected"] == pytest.approx(0.05, abs=0.002)


def test_uncertainty_decomposition():
    agree = np.array([[[0.5, 0.5]], [[0.5, 0.5]]])
    disagree = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    assert uncertainty(agree)["epistemic"][0] == pytest.approx(0, abs=1e-9)
    assert uncertainty(disagree)["epistemic"][0] == pytest.approx(np.log(2), rel=1e-6)


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 1.5, (5000, 3))
    p_true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=r) for r in p_true])
    p_over = apply_temperature(p_true, 0.5)                    # sharpened = over-confident
    T = fit_temperature(y, p_over)
    assert T == pytest.approx(2.0, rel=0.15)
    assert expected_calibration_error(y, apply_temperature(p_over, T)) < expected_calibration_error(y, p_over)


def test_evaluate_and_risk_coverage():
    y = np.array([0, 1, 2, 2])
    p = np.array([[.9, .05, .05], [.1, .8, .1], [.2, .2, .6], [.6, .3, .1]])
    m, cm = evaluate(y, p, ["a", "b", "c"])
    assert m["accuracy"] == 0.75 and cm[2, 0] == 1
    assert m["sensitivity[c]"] == 0.5 and m["specificity[a]"] == pytest.approx(2 / 3)
    cov, risk, aurc = risk_coverage(p.argmax(1) == y, p.max(1))
    assert risk[-1] == 0.25 and risk[0] == 0.0
