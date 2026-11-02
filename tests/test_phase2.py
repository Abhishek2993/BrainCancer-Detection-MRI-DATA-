import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bcdc.edge import pareto_front
from bcdc.federated import dirichlet_partition, iid_partition

REPO = Path(__file__).resolve().parents[1]


def test_partitions_cover_everything_once():
    labels = np.repeat([0, 1, 2], [300, 200, 100])
    for parts in (iid_partition(labels, 4), dirichlet_partition(labels, 4, alpha=0.3, min_size=5)):
        allidx = np.concatenate(parts)
        assert len(allidx) == len(labels) and len(np.unique(allidx)) == len(labels)


def test_dirichlet_is_skewed_and_alpha_controls_it():
    labels = np.repeat([0, 1, 2], 400)

    def skew(alpha):
        parts = dirichlet_partition(labels, 5, alpha=alpha, seed=1, min_size=5)
        fr = np.array([[np.mean(labels[p] == c) for c in range(3)] for p in parts])
        return np.abs(fr - 1 / 3).mean()

    assert skew(0.1) > skew(100.0) and skew(100.0) < 0.05


def test_pareto_front():
    df = pd.DataFrame({"accuracy": [0.9, 0.95, 0.85, 0.95], "latency_ms": [10, 50, 20, 40], "size_mb": [2, 10, 3, 10]})
    # row 2 is dominated by row 0; row 1 is dominated by row 3
    assert pareto_front(df).tolist() == [True, False, False, True]


tf = pytest.importorskip("tensorflow")

from bcdc.backbones import BACKBONES, build_classifier, set_finetune_trainable  # noqa: E402
from bcdc.edge import TFLiteClassifier, inference_model, to_tflite  # noqa: E402
from bcdc.federated import FederatedTrainer  # noqa: E402
from bcdc.model import model_parts, softmax  # noqa: E402


@pytest.fixture(scope="module")
def v3():
    model, base = build_classifier("mobilenet_v3_small", 3, img_size=(64, 64), weights=None, seed=0)
    W, b = model.layers[-1].get_weights()
    model.layers[-1].set_weights([np.random.default_rng(0).normal(0, 0.3, W.shape).astype("float32"), b])
    return model, base


def test_builtin_preprocessing_backbone_parts(v3):
    model, _ = v3
    parts = model_parts(model)                     # no Rescaling layer: identity preprocessing
    x = np.random.default_rng(0).uniform(0, 255, (4, 64, 64, 3)).astype("float32")
    from tensorflow import keras
    inp = keras.Input((64, 64, 3))
    fm = keras.Model(inp, parts.pool(parts.backbone(parts.preprocess(inp), training=False)))
    np.testing.assert_allclose(softmax(fm.predict(x, verbose=0) @ parts.W + parts.b),
                               model.predict(x, verbose=0), atol=1e-5)


def test_finetune_fraction_matches_phase0():
    base = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=(64, 64, 3))
    assert set_finetune_trainable(base, 0.35) == 100
    assert not any(l.trainable for l in base.layers if isinstance(l, tf.keras.layers.BatchNormalization))
    assert base.layers[120].trainable or isinstance(base.layers[120], tf.keras.layers.BatchNormalization)


@pytest.mark.parametrize("mode", ["fp32", "fp16", "dynamic", "int8"])
def test_tflite_matches_keras(v3, mode):
    model, _ = v3
    x = np.random.default_rng(1).integers(0, 255, (6, 64, 64, 3)).astype("float32")
    ref = inference_model(model).predict(x, verbose=0)
    np.testing.assert_allclose(ref, model.predict(x, verbose=0), atol=1e-5)
    clf = TFLiteClassifier(model_content=to_tflite(model, mode, representative_images=x))
    out = clf.predict(x, batch_size=4)
    assert out.shape == (6, 3) and np.allclose(out.sum(1), 1, atol=1e-2)
    tol = {"fp32": 1e-4, "fp16": 1e-2, "dynamic": 0.1, "int8": 0.25}[mode]
    assert np.abs(out - ref).max() < tol
    assert clf.latency_ms(n_runs=3, warmup=1)["median_ms"] > 0


def test_device_script_runs(v3, tmp_path):
    model, _ = v3
    x = np.random.default_rng(2).integers(0, 255, (8, 64, 64, 3)).astype(np.uint8)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "v3_int8.tflite").write_bytes(to_tflite(model, "int8", x.astype("float32")))
    np.savez_compressed(tmp_path / "testset.npz", images=x, labels=np.array([0, 1, 2, 0, 1, 2, 0, 1]))
    (tmp_path / "calibration.json").write_text(json.dumps({
        "class_names": ["brain_glioma", "brain_menin", "brain_tumor"],
        "models": {"v3_int8": {"temperature": 1.0, "qhat": [0.9, 0.9, 0.9]}}}))
    r = subprocess.run([sys.executable, str(REPO / "edge" / "run_on_device.py"), "--bundle", str(tmp_path),
                        "--runs", "3", "--threads", "1"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((tmp_path / "device_results.json").read_text())["results"][0]
    assert res["model"] == "v3_int8" and 0 <= res["accuracy"] <= 1 and "conformal_coverage" in res


def test_fedavg_equals_weighted_average_and_learns():
    from tensorflow import keras
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (300, 4)).astype("float32")
    y = (x[:, 0] + x[:, 1] > 0).astype("int32")
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(2, activation="softmax")])
    parts = [np.arange(0, 100), np.arange(100, 300)]
    dss = [tf.data.Dataset.from_tensor_slices((x[p], y[p])).batch(32) for p in parts]
    ft = FederatedTrainer(model, lr=0.05, mu=0.0)
    w0 = model.get_weights()
    # reference: train each client separately from w0, then weight by size
    local = []
    for ds in dss:
        model.set_weights(w0)
        local.append(ft.local_update(ds, 1))
    expected = [(1 / 3) * a + (2 / 3) * b for a, b in zip(*local)]
    model.set_weights(w0)
    ft.run_round(dss, [100, 200], local_epochs=1)
    for got, exp in zip(model.get_weights(), expected):
        np.testing.assert_allclose(got, exp, atol=1e-5)
    for _ in range(15):
        ft.run_round(dss, [100, 200])
    acc = (model.predict(x, verbose=0).argmax(1) == y).mean()
    assert acc > 0.9
    assert ft.communication_mb(2, 10) == pytest.approx(2 * 2 * 10 * 10 * 4 / 1e6)


def test_fedprox_pulls_towards_global():
    from tensorflow import keras
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (200, 4)).astype("float32")
    y = (x[:, 0] > 0).astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
    drift = {}
    for mu in (0.0, 10.0):
        keras.utils.set_random_seed(0)
        model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(2, activation="softmax")])
        w0 = [w.copy() for w in model.get_weights()]
        ft = FederatedTrainer(model, lr=0.05, mu=mu)
        ft.run_round([ds], [200], local_epochs=3)
        drift[mu] = sum(np.abs(a - b).sum() for a, b in zip(model.get_weights(), w0))
    assert drift[10.0] < drift[0.0]
