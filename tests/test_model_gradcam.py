import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bcdc.data import array_dataset
from bcdc.gradcam import gradcam
from bcdc.model import build_model, feature_extractor, features_and_logits, model_parts, softmax


@pytest.fixture(scope="module")
def tiny():
    model, _ = build_model(3, img_size=(64, 64), weights=None, seed=0)
    # make the head non-trivial
    W, b = model.layers[-1].get_weights()
    model.layers[-1].set_weights([np.random.default_rng(0).normal(0, 0.5, W.shape).astype("float32"), b])
    return model


def test_features_and_logits_match_model(tiny, tmp_path):
    imgs = np.random.default_rng(1).integers(0, 255, (5, 70, 50), dtype=np.uint8)
    ds = array_dataset(imgs, (64, 64), batch_size=2)
    parts = model_parts(tiny)
    feats, logits = features_and_logits(parts, ds)
    np.testing.assert_allclose(softmax(logits), tiny.predict(ds, verbose=0), atol=1e-5)
    # also after a save / load round trip
    tiny.save(tmp_path / "m.keras")
    reloaded = tf.keras.models.load_model(tmp_path / "m.keras")
    _, logits2 = features_and_logits(model_parts(reloaded), ds)
    np.testing.assert_allclose(logits, logits2, atol=1e-5)


def test_gradcam_equals_cam(tiny):
    parts = model_parts(tiny)
    ext = feature_extractor(parts)
    x = tf.constant(np.random.default_rng(2).uniform(0, 255, (3, 64, 64, 3)), tf.float32)
    cls = np.array([0, 1, 2])
    cams = gradcam(ext, parts, x, cls)
    conv, _ = ext(x)
    conv = conv.numpy()
    for i, c in enumerate(cls):
        ref = np.maximum((conv[i] * parts.W[:, c]).sum(-1), 0)   # CAM: GAP + linear head
        ref = ref / max(ref.max(), 1e-12)
        np.testing.assert_allclose(cams[i], ref, atol=1e-4)
    up = gradcam(ext, parts, x, cls, out_size=(32, 32))
    assert up.shape == (3, 32, 32) and up.max() <= 1 + 1e-6 and up.min() >= 0
