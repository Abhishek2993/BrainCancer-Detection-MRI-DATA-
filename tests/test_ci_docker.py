import io
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
tf = pytest.importorskip("tensorflow")


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    from bcdc.smoke import build
    out = tmp_path_factory.mktemp("smoke")
    build(out)
    return out


def test_smoke_bundle(smoke):
    cfg = json.loads((smoke / "bundle" / "deploy.json").read_text())
    assert cfg["backend"] == "keras" and "UNTRAINED" in cfg["notes"] and (smoke / "sample.png").exists()


def test_concurrent_requests_are_safe(smoke, tmp_path):
    """The API serves requests from a thread pool; results must not depend on concurrency."""
    from bcdc.deploy import build_bundle  # noqa: F401  (import check)
    from bcdc.edge import to_tflite
    from bcdc.inference import Predictor

    keras_pred = Predictor(smoke / "bundle")
    # a TFLite bundle (interpreters are not thread-safe without the Predictor lock)
    model = tf.keras.models.load_model(smoke / "bundle" / "models" / "seed_0.keras")
    tfl = tmp_path / "tfl"
    (tfl / "models").mkdir(parents=True)
    calib = np.random.default_rng(0).uniform(0, 255, (8, 32, 32, 3)).astype("float32")
    (tfl / "models" / "m.tflite").write_bytes(to_tflite(model, "int8", calib))
    cfg = json.loads((smoke / "bundle" / "deploy.json").read_text())
    cfg.update(backend="tflite", models=["models/m.tflite"], ood=None)
    (tfl / "deploy.json").write_text(json.dumps(cfg))
    tfl_pred = Predictor(tfl)

    rng = np.random.default_rng(1)
    imgs = []
    for _ in range(12):
        buf = io.BytesIO()
        Image.fromarray(rng.integers(0, 255, (40, 40), dtype=np.uint8)).save(buf, "PNG")
        imgs.append(buf.getvalue())
    for pred in (keras_pred, tfl_pred):
        serial = [pred.predict_bytes(b).to_dict() for b in imgs]
        with ThreadPoolExecutor(max_workers=6) as ex:
            parallel = list(ex.map(lambda b: pred.predict_bytes(b).to_dict(), imgs))
        for a, b in zip(serial, parallel):
            assert a["rejected"] == b["rejected"]
            for k in a["probabilities"]:
                assert abs(a["probabilities"][k] - b["probabilities"][k]) < 1e-5


def test_notebook_check_script():
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "check_notebooks.py"), str(REPO)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_docker_constraints_match_pyproject():
    """Every runtime dependency of the API image is pinned in constraints-docker.txt."""
    pins = {l.split("==")[0].lower().replace("_", "-") for l in (REPO / "constraints-docker.txt").read_text().splitlines()
            if "==" in l and not l.startswith("#")}
    for dep in ["tensorflow", "numpy", "pandas", "pillow", "scikit-learn", "pyyaml", "fastapi", "uvicorn",
                "python-multipart", "pydicom", "ai-edge-litert"]:
        assert dep in pins, dep
