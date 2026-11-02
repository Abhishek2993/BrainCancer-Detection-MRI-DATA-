import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bcdc.inference import Prediction, load_image, looks_like_grayscale
from bcdc.report import build_report, load_kb, retrieve
from bcdc.split import make_split
from bcdc.tracking import Tracker

REPO = Path(__file__).resolve().parents[1]
CLASSES = ["brain_glioma", "brain_menin", "brain_tumor"]


def make_dataset(root, n=14, size=48, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    for k, c in enumerate(CLASSES):
        d = root / c
        d.mkdir(parents=True)
        for i in range(n):
            cy = 10 + 14 * k + rng.integers(-3, 3)
            a = rng.normal(60, 20, (size, size)) + 150 * np.exp(-((yy - cy) ** 2 + (xx - size / 2) ** 2) / 30)
            Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGB").save(d / f"{c}_{i}.png")
        (d / f"{c}_dup.png").write_bytes((d / f"{c}_0.png").read_bytes())       # exact duplicate
    (root / "brain_menin" / "conflict.png").write_bytes((root / "brain_glioma" / "brain_glioma_1.png").read_bytes())
    return root


# ---------------------------------------------------------------- split, tracking, report (no TensorFlow)
def test_split_groups_duplicates_and_drops_conflicts(tmp_path):
    data = make_dataset(tmp_path / "data")
    out, stats = make_split(data, tmp_path / "split.csv")
    assert stats["exact_duplicates"] == 4 and stats["conflict_groups"] == 1
    assert out.groupby("group")["split"].nunique().max() == 1
    assert "brain_menin/conflict.png" not in set(out["path"])
    assert set(out["split"]) == {"train", "val", "test"}
    again, _ = make_split(data, tmp_path / "split2.csv")
    assert again["split"].tolist() == out["split"].tolist()           # deterministic


def test_local_tracker(tmp_path):
    t = Tracker("local", run_name="r1", root=tmp_path)
    t.log_params({"model": {"backbone": "x"}, "lr": 0.1})
    t.log_metrics({"acc": 0.5}, step=0)
    t.log_metrics({"acc": 0.7}, step=1)
    t.close()
    assert json.loads((tmp_path / "r1" / "params.json").read_text())["model.backbone"] == "x"
    assert len((tmp_path / "r1" / "metrics.jsonl").read_text().splitlines()) == 2


def test_mlflow_tracker(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())   # file store: works without SQLAlchemy
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    t = Tracker("mlflow", project="bcdc-test", run_name="r")
    t.log_params({"a": 1})
    t.log_metrics({"sensitivity[brain_glioma]": 0.9}, step=0)
    run_id = mlflow.active_run().info.run_id
    t.close()
    assert mlflow.get_run(run_id).data.metrics["sensitivity_brain_glioma"] == 0.9


def test_report_is_grounded_and_cited():
    kb = load_kb()
    texts = {p["text"] for p in kb["passages"]}
    pred = Prediction(rejected=False, probabilities={"brain_glioma": 0.6, "brain_menin": 0.35, "brain_tumor": 0.05},
                      top_class="brain_glioma", top_probability=0.6, prediction_set=["brain_glioma", "brain_menin"],
                      coverage_target=0.9, epistemic_uncertainty=0.2, warnings=["Low confidence"])
    rep = build_report(pred, {"brain_glioma": "Glioma", "brain_menin": "Meningioma"})
    md = rep["markdown"]
    assert "could not narrow the result down" in md and "Glioma, Meningioma" in md and "not a diagnosis" in md
    background = md.split("## Background")[1].split("## Questions")[0]
    bullets = [l[2:] for l in background.splitlines() if l.startswith("- ")]
    assert bullets, "no background passages"
    for b in bullets:                                        # every background sentence is a KB passage + citation
        text, cite = b.rsplit(" [", 1)
        assert text in texts and 1 <= int(cite.rstrip("]")) <= len(rep["citations"])
    assert {"glioma_diagnosis", "meningioma_definition"} <= set(rep["passages"]) or len(rep["passages"]) >= 4
    for c in rep["citations"]:
        assert c["url"].startswith("https://") and c["citation"]
    assert "cost" not in md.lower() and "hospital" not in md.lower()


def test_report_for_rejection_has_no_medical_content():
    rep = build_report(Prediction(rejected=True, reason="Not like the training MRIs."))
    assert rep["citations"] == [] and "No prediction was made" in rep["markdown"]


def test_retrieve_respects_class_filter():
    hits = retrieve("meningioma treatment monitoring", classes=["brain_glioma"], k=10)
    assert all(set(h["classes"]) & {"brain_glioma", "*"} for h in hits)


# ---------------------------------------------------------------- DICOM
def make_dicom(path_or_buf, arr, modality="MR", photometric="MONOCHROME2", window=None, slope=1.0, intercept=0.0):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName, ds.PatientID, ds.Modality = "SECRET^PATIENT", "ID-12345", modality
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, photometric
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
    if window:
        ds.WindowCenter, ds.WindowWidth = window
    ds.PixelData = arr.astype(np.uint16).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def test_dicom_conversion_and_privacy():
    arr = np.zeros((40, 50), np.uint16)
    arr[10:20, 30:40] = 3000
    data = make_dicom(None, arr, window=(1500, 3000))
    img, info = load_image(data, "scan.dcm")
    a = np.asarray(img)
    assert img.mode == "RGB" and a.shape == (40, 50, 3)
    assert a[15, 35, 0] > 200 and a[0, 0, 0] < 30                 # bright region stays bright, orientation kept
    assert info["format"] == "DICOM" and info["Modality"] == "MR"
    assert "SECRET" not in json.dumps(info) and "12345" not in json.dumps(info)
    inv, _ = load_image(make_dicom(None, arr, photometric="MONOCHROME1"), "x.dcm")
    assert np.asarray(inv)[15, 35, 0] < 50                        # MONOCHROME1 inverted
    _, ct = load_image(make_dicom(None, arr, modality="CT"), "x.dcm")
    assert any("CT" in w for w in ct["warnings"])


def test_load_image_errors_and_modes():
    with pytest.raises(ValueError):
        load_image(b"not an image", "x.png")
    buf = io.BytesIO()
    Image.new("RGBA", (10, 10), (255, 0, 0, 128)).save(buf, "PNG")
    img, info = load_image(buf.getvalue(), "x.png")
    assert img.mode == "RGB" and not looks_like_grayscale(img)


# ---------------------------------------------------------------- bundle, predictor, API, demo, CLI (TensorFlow)
tf = pytest.importorskip("tensorflow")


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    from bcdc.deploy import build_bundle
    from bcdc.edge import to_tflite
    from bcdc.model import build_model

    root = tmp_path_factory.mktemp("p3")
    data = make_dataset(root / "data")
    art = root / "outputs"
    (art / "models").mkdir(parents=True)
    make_split(data, art / "split.csv")
    for seed in (0, 1):
        model, _ = build_model(3, img_size=(32, 32), weights=None, seed=seed)
        model.save(art / "models" / f"seed_{seed}.keras")
    (art / "model_meta.json").write_text(json.dumps({"class_names": CLASSES, "img_size": [32, 32]}))
    build_bundle(art, data, root / "deploy_keras")
    calib = np.random.default_rng(0).uniform(0, 255, (8, 32, 32, 3)).astype("float32")
    (root / "m.tflite").write_bytes(to_tflite(tf.keras.models.load_model(art / "models" / "seed_0.keras"), "int8", calib))
    build_bundle(art, data, root / "deploy_tflite", tflite_file=root / "m.tflite")
    return {"keras": root / "deploy_keras", "tflite": root / "deploy_tflite", "data": data, "art": art}


def png_bytes(path):
    return Path(path).read_bytes()


def test_bundle_contents(bundles):
    cfg = json.loads((bundles["keras"] / "deploy.json").read_text())
    assert cfg["backend"] == "keras" and len(cfg["models"]) == 2 and cfg["ood"]["threshold"] is not None
    assert len(cfg["conformal"]["qhat"]) == 3 and 0 <= cfg["evaluation"]["accuracy"] <= 1
    t = json.loads((bundles["tflite"] / "deploy.json").read_text())
    assert t["backend"] == "tflite" and t["ood"] is None


@pytest.mark.parametrize("kind", ["keras", "tflite"])
def test_predictor(bundles, kind):
    from bcdc.inference import Predictor
    p = Predictor(bundles[kind])
    pred = p.predict_bytes(png_bytes(next((bundles["data"] / "brain_glioma").glob("*.png"))), "a.png")
    if not pred.rejected:
        assert abs(sum(pred.probabilities.values()) - 1) < 1e-4
        assert set(pred.prediction_set) <= set(CLASSES) and pred.top_class in CLASSES
    if kind == "tflite":
        assert any("no out-of-distribution gate" in w for w in pred.warnings)
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO()
    noise.save(buf, "PNG")
    out = p.predict_bytes(buf.getvalue(), "noise.png")
    assert any("colour" in w for w in out.warnings) or out.rejected


def test_api(bundles):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from bcdc.api import create_app
    from bcdc.inference import Predictor
    client = TestClient(create_app(Predictor(bundles["keras"])))
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/info").json()
    assert len(info["classes"]) == 3 and info["ood_gate"] and "not a medical device" in info["disclaimer"]
    img = next((bundles["data"] / "brain_menin").glob("*.png"))
    r = client.post("/predict", files={"file": ("x.png", png_bytes(img), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body and "markdown" in body["report"] and body["disclaimer"]
    arr = (np.asarray(Image.open(img).convert("L")).astype(np.uint16) * 10)
    r = client.post("/predict", files={"file": ("scan.dcm", make_dicom(None, arr), "application/dicom")})
    assert r.status_code == 200 and r.json()["prediction"]["input_info"]["format"] == "DICOM"
    assert "SECRET" not in r.text and "ID-12345" not in r.text
    assert client.post("/predict", files={"file": ("x.png", b"garbage", "image/png")}).status_code == 415
    assert client.post("/predict", files={"file": ("x.png", b"", "image/png")}).status_code == 400


def test_demo_builds(bundles):
    pytest.importorskip("gradio")
    from bcdc.demo import build_demo
    from bcdc.inference import Predictor
    demo = build_demo(Predictor(bundles["keras"]))
    assert demo is not None


def test_cli_predict_json(bundles):
    img = next((bundles["data"] / "brain_tumor").glob("*.png"))
    r = subprocess.run([sys.executable, "-m", "bcdc.cli", "predict", str(img), "--bundle", str(bundles["keras"]),
                        "--json"], capture_output=True, text=True, cwd=REPO, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout[r.stdout.index("["):])
    assert out[0]["file"] == str(img) and "report" in out[0]


def test_cli_train_quick(bundles, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
data: {{split_csv: {bundles['art'] / 'split.csv'}}}
model: {{backbone: mobilenet_v3_small, img_size: [32, 32], weights: null}}
training: {{seeds: [0], batch_size: 8}}
output_dir: {tmp_path / 'out'}
tracking: {{backend: local, run_name: t}}
""")
    r = subprocess.run([sys.executable, "-m", "bcdc.cli", "train", "--config", str(cfg), "--data",
                        str(bundles["data"]), "--quick"], capture_output=True, text=True, cwd=tmp_path, timeout=900,
                       env={**__import__("os").environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0, r.stderr[-3000:]
    meta = json.loads((tmp_path / "out" / "model_meta.json").read_text())
    assert meta["backbone"] == "mobilenet_v3_small" and (tmp_path / "out" / "models" / "seed_0.keras").exists()
    assert (tmp_path / "out" / "split.csv").exists() and (tmp_path / "runs" / "t" / "metrics.jsonl").exists()
