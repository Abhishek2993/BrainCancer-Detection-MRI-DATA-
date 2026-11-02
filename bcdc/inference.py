"""One inference engine for the CLI, the API and the demo."""
from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .conformal import score_matrix
from .dicom import dicom_to_image, is_dicom
from .ensemble import ensemble_probs, uncertainty
from .metrics import apply_temperature

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
DISCLAIMER = ("Research prototype, not a medical device. This output is not a diagnosis. "
              "Please discuss any MRI findings with a qualified doctor, such as a neurologist or neuro-oncologist.")


@dataclass
class Prediction:
    rejected: bool
    reason: str | None = None
    probabilities: dict = field(default_factory=dict)
    top_class: str | None = None
    top_probability: float | None = None
    prediction_set: list = field(default_factory=list)
    coverage_target: float | None = None
    epistemic_uncertainty: float | None = None
    ood_score: float | None = None
    ood_threshold: float | None = None
    warnings: list = field(default_factory=list)
    input_info: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def load_image(data: bytes, filename: str | None = None) -> tuple[Image.Image, dict]:
    """Reads PNG/JPEG/BMP/TIFF or DICOM bytes into an RGB PIL image. Returns (image, info)."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    if is_dicom(data, filename):
        img, info = dicom_to_image(data)
        return img.convert("RGB"), {"format": "DICOM", **info}
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        raise ValueError("Unsupported or corrupt image. Use PNG, JPEG, BMP, TIFF or DICOM.") from e
    fmt = img.format
    return ImageOps.exif_transpose(img).convert("RGB"), {"format": fmt, "warnings": []}


def looks_like_grayscale(img_rgb: Image.Image, max_mean_chroma=8.0) -> bool:
    a = np.asarray(img_rgb, dtype=np.float32)
    return float((np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])).mean() / 2) <= max_mean_chroma


class Predictor:
    def __init__(self, bundle_dir):
        self.dir = Path(bundle_dir)
        self.cfg = json.loads((self.dir / "deploy.json").read_text())
        self.class_names = self.cfg["class_names"]
        self.display_names = self.cfg.get("display_names", {})
        self.img_size = tuple(self.cfg["img_size"])
        self.T = float(self.cfg["temperature"])
        self.qhat = np.asarray([np.inf if q is None else q for q in self.cfg["conformal"]["qhat"]], dtype=float)
        self.alpha = float(self.cfg["conformal"]["alpha"])
        self.backend = self.cfg["backend"]
        if self.backend == "keras":
            from tensorflow import keras

            from .model import feature_model, model_parts
            from .ood import Mahalanobis
            self.members = []
            for rel in self.cfg["models"]:
                parts = model_parts(keras.models.load_model(self.dir / rel))
                self.members.append((parts, feature_model(parts)))
            ood = self.cfg.get("ood")
            self.maha = [Mahalanobis.load(self.dir / f) for f in ood["files"]] if ood else None
            self.ood_threshold = float(ood["threshold"]) if ood else None
        elif self.backend == "tflite":
            from .edge import TFLiteClassifier
            self.clf = TFLiteClassifier(model_path=self.dir / self.cfg["models"][0])
            self.maha, self.ood_threshold = None, None
        else:
            raise ValueError(f"unknown backend {self.backend!r}")

    def name(self, cls):
        return self.display_names.get(cls, cls)

    def _array(self, img: Image.Image):
        import tensorflow as tf
        x = tf.image.resize(np.asarray(img.convert("RGB"), dtype=np.float32)[None], self.img_size)
        return np.clip(np.round(x.numpy()), 0, 255)

    def predict(self, img: Image.Image, info: dict | None = None) -> Prediction:
        info = dict(info or {})
        warnings = list(info.pop("warnings", []) or [])
        if not looks_like_grayscale(img):
            warnings.append("The image is in colour; MRI slices are grayscale.")
        x = self._array(img)
        epistemic, ood_score = None, None
        if self.backend == "keras":
            feats = [fm(x, training=False).numpy() for _, fm in self.members]
            member_p = np.stack([_softmax(f @ p.W + p.b) for f, (p, _) in zip(feats, self.members)])
            if self.maha is not None:
                ood_score = float(np.mean([m.score(f)[0] for m, f in zip(self.maha, feats)]))
                if ood_score < self.ood_threshold:
                    return Prediction(True, "This image doesn't look like the brain MRI slices the model was "
                                            "trained on, so no prediction is made.",
                                      ood_score=ood_score, ood_threshold=self.ood_threshold,
                                      warnings=warnings, input_info=info)
            probs = apply_temperature(ensemble_probs(member_p), self.T)[0]
            epistemic = float(uncertainty(member_p)["epistemic"][0]) if len(self.members) > 1 else None
        else:
            warnings.append("This bundle has no out-of-distribution gate; any image will get a label.")
            probs = apply_temperature(self.clf.predict(x), self.T)[0]
        in_set = score_matrix(probs[None], "lac")[0] <= self.qhat
        order = np.argsort(-probs)
        top = int(order[0])
        if probs[top] < self.cfg.get("min_confidence", 0.7):
            warnings.append(f"Low confidence: the top probability is {100 * probs[top]:.0f}%.")
        return Prediction(
            rejected=False,
            probabilities={self.class_names[k]: float(probs[k]) for k in order},
            top_class=self.class_names[top], top_probability=float(probs[top]),
            prediction_set=[self.class_names[k] for k in order if in_set[k]],
            coverage_target=1 - self.alpha, epistemic_uncertainty=epistemic,
            ood_score=ood_score, ood_threshold=self.ood_threshold, warnings=warnings, input_info=info)

    def predict_bytes(self, data: bytes, filename: str | None = None) -> Prediction:
        img, info = load_image(data, filename)
        return self.predict(img, info)


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
