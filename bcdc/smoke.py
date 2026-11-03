"""Builds a tiny, UNTRAINED deployment bundle for smoke tests (CI, Docker image checks).

    python -m bcdc.smoke OUT_DIR

Creates a synthetic 3-class image set, two untrained 32x32 MobileNetV2 models, a split, and a
bundle in OUT_DIR/bundle, plus OUT_DIR/sample.png to send to /predict. The predictions are
meaningless; it only checks that the serving stack works end to end.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["brain_glioma", "brain_menin", "brain_tumor"]


def make_images(root: Path, n=12, size=48, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    for k, c in enumerate(CLASSES):
        (root / c).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            a = rng.normal(60, 20, (size, size)) + 150 * np.exp(-((yy - 10 - 14 * k) ** 2 + (xx - size / 2) ** 2) / 30)
            Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(root / c / f"{c}_{i}.png")
    return root


def build(out_dir) -> Path:
    from .deploy import build_bundle
    from .model import build_model
    from .split import make_split

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = make_images(tmp / "data")
        art = tmp / "outputs"
        (art / "models").mkdir(parents=True)
        make_split(data, art / "split.csv")
        for seed in (0, 1):
            model, _ = build_model(len(CLASSES), img_size=(32, 32), weights=None, seed=seed)
            model.save(art / "models" / f"seed_{seed}.keras")
        (art / "model_meta.json").write_text(json.dumps({"class_names": CLASSES, "img_size": [32, 32]}))
        build_bundle(art, data, out_dir / "bundle", notes="UNTRAINED smoke-test bundle: predictions are meaningless.")
        Image.open(next((data / "brain_glioma").glob("*.png"))).save(out_dir / "sample.png")
    return out_dir / "bundle"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m bcdc.smoke OUT_DIR")
    print("Smoke bundle written to", build(sys.argv[1]))
