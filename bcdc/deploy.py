"""Deployment bundles: everything the API, demo and CLI need to make a prediction.

A bundle is a folder with ``deploy.json`` plus model files::

    deploy.json         format version, classes, image size, backend, temperature,
                        conformal thresholds, OOD detector settings, evaluation summary
    models/             Keras members (ensemble) or one .tflite file
    ood/                Mahalanobis statistics per member (Keras backend only)

``build_bundle`` reproduces the Phase 1 deployment steps from trained models and the split:
temperature on val-A, class-conditional conformal thresholds on val-B, Mahalanobis OOD
statistics on the training set and an OOD threshold keeping ``ood_retain`` of val-A.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from .conformal import ConformalClassifier, set_metrics
from .data import DISPLAY_NAMES, labeled_path_dataset, load_split, path_dataset, split_validation
from .ensemble import ensemble_probs
from .metrics import apply_temperature, evaluate, fit_temperature
from .model import feature_model, model_parts, softmax
from .ood import Mahalanobis, threshold_at_retention

FORMAT_VERSION = 1


def build_bundle(artefact_dir, dataset_path, out_dir, model_files=None, tflite_file=None, alpha=0.10,
                 ood_retain=0.95, batch_size=32, val_seed=0, notes=""):
    """Builds a bundle from Phase 0/CLI artefacts (``split.csv``, ``model_meta.json``, ``models/``).

    Keras backend (default): every ``models/seed_*.keras`` becomes an ensemble member.
    TFLite backend: pass ``tflite_file``; there is no OOD gate (the TFLite graph exposes only
    probabilities, not features).
    """
    from tensorflow import keras

    artefact_dir, out_dir = Path(artefact_dir), Path(out_dir)
    meta = json.loads((artefact_dir / "model_meta.json").read_text())
    classes, img_size = meta["class_names"], tuple(meta["img_size"])
    split = split_validation(load_split(artefact_dir / "split.csv", dataset_path), seed=val_seed)
    sets = {s: split[split["split"] == s] for s in ("train", "valA", "valB", "test")}
    Y = {s: d["label"].to_numpy() for s, d in sets.items()}
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "models").mkdir(parents=True)

    cfg = {"format_version": FORMAT_VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "notes": notes,
           "class_names": classes, "display_names": {c: DISPLAY_NAMES.get(c, c) for c in classes},
           "img_size": list(img_size), "input": "RGB pixels in [0, 255]"}

    if tflite_file is not None:
        from .edge import TFLiteClassifier
        dst = out_dir / "models" / Path(tflite_file).name
        shutil.copy(tflite_file, dst)
        clf = TFLiteClassifier(model_path=dst)

        def probs(s):
            ds = labeled_path_dataset(sets[s]["abs_path"], Y[s], img_size, batch_size, cache=False)
            return clf.predict(np.concatenate([x.numpy() for x, _ in ds]).astype(np.uint8))

        P = {s: probs(s) for s in ("valA", "valB", "test")}
        cfg.update(backend="tflite", models=[f"models/{dst.name}"], ood=None)
        member_p = None
    else:
        model_files = sorted(model_files or (artefact_dir / "models").glob("seed_*.keras"))
        if not model_files:
            raise FileNotFoundError(f"No models in {artefact_dir / 'models'}")
        member_p, maha_files, maha_val = {s: [] for s in ("valA", "valB", "test")}, [], []
        (out_dir / "ood").mkdir()
        for f in model_files:
            dst = out_dir / "models" / Path(f).name
            shutil.copy(f, dst)
            model = keras.models.load_model(dst)
            parts = model_parts(model)
            fm = feature_model(parts)
            feats = {s: fm.predict(path_dataset(sets[s]["abs_path"], img_size, batch_size), verbose=0)
                     for s in ("train", "valA", "valB", "test")}
            for s in member_p:
                member_p[s].append(softmax(feats[s] @ parts.W + parts.b))
            maha = Mahalanobis().fit(feats["train"], Y["train"])
            mf = out_dir / "ood" / f"mahalanobis_{Path(f).stem}.npz"
            maha.save(mf)
            maha_files.append(f"ood/{mf.name}")
            maha_val.append(maha.score(feats["valA"]))
        P = {s: ensemble_probs(np.stack(v)) for s, v in member_p.items()}
        ood_thr = threshold_at_retention(np.mean(maha_val, axis=0), ood_retain)
        cfg.update(backend="keras", models=[f"models/{Path(f).name}" for f in model_files],
                   ood={"type": "mahalanobis_mean_over_members", "files": maha_files,
                        "threshold": float(ood_thr), "retain": ood_retain})

    T = fit_temperature(Y["valA"], P["valA"])
    cal = {s: apply_temperature(p, T) for s, p in P.items()}
    cp = ConformalClassifier(alpha, "lac", class_conditional=True).fit(cal["valB"], Y["valB"])
    test_m, _ = evaluate(Y["test"], cal["test"], classes)
    sm = set_metrics(cp.predict_sets(cal["test"]), Y["test"], classes)
    cfg.update(temperature=T, conformal=cp.to_dict(), min_confidence=0.70,
               evaluation={"split": "internal test", "n": int(len(Y["test"])),
                           **{k: test_m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc", "ece")},
                           **{f"sensitivity[{c}]": test_m[f"sensitivity[{c}]"] for c in classes},
                           "conformal_coverage": sm["coverage"], "conformal_avg_set_size": sm["avg_set_size"]})
    (out_dir / "deploy.json").write_text(json.dumps(cfg, indent=2))
    return cfg
