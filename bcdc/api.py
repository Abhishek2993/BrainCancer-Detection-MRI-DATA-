"""FastAPI backend.

Run with ``bcdc serve --bundle deploy/`` or
``BCDC_BUNDLE=deploy uvicorn bcdc.api:app_from_env --factory``.

Endpoints:
* ``GET  /health``  - liveness and the loaded backend
* ``GET  /info``    - classes, bundle settings, evaluation summary and the disclaimer
* ``POST /predict`` - multipart upload (PNG, JPEG, BMP, TIFF or DICOM) -> prediction + grounded report

Uploaded images are processed in memory and never stored. For DICOM, only technical
fields (modality, size) are returned; patient fields are never read into the response.
"""
import os

from . import __version__
from .inference import DISCLAIMER, MAX_UPLOAD_BYTES, Predictor
from .report import build_report


def create_app(predictor: Predictor):
    # NB: no "from __future__ import annotations" in this module: FastAPI must see real types.
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile

    app = FastAPI(title="BCDC: brain-tumour MRI classifier (research prototype)", version=__version__,
                  description=DISCLAIMER)

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": predictor.backend, "version": __version__}

    @app.get("/info")
    def info():
        cfg = predictor.cfg
        return {"classes": [{"id": c, "name": predictor.name(c)} for c in predictor.class_names],
                "img_size": cfg["img_size"], "backend": cfg["backend"], "ensemble_members": len(cfg["models"]),
                "conformal_coverage_target": 1 - predictor.alpha, "ood_gate": cfg.get("ood") is not None,
                "evaluation": cfg.get("evaluation"), "created": cfg.get("created"), "disclaimer": DISCLAIMER}

    @app.post("/predict")
    def predict(file: UploadFile = File(...), report: bool = Query(True, description="include the grounded report")):
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
        if not data:
            raise HTTPException(400, "Empty file.")
        try:
            pred = predictor.predict_bytes(data, file.filename)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(415, str(e))
        out = {"prediction": pred.to_dict(), "display_names": predictor.display_names, "disclaimer": DISCLAIMER}
        if report:
            out["report"] = build_report(pred, predictor.display_names)
        return out

    return app


def app_from_env():
    """App factory for uvicorn: reads the bundle folder from the BCDC_BUNDLE environment variable."""
    return create_app(Predictor(os.environ.get("BCDC_BUNDLE", "deploy")))
