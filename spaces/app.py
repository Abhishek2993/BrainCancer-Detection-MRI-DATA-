"""Hugging Face Spaces entry point for the BCDC Gradio demo.

The deployment bundle (made with ``bcdc calibrate``) is loaded from, in order:
1. the folder in the BCDC_BUNDLE environment variable (default: ./deploy, uploaded with the Space), or
2. a Hugging Face model repository named in BCDC_HF_MODEL_REPO (downloaded at start-up).
"""
import os
from pathlib import Path

from bcdc.demo import build_demo
from bcdc.inference import Predictor

bundle = Path(os.environ.get("BCDC_BUNDLE", "deploy"))
if not (bundle / "deploy.json").exists():
    repo = os.environ.get("BCDC_HF_MODEL_REPO")
    if not repo:
        raise SystemExit("No bundle found: upload a 'deploy/' folder or set BCDC_HF_MODEL_REPO.")
    from huggingface_hub import snapshot_download
    bundle = Path(snapshot_download(repo))

build_demo(Predictor(bundle)).launch()
