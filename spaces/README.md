---
title: BCDC Brain Tumour MRI Classifier
emoji: 🧠
colorFrom: indigo
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Research prototype, not a medical device
---

# BCDC demo (Hugging Face Space)

A research prototype that classifies a 2D brain MRI slice as glioma, meningioma or an unspecified tumour class. It refuses images that don't look like its training data, gives a 90% prediction set, and writes a cited background report.

**Not a medical device and not a diagnosis.** See the [model card](https://github.com/Abhishek2993/BrainCancer-Detection-MRI-DATA-/blob/main/MODEL_CARD.md).

## Deploying this Space

1. Build a bundle from the trained models: `bcdc calibrate --artefacts outputs --out deploy`.
2. Create a Gradio Space and upload `app.py`, `requirements.txt`, this `README.md` and the `deploy/` folder.
   - Alternatively, push `deploy/` to a Hugging Face **model** repo and set the Space variable `BCDC_HF_MODEL_REPO=<user>/<repo>`.
3. The Space installs the `bcdc` package from GitHub and starts the demo.
