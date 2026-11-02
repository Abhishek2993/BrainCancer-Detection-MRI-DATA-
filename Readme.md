# BCDC: Brain Cancer Detection & Classification

### 3-class brain-tumour MRI classification with leakage-aware splitting, multi-seed evaluation and calibrated confidence

![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras%203-orange) ![Model](https://img.shields.io/badge/Model-MobileNetV2-purple) ![Platform](https://img.shields.io/badge/Platform-Google%20Colab-blue) ![Status](https://img.shields.io/badge/Status-Phase%200%20(correctness)-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

<a href="https://colab.research.google.com/github/Abhishek2993/BrainCancer-Detection-MRI-DATA-/blob/main/Project_BCDCipynb.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---

> ⚠️ **Disclaimer:** BCDC is a research and educational project. It is **not** a medical device and must not be used for diagnosis. The model has **no "no tumour" class**: it assumes a tumour is present and only estimates which of three labels fits best. Always consult a qualified medical professional.

---

## Overview

BCDC classifies a 2D brain MRI slice into one of the three classes in the Kaggle *Brain Cancer MRI* dataset, using a transfer-learned **MobileNetV2**.

Most student projects aim for the highest possible accuracy number. This one aims for an **honest, reproducible** number. It checks the data for duplicates, splits so copies can't leak between train and test, evaluates on a held-out test set across several random seeds, reports the per-class metrics that matter clinically, and calibrates its confidence scores.

The project runs in phases (see the [Roadmap](#roadmap)). **Phase 0, making the pipeline correct, is what's in this repository today.**

---

## What the notebook does

`Project_BCDCipynb.ipynb` runs top to bottom in Google Colab:

| Step | What happens |
|---|---|
| **1. Data audit** | Counts images per class, image modes and sizes. Finds **exact duplicates** (MD5) and **near-duplicates** (64-bit dHash compared across all 8 flips/90° rotations, ≤ 4 differing bits). |
| **2. Leakage-aware split** | Near-duplicates are merged into groups, and each group goes entirely into **train, validation or test** (~70/15/15, stratified by class). Copies filed under two different classes are dropped. The split is saved to `outputs/split.csv`. |
| **3. Model** | MobileNetV2 (ImageNet weights). Preprocessing is built into the model and scales inputs to **[-1, 1]**, the range MobileNetV2 was pretrained on. Training-only augmentation: flip, small rotation, zoom, contrast. |
| **4. Two-stage training** | Stage 1 trains the new head with the backbone frozen (Adam 1e-3). Stage 2 unfreezes the backbone from layer 100 up (BatchNorm stays frozen) with Adam 1e-5. Early stopping on validation loss, best weights restored. |
| **5. Multiple seeds** | The whole training run is repeated for each seed in `TRAIN_SEEDS` (default 3; use 5 for final numbers). All metrics are reported as **mean ± std**. |
| **6. Evaluation (test set)** | Accuracy, balanced accuracy, macro-F1, per-class **sensitivity / specificity / precision**, one-vs-rest **AUROC**, confusion matrix, and expected calibration error (**ECE**). |
| **7. Calibration** | **Temperature scaling** fitted on the validation set only. Reliability diagrams before and after. |
| **8. Inference** | `predict_image()` accepts any image mode (RGBA, grayscale, palette), applies softmax **once**, reports calibrated probabilities, says **"uncertain"** below 70%, and warns when an input is in colour and so unlikely to be an MRI. |

All results, figures and the trained model are written to `outputs/`:

```
outputs/
├── split.csv                    # which image is in train / val / test
├── results_summary.md           # the results table below, ready to paste
├── test_metrics_per_seed.csv
├── test_metrics_summary.csv
├── bcdc_mobilenetv2.keras       # model chosen by lowest validation loss
├── model_meta.json              # class names, image size, temperature
└── figures/                     # training curves, confusion matrix, reliability diagram, duplicate examples
```

---

## Dataset

- **Source:** [Brain Cancer MRI dataset](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset) on Kaggle (`orvile/brain-cancer-mri-dataset`), downloaded automatically with `kagglehub`.
- **Size:** 6,056 2D MRI images.
- **Classes (3):**

| Folder | Shown as | Note |
|---|---|---|
| `brain_glioma` | Glioma | |
| `brain_menin` | Meningioma | |
| `brain_tumor` | Brain tumour (unspecified type) | The dataset doesn't say which tumour type this is, so it is **not** treated as pituitary. |

- **No healthy / "no tumour" class.**
- **No patient IDs** are provided, so a true patient-level split isn't possible. See [Limitations](#limitations).

---

## Results

> **Not yet reported.** Numbers will be added here after the first full GPU run of the corrected notebook. The notebook writes the finished table to `outputs/results_summary.md`.

The table will contain (test set, mean ± std over seeds):

| Metric | Value |
|---|---|
| Accuracy (%) | — |
| Balanced accuracy (%) | — |
| Macro-F1 (%) | — |
| Macro AUROC (one-vs-rest) | — |
| ECE before → after temperature scaling | — |

| Class | Sensitivity (%) | Specificity (%) | Precision (%) | AUROC |
|---|---|---|---|---|
| Glioma | — | — | — | — |
| Meningioma | — | — | — | — |
| Brain tumour (unspecified type) | — | — | — | — |

**About the earlier 92.9% figure:** a previous version reported 92.9% *validation* accuracy. That number came from a frozen MobileNetV2 trained for only 3 epochs, with a random image-level split (duplicates could leak between train and validation), the wrong input scaling, and no held-out test set. It has been withdrawn and shouldn't be compared with the new results.

---

## What changed from the first version

| Problem in v1 | Fix |
|---|---|
| README described 4 classes (incl. "No Tumour") and a trained ResNet50 | README matches the code: 3 classes, MobileNetV2 only. The untrained ResNet50 was removed. |
| `Rescaling(1/255)` fed pretrained backbones the wrong input range | Uses MobileNetV2's own scaling to [-1, 1], built into the model |
| Softmax applied twice, so every "confidence" fell between ~21% and 58% | Softmax applied once, plus temperature scaling |
| Only validation accuracy; `y_true`/`y_pred` collected but never used | Held-out test set, confusion matrix, per-class sensitivity/specificity, macro-F1, AUROC, ECE |
| Random image-level split, so duplicates could leak | Duplicate audit and a group-aware, stratified split |
| 3 epochs, 1 run | Two-stage training with early stopping, several seeds, mean ± std |
| RGBA/grayscale uploads crashed; any image got a confident label | Images are converted to RGB, with an "uncertain" threshold and a colour-image warning |
| Hard-coded treatment costs and "best hospitals" shown as "Estimated Insights" | **Removed.** Unsourced figures don't belong next to a medical prediction. |

---

## How to run

1. Click **Open in Colab** above.
2. Choose *Runtime → Change runtime type → **T4 GPU***.
3. *Runtime → Run all*. The dataset downloads automatically. A full 3-seed run takes roughly 30–60 minutes.
4. To try your own image, uncomment `run_app()` in section 11, or call `predict_image("path/to/scan.png")`.

Running locally: `pip install -r requirements.txt`, then open the notebook in Jupyter. You'll need Kaggle API credentials for `kagglehub`.

All settings (seeds, split fractions, epochs, learning rates, duplicate threshold, confidence threshold) are in the **Configuration** cell at the top of the notebook.

---

## Limitations

- **No patient-level split.** Grouping near-duplicates removes copied images but not *different slices of the same patient*, so test scores may still be optimistic.
- **Single dataset.** Performance on scans from other hospitals or scanners is unknown until external validation is done (Phase 1).
- **No healthy class.** The model always names a tumour type, even for a normal scan.
- **Colour check ≠ OOD detection.** The "doesn't look like an MRI" warning only catches colour images. A grayscale photo of anything will still get a label.
- **2D slices only.** The model doesn't use full 3D, multi-sequence MRI studies (T1, T1ce, T2, FLAIR).
- **The `brain_tumor` class is underspecified** in the source data, which limits its clinical meaning.

---

## Roadmap

### Phase 0: Correctness ✅ *(this version)*
- [x] Correct MobileNetV2 preprocessing; single softmax; RGB conversion of uploads
- [x] README matches the code
- [x] Train / validation / test split, grouped by near-duplicate cluster (no patient IDs available; documented as a limitation)
- [x] Confusion matrix, per-class sensitivity and specificity, macro-F1, one-vs-rest AUROC
- [x] Two-stage training (frozen head → fine-tune top blocks), early stopping, multi-seed mean ± std
- [x] Temperature scaling and ECE
- [ ] Publish the first full-run numbers in the Results table

### Phase 1: Rigour
- [ ] **External validation:** train here, test on a different public dataset (e.g. the Figshare T1-contrast set by Cheng et al.), then measure and reduce the drop
- [ ] **"No tumour" class** without source shortcuts: intensity normalisation, skull-stripping, and a check of whether the model can guess which dataset an image came from
- [ ] **Uncertainty:** deep ensembles or MC-dropout; **conformal prediction** sets with guaranteed coverage
- [ ] **Out-of-distribution rejection** ("this doesn't look like a brain MRI")
- [ ] **Explainability that's verified:** Grad-CAM checked against tumour masks (pointing game / IoU), not just shown

### Phase 2: Headline direction: edge-efficient, trustworthy diagnosis
- [ ] Benchmark lightweight backbones (MobileNetV3, EfficientNet-Lite, small ConvNeXt)
- [ ] INT8 quantisation with TFLite / ONNX; run on Raspberry Pi or Jetson
- [ ] Plot accuracy, calibration and latency against each other
- [ ] Federated learning (e.g. Flower) to simulate several hospitals training without sharing scans

*Alternatives considered: 3D segmentation on BraTS with MONAI/nnU-Net, or comparing medical foundation-model pretraining (RadImageNet, BiomedCLIP). Edge deployment was chosen because it best fits an AI + edge-computing background.*

### Phase 3: Product
- [ ] Move from one notebook to a proper repo: `src/`, configs, training script, tests, fixed seeds, experiment tracking (Weights & Biases / MLflow)
- [ ] FastAPI backend and a Gradio demo on Hugging Face Spaces, with DICOM input
- [ ] Model card: intended use, out-of-scope use, data, failure cases
- [ ] A cited, retrieval-grounded explanation report (WHO CNS tumour classification, clinical guidelines), framed as *"questions to ask your neurologist"*, not advice. No cost or doctor information unless it comes from live, cited sources.

The long-term goal is a write-up on cross-dataset generalisation and calibration of edge models, and later a model trained across many kinds of medical data.

---

## Tech stack

Python · TensorFlow / Keras 3 · MobileNetV2 · scikit-learn · NumPy · pandas · Pillow · Matplotlib · kagglehub · Google Colab

## License

[MIT](LICENSE)
