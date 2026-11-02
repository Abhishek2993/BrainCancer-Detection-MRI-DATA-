# BCDC: Brain Cancer Detection & Classification

### Trustworthy, edge-ready 3-class brain-tumour MRI classification: leakage-aware evaluation, external validation, calibrated uncertainty, conformal prediction, OOD rejection, verified explanations, INT8 edge deployment and federated learning

![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras%203-orange) ![Model](https://img.shields.io/badge/Models-MobileNet%20%7C%20EfficientNet%20%7C%20ConvNeXt-purple) ![Edge](https://img.shields.io/badge/Edge-TFLite%20%2F%20LiteRT%20INT8-teal) ![Platform](https://img.shields.io/badge/Platform-Google%20Colab-blue) ![Status](https://img.shields.io/badge/Status-Phase%202%20(edge)-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

| Notebook | Open |
|---|---|
| **Phase 0:** training and evaluation (`Project_BCDCipynb.ipynb`) | <a href="https://colab.research.google.com/github/Abhishek2993/BrainCancer-Detection-MRI-DATA-/blob/main/Project_BCDCipynb.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| **Phase 1:** rigour (`Phase1_Rigour.ipynb`) | <a href="https://colab.research.google.com/github/Abhishek2993/BrainCancer-Detection-MRI-DATA-/blob/main/Phase1_Rigour.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| **Phase 2:** edge deployment and federated learning (`Phase2_Edge.ipynb`) | <a href="https://colab.research.google.com/github/Abhishek2993/BrainCancer-Detection-MRI-DATA-/blob/main/Phase2_Edge.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

---

> ⚠️ **Disclaimer:** BCDC is a research and educational project. It is **not** a medical device and must not be used for diagnosis. The model has **no "no tumour" class**: it assumes a tumour is present and only estimates which of three labels fits best. Always consult a qualified medical professional.

---

## Overview

BCDC classifies a 2D brain MRI slice into one of the three classes in the Kaggle *Brain Cancer MRI* dataset, using transfer-learned **MobileNetV2** models.

Most student projects aim for the highest possible accuracy number. This one aims for an **honest, reproducible** number and a model that knows its limits:

- **Phase 0 (correctness):** duplicate audit, leakage-aware split, held-out test set, multi-seed results, clinically relevant per-class metrics, calibrated confidence.
- **Phase 1 (rigour):** a deep ensemble with uncertainty estimates, **conformal prediction sets** with a coverage guarantee, **external validation** on a second dataset from different hospitals, **out-of-distribution rejection**, and **Grad-CAM checked against radiologist tumour masks**.
- **Phase 2 (edge):** five lightweight backbones × four TFLite precisions (FP32, FP16, dynamic, **INT8**). For each variant: accuracy, glioma sensitivity, **calibration and conformal coverage**, size and latency, plus a Pareto front. Includes a **Raspberry Pi / Jetson benchmark bundle** and a **federated learning** simulation of hospitals training together without sharing scans.

See the [Roadmap](#roadmap) for what comes next.

---

## Repository layout

```
├── Project_BCDCipynb.ipynb   # Phase 0: audit, split, train (multi-seed), evaluate, calibrate
├── Phase1_Rigour.ipynb       # Phase 1: ensemble, conformal, external validation, OOD, Grad-CAM
├── Phase2_Edge.ipynb         # Phase 2: backbones × quantisation trade-offs, edge bundle, federated learning
├── bcdc/                     # shared Python package used by the Phase 1 and 2 notebooks
│   ├── data.py               #   dataset discovery, image loading, duplicate hashing / matching
│   ├── model.py              #   model architecture, features and logits from saved models
│   ├── metrics.py            #   metrics, ECE, temperature scaling, risk–coverage
│   ├── ensemble.py           #   ensemble averaging, aleatoric / epistemic uncertainty
│   ├── conformal.py          #   split-conformal sets: LAC, APS, class-conditional
│   ├── ood.py                #   MSP, energy and Mahalanobis OOD scores + metrics
│   ├── gradcam.py            #   Grad-CAM + pointing game, energy-in-mask, IoU, baselines
│   ├── figshare.py           #   loader for the Figshare (Cheng et al.) dataset
│   ├── backbones.py          #   MobileNetV2/V3, EfficientNetB0, ConvNeXt-Tiny with correct input scaling
│   ├── train.py              #   shared two-stage transfer-learning recipe
│   ├── edge.py               #   TFLite export (FP32/FP16/dynamic/INT8), inference, latency, Pareto front
│   └── federated.py          #   FedAvg / FedProx, IID and Dirichlet (non-IID) hospital splits
├── edge/
│   └── run_on_device.py      # standalone benchmark for Raspberry Pi / Jetson (NumPy + LiteRT only)
├── tests/                    # unit tests (pytest), run without any dataset
├── requirements.txt
└── LICENSE
```

---

## Phase 0: training and evaluation

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
| **9. Save** | The validation-selected model, **every seed's model** (`outputs/models/seed_<n>.keras`, used as the Phase 1 ensemble), the split and the metadata. In Colab, `outputs/` is also copied to Google Drive (`MyDrive/BCDC/outputs`). |

---

## Phase 1: rigour

`Phase1_Rigour.ipynb` loads the Phase 0 models from Google Drive and answers five questions:

| Section | Question | How |
|---|---|---|
| **A. Deep ensemble** | Does combining the seed models help, and does the ensemble know when it's unsure? | Average the members' probabilities, then apply one temperature. Split uncertainty into *aleatoric* (the case is ambiguous) and *epistemic* (the members disagree). Risk–coverage curve and AURC: how fast the error falls if the least-certain cases are referred to a human. |
| **B. Conformal prediction** | Can the model output a *set* of labels that contains the true class ≥ 90% (or 95%) of the time? | Split-conformal **LAC**, **APS** and **class-conditional LAC**, which guarantees coverage *for each tumour type separately*. Reports coverage, average set size and per-class coverage. |
| **C. External validation** | How much does performance drop on a different dataset? | Test on the **Figshare** dataset (Cheng et al.: 3,064 slices, 233 patients, other hospitals). Uses the classes both datasets share (glioma, meningioma), at image and **patient** level, comparing internal vs external accuracy, sensitivity, AUROC, calibration and conformal coverage. Figshare slices that copy a training image are **excluded** first (hash *or* thumbnail-correlation match, across all flips and rotations). |
| **D. OOD rejection** | Can it say *"this isn't like my training MRIs"* instead of guessing? | Compares **MSP**, **energy**, **ensemble disagreement** and **Mahalanobis** (feature-space) scores on colour and **grayscale** CIFAR-10 photos, noise, and an **unseen tumour type** (Figshare pituitary). Threshold set to keep 95% of validation images. It also measures how many *legitimate* external scans get wrongly rejected. |
| **E. Verified explainability** | Do Grad-CAM heatmaps actually fall on the tumour? | Compared with radiologist **tumour masks**: pointing game, energy-in-mask, area-matched IoU and border energy. Each is benchmarked against a **centre-prior** and a **random** baseline. |

The validation set is split in two, by duplicate group:

- **val-A** fits the temperature and the OOD threshold.
- **val-B** calibrates the conformal sets.

The two halves are kept separate so that the conformal guarantee holds.

Phase 1 ends with `predict_image_v2()`, which runs four steps:

1. **OOD gate:** rejects the image, with no prediction, if it doesn't look like a training MRI.
2. **Ensemble probabilities,** calibrated with the ensemble temperature.
3. **Conformal prediction set** at 90% coverage, e.g. *{Glioma, Meningioma}* when the model can't tell them apart.
4. **Member disagreement** (epistemic uncertainty), reported alongside the result.

Outputs are written to `outputs/phase1/` (on Drive in Colab). They include `results_phase1.md` with every table, ready to paste, plus the deployed conformal and OOD settings and figures.

---

## Phase 2: edge-efficient, trustworthy diagnosis

`Phase2_Edge.ipynb` uses the same split and training recipe as Phase 0:

| Section | What it does |
|---|---|
| **A. Backbone benchmark** | Trains **MobileNetV2, MobileNetV3-Small, MobileNetV3-Large, EfficientNetB0 and ConvNeXt-Tiny**, each with the input scaling it was pretrained with. The Phase 0 MobileNetV2 is reused. Models are saved to Drive as they finish, so an interrupted run can resume. |
| **B. Quantisation** | Exports each model to TensorFlow Lite as **FP32, FP16, dynamic-range and full INT8**. INT8 is calibrated on 200 training images and takes uint8 RGB input. |
| **C. Trade-offs** | For every variant: test accuracy, macro-F1, **glioma sensitivity**, ECE (after a per-variant temperature), **class-conditional conformal coverage and set size**, agreement with FP32, file size, and single-image CPU latency (1 and 4 threads). Plots accuracy, calibration and set size against latency and size, and marks the **Pareto front**. Also tests whether **conformal thresholds survive INT8** or must be recalibrated. The deployed model is the most accurate on *validation* within a latency budget. |
| **D. Edge bundle** | `bcdc_edge_bundle.zip` contains the INT8 and FP16 models, 300 test images, the per-model temperatures and conformal thresholds, and `run_on_device.py`. On a **Raspberry Pi or Jetson** it measures real latency, accuracy, ECE and coverage with only `numpy` + `ai-edge-litert`. |
| **E. Federated learning** | Splits the training set across 4 simulated hospitals: **IID**, and **non-IID** with Dirichlet(0.5) label skew. Compares **centralised**, **FedAvg**, **FedProx** and **local-only** training, with the same model, learning rate and data passes. Reports accuracy, glioma sensitivity, calibration, conformal coverage and **communication cost**. |

Some INT8 graphs, such as MobileNetV3's hard-swish, fail to prepare in LiteRT's default XNNPACK CPU delegate. The code then falls back to the built-in kernels and says so in the results, because this changes real-device latency.

---

## Datasets

**Training and internal test: [Brain Cancer MRI dataset](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset)** (Kaggle, `orvile/brain-cancer-mri-dataset`)

- 6,056 2D MRI images, downloaded automatically with `kagglehub`. No healthy class and no patient IDs.

| Folder | Shown as | Note |
|---|---|---|
| `brain_glioma` | Glioma | |
| `brain_menin` | Meningioma | |
| `brain_tumor` | Brain tumour (unspecified type) | The dataset doesn't say which tumour type this is, so it is **not** treated as pituitary. |

**External validation: [Figshare brain-tumour dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)** (Cheng et al.), downloaded automatically by `bcdc.figshare` (~880 MB)

- 3,064 T1-weighted contrast-enhanced slices from 233 patients: glioma, meningioma and pituitary, with a tumour mask for every slice.
- Please cite: Cheng J. et al., *PLoS ONE* 10(10), 2015, and *PLoS ONE* 11(6), 2016.

**OOD examples:** CIFAR-10 test images (via `keras.datasets`) and synthetic noise.

---

## Results

> **Not yet reported.** Numbers will be added after the first full GPU run of the notebooks. Phase 0 writes `outputs/results_summary.md`, Phase 1 writes `outputs/phase1/results_phase1.md`, Phase 2 writes `outputs/phase2/results_phase2.md`, and the device script writes `device_results.json`. Paste them here.

**Phase 0** (internal test set, mean ± std over seeds):

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

**Phase 1** (headline numbers):

| | Internal test | Figshare (external) |
|---|---|---|
| Ensemble accuracy (glioma + meningioma) | — | — |
| Glioma sensitivity | — | — |
| Conformal coverage at 90% target (class-conditional) | — | — |
| Average prediction-set size | — | — |
| OOD: grayscale CIFAR-10 accepted (Mahalanobis) | — | n/a |
| Legitimate scans wrongly rejected by OOD gate | — | — |
| Grad-CAM pointing game (vs centre-prior baseline) | n/a | — |

**Phase 2** (internal test set; latency on the Colab CPU and on a Raspberry Pi):

| Variant | Accuracy | Glioma sensitivity | ECE | Conformal coverage / set size | Size (MB) | Latency, Colab CPU (ms) | Latency, Raspberry Pi (ms) |
|---|---|---|---|---|---|---|---|
| MobileNetV2 FP32 | — | — | — | — | — | — | — |
| MobileNetV2 INT8 | — | — | — | — | — | — | — |
| MobileNetV3-Small INT8 | — | — | — | — | — | — | — |
| EfficientNetB0 INT8 | — | — | — | — | — | — | — |
| ConvNeXt-Tiny FP16 | — | — | — | — | — | — | — |

| Federated (4 hospitals) | Accuracy | Glioma sensitivity | Communication (MB) |
|---|---|---|---|
| Centralised | — | — | 0 |
| FedAvg (non-IID) | — | — | — |
| FedProx (non-IID) | — | — | — |
| Local only (non-IID, mean) | — | — | 0 |

**About the earlier 92.9% figure:** a previous version reported 92.9% *validation* accuracy. That number came from a frozen MobileNetV2 trained for only 3 epochs, with a random image-level split (duplicates could leak between train and validation), the wrong input scaling, and no held-out test set. It has been withdrawn and shouldn't be compared with the new results.

---

## How to run

1. Open **Phase 0** in Colab (badge above). Choose *Runtime → Change runtime type → **T4 GPU***, then *Run all* (about 30–60 minutes for 3 seeds). Allow Google Drive access when asked, so the models are saved to `MyDrive/BCDC/outputs`.
2. Open **Phase 1** in Colab, with the same GPU setting, and *Run all* (about 20–40 minutes). It clones this repo for the `bcdc` package, loads the Phase 0 models from Drive, and downloads Figshare and CIFAR-10.
3. Open **Phase 2** in Colab (T4 GPU) and *Run all*. Training five backbones and the federated runs take roughly 1.5–2.5 hours. Everything is saved to Drive as it finishes, so after a disconnect you can just *Run all* again.
4. **On a Raspberry Pi / Jetson:** copy `MyDrive/BCDC/outputs/phase2/bcdc_edge_bundle.zip` to the device, then:
   ```bash
   pip install numpy ai-edge-litert
   unzip bcdc_edge_bundle.zip && cd bcdc_edge_bundle
   python run_on_device.py --threads 4
   ```
5. Paste `results_summary.md`, `phase1/results_phase1.md`, `phase2/results_phase2.md` and the device numbers into the Results section above.

All settings (seeds, split fractions, epochs, learning rates, duplicate thresholds, conformal α, OOD retention) are in each notebook's **Configuration** cell.

**Running locally:** `pip install -r requirements.txt`. You'll need Kaggle API credentials for `kagglehub`. Unit tests (no datasets needed):

```bash
python -m pytest tests
```

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
| RGBA/grayscale uploads crashed; any image got a confident label | Images converted to RGB; "uncertain" threshold; Phase 1 adds a feature-space OOD gate and conformal prediction sets |
| Hard-coded treatment costs and "best hospitals" shown as "Estimated Insights" | **Removed.** Unsourced figures don't belong next to a medical prediction. |

---

## Limitations

- **No patient-level split on the training dataset.** It has no patient IDs, so different slices of the same patient can still be split across train and test, which may make internal scores optimistic. Figshare *does* have patient IDs, so Phase 1 also reports patient-level external results.
- **External validation covers 2 of 3 classes.** Figshare has no `brain_tumor` class, and the training data has no pituitary class.
- **Figshare slices are converted** from raw MRI intensities to 8-bit with a per-slice percentile window. Part of any external performance drop may come from this conversion.
- **The cross-dataset duplicate check catches copied images, not shared patients.**
- **Conformal guarantees assume exchangeability:** they hold for data like the calibration set. Phase 1 measures how coverage changes on the external dataset.
- **OOD rejection is tested on examples, not proven.** It is not tested on, e.g., CT or X-ray images, and may also reject legitimate scans from new scanners.
- **Grad-CAM is coarse** (a 7×7 map). Good localisation shows the model *uses* the tumour region, not *how* it decides.
- **No healthy class.** The model always names a tumour type for images it accepts.
- **2D slices only:** no 3D, multi-sequence MRI (T1, T1ce, T2, FLAIR).
- **Phase 2 latency on Colab is only a ranking.** Edge claims rest on the Raspberry Pi / Jetson numbers from `run_on_device.py`.
- **One training seed per backbone in Phase 2,** so small differences between backbones may be noise.
- **EfficientNetB0 stands in for EfficientNet-Lite,** which isn't in `keras.applications` and is designed to quantise better.
- **The federated learning is a single-machine simulation:** hospitals are a split of one dataset, and there's no secure aggregation or differential privacy.

---

## Roadmap

### Phase 0: Correctness ✅
- [x] Correct MobileNetV2 preprocessing; single softmax; RGB conversion of uploads
- [x] README matches the code
- [x] Train / validation / test split, grouped by near-duplicate cluster (no patient IDs available; documented as a limitation)
- [x] Confusion matrix, per-class sensitivity and specificity, macro-F1, one-vs-rest AUROC
- [x] Two-stage training (frozen head → fine-tune top blocks), early stopping, multi-seed mean ± std
- [x] Temperature scaling and ECE
- [ ] Publish the first full-run numbers in the Results table

### Phase 1: Rigour 🔄 *(code complete; awaiting first full run)*
- [x] **External validation** on the Figshare T1-contrast set (Cheng et al.), at image and patient level, with cross-dataset duplicate exclusion
- [x] **Uncertainty:** deep ensemble with aleatoric/epistemic decomposition and selective classification (risk–coverage)
- [x] **Conformal prediction** (LAC, APS, class-conditional) with coverage measured internally and under dataset shift
- [x] **Out-of-distribution rejection** (MSP, energy, ensemble disagreement, Mahalanobis), tested on colour and grayscale natural images, noise and an unseen tumour type
- [x] **Verified explainability:** Grad-CAM vs tumour masks (pointing game, energy-in-mask, IoU), compared against centre-prior and random baselines
- [ ] Publish the Phase 1 numbers in the Results table
- [ ] **Phase 1b: "no tumour" class** without dataset shortcuts: intensity normalisation, skull-stripping, and a check of whether the model can guess which dataset an image came from
- [ ] Reduce the external performance drop (e.g. intensity standardisation, stronger augmentation), then re-measure

### Phase 2: Headline direction: edge-efficient, trustworthy diagnosis 🔄 *(code complete; awaiting first full run)*
- [x] Benchmark lightweight backbones: MobileNetV2, MobileNetV3-Small/Large, EfficientNetB0 (EfficientNet-Lite stand-in), ConvNeXt-Tiny
- [x] TFLite export in FP32, FP16, dynamic-range and full INT8, with XNNPACK fallback detection
- [x] Plot accuracy, calibration, conformal set size and latency against each other; Pareto front; conformal coverage under INT8 (reused vs recalibrated thresholds)
- [x] Edge bundle + `run_on_device.py` benchmark (NumPy + LiteRT only)
- [x] Federated learning simulation: FedAvg and FedProx vs centralised and local-only, IID and non-IID hospitals, communication cost
- [ ] Run the bundle on a Raspberry Pi / Jetson and publish the device numbers
- [ ] ONNX Runtime export as a second deployment path
- [ ] Real multi-machine federation with Flower, plus secure aggregation / differential privacy
- [ ] Phase 1 external validation (Figshare) of the chosen edge model

*Alternatives considered: 3D segmentation on BraTS with MONAI/nnU-Net, or comparing medical foundation-model pretraining (RadImageNet, BiomedCLIP). Edge deployment was chosen because it best fits an AI + edge-computing background.*

### Phase 3: Product
- [ ] Finish moving from notebooks to a proper repo: training script, configs, experiment tracking (Weights & Biases / MLflow). The `bcdc/` package and tests are the start of this.
- [ ] FastAPI backend and a Gradio demo on Hugging Face Spaces, with DICOM input
- [ ] Model card: intended use, out-of-scope use, data, failure cases
- [ ] A cited, retrieval-grounded explanation report (WHO CNS tumour classification, clinical guidelines), framed as *"questions to ask your neurologist"*, not advice. No cost or doctor information unless it comes from live, cited sources.

The long-term goal is a write-up on cross-dataset generalisation and calibration of edge models, and later a model trained across many kinds of medical data.

---

## Tech stack

Python · TensorFlow / Keras 3 · MobileNetV2/V3 · EfficientNet · ConvNeXt · TensorFlow Lite / LiteRT · scikit-learn · NumPy · pandas · Pillow · Matplotlib · h5py · SciPy · kagglehub · pytest · Google Colab

## License

[MIT](LICENSE)
