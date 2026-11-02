# Model card: BCDC brain-tumour MRI classifier

*Following the structure of "Model Cards for Model Reporting" (Mitchell et al., 2019). Numbers marked "—" are filled in after the first full training run; see the README's Results section.*

## Model details

| | |
|---|---|
| **Developer** | Abhishek Maurya (research and educational project) |
| **Version** | `bcdc` 0.4.0 |
| **Type** | Image classifier. Transfer-learned CNNs: MobileNetV2 by default, plus MobileNetV3, EfficientNetB0 and ConvNeXt-Tiny variants. Deployed either as a deep ensemble of seed models (Keras) or as a single TFLite model (FP32 / FP16 / INT8). |
| **Input** | One 2D brain MRI slice: PNG, JPEG, BMP, TIFF or single-frame DICOM, converted to RGB 224×224, with pixel values 0–255. |
| **Output** | Calibrated probabilities for 3 classes; a conformal **prediction set** (class-conditional, 90% target coverage); ensemble disagreement; an **out-of-distribution (OOD) rejection** for inputs unlike the training data (Keras bundles only); and a cited background report. |
| **Training** | Two stages: train the new head with the backbone frozen, then fine-tune the top 35% of the backbone with normalisation layers frozen. Early stopping on validation loss. Split grouped by near-duplicate images. |
| **License** | MIT |
| **Contact** | GitHub issues on this repository |

## Intended use

- **Intended:** research and education on trustworthy medical-image classification, including calibration, conformal prediction, OOD rejection, external validation, edge deployment and federated learning, plus demonstrations of these ideas.
- **Intended users:** students, researchers and engineers.

## Out-of-scope uses

- **Any clinical use**: diagnosis, triage, screening, or treatment decisions. This model is **not a medical device** and has not been clinically validated or approved by any regulator.
- **Deciding whether a tumour is present.** There is **no "no tumour" class**, so the model always names a tumour type for images it accepts.
- Use on CT, X-ray, other body parts, 3D volumes, or MRI sequences unlike the training data. The OOD gate is meant to reject these but is not guaranteed to.
- Use on patients from populations, hospitals or scanners not represented in the training data without new validation.

## Training data

- **Kaggle *Brain Cancer MRI* dataset** (`orvile/brain-cancer-mri-dataset`): 6,056 2D MRI images in 3 folders: `brain_glioma`, `brain_menin` and `brain_tumor`.
- The `brain_tumor` label's tumour type is **not documented** by the source, and it is shown as "Brain tumour (unspecified type)".
- **No patient IDs, no demographics, no scanner metadata** are provided. The dataset's demographic make-up (age, sex, ethnicity) is therefore **unknown**, and performance across groups **cannot be assessed**.
- **Preprocessing:** a duplicate audit (MD5 plus dHash across flips and rotations). Near-duplicates are grouped, and each group is kept entirely in train, validation or test, about 70 / 15 / 15. Label conflicts are dropped.

## Evaluation data

- **Internal test set:** a held-out 15% of the Kaggle dataset, split by duplicate group.
- **External test set:** the Figshare brain-tumour dataset (Cheng et al., 2015/2016), with 3,064 T1-weighted contrast-enhanced slices from 233 patients. Only glioma and meningioma are shared with the training classes. Slices that copy a training image are excluded first.
- **OOD sets:** CIFAR-10 photos (in colour and grayscale), noise, and Figshare pituitary tumours (an unseen tumour type).

## Metrics

Accuracy is not enough for this task. Missing a glioma is costlier than a false alarm, so the evaluation reports:

- per-class **sensitivity**, specificity and precision, plus one-vs-rest AUROC and macro-F1;
- **expected calibration error (ECE)** before and after temperature scaling;
- **conformal coverage** and average set size, overall and per class;
- **OOD** AUROC, FPR at 95% TPR, and the share of legitimate external scans wrongly rejected;
- **Grad-CAM localisation** against tumour masks (pointing game, energy-in-mask, IoU), compared with centre-prior and random baselines;
- for edge variants: model size, single-image latency, and agreement with FP32.

## Quantitative analyses

| | Internal test | External (Figshare) |
|---|---|---|
| Accuracy | — | — |
| Glioma sensitivity | — | — |
| Meningioma sensitivity | — | — |
| ECE (after temperature scaling) | — | — |
| Conformal coverage, 90% target | — | — |
| Average prediction-set size | — | — |

Results are mean ± std over seeds for the internal test set. The full tables are written by the notebooks to `outputs/results_summary.md`, `outputs/phase1/results_phase1.md` and `outputs/phase2/results_phase2.md`.

## Ethical considerations

- **Risk of over-trust.** A confident-looking label next to a scan can be mistaken for a diagnosis. The demo, API and reports state on every output that this is not a diagnosis. They show prediction sets rather than a single label, and refuse inputs unlike the training data.
- **No invented medical claims.** The explanation report only quotes curated passages from cited sources (WHO 2021 CNS classification, EANO guidelines, NCI PDQ, CBTRUS). It frames them as background and *questions to ask a doctor*. It gives **no** cost, hospital or doctor recommendations.
- **Privacy.** Uploaded images are processed in memory and not stored. For DICOM uploads, only technical fields (modality, size) are read into responses; patient fields are never returned or logged.
- **Fairness.** Without demographic data, performance gaps between patient groups cannot be measured. This is a known, unaddressed risk.
- **Federated learning** (Phase 2) is a simulation. It is not a privacy guarantee without secure aggregation or differential privacy.

## Caveats and recommendations

- Internal scores may be **optimistic**, because different slices of the same patient can fall in both train and test (there are no patient IDs).
- Expect a **performance drop on new hospitals and scanners**. The Figshare external test measures this for two classes.
- Conformal coverage is only guaranteed for data like the calibration set. **Recalibrate** after changing the model (e.g. INT8 quantisation) or the data source.
- The OOD gate has been tested on examples, not proven in general.
- Grad-CAM shows *where* evidence is, not *how* the model decides.
- **Recommended before any real-world study:** patient-level splits, multi-site external validation, a reader study with radiologists, subgroup analysis, and regulatory review.
