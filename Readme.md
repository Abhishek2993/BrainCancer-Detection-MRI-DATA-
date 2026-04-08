# BCDC
### Brain Cancer Detection & Care · MRI-based early detection · diagnosis · treatment cost · doctor recommendations

![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) ![Model](https://img.shields.io/badge/Model-ResNet50%20%7C%20MobileNetV2-purple) ![Platform](https://img.shields.io/badge/Platform-Google%20Colab-blue) ![License](https://img.shields.io/badge/License-MIT-green)
.
---

> ⚠️ **Disclaimer:** BCDC is a research and educational tool. It is not a substitute for professional medical diagnosis. Always consult a qualified medical professional for any health concerns.

---

## Overview

BCDC (Brain Cancer Detection & Care) is a deep learning-powered tool for early detection of brain cancer from MRI scans. Users upload an MRI image; the system classifies the scan, identifies the likely tumour type, and then provides actionable next steps — including estimated treatment costs, affordable care locations, and recommended specialists to contact.

The model was trained on the **Kaggle Brain Tumour MRI Dataset** using transfer learning with **ResNet50** and **MobileNetV2**, achieving strong classification performance across multiple tumour categories.

---

## Features

- 🧠 **MRI Classification** — Upload an MRI scan and get instant classification of tumour type using deep learning
- 🔬 **Diagnosis Output** — Identifies glioma, meningioma, pituitary tumour, or no tumour with confidence scores
- 💰 **Treatment Cost Lookup** — Surfaces estimated treatment costs and identifies affordable care options globally
- 👨‍⚕️ **Doctor Recommendations** — Suggests relevant specialists and hospitals based on the detected condition

---

## How It Works
Upload MRI scan → Preprocessing → ResNet50 / MobileNetV2 classification → Diagnosis output → Cost & doctor recommendations

The uploaded MRI image is preprocessed and passed through a fine-tuned transfer learning model. The predicted tumour class is then used to query a recommendation engine that returns treatment cost estimates and specialist contacts. In production, this recommendation step can be powered by the **Serper API** or **Google Search API** for live results.

---

## Model Details

- **Dataset:** Kaggle Brain Tumour MRI Dataset
- **Classes:** Glioma · Meningioma · Pituitary Tumour · No Tumour
- **Architecture:** Transfer learning with ResNet50 and MobileNetV2
- **Framework:** TensorFlow / Keras
- **Environment:** Google Colab (GPU-accelerated training)

---

## Tech Stack

- Python
- TensorFlow / Keras
- ResNet50 · MobileNetV2
- Google Colab
- NumPy · Matplotlib
- Serper API *(planned)*

---

[Still working on this one, part of a broader plan of a model that is strictly trained on medical data of all sorts]
