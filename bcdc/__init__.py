"""BCDC: brain-tumour MRI classification utilities.

Shared code for the project notebooks:

* ``data``      - dataset discovery, image loading, duplicate hashing
* ``model``     - the MobileNetV2 classifier and helpers to get features / logits
* ``metrics``   - classification metrics, calibration (ECE, temperature scaling)
* ``ensemble``  - deep-ensemble prediction and uncertainty decomposition
* ``conformal`` - split-conformal prediction sets (LAC, APS, class-conditional)
* ``ood``       - out-of-distribution scores (MSP, energy, Mahalanobis) and metrics
* ``gradcam``   - Grad-CAM and metrics that check heatmaps against tumour masks
* ``figshare``  - loader for the Figshare brain-tumour dataset (Cheng et al.)
* ``backbones`` - lightweight backbones compared in Phase 2 (MobileNetV2/V3, EfficientNetB0, ConvNeXt-Tiny)
* ``train``     - the shared two-stage transfer-learning recipe
* ``edge``      - TensorFlow Lite export (FP32/FP16/dynamic/INT8), inference, latency, Pareto front
* ``federated`` - FedAvg / FedProx simulation with IID and Dirichlet (non-IID) hospital splits
"""

__version__ = "0.3.0"
