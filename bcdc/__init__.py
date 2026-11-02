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
"""

__version__ = "0.2.0"
