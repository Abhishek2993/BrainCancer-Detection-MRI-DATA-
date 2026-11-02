"""DICOM input.

Converts a single DICOM image to an 8-bit grayscale PIL image the way a viewer would:
modality LUT (rescale slope/intercept), then the VOI window stored in the file (or a
0.5-99.5 percentile window if none), inverting MONOCHROME1. Patient-identifying fields are
never read into the returned metadata; only technical fields (modality, size) are kept.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

SAFE_FIELDS = ("Modality", "Rows", "Columns", "PhotometricInterpretation", "BodyPartExamined", "MRAcquisitionType")


def is_dicom(data: bytes, filename: str | None = None) -> bool:
    if len(data) >= 132 and data[128:132] == b"DICM":
        return True
    return bool(filename) and filename.lower().endswith((".dcm", ".dicom"))


def _luts():
    try:                                         # pydicom >= 3
        from pydicom.pixels import apply_modality_lut, apply_voi_lut
    except ImportError:                          # pydicom 2.x
        from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
    return apply_modality_lut, apply_voi_lut


def _window_to_uint8(arr, lo_pct=0.5, hi_pct=99.5):
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    return (np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255).round().astype(np.uint8)


def dicom_to_image(data: bytes) -> tuple[Image.Image, dict]:
    """Returns (PIL image in mode 'L' or 'RGB', info) where info has safe technical fields and warnings."""
    try:
        import pydicom
    except ImportError as e:
        raise RuntimeError("DICOM support needs pydicom: pip install pydicom") from e
    apply_modality_lut, apply_voi_lut = _luts()

    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    info = {k: (str(ds.get(k)) if ds.get(k) is not None else None) for k in SAFE_FIELDS}
    warnings = []
    if info["Modality"] and info["Modality"] != "MR":
        warnings.append(f"DICOM modality is {info['Modality']}, not MR; the model was trained on MRI only.")
    try:
        arr = ds.pixel_array
    except Exception as e:
        raise ValueError("Could not decode the DICOM pixel data (compressed DICOM may need extra packages "
                         "such as pylibjpeg or gdcm).") from e

    if int(ds.get("SamplesPerPixel", 1)) == 3:
        return Image.fromarray(_window_to_uint8(arr, 0, 100)).convert("RGB"), {**info, "warnings": warnings}
    if arr.ndim == 3:                            # multi-frame: take the middle slice
        warnings.append(f"Multi-frame DICOM ({arr.shape[0]} frames): the middle frame was used.")
        arr = arr[arr.shape[0] // 2]
    arr = apply_modality_lut(arr, ds)
    if "WindowCenter" in ds and "WindowWidth" in ds:
        arr = apply_voi_lut(arr, ds)
        img = _window_to_uint8(arr, 0, 100)
    else:
        img = _window_to_uint8(arr)
    if str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        img = 255 - img
    return Image.fromarray(img, mode="L"), {**info, "warnings": warnings}
