"""Loader for the Figshare brain-tumour dataset (Cheng et al., 2015 / 2016).

3,064 T1-weighted contrast-enhanced MRI slices from 233 patients, labelled
1 = meningioma, 2 = glioma, 3 = pituitary tumour, with a radiologist-drawn tumour mask
for every slice. Each slice is a MATLAB v7.3 (.mat / HDF5) file with a ``cjdata`` struct:
``label``, ``PID`` (patient ID), ``image``, ``tumorBorder`` and ``tumorMask``.

Source: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
Please cite:
  Cheng J. et al. "Enhanced Performance of Brain Tumor Classification via Tumor Region
  Augmentation and Partition." PLoS ONE 10(10), 2015.
  Cheng J. et al. "Retrieval of Brain Tumors by Adaptive Spatial Pooling and Fisher Vector
  Representation." PLoS ONE 11(6), 2016.
"""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

FIGSHARE_URL = "https://figshare.com/ndownloader/articles/1512427/versions/5"
LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}
# Figshare label -> Kaggle (training) class folder. Pituitary has no counterpart.
TO_TRAINING_CLASS = {"glioma": "brain_glioma", "meningioma": "brain_menin"}


def _extract_nested(root: Path):
    """Extract every .zip under ``root`` (zips inside zips too), each into a folder of its own."""
    done = set()
    while True:
        todo = [z for z in root.rglob("*.zip") if z not in done]
        if not todo:
            return
        for z in todo:
            target = z.with_suffix("")
            if not target.exists():
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(target)
            done.add(z)


def download(dest="data/figshare", url=FIGSHARE_URL) -> Path:
    """Downloads (~880 MB) and extracts the dataset unless .mat files are already in ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not any(dest.rglob("*.mat")):
        archive = dest / "figshare_1512427.zip"
        if not archive.exists():
            print(f"Downloading {url} ...")
            tmp = archive.with_suffix(".part")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (bcdc research notebook)"})
            try:
                with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f, length=1 << 20)
            except Exception as e:
                raise RuntimeError(
                    f"Could not download the Figshare dataset ({e}). Download it manually from "
                    "https://figshare.com/articles/dataset/brain_tumor_dataset/1512427, unzip it, and set "
                    "the BCDC_FIGSHARE_DIR environment variable to that folder.") from e
            tmp.rename(archive)
        _extract_nested(dest)
    n = sum(1 for _ in _mat_files(dest))
    print(f"Figshare dataset ready: {n} slice files under {dest}")
    return dest


def _mat_files(root: Path):
    for p in sorted(Path(root).rglob("*.mat")):
        if p.name.lower() != "cvind.mat":      # cross-validation indices, not an image
            yield p


def _matlab_string(arr) -> str:
    codes = np.asarray(arr).astype(np.int64).ravel()
    return "".join(chr(c) for c in codes if c > 0).strip()


def read_slice(path) -> dict:
    """Reads one .mat slice. Returns label name, patient ID, image (float) and mask (bool)."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            g = f["cjdata"]
            label = int(np.asarray(g["label"]).ravel()[0])
            pid = _matlab_string(g["PID"])
            image = np.asarray(g["image"], dtype=np.float32).T   # MATLAB is column-major
            mask = np.asarray(g["tumorMask"]).T.astype(bool)
    except OSError:                                              # older, non-HDF5 .mat file
        from scipy.io import loadmat
        c = loadmat(path, squeeze_me=True, struct_as_record=False)["cjdata"]
        label, pid = int(c.label), str(c.PID).strip()
        image, mask = np.asarray(c.image, dtype=np.float32), np.asarray(c.tumorMask).astype(bool)
    return {"label": LABELS[label], "pid": pid, "image": image, "mask": mask}


def to_uint8(image, lo_pct=0.5, hi_pct=99.5) -> np.ndarray:
    """Maps raw MRI intensities to 0-255 with a per-slice percentile window."""
    lo, hi = np.percentile(image, [lo_pct, hi_pct])
    scaled = (np.asarray(image, dtype=np.float32) - lo) / max(hi - lo, 1e-6)
    return (np.clip(scaled, 0, 1) * 255).round().astype(np.uint8)


def load(root, size=256):
    """Loads every slice, resized to ``size`` x ``size``.

    Returns (meta DataFrame [file, label, pid], images uint8 (N,size,size), masks bool (N,size,size)).
    """
    rows, images, masks = [], [], []
    for p in _mat_files(Path(root)):
        s = read_slice(p)
        img = Image.fromarray(to_uint8(s["image"])).resize((size, size), Image.BILINEAR)
        msk = Image.fromarray(s["mask"].astype(np.uint8) * 255).resize((size, size), Image.NEAREST)
        rows.append({"file": p.name, "label": s["label"], "pid": s["pid"]})
        images.append(np.asarray(img))
        masks.append(np.asarray(msk) > 127)
    if not rows:
        raise FileNotFoundError(f"No Figshare .mat slices found under {root}")
    return pd.DataFrame(rows), np.stack(images), np.stack(masks)
