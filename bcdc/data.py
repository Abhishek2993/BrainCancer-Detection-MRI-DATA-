"""Dataset discovery, image loading and duplicate hashing."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Folder names in the Kaggle dataset -> readable names. The third class is labelled only
# "brain_tumor" in the source and its tumour type is not documented, so it is NOT pituitary.
DISPLAY_NAMES = {
    "brain_glioma": "Glioma",
    "brain_menin": "Meningioma",
    "brain_tumor": "Brain tumour (unspecified type)",
}


def display_name(cls: str) -> str:
    return DISPLAY_NAMES.get(cls, cls)


def find_class_root(root) -> Path:
    """First directory under ``root`` whose sub-folders all contain images."""
    for dirpath, dirnames, _ in os.walk(root):
        subs = sorted(d for d in dirnames if not d.startswith("."))
        if len(subs) >= 2 and all(
            any(f.lower().endswith(IMG_EXTS) for f in os.listdir(os.path.join(dirpath, d)))
            for d in subs
        ):
            return Path(dirpath)
    raise FileNotFoundError(f"No folder of class sub-folders found under {root}")


def load_split(split_csv, dataset_path) -> pd.DataFrame:
    """Reads the Phase 0 ``split.csv`` and adds absolute image paths."""
    df = pd.read_csv(split_csv)
    df["abs_path"] = [str(Path(dataset_path) / p) for p in df["path"]]
    missing = [p for p in df["abs_path"].head(50) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Images listed in {split_csv} not found, e.g. {missing[0]}")
    return df


# ---------------------------------------------------------------- tf.data pipelines
def _tf():
    import tensorflow as tf  # imported lazily so hashing/metrics work without TensorFlow
    return tf


def _resize_round(img, img_size):
    """Same resize as Phase 0 training: bilinear, then rounded/clipped to the uint8 range."""
    tf = _tf()
    img = tf.image.resize(img, img_size)
    return tf.clip_by_value(tf.round(img), 0.0, 255.0)


def path_dataset(paths, img_size, batch_size=32):
    """Batches of float32 RGB images in [0, 255] from image files (any colour mode)."""
    tf = _tf()

    def load(p):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        return _resize_round(tf.cast(img, tf.float32), img_size)

    ds = tf.data.Dataset.from_tensor_slices([str(p) for p in paths])
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def labeled_path_dataset(paths, labels, img_size, batch_size=32, shuffle=False, seed=0, cache=True):
    """(image, label) batches for training/evaluation. Images are float32 RGB in [0, 255];
    decoded images are cached as uint8 to keep memory small (same pipeline as Phase 0)."""
    tf = _tf()

    def load(p, y):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        return tf.saturate_cast(_resize_round(tf.cast(img, tf.float32), img_size), tf.uint8), y

    ds = tf.data.Dataset.from_tensor_slices(([str(p) for p in paths], np.asarray(labels, dtype="int32")))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_validation(split: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Splits the Phase 0 validation set, by duplicate group, into halves "valA" and "valB"
    (identical to the Phase 1 notebook). val-A tunes temperatures/thresholds; val-B calibrates
    conformal sets."""
    split = split.copy()
    groups = np.array(sorted(split.loc[split["split"] == "val", "group"].unique()))
    np.random.default_rng(seed).shuffle(groups)
    half_a = set(groups[: len(groups) // 2])
    is_val = split["split"] == "val"
    split.loc[is_val & split["group"].isin(half_a), "split"] = "valA"
    split.loc[is_val & ~split["group"].isin(half_a), "split"] = "valB"
    return split


def array_dataset(images, img_size, batch_size=32):
    """Batches of float32 RGB images in [0, 255] from a uint8 array (N,H,W) or (N,H,W,3)."""
    tf = _tf()
    images = np.asarray(images)
    if images.ndim == 3:
        images = np.repeat(images[..., None], 3, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.map(lambda x: _resize_round(tf.cast(x, tf.float32), img_size),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------- duplicate hashing
def dhash_bits(img_l: Image.Image, size: int = 8) -> int:
    """64-bit difference hash of a grayscale PIL image."""
    small = np.asarray(img_l.resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


_ROTATIONS = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270]


def dihedral_hashes(img_l: Image.Image) -> list[int]:
    """dHashes under all 8 flips/90-degree rotations (index 0 = original)."""
    small = img_l.resize((64, 64), Image.BILINEAR)
    out = []
    for base in (small, small.transpose(Image.Transpose.FLIP_LEFT_RIGHT)):
        for rot in _ROTATIONS:
            out.append(dhash_bits(base if rot is None else base.transpose(rot)))
    return out


def query_hash(img_l: Image.Image) -> int:
    """dHash computed the same way as index 0 of :func:`dihedral_hashes`."""
    return dhash_bits(img_l.resize((64, 64), Image.BILINEAR))


_BYTE_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return _BYTE_POPCOUNT[x.view(np.uint8)].reshape(x.shape + (8,)).sum(axis=-1)


def cross_near_duplicates(query_hashes, ref_hashes, max_bits=4, chunk=64) -> np.ndarray:
    """For each query image, True if it near-duplicates any reference image.

    ``query_hashes``: (n,) original dHashes. ``ref_hashes``: (m, 8) dihedral dHashes.
    """
    q = np.asarray(query_hashes, dtype=np.uint64)
    r = np.asarray(ref_hashes, dtype=np.uint64)
    hit = np.zeros(len(q), dtype=bool)
    for s in range(0, len(q), chunk):
        d = popcount64(np.bitwise_xor(q[s:s + chunk, None, None], r[None])).min(axis=(1, 2))
        hit[s:s + chunk] = d <= max_bits
    return hit


# ---------------------------------------------------------------- thumbnail correlation
def _znorm(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32).ravel()
    a = a - a.mean()
    return a / max(float(np.linalg.norm(a)), 1e-6)


def thumbnail_variants(img_l: Image.Image, size: int = 32) -> np.ndarray:
    """Zero-mean, unit-norm thumbnails under all 8 flips/rotations: shape (8, size*size).
    The dot product of two thumbnails is their Pearson correlation."""
    small = img_l.resize((size, size), Image.BILINEAR)
    out = []
    for base in (small, small.transpose(Image.Transpose.FLIP_LEFT_RIGHT)):
        for rot in _ROTATIONS:
            out.append(_znorm(np.asarray(base if rot is None else base.transpose(rot))))
    return np.stack(out)


def thumbnail(img_l: Image.Image, size: int = 32) -> np.ndarray:
    return _znorm(np.asarray(img_l.resize((size, size), Image.BILINEAR)))


def best_thumbnail_match(query_thumbs, ref_variants, chunk=256):
    """For each query thumbnail: (highest correlation with any reference variant, index of that reference)."""
    q = np.asarray(query_thumbs, dtype=np.float32)
    r = np.asarray(ref_variants, dtype=np.float32)
    m, v, d = r.shape
    flat = r.reshape(m * v, d)
    best = np.empty(len(q), dtype=np.float32)
    which = np.empty(len(q), dtype=np.int64)
    for s in range(0, len(q), chunk):
        sims = q[s:s + chunk] @ flat.T
        j = sims.argmax(axis=1)
        best[s:s + chunk] = sims[np.arange(len(j)), j]
        which[s:s + chunk] = j // v
    return best, which
