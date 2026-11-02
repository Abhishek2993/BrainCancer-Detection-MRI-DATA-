"""Grad-CAM heatmaps and metrics that check them against ground-truth tumour masks.

A heatmap that "looks right" proves little. These metrics measure whether the model's
evidence actually falls on the tumour, and compare it with trivial baselines (a centred
blob, a random map) that any useful explanation must beat.

Note: with a GlobalAveragePooling + linear head, Grad-CAM is identical to CAM.
"""
from __future__ import annotations

import numpy as np


def gradcam(extractor, parts, images, class_idx, out_size=None):
    """Grad-CAM maps in [0, 1], shape (B, H, W). ``images``: float batch in [0, 255]."""
    import tensorflow as tf

    conv, _ = extractor(images, training=False)
    W = tf.constant(parts.W, dtype=conv.dtype)
    b = tf.constant(parts.b, dtype=conv.dtype)
    with tf.GradientTape() as tape:
        tape.watch(conv)
        logits = tf.matmul(parts.pool(conv), W) + b
        selected = tf.gather(logits, np.asarray(class_idx), axis=1, batch_dims=1)
    grads = tape.gradient(selected, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))                         # (B, C)
    cam = tf.nn.relu(tf.einsum("bhwc,bc->bhw", conv, weights))
    if out_size is not None:
        cam = tf.image.resize(cam[..., None], out_size, method="bilinear")[..., 0]
    cam = cam.numpy()
    peak = cam.reshape(len(cam), -1).max(axis=1).reshape(-1, 1, 1)
    return cam / np.maximum(peak, 1e-12)


def pointing_game(cam, mask, tolerance_frac=15 / 512) -> bool:
    """Hit if the heatmap's peak lies on the tumour, or within ``tolerance`` pixels of it
    (15 px at 512 px, following Zhang et al., 2018)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return False
    py, px = np.unravel_index(int(np.argmax(cam)), cam.shape)
    if mask[py, px]:
        return True
    ys, xs = np.nonzero(mask)
    tol = tolerance_frac * max(mask.shape)
    return bool(np.min((ys - py) ** 2 + (xs - px) ** 2) <= tol ** 2)


def energy_in_mask(cam, mask) -> float:
    """Share of the heatmap's total mass that falls inside the tumour."""
    cam = np.asarray(cam, dtype=float)
    return float((cam * mask).sum() / max(cam.sum(), 1e-12))


def iou_area_matched(cam, mask) -> float:
    """IoU between the tumour and the heatmap's top-k pixels, where k = tumour area."""
    mask = np.asarray(mask, dtype=bool)
    k = int(mask.sum())
    if k == 0:
        return float("nan")
    flat = np.asarray(cam, dtype=float).ravel()
    top = np.zeros(flat.shape, dtype=bool)
    top[np.argpartition(-flat, k - 1)[:k]] = True
    top = top.reshape(mask.shape)
    return float((top & mask).sum() / (top | mask).sum())


def border_energy(cam, frac=0.1) -> float:
    """Share of heatmap mass in the outer ``frac`` ring of the image (borders, text, skull edge)."""
    cam = np.asarray(cam, dtype=float)
    h, w = cam.shape
    bh, bw = max(1, int(h * frac)), max(1, int(w * frac))
    inner = cam[bh:h - bh, bw:w - bw].sum()
    return float(1 - inner / max(cam.sum(), 1e-12))


def centre_prior(shape, sigma_frac=0.25):
    """Baseline map: a Gaussian blob in the image centre."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    s = sigma_frac * max(h, w)
    return np.exp(-(((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2) / (2 * s ** 2)))


def random_map(shape, rng):
    """Baseline map: uniform noise (its peak is a random pixel)."""
    return rng.uniform(size=shape)


def localisation_metrics(cam, mask) -> dict:
    return {"pointing_game": float(pointing_game(cam, mask)),
            "energy_in_mask": energy_in_mask(cam, mask),
            "iou_area_matched": iou_area_matched(cam, mask),
            "border_energy": border_energy(cam)}
