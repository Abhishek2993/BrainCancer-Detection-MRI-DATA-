"""The BCDC MobileNetV2 classifier and helpers to reach its internals.

The Phase 0 model is: input (RGB, 0-255) -> augmentation -> Rescaling to [-1, 1]
-> MobileNetV2 backbone -> GlobalAveragePooling2D -> Dropout -> Dense(softmax).
Augmentation and dropout do nothing at inference, so features and logits can be
computed by calling the preprocessing, backbone, pooling and dense layers directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def build_model(num_classes, img_size=(224, 224), weights="imagenet", seed=0):
    """The Phase 0 MobileNetV2 classifier. Returns (model, backbone)."""
    from .backbones import build_classifier

    return build_classifier("mobilenet_v2", num_classes, img_size, weights, seed)


def _identity(x):
    return x


@dataclass
class ModelParts:
    preprocess: object   # Rescaling layer, or identity when the backbone scales its own input
    backbone: object     # MobileNetV2 (include_top=False)
    pool: object         # GlobalAveragePooling2D
    W: np.ndarray        # (features, classes) dense kernel
    b: np.ndarray        # (classes,) dense bias
    img_size: tuple


def model_parts(model) -> ModelParts:
    from tensorflow import keras
    from tensorflow.keras import layers

    pre = next((l for l in model.layers if isinstance(l, layers.Rescaling)), None)
    pre = _identity if pre is None else pre
    base = next(l for l in model.layers if isinstance(l, keras.Model) and l.name != "augmentation"
                and not isinstance(l, keras.Sequential))
    pool = next(l for l in model.layers if isinstance(l, layers.GlobalAveragePooling2D))
    dense = model.layers[-1]
    W, b = dense.get_weights()
    return ModelParts(pre, base, pool, W, b, tuple(model.input_shape[1:3]))


def feature_extractor(parts: ModelParts):
    """Keras model: image batch (0-255) -> (conv feature map, pooled features)."""
    from tensorflow import keras

    inp = keras.Input(shape=parts.img_size + (3,))
    conv = parts.backbone(parts.preprocess(inp), training=False)
    feat = parts.pool(conv)
    return keras.Model(inp, [conv, feat])


def feature_model(parts: ModelParts):
    """Keras model: image batch (0-255) -> pooled penultimate features (N, F)."""
    from tensorflow import keras

    inp = keras.Input(shape=parts.img_size + (3,))
    return keras.Model(inp, parts.pool(parts.backbone(parts.preprocess(inp), training=False)))


def features_and_logits(parts: ModelParts, dataset, fmodel=None):
    """Pooled features (N, F) and logits (N, K) for every image in a tf.data dataset."""
    fmodel = fmodel or feature_model(parts)
    feats = fmodel.predict(dataset, verbose=0)
    return feats, feats @ parts.W + parts.b


def softmax(logits, axis=-1):
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
