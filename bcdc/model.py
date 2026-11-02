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
    """Same architecture as the Phase 0 notebook. Returns (model, backbone)."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    augment = keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.05, fill_mode="constant", seed=seed),
        layers.RandomZoom(0.1, fill_mode="constant", seed=seed),
        layers.RandomContrast(0.1, seed=seed),
    ], name="augmentation")
    base = keras.applications.MobileNetV2(include_top=False, weights=weights,
                                          input_shape=tuple(img_size) + (3,))
    base.trainable = False
    inputs = keras.Input(shape=tuple(img_size) + (3,), name="image_rgb_0_255")
    x = augment(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_v2_preprocess")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3, seed=seed)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name="bcdc_mobilenetv2"), base


@dataclass
class ModelParts:
    preprocess: object   # Rescaling layer
    backbone: object     # MobileNetV2 (include_top=False)
    pool: object         # GlobalAveragePooling2D
    W: np.ndarray        # (features, classes) dense kernel
    b: np.ndarray        # (classes,) dense bias
    img_size: tuple


def model_parts(model) -> ModelParts:
    from tensorflow import keras
    from tensorflow.keras import layers

    pre = next(l for l in model.layers if isinstance(l, layers.Rescaling))
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
