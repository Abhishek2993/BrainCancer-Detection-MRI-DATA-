"""Lightweight backbones compared in Phase 2.

Every classifier takes RGB pixels in [0, 255]. Each backbone gets the input scaling it was
pretrained with:

* MobileNetV2: an explicit Rescaling layer to [-1, 1] (same model as Phase 0);
* MobileNetV3, EfficientNet, ConvNeXt: Keras builds their preprocessing into the backbone.

Note: EfficientNet-Lite is not in ``keras.applications``. EfficientNetB0 is the closest
available model (it uses Swish and squeeze-excitation, which Lite removes for
quantisation-friendliness). This difference matters for INT8 results.
"""
from __future__ import annotations

BACKBONES = {
    #  name                  keras.applications class   input scaling           approx. params (no head)
    "mobilenet_v2":        {"cls": "MobileNetV2",       "scaling": "minus1_1", "params_m": 2.3},
    "mobilenet_v3_small":  {"cls": "MobileNetV3Small",  "scaling": "builtin",  "params_m": 0.9},
    "mobilenet_v3_large":  {"cls": "MobileNetV3Large",  "scaling": "builtin",  "params_m": 3.0},
    "efficientnet_b0":     {"cls": "EfficientNetB0",    "scaling": "builtin",  "params_m": 4.0},
    "convnext_tiny":       {"cls": "ConvNeXtTiny",      "scaling": "builtin",  "params_m": 27.8},
}


def build_backbone(name, img_size=(224, 224), weights="imagenet"):
    from tensorflow import keras

    spec = BACKBONES[name]
    ctor = getattr(keras.applications, spec["cls"])
    kwargs = dict(include_top=False, weights=weights, input_shape=tuple(img_size) + (3,))
    if spec["scaling"] == "builtin" and name != "efficientnet_b0":
        kwargs["include_preprocessing"] = True
    return ctor(**kwargs)


def build_classifier(name, num_classes, img_size=(224, 224), weights="imagenet", seed=0, dropout=0.3):
    """Input (RGB 0-255) -> augmentation -> [scaling] -> backbone -> GAP -> dropout -> softmax.

    Returns (model, backbone). The backbone starts frozen.
    """
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
    base = build_backbone(name, img_size, weights)
    base.trainable = False
    inputs = keras.Input(shape=tuple(img_size) + (3,), name="image_rgb_0_255")
    x = augment(inputs)
    if BACKBONES[name]["scaling"] == "minus1_1":
        x = layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_v2_preprocess")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout, seed=seed)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name=f"bcdc_{name}"), base


def set_finetune_trainable(base, top_fraction=0.35):
    """Unfreezes the top ``top_fraction`` of the backbone's layers. Normalisation layers
    (BatchNorm, LayerNorm) stay frozen. For MobileNetV2 (154 layers), 0.35 unfreezes
    layers >= 100, as in Phase 0."""
    from tensorflow.keras import layers

    start = int(round(len(base.layers) * (1 - top_fraction)))
    base.trainable = True
    for i, layer in enumerate(base.layers):
        norm = isinstance(layer, (layers.BatchNormalization, layers.LayerNormalization))
        layer.trainable = i >= start and not norm
    return start
