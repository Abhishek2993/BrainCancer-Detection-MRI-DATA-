"""Edge deployment: TensorFlow Lite export, quantisation, inference and latency.

Quantisation modes:

* ``fp32``    - plain TFLite conversion (reference);
* ``fp16``    - weights stored as float16 (about half the size; GPU delegates run it natively);
* ``dynamic`` - weights int8, activations float (dynamic-range quantisation);
* ``int8``    - full integer: weights and activations int8, calibrated on representative images.
               Input is uint8 RGB (0-255, exactly the camera/PNG range); output is float32 probabilities.
"""
from __future__ import annotations

import time

import numpy as np

from .model import model_parts

QUANT_MODES = ("fp32", "fp16", "dynamic", "int8")


def inference_model(model):
    """Strips augmentation and dropout: input (RGB 0-255) -> scaling -> backbone -> GAP -> softmax."""
    from tensorflow import keras
    from tensorflow.keras import layers

    parts = model_parts(model)
    inp = keras.Input(shape=parts.img_size + (3,), name="image_rgb_0_255")
    x = parts.backbone(parts.preprocess(inp), training=False)
    x = parts.pool(x)
    dense = layers.Dense(parts.W.shape[1], activation="softmax", name="probs")
    out = dense(x)
    dense.set_weights([parts.W, parts.b])
    return keras.Model(inp, out, name=model.name + "_inference")


def to_tflite(model, mode="fp32", representative_images=None, n_calibration=200, seed=0) -> bytes:
    """Converts a Keras classifier to a TFLite flatbuffer. ``representative_images``
    (uint8 or float in [0, 255], shape (N, H, W, 3)) is required for ``int8``."""
    import tensorflow as tf

    if mode not in QUANT_MODES:
        raise ValueError(f"mode must be one of {QUANT_MODES}")
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model(model))
    if mode == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif mode == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif mode == "int8":
        if representative_images is None:
            raise ValueError("int8 needs representative_images")
        imgs = np.asarray(representative_images, dtype=np.float32)
        idx = np.random.default_rng(seed).permutation(len(imgs))[:n_calibration]

        def rep():
            for i in idx:
                yield [imgs[i:i + 1]]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32
    return converter.convert()


def _interpreter_api():
    """(Interpreter class, OpResolverType enum) from LiteRT, falling back to tf.lite."""
    try:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType   # LiteRT (successor of tf.lite)
    except ImportError:
        import tensorflow as tf
        Interpreter, OpResolverType = tf.lite.Interpreter, tf.lite.experimental.OpResolverType
    return Interpreter, OpResolverType


def make_interpreter(model_content=None, model_path=None, num_threads=None):
    """Creates and allocates an interpreter. Uses the default XNNPACK CPU delegate when it can;
    some quantised graphs (e.g. INT8 MobileNetV3 hard-swish) make XNNPACK fail to prepare, and
    then the plain built-in kernels are used instead. Returns (interpreter, delegate name)."""
    Interpreter, OpResolverType = _interpreter_api()
    kw = {"num_threads": num_threads} if num_threads else {}
    src = {"model_content": model_content} if model_content is not None else {"model_path": str(model_path)}
    try:
        it = Interpreter(**src, **kw)
        it.allocate_tensors()
        return it, "xnnpack"
    except RuntimeError:
        it = Interpreter(**src, **kw, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        it.allocate_tensors()
        return it, "builtin (XNNPACK failed)"


class TFLiteClassifier:
    """Runs a TFLite classifier on RGB images in [0, 255]."""

    def __init__(self, model_content=None, model_path=None, num_threads=None):
        self.interp, self.delegate = make_interpreter(model_content, model_path, num_threads)
        self._in = self.interp.get_input_details()[0]
        self._out = self.interp.get_output_details()[0]
        self.input_dtype = self._in["dtype"]
        self.img_size = tuple(int(v) for v in self._in["shape"][1:3])
        self.batch = int(self._in["shape"][0])

    def _prepare(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.input_dtype == np.float32:
            return x
        scale, zero = self._in["quantization"]
        info = np.iinfo(self.input_dtype)
        q = np.round(x / scale + zero) if scale else np.round(x)
        return np.clip(q, info.min, info.max).astype(self.input_dtype)

    def _set_batch(self, n):
        if n != self.batch:
            self.interp.resize_tensor_input(self._in["index"], [n, *self.img_size, 3])
            self.interp.allocate_tensors()
            self._in = self.interp.get_input_details()[0]
            self._out = self.interp.get_output_details()[0]
            self.batch = n

    def _run(self, x):
        self._set_batch(len(x))
        self.interp.set_tensor(self._in["index"], self._prepare(x))
        self.interp.invoke()
        out = self.interp.get_tensor(self._out["index"]).astype(np.float32)
        scale, zero = self._out["quantization"]
        return (out - zero) * scale if self._out["dtype"] != np.float32 and scale else out

    def predict(self, images, batch_size=32):
        """Probabilities (N, K). Uses batches when the model allows resizing, else one image at a time."""
        images = np.asarray(images)
        outs = []
        try:
            for s in range(0, len(images), batch_size):
                outs.append(self._run(images[s:s + batch_size]))
        except (RuntimeError, ValueError):
            outs = [self._run(images[i:i + 1]) for i in range(len(images))]
        return np.concatenate(outs)

    def latency_ms(self, n_runs=50, warmup=5, seed=0):
        """Single-image latency (batch 1): median and 90th percentile in milliseconds."""
        x = np.random.default_rng(seed).uniform(0, 255, (1, *self.img_size, 3))
        self._set_batch(1)
        for _ in range(warmup):
            self._run(x)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self._run(x)
            times.append((time.perf_counter() - t0) * 1000)
        return {"median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90))}


def keras_latency_ms(model, n_runs=30, warmup=3, seed=0):
    """Single-image latency of a Keras model (eager call, no tf.data), in milliseconds."""
    import tensorflow as tf

    size = tuple(model.input_shape[1:3])
    x = tf.constant(np.random.default_rng(seed).uniform(0, 255, (1, *size, 3)), tf.float32)
    fn = tf.function(lambda t: model(t, training=False))
    for _ in range(warmup):
        fn(x).numpy()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(x).numpy()
        times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90))}


def pareto_front(df, maximise=("accuracy",), minimise=("latency_ms", "size_mb")):
    """Boolean mask of rows not dominated by any other row (better or equal on every
    objective and strictly better on at least one)."""
    vals = np.column_stack([df[c].to_numpy(dtype=float) for c in maximise] +
                           [-df[c].to_numpy(dtype=float) for c in minimise])
    keep = np.ones(len(vals), dtype=bool)
    for i in range(len(vals)):
        dominated_by = np.all(vals >= vals[i], axis=1) & np.any(vals > vals[i], axis=1)
        keep[i] = not dominated_by.any()
    return keep
