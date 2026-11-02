#!/usr/bin/env python3
"""Benchmark BCDC TFLite models on an edge device (Raspberry Pi, Jetson, laptop).

Needs only NumPy and the LiteRT runtime (no TensorFlow):

    pip install numpy ai-edge-litert
    python run_on_device.py --bundle bcdc_edge_bundle --threads 4

The bundle (made by Phase2_Edge.ipynb) contains:
    models/*.tflite     exported classifiers
    testset.npz         images (uint8, N x H x W x 3) and labels from the held-out test set
    calibration.json    class names, plus per-model temperature and conformal thresholds
                        fitted on validation data in the notebook

For every model it measures single-image latency, model size, accuracy, glioma sensitivity,
calibration error and conformal coverage, and writes device_results.json.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
except ImportError:                                   # older devices: tflite-runtime or full TF
    try:
        from tflite_runtime.interpreter import Interpreter, OpResolverType
    except ImportError:
        import tensorflow as tf
        Interpreter, OpResolverType = tf.lite.Interpreter, tf.lite.experimental.OpResolverType


def device_name():
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            return Path(p).read_text().strip("\x00\n ")
        except OSError:
            pass
    return f"{platform.system()} {platform.machine()}"


class Model:
    def __init__(self, path, threads):
        try:                                           # default XNNPACK CPU delegate
            self.it = Interpreter(model_path=str(path), num_threads=threads)
            self.it.allocate_tensors()
            self.delegate = "xnnpack"
        except RuntimeError:                           # some INT8 graphs fail in XNNPACK: plain kernels
            self.it = Interpreter(model_path=str(path), num_threads=threads,
                                  experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
            self.it.allocate_tensors()
            self.delegate = "builtin (XNNPACK failed)"
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]

    def __call__(self, img):                           # img: uint8 (H, W, 3)
        x = img[None].astype(np.float32)
        if self.inp["dtype"] != np.float32:
            scale, zero = self.inp["quantization"]
            info = np.iinfo(self.inp["dtype"])
            x = np.clip(np.round(x / scale + zero) if scale else np.round(x), info.min, info.max)
            x = x.astype(self.inp["dtype"])
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"]).astype(np.float32)[0]
        if self.out["dtype"] != np.float32:
            scale, zero = self.out["quantization"]
            y = (y - zero) * scale
        return y


def temperature(p, T):
    z = np.log(np.clip(p, 1e-12, 1)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(y, p, bins=15):
    """Expected calibration error with equal-width confidence bins."""
    conf, correct = p.max(1), p.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=".", help="folder with models/, testset.npz, calibration.json")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--runs", type=int, default=50, help="timed single-image runs per model")
    ap.add_argument("--max-images", type=int, default=0, help="limit test images (0 = all)")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    data = np.load(bundle / "testset.npz")
    images, labels = data["images"], data["labels"]
    if args.max_images:
        images, labels = images[:args.max_images], labels[:args.max_images]
    calib = json.loads((bundle / "calibration.json").read_text())
    classes = calib["class_names"]
    g = classes.index("brain_glioma") if "brain_glioma" in classes else 0
    print(f"Device: {device_name()} | threads: {args.threads} | test images: {len(images)}\n")

    results = []
    for path in sorted((bundle / "models").glob("*.tflite")):
        name = path.stem
        m = Model(path, args.threads)
        for _ in range(5):
            m(images[0])
        times = []
        for i in range(args.runs):
            t0 = time.perf_counter()
            m(images[i % len(images)])
            times.append((time.perf_counter() - t0) * 1000)
        probs = np.stack([m(img) for img in images])
        c = calib["models"].get(name, {})
        if "temperature" in c:
            probs = temperature(probs, c["temperature"])
        pred = probs.argmax(1)
        r = {"model": name, "delegate": m.delegate, "size_mb": path.stat().st_size / 1e6,
             "latency_median_ms": float(np.median(times)), "latency_p90_ms": float(np.percentile(times, 90)),
             "accuracy": float((pred == labels).mean()),
             "glioma_sensitivity": float((pred[labels == g] == g).mean()) if (labels == g).any() else float("nan"),
             "ece": ece(labels, probs)}
        if "qhat" in c:
            sets = (1 - probs) <= np.asarray(c["qhat"])[None, :]       # class-conditional LAC
            r["conformal_coverage"] = float(sets[np.arange(len(labels)), labels].mean())
            r["avg_set_size"] = float(sets.sum(1).mean())
        results.append(r)
        print(f"{name:<34s} {r['size_mb']:6.2f} MB  {r['latency_median_ms']:8.2f} ms  "
              f"acc {100 * r['accuracy']:5.1f}%  glioma sens {100 * r['glioma_sensitivity']:5.1f}%  "
              f"ECE {r['ece']:.3f}" + (f"  coverage {r['conformal_coverage']:.3f}" if "conformal_coverage" in r else "")
              + ("" if m.delegate == "xnnpack" else f"  [{m.delegate}]"))

    out = {"device": device_name(), "machine": platform.machine(), "threads": args.threads,
           "n_images": int(len(images)), "results": results}
    (bundle / "device_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {bundle / 'device_results.json'}")


if __name__ == "__main__":
    main()
