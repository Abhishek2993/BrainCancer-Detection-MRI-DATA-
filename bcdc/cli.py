"""Command-line interface.

    bcdc split      --data DIR --out outputs/split.csv
    bcdc train      --config configs/mobilenet_v2.yaml
    bcdc calibrate  --artefacts outputs --data DIR --out deploy [--tflite model.tflite]
    bcdc predict    IMAGE [IMAGE ...] --bundle deploy [--json]
    bcdc serve      --bundle deploy [--host 0.0.0.0 --port 8000]
    bcdc demo       --bundle deploy [--share]

``--data`` may be omitted to download the Kaggle dataset with kagglehub.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_CONFIG = {
    "data": {"dataset_path": None, "split_csv": "outputs/split.csv", "dup_hamming": 4,
             "val_frac": 0.15, "test_frac": 0.15, "split_seed": 42},
    "model": {"backbone": "mobilenet_v2", "img_size": [224, 224], "weights": "imagenet"},
    "training": {"seeds": [0, 1, 2], "batch_size": 32, "head_epochs": 15, "head_lr": 1e-3,
                 "finetune_epochs": 20, "finetune_lr": 1e-5, "top_fraction": 0.35, "patience": 4},
    "output_dir": "outputs",
    "tracking": {"backend": "local", "project": "bcdc", "run_name": None},
}


def _merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path=None, overrides=None):
    cfg = DEFAULT_CONFIG
    if path:
        import yaml
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    return _merge(cfg, overrides or {})


def resolve_dataset(path=None):
    from .data import find_class_root
    if path:
        return find_class_root(path)
    import kagglehub
    root = Path(kagglehub.dataset_download("orvile/brain-cancer-mri-dataset"))
    expected = root / "Brain_Cancer raw MRI data" / "Brain_Cancer"
    return expected if expected.is_dir() else find_class_root(root)


# ---------------------------------------------------------------- commands
def cmd_split(args):
    from .split import make_split
    data = resolve_dataset(args.data)
    _, stats = make_split(data, args.out, args.dup_hamming, args.val_frac, args.test_frac, args.seed)
    print(json.dumps(stats, indent=2, default=str))


def cmd_train(args):
    import numpy as np
    import pandas as pd

    from .backbones import build_classifier
    from .data import labeled_path_dataset, load_split
    from .metrics import apply_temperature, evaluate, fit_temperature
    from .split import make_split
    from .tracking import Tracker
    from .train import train_two_stage

    cfg = load_config(args.config)
    if args.quick:
        cfg = _merge(cfg, {"training": {"seeds": cfg["training"]["seeds"][:1], "head_epochs": 1, "finetune_epochs": 1}})
    d, m, t = cfg["data"], cfg["model"], cfg["training"]
    out = Path(cfg["output_dir"])
    (out / "models").mkdir(parents=True, exist_ok=True)
    data = resolve_dataset(args.data or d["dataset_path"])
    split_csv = Path(d["split_csv"])
    if not split_csv.exists():
        print("No split found; creating", split_csv)
        make_split(data, split_csv, d["dup_hamming"], d["val_frac"], d["test_frac"], d["split_seed"])
    split = load_split(split_csv, data)
    if not (out / "split.csv").exists():              # keep the split next to the models (used by calibrate)
        split[["path", "cls", "label", "group", "split"]].to_csv(out / "split.csv", index=False)
    img_size = tuple(m["img_size"])
    P = {s: split[split["split"] == s] for s in ("train", "val", "test")}
    val_ds = labeled_path_dataset(P["val"]["abs_path"], P["val"]["label"], img_size, t["batch_size"])
    test_ds = labeled_path_dataset(P["test"]["abs_path"], P["test"]["label"], img_size, t["batch_size"])
    classes = sorted(split["cls"].unique(), key=lambda c: split.loc[split["cls"] == c, "label"].iloc[0])

    tracker = Tracker(cfg["tracking"]["backend"], cfg["tracking"]["project"],
                      cfg["tracking"]["run_name"] or f"{m['backbone']}-{time.strftime('%Y%m%d-%H%M%S')}")
    tracker.log_params(cfg)
    rows, seeds_meta = [], {}
    for seed in t["seeds"]:
        t0 = time.time()
        model, base = build_classifier(m["backbone"], len(classes), img_size, m["weights"], seed)
        train_ds = labeled_path_dataset(P["train"]["abs_path"], P["train"]["label"], img_size, t["batch_size"],
                                        shuffle=True, seed=seed)
        hist = train_two_stage(model, base, train_ds, val_ds, t["head_epochs"], t["head_lr"], t["finetune_epochs"],
                               t["finetune_lr"], t["top_fraction"], t["patience"], verbose=2 if args.verbose else 0)
        for i, (l, vl) in enumerate(zip(hist["loss"], hist["val_loss"])):
            tracker.log_metrics({f"seed{seed}.loss": l, f"seed{seed}.val_loss": vl}, step=i)
        p_val, p_test = model.predict(val_ds, verbose=0), model.predict(test_ds, verbose=0)
        T = fit_temperature(P["val"]["label"].to_numpy(), p_val)
        metrics, _ = evaluate(P["test"]["label"].to_numpy(), apply_temperature(p_test, T), classes)
        model.save(out / "models" / f"seed_{seed}.keras")
        seeds_meta[str(seed)] = {"temperature": T, "val_loss": float(min(hist["val_loss"]))}
        rows.append({"seed": seed, **metrics, "temperature": T, "minutes": (time.time() - t0) / 60})
        tracker.log_metrics({f"test.{k}": v for k, v in metrics.items()}, step=seed)
        print(f"seed {seed}: test accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}, "
              f"ECE {metrics['ece']:.4f} [{rows[-1]['minutes']:.1f} min]")

    per_seed = pd.DataFrame(rows).set_index("seed")
    per_seed.to_csv(out / "test_metrics_per_seed.csv")
    summary = pd.DataFrame({"mean": per_seed.mean(), "std": per_seed.std(ddof=1) if len(rows) > 1 else 0.0})
    summary.to_csv(out / "test_metrics_summary.csv")
    best = min(seeds_meta, key=lambda s: seeds_meta[s]["val_loss"])
    (out / "model_meta.json").write_text(json.dumps({
        "class_names": classes, "img_size": list(img_size), "backbone": m["backbone"],
        "input": "RGB pixels in [0, 255]; preprocessing is inside the model",
        "temperature": seeds_meta[best]["temperature"], "seed": int(best), "seeds": seeds_meta,
        "split_seed": d["split_seed"], "config": cfg}, indent=2, default=str))
    tracker.log_metrics({f"mean.{k}": float(v) for k, v in summary["mean"].items()})
    tracker.log_artifact(out / "test_metrics_summary.csv")
    tracker.close()
    print(f"Saved {len(rows)} model(s) and metrics to {out}")


def cmd_calibrate(args):
    from .deploy import build_bundle
    cfg = build_bundle(args.artefacts, resolve_dataset(args.data), args.out, tflite_file=args.tflite,
                       alpha=args.alpha, ood_retain=args.ood_retain)
    print(json.dumps({k: cfg[k] for k in ("backend", "models", "temperature", "evaluation")}, indent=2))
    print("Bundle written to", args.out)


def cmd_predict(args):
    from .inference import Predictor
    from .report import build_report
    predictor = Predictor(args.bundle)
    results = []
    for path in args.images:
        pred = predictor.predict_bytes(Path(path).read_bytes(), Path(path).name)
        rep = build_report(pred, predictor.display_names)
        results.append({"file": str(path), "prediction": pred.to_dict(), "report": rep})
        if not args.json:
            print(f"\n=== {path} ===\n{rep['markdown']}")
    if args.json:
        print(json.dumps(results, indent=2))


def cmd_serve(args):
    import uvicorn

    from .api import create_app
    from .inference import Predictor
    uvicorn.run(create_app(Predictor(args.bundle)), host=args.host, port=args.port)


def cmd_demo(args):
    from .demo import build_demo
    from .inference import Predictor
    build_demo(Predictor(args.bundle)).launch(share=args.share, server_name=args.host, server_port=args.port)


def main(argv=None):
    p = argparse.ArgumentParser(prog="bcdc", description="BCDC brain-tumour MRI classifier (research prototype)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("split", help="audit the dataset and write a leakage-aware split.csv")
    s.add_argument("--data", help="dataset folder (class sub-folders); omit to download from Kaggle")
    s.add_argument("--out", default="outputs/split.csv")
    s.add_argument("--dup-hamming", type=int, default=4)
    s.add_argument("--val-frac", type=float, default=0.15)
    s.add_argument("--test-frac", type=float, default=0.15)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_split)

    s = sub.add_parser("train", help="train one backbone over several seeds from a YAML config")
    s.add_argument("--config", help="YAML config (see configs/)")
    s.add_argument("--data", help="dataset folder (overrides the config)")
    s.add_argument("--quick", action="store_true", help="1 seed, 1+1 epochs (smoke test)")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("calibrate", help="build a deployment bundle (temperature, conformal, OOD)")
    s.add_argument("--artefacts", default="outputs", help="folder with split.csv, model_meta.json, models/")
    s.add_argument("--data", help="dataset folder; omit to download from Kaggle")
    s.add_argument("--out", default="deploy")
    s.add_argument("--tflite", help="use this .tflite model instead of the Keras ensemble (no OOD gate)")
    s.add_argument("--alpha", type=float, default=0.10)
    s.add_argument("--ood-retain", type=float, default=0.95)
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("predict", help="classify image or DICOM files")
    s.add_argument("images", nargs="+")
    s.add_argument("--bundle", default="deploy")
    s.add_argument("--json", action="store_true", help="print JSON instead of the report")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("serve", help="run the FastAPI backend")
    s.add_argument("--bundle", default="deploy")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("demo", help="run the Gradio demo")
    s.add_argument("--bundle", default="deploy")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("--share", action="store_true")
    s.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
