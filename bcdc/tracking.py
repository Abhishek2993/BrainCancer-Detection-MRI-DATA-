"""Experiment tracking with a small common interface.

Backends:
* ``none``   - do nothing;
* ``local``  - JSON files under ``runs/<run name>/`` (params.json, metrics.jsonl), no extra dependency;
* ``mlflow`` - MLflow (``pip install mlflow``); uses MLFLOW_TRACKING_URI, or a local SQLite
  store ``sqlite:///mlflow.db`` (browse with ``mlflow ui --backend-store-uri sqlite:///mlflow.db``);
* ``wandb``  - Weights & Biases (``pip install wandb``; needs ``wandb login``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class Tracker:
    def __init__(self, backend="local", project="bcdc", run_name=None, root="runs"):
        self.backend = backend
        self.run_name = run_name or time.strftime("run-%Y%m%d-%H%M%S")
        if backend == "local":
            self.dir = Path(root) / self.run_name
            self.dir.mkdir(parents=True, exist_ok=True)
        elif backend == "mlflow":
            import os

            import mlflow
            if not os.environ.get("MLFLOW_TRACKING_URI"):
                mlflow.set_tracking_uri("sqlite:///mlflow.db")   # recent MLflow deprecates the ./mlruns file store
            mlflow.set_experiment(project)
            self._mlflow = mlflow
            self._run = mlflow.start_run(run_name=self.run_name)
        elif backend == "wandb":
            import wandb
            self._wandb = wandb
            self._run = wandb.init(project=project, name=self.run_name)
        elif backend != "none":
            raise ValueError(f"unknown tracking backend {backend!r}")

    @staticmethod
    def _flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(Tracker._flat(v, key + "."))
            else:
                out[key] = v
        return out

    def log_params(self, params: dict):
        flat = self._flat(params)
        if self.backend == "local":
            path = self.dir / "params.json"
            old = json.loads(path.read_text()) if path.exists() else {}
            path.write_text(json.dumps({**old, **{k: v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
                                                  for k, v in flat.items()}}, indent=2))
        elif self.backend == "mlflow":
            self._mlflow.log_params({k: str(v)[:500] for k, v in flat.items()})
        elif self.backend == "wandb":
            self._wandb.config.update(flat, allow_val_change=True)

    def log_metrics(self, metrics: dict, step=None):
        clean = {k: float(v) for k, v in self._flat(metrics).items() if isinstance(v, (int, float))}
        if self.backend == "local":
            with open(self.dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": step, **clean}) + "\n")
        elif self.backend == "mlflow":
            self._mlflow.log_metrics({k.replace("[", "_").replace("]", ""): v for k, v in clean.items()}, step=step)
        elif self.backend == "wandb":
            self._wandb.log(clean, step=step)

    def log_artifact(self, path):
        if self.backend == "mlflow":
            self._mlflow.log_artifact(str(path))
        elif self.backend == "wandb":
            self._wandb.save(str(path), policy="now")

    def close(self):
        if self.backend == "mlflow":
            self._mlflow.end_run()
        elif self.backend == "wandb":
            self._run.finish()
