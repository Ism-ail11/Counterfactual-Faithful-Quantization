"""Portable checkpoint metadata and exact preprocessed data splits."""
from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

import joblib
import torch

from .constraints import ActionSet
from .data.tabular import DatasetBundle
from .utils import save_json


def save_tabular_data(bundle: DatasetBundle, run_dir: str | Path) -> None:
    directory = Path(run_dir)
    payload = {}
    for field in fields(bundle):
        if field.name not in {"metadata", "action_set"}:
            value = getattr(bundle, field.name)
            payload[field.name] = value.cpu() if isinstance(value, torch.Tensor) else value
    payload["action_set"] = asdict(bundle.action_set.to("cpu"))
    payload["metadata"] = {key: value for key, value in (bundle.metadata or {}).items() if key != "preprocessor"}
    torch.save(payload, directory / "dataset.pt")
    preprocessor = (bundle.metadata or {}).get("preprocessor")
    if preprocessor is not None:
        joblib.dump(preprocessor, directory / "preprocessor.joblib")


def load_tabular_data(run_dir: str | Path) -> DatasetBundle:
    payload = torch.load(Path(run_dir) / "dataset.pt", map_location="cpu", weights_only=True)
    payload["action_set"] = ActionSet(**payload["action_set"])
    return DatasetBundle(**payload)


def save_quantization_settings(model, run_dir: str | Path) -> None:
    names = ("temperature", "hard", "stochastic", "fixed_bit", "quantize_activations")
    save_json({name: getattr(model, name) for name in names if hasattr(model, name)},
              Path(run_dir) / "quantization.json")


def load_quantization_settings(model, run_dir: str | Path) -> None:
    from .quantization import configure_quantization

    path = Path(run_dir) / "quantization.json"
    if path.exists():
        settings = json.loads(path.read_text(encoding="utf-8"))
        for name, value in settings.items():
            setattr(model, name, value)
        if "temperature" in settings:
            configure_quantization(model, settings["temperature"], settings["hard"],
                                   settings["stochastic"], settings["fixed_bit"])
