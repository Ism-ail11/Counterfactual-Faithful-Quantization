from __future__ import annotations

import json
from pathlib import Path

import torch

from cfq.config import ExperimentConfig, prepare_config
from cfq.cli import config_from_mapping
from cfq.checkpoints import load_tabular_data, load_quantization_settings
from cfq.data import load_tabular_dataset
from cfq.models import make_quant_tabular_model, make_tabular_model


def load_models(config: ExperimentConfig, run_dir: str | Path, max_samples: int | None = None, cache_dir="data"):
    run_dir = Path(run_dir)
    config = prepare_config(config)
    saved_path = run_dir / "config.json"
    if saved_path.exists():
        saved = config_from_mapping(json.loads(saved_path.read_text(encoding="utf-8")))
        if (saved.dataset, saved.method, saved.model, saved.quant, saved.train.seed) != (
            config.dataset, config.method, config.model, config.quant, config.train.seed
        ):
            raise ValueError("Checkpoint configuration differs from the requested run; use a new output directory or rerun training")
        config = saved
    bundle = (load_tabular_data(run_dir) if (run_dir / "dataset.pt").exists() else
              load_tabular_dataset(config.dataset, cache_dir=cache_dir, seed=config.train.seed, max_samples=max_samples))
    if max_samples is not None and sum(len(split) for split in (bundle.x_train, bundle.x_val, bundle.x_test)) > max_samples:
        raise ValueError("Checkpoint data contains more samples than requested; rerun training")
    fp_model = make_tabular_model(
        config.model.backbone,
        bundle.input_dim,
        config.model.hidden_dims,
        config.model.num_classes,
        dropout=config.model.dropout,
    )
    q_model = make_quant_tabular_model(
        config.model.backbone,
        bundle.input_dim,
        config.model.hidden_dims,
        config.model.num_classes,
        bits=config.quant.bits,
        init_bit=config.quant.uniform_bit,
        quantize_activations=config.quant.quantize_activations,
        dropout=config.model.dropout,
    )
    if config.method == "fp32":
        q_model = make_tabular_model(config.model.backbone, bundle.input_dim, config.model.hidden_dims,
                                     config.model.num_classes, dropout=config.model.dropout)
    fp_model.load_state_dict(torch.load(run_dir / "fp_model.pt", map_location="cpu", weights_only=True))
    q_model.load_state_dict(torch.load(run_dir / "quantized_model.pt", map_location="cpu", weights_only=True))
    q_model.temperature = config.quant.temperature_end
    q_model.hard = True
    q_model.stochastic = False
    q_model.fixed_bit = config.quant.uniform_bit if config.method in {"lsq", "pact", "cfq_uniform", "ptq4"} else (8 if config.method == "ptq8" else None)
    if not config.quant.mixed_precision and config.method != "fp32":
        q_model.fixed_bit = config.quant.uniform_bit
    load_quantization_settings(q_model, run_dir)
    fp_model.eval()
    q_model.eval()
    return bundle, fp_model, q_model
