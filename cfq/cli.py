from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

from .config import ExperimentConfig, ModelConfig, QuantConfig, RecourseConfig, TrainConfig
from .experiments import run_tabular_experiment
from .utils import load_yaml

T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any] | None) -> T:
    values = {} if values is None else values
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} options must be a mapping")
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} options: {sorted(unknown)}")
    clean = {key: value for key, value in values.items() if key in allowed}
    for key in ("hidden_dims", "bits"):
        if key in clean and isinstance(clean[key], list):
            clean[key] = tuple(clean[key])
    return cls(**clean)


def config_from_mapping(mapping: dict[str, Any]) -> ExperimentConfig:
    if not isinstance(mapping, dict):
        raise ValueError("Experiment configuration must be a mapping")
    unknown = set(mapping) - {field.name for field in fields(ExperimentConfig)}
    if unknown:
        raise ValueError(f"Unknown experiment options: {sorted(unknown)}")
    return ExperimentConfig(
        dataset=mapping.get("dataset", "synthetic"),
        method=mapping.get("method", "cfq"),
        model=_construct(ModelConfig, mapping.get("model")),
        recourse=_construct(RecourseConfig, mapping.get("recourse")),
        quant=_construct(QuantConfig, mapping.get("quant")),
        train=_construct(TrainConfig, mapping.get("train")),
        output_dir=mapping.get("output_dir", "results/run"),
        device=mapping.get("device"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfq", description="Counterfactual-Faithful Quantization experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    tabular = subparsers.add_parser("tabular", help="Run a tabular experiment")
    tabular.add_argument("--config", type=Path, help="YAML config; defaults to the built-in experiment settings")
    tabular.add_argument("--dataset")
    tabular.add_argument("--method")
    tabular.add_argument("--output-dir")
    tabular.add_argument("--device")
    tabular.add_argument("--seed", type=int)
    tabular.add_argument("--max-samples", type=int)
    tabular.add_argument("--max-eval-examples", type=int, default=512)
    tabular.add_argument("--cache-dir", default="data")

    smoke = subparsers.add_parser("smoke", help="Run a fast end-to-end synthetic smoke test")
    smoke.add_argument("--output-dir", default="results/smoke")
    smoke.add_argument("--device")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command == "tabular":
        mapping = load_yaml(args.config) if args.config is not None else {}
        config = config_from_mapping(mapping)
        if args.dataset:
            config.dataset = args.dataset
        if args.method:
            config.method = args.method
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.device:
            config.device = args.device
        if args.seed is not None:
            config.train.seed = args.seed
        result = run_tabular_experiment(
            config,
            max_samples=args.max_samples,
            max_eval_examples=args.max_eval_examples,
            cache_dir=args.cache_dir,
        )
        print(result)
        return

    config = ExperimentConfig(dataset="synthetic", method="cfq", output_dir=args.output_dir, device=args.device)
    config.model.hidden_dims = (32, 16)
    config.train.epochs_fp = 3
    config.train.epochs_qat = 2
    config.train.batch_size = 128
    config.train.early_stopping_patience = 3
    config.recourse.train_steps = 1
    config.recourse.eval_steps = 8
    config.recourse.eval_restarts = 1
    config.quant.bits = (2, 4, 8)
    result = run_tabular_experiment(config, max_samples=600, max_eval_examples=48)
    print(result)


if __name__ == "__main__":
    main()
