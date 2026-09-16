from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import copy
import math


@dataclass
class ModelConfig:
    backbone: Literal["logreg", "mlp", "deep_mlp"] = "mlp"
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.0
    num_classes: int = 2


@dataclass
class RecourseConfig:
    train_steps: int = 3
    eval_steps: int = 80
    train_step_size: float = 0.08
    eval_step_size: float = 0.04
    train_restarts: int = 1
    eval_restarts: int = 3
    cost_kind: Literal["l1", "l2", "mixed"] = "l1"
    cost_weight: float = 0.02
    mixed_l1: float = 0.5
    mixed_l2: float = 0.5
    margin: float = 0.0
    support_threshold: float = 1e-4


@dataclass
class QuantConfig:
    bits: tuple[int, ...] = (2, 3, 4, 8)
    uniform_bit: int = 4
    mixed_precision: bool = True
    temperature_start: float = 5.0
    temperature_end: float = 0.25
    target_avg_bits: float = 4.0
    budget_tolerance: float = 0.05
    quantize_activations: bool = True


@dataclass
class TrainConfig:
    epochs_fp: int = 30
    epochs_qat: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    eta: float = 1.0
    bit_lambda: float = 1e-2
    hinge_beta: float = 0.0
    hinge_gamma: float = 0.25
    match_alpha1: float = 0.0
    match_alpha2: float = 0.0
    early_stopping_patience: int = 8
    seed: int = 42


@dataclass
class ExperimentConfig:
    dataset: str = "synthetic"
    method: str = "cfq"
    model: ModelConfig = field(default_factory=ModelConfig)
    recourse: RecourseConfig = field(default_factory=RecourseConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "results/run"
    device: str | None = None


TABULAR_METHODS = {
    "fp32", "lsq", "pact", "mixedprec", "cfq", "cfq_uniform", "cfq_match",
    "r_margin", "r_consistency", "prune_quant", "kd_quant", "ptq4", "ptq8",
    "mixedptq", "cfptq", "cfptq_sensitivity",
}


def prepare_config(config: ExperimentConfig) -> ExperimentConfig:
    """Validate mutable experiment settings and resolve method-specific behavior."""
    config = copy.deepcopy(config)
    config.method = config.method.lower()
    if config.method not in TABULAR_METHODS:
        raise ValueError(f"Unknown method {config.method!r}; choose from {sorted(TABULAR_METHODS)}")
    model, quant, train, recourse = config.model, config.quant, config.train, config.recourse
    if model.backbone not in {"logreg", "mlp", "deep_mlp"}:
        raise ValueError(f"Unknown backbone: {model.backbone}")
    if model.num_classes != 2:
        raise ValueError("The tabular datasets use binary labels; num_classes must be 2")
    if not 0 <= model.dropout < 1 or any(not isinstance(d, int) or d < 1 for d in model.hidden_dims):
        raise ValueError("Invalid hidden_dims or dropout")
    if not quant.bits or any(not isinstance(bit, int) or bit < 2 for bit in quant.bits) or len(set(quant.bits)) != len(quant.bits):
        raise ValueError("bits must contain distinct integers of at least 2")
    if config.method in {"ptq4", "ptq8"}:
        quant.uniform_bit = 4 if config.method == "ptq4" else 8
        quant.target_avg_bits = float(quant.uniform_bit)
        quant.bits = tuple(sorted(set((*quant.bits, quant.uniform_bit))))
    if quant.uniform_bit not in quant.bits:
        raise ValueError("uniform_bit must be present in bits")
    if config.method == "lsq":
        quant.quantize_activations = False
    if min(quant.temperature_start, quant.temperature_end) <= 0:
        raise ValueError("Quantization temperatures must be positive")
    if not math.isfinite(quant.target_avg_bits) or quant.target_avg_bits < min(quant.bits):
        raise ValueError("target_avg_bits must be finite and at least the smallest candidate")
    if min(train.epochs_fp, train.epochs_qat) < 1 or train.batch_size < 1 or train.early_stopping_patience < 1:
        raise ValueError("Epochs, batch_size and early_stopping_patience must be positive")
    if train.learning_rate <= 0 or train.weight_decay < 0:
        raise ValueError("Invalid optimizer learning_rate or weight_decay")
    if recourse.train_steps < 0 or recourse.eval_steps < 0 or min(recourse.train_restarts, recourse.eval_restarts) < 1:
        raise ValueError("Recourse steps must be nonnegative and restarts must be positive")
    if min(recourse.train_step_size, recourse.eval_step_size) <= 0:
        raise ValueError("Recourse step sizes must be positive")
    if recourse.cost_kind not in {"l1", "l2", "mixed"} or min(recourse.cost_weight, recourse.mixed_l1, recourse.mixed_l2, recourse.margin) < 0:
        raise ValueError("Invalid recourse cost or margin")
    return config
