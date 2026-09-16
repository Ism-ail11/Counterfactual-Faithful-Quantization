from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn

from .quantization import iter_quant_layers
from .ptq import layer_sensitivity
from .recourse import target_margin


@dataclass
class MarginDiagnostics:
    empirical_epsilon: float
    mean_teacher_margin: float
    mean_quantized_margin: float
    certificate_rate: float
    observed_transfer_rate: float

    def to_dict(self):
        return asdict(self)


@torch.no_grad()
def empirical_logit_perturbation(
    fp_model: nn.Module,
    q_model: nn.Module,
    points: torch.Tensor,
    radius: float = 0.05,
    neighborhood_samples: int = 8,
) -> float:
    if len(points) == 0:
        return float("nan")
    maximum = torch.tensor(0.0, device=points.device)
    candidates = [points]
    for _ in range(neighborhood_samples):
        noise = torch.randn_like(points)
        norm = torch.linalg.vector_norm(noise, dim=-1, keepdim=True).clamp_min(1e-8)
        noise = noise / norm * radius * torch.rand(points.shape[0], 1, device=points.device)
        candidates.append(points + noise)
    for candidate in candidates:
        deviation = (q_model(candidate) - fp_model(candidate)).abs().amax()
        maximum = torch.maximum(maximum, deviation)
    return float(maximum.item())


@torch.no_grad()
def margin_diagnostics(
    fp_model: nn.Module,
    q_model: nn.Module,
    points: torch.Tensor,
    target: torch.Tensor,
    radius: float = 0.05,
    neighborhood_samples: int = 8,
) -> MarginDiagnostics:
    epsilon = empirical_logit_perturbation(
        fp_model, q_model, points, radius=radius, neighborhood_samples=neighborhood_samples
    )
    fp_logits = fp_model(points)
    q_logits = q_model(points)
    teacher_margin = target_margin(fp_logits, target)
    q_margin = target_margin(q_logits, target)
    certificate = teacher_margin > 2 * epsilon
    transferred = q_logits.argmax(1).eq(target)
    return MarginDiagnostics(
        empirical_epsilon=epsilon,
        mean_teacher_margin=float(teacher_margin.mean().item()),
        mean_quantized_margin=float(q_margin.mean().item()),
        certificate_rate=float(certificate.float().mean().item()),
        observed_transfer_rate=float(transferred.float().mean().item()),
    )


def counterfactual_layer_sensitivity(
    model: nn.Module,
    points: torch.Tensor,
    target: torch.Tensor,
) -> list[float]:
    if len(points) == 0:
        return [float("nan") for module in model.modules() if isinstance(module, (nn.Linear, nn.Conv2d))]
    return layer_sensitivity(model, points, target)


@torch.no_grad()
def quantization_error_by_layer(q_model: nn.Module) -> list[dict[str, float | int]]:
    rows = []
    for index, layer in enumerate(iter_quant_layers(q_model)):
        quantized, expected_bit, hard_bit = layer.quantizer(
            layer.weight,
            temperature=0.1,
            hard=True,
            stochastic=False,
            fixed_bit=layer.fixed_bit,
        )
        error = quantized - layer.weight
        rows.append(
            {
                "layer": index,
                "bit": int(hard_bit),
                "expected_bit": float(expected_bit.item()),
                "linf_error": float(error.abs().max().item()),
                "l2_error": float(torch.linalg.vector_norm(error).item()),
            }
        )
    return rows
