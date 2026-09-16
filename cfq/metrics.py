from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn

from .recourse import PGDRecourseSolver, RecourseResult, target_margin


@dataclass
class RecourseMetrics:
    accuracy: float
    validity_drop: float
    crg: float
    direction_similarity: float
    action_overlap: float
    fp_feasible_rate: float
    q_feasible_rate: float
    transferred_success_rate: float
    mean_fp_cost: float
    mean_q_cost: float
    mean_quantized_margin_at_fp_cf: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 2048) -> float:
    correct = 0
    total = 0
    model.eval()
    for start in range(0, x.shape[0], batch_size):
        logits = model(x[start : start + batch_size])
        labels = y[start : start + batch_size]
        correct += int(logits.argmax(dim=1).eq(labels).sum().item())
        total += labels.numel()
    return correct / max(total, 1)


def direction_similarity(delta_a: torch.Tensor, delta_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    numerator = (delta_a * delta_b).sum(dim=-1)
    denominator = torch.linalg.vector_norm(delta_a, dim=-1) * torch.linalg.vector_norm(delta_b, dim=-1)
    return numerator / denominator.clamp_min(eps)


def action_overlap(delta_a: torch.Tensor, delta_b: torch.Tensor, threshold: float = 1e-4, eps: float = 1e-8):
    support_a = delta_a.abs() > threshold
    support_b = delta_b.abs() > threshold
    intersection = (support_a & support_b).sum(dim=-1).float()
    union = (support_a | support_b).sum(dim=-1).float()
    return intersection / union.clamp_min(eps)


def evaluate_recourse(
    fp_model: nn.Module,
    q_model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    target: int | torch.Tensor,
    action_set,
    solver: PGDRecourseSolver,
    feature_weights: torch.Tensor | None = None,
    support_threshold: float = 1e-4,
    max_examples: int | None = None,
) -> tuple[RecourseMetrics, RecourseResult, RecourseResult]:
    fp_model.eval()
    q_model.eval()
    if max_examples is not None:
        x = x[:max_examples]
        y = y[:max_examples]
    if isinstance(target, int):
        target_tensor = torch.full((x.shape[0],), target, dtype=torch.long, device=x.device)
    else:
        target_tensor = target[: x.shape[0]].to(x.device)

    with torch.enable_grad():
        fp_result = solver.solve(fp_model, x, target_tensor, action_set, feature_weights)
        q_result = fp_result if q_model is fp_model else solver.solve(q_model, x, target_tensor, action_set, feature_weights)

    with torch.no_grad():
        q_logits_at_fp = q_model(x + fp_result.delta)
        transferred = q_logits_at_fp.argmax(dim=1).eq(target_tensor)
        fp_valid = fp_result.success
        valid_mask = fp_valid
        vd = (~transferred[valid_mask]).float().mean()

        both_feasible = fp_result.success & q_result.success
        fp_cost = fp_result.cost
        q_cost = q_result.cost
        relative_gap = (q_cost - fp_cost) / fp_cost.clamp_min(1e-8)
        # Relative change has no denominator when the teacher requires no action.
        positive_cost = both_feasible & (fp_cost > 1e-8)
        crg = relative_gap[positive_cost].mean()
        dirs = direction_similarity(fp_result.delta, q_result.delta)
        overlap = action_overlap(fp_result.delta, q_result.delta, support_threshold)
        margin = target_margin(q_logits_at_fp, target_tensor)

        metrics = RecourseMetrics(
            accuracy=accuracy(q_model, x, y),
            validity_drop=float(vd.item()),
            crg=float(crg.item()),
            direction_similarity=float(dirs[both_feasible].mean().item()) if both_feasible.any() else float("nan"),
            action_overlap=float(overlap[both_feasible].mean().item()) if both_feasible.any() else float("nan"),
            fp_feasible_rate=float(fp_result.success.float().mean().item()),
            q_feasible_rate=float(q_result.success.float().mean().item()),
            transferred_success_rate=float(transferred[valid_mask].float().mean().item()) if valid_mask.any() else float("nan"),
            mean_fp_cost=float(fp_cost[fp_result.success].mean().item()) if fp_result.success.any() else float("nan"),
            mean_q_cost=float(q_cost[q_result.success].mean().item()) if q_result.success.any() else float("nan"),
            mean_quantized_margin_at_fp_cf=float(margin[valid_mask].mean().item()) if valid_mask.any() else float("nan"),
        )
    return metrics, fp_result, q_result


def subgroup_metrics(
    group_values: torch.Tensor,
    fp_result: RecourseResult,
    q_result: RecourseResult,
    q_model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    feature_weights: torch.Tensor | None,
    cost_kind: str,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        transferred = q_model(x + fp_result.delta).argmax(dim=1).eq(target)
        fp_cost = fp_result.cost
        q_cost = q_result.cost
        for value in group_values.unique(sorted=True):
            mask = group_values.eq(value)
            fp_valid = mask & fp_result.success
            both = fp_valid & q_result.success & (fp_cost > 1e-8)
            vd = (~transferred[fp_valid]).float().mean()
            gaps = (q_cost - fp_cost) / fp_cost.clamp_min(1e-8)
            output[str(int(value.item()))] = {
                "n": int(mask.sum().item()),
                "vd": float(vd.item()),
                "crg": float(gaps[both].mean().item()) if both.any() else float("nan"),
                "fp_feasible_rate": float(fp_result.success[mask].float().mean().item()),
            }
    return output
