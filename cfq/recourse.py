from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .constraints import ActionSet
from .costs import recourse_cost


@dataclass
class RecourseResult:
    delta: torch.Tensor
    success: torch.Tensor
    cost: torch.Tensor
    target_loss: torch.Tensor


def target_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_logit = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, target.unsqueeze(1), float("-inf"))
    competitor = masked.max(dim=1).values
    return target_logit - competitor


class PGDRecourseSolver:
    def __init__(
        self, steps: int = 80, step_size: float = 0.04, restarts: int = 3,
        cost_kind: str = "l1", cost_weight: float = 0.02,
        mixed_l1: float = 0.5, mixed_l2: float = 0.5, margin: float = 0.0,
    ) -> None:
        if steps < 0 or step_size <= 0 or restarts < 1:
            raise ValueError("steps must be nonnegative; step_size and restarts must be positive")
        if cost_kind not in {"l1", "l2", "mixed"}:
            raise ValueError(f"Unknown cost kind: {cost_kind}")
        if min(cost_weight, mixed_l1, mixed_l2, margin) < 0:
            raise ValueError("Cost coefficients and margin must be nonnegative")
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.restarts = int(restarts)
        self.cost_kind = cost_kind
        self.cost_weight = float(cost_weight)
        self.mixed_l1 = float(mixed_l1)
        self.mixed_l2 = float(mixed_l2)
        self.margin = float(margin)

    def _cost(self, delta, weights, smooth=False):
        return recourse_cost(delta, weights, self.cost_kind, self.mixed_l1,
                             self.mixed_l2, smooth_eps=1e-6 if smooth else 0.0)

    @torch.enable_grad()
    def _search(self, models, x, target, action_set, weights, create_graph=False,
                detach_result=True, initial_noise=0.02, worst_case=True):
        if not models:
            raise ValueError("At least one model is required")
        if x.ndim != 2 or x.shape[1] != action_set.dimension:
            raise ValueError("x must be a batch with the action set's feature dimension")
        if isinstance(target, int):
            target = torch.full((len(x),), target, device=x.device, dtype=torch.long)
        else:
            target = target.to(device=x.device, dtype=torch.long)
        if target.shape != (len(x),):
            raise ValueError("target must have one class index per example")
        if len(x) == 0:
            return RecourseResult(torch.zeros_like(x), torch.empty(0, dtype=torch.bool, device=x.device),
                                  x.new_empty(0), x.new_empty(0))
        action = action_set.to(x.device)
        weights = None if weights is None else weights.to(x)
        best_delta = torch.zeros_like(x)
        best_success = torch.zeros(len(x), dtype=torch.bool, device=x.device)
        best_cost = x.new_full((len(x),), float("inf"))
        best_loss = x.new_full((len(x),), float("inf"))
        # Search a continuous relaxation so small steps can accumulate enough to
        # change a category/ordinal value. Every evaluated and returned action is
        # hard-projected and checked against all constraints.
        for restart in range(self.restarts):
            proposal = torch.zeros_like(x) if restart == 0 else torch.randn_like(x) * initial_noise
            proposal = action.project(x, proposal, relax_discrete=True).detach().requires_grad_(True)
            for step in range(self.steps + 1):
                delta = action.project(x, proposal, ste=True)
                logits = [model(x + delta) for model in models]
                losses = torch.stack([F.cross_entropy(value, target, reduction="none") for value in logits])
                loss = losses.max(dim=0).values if worst_case else losses.mean(dim=0)
                cost = self._cost(delta, weights, smooth=create_graph)
                success = torch.stack([value.argmax(1).eq(target) for value in logits]).all(dim=0)
                margins = torch.stack([target_margin(value, target) for value in logits])
                if self.margin > 0:
                    success = success & (margins >= self.margin).all(dim=0)
                success = success & action.is_feasible(x, delta.detach())
                better = (success & (~best_success | (cost < best_cost))) | (
                    ~success & ~best_success & (loss < best_loss))
                # Keep the least-cost feasible iterate, including a zero action.
                candidate_delta = delta if create_graph else delta.detach()
                candidate_cost = cost if create_graph else cost.detach()
                candidate_loss = loss if create_graph else loss.detach()
                best_delta = torch.where(better[:, None], candidate_delta, best_delta)
                best_success = torch.where(better, success, best_success)
                best_cost = torch.where(better, candidate_cost, best_cost)
                best_loss = torch.where(better, candidate_loss, best_loss)
                if step == self.steps:
                    break
                penalties = F.relu(self.margin - margins) if self.margin > 0 else torch.zeros_like(margins)
                penalty = penalties.max(dim=0).values if worst_case else penalties.mean(dim=0)
                objective = loss + self.cost_weight * cost + penalty
                gradient = torch.autograd.grad(objective.sum(), proposal, create_graph=create_graph,
                                               retain_graph=create_graph)[0]
                proposal = action.project(x, proposal - self.step_size * gradient, relax_discrete=True)
                if not create_graph:
                    proposal = proposal.detach().requires_grad_(True)
        # Evaluation uses the exact norm; differentiable training results retain
        # the smoothed norm so their backward pass stays finite at zero.
        best_cost = self._cost(best_delta, weights, smooth=create_graph)
        result = RecourseResult(best_delta, best_success, best_cost, best_loss)
        if detach_result:
            result = RecourseResult(*(value.detach() for value in
                (result.delta, result.success, result.cost, result.target_loss)))
        return result

    def solve(self, model: nn.Module, x: torch.Tensor, target: torch.Tensor | int,
              action_set: ActionSet, weights: torch.Tensor | None = None,
              create_graph: bool = False, detach_result: bool = True,
              initial_noise: float = 0.02) -> RecourseResult:
        return self._search([model], x, target, action_set, weights, create_graph,
                            detach_result, initial_noise)


def solve_linear_l2(
    weight: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    target_class: int = 1,
    feature_weights: torch.Tensor | None = None,
    margin: float = 1e-6,
) -> torch.Tensor:
    """Closed-form unconstrained binary affine recourse under weighted L2 cost.

    This is a sanity-check solver. Box, categorical, immutable, and sparsity
    constraints should be enforced afterwards with ``ActionSet.project`` or by
    the PGD solver.
    """
    if target_class not in (0, 1) or margin < 0:
        raise ValueError("target_class must be 0 or 1 and margin must be nonnegative")
    if weight.ndim == 2 and weight.shape[0] == 2:
        direction = weight[target_class] - weight[1 - target_class]
        offset = bias[target_class] - bias[1 - target_class]
    elif weight.ndim == 1:
        direction = weight if target_class == 1 else -weight
        offset = bias if target_class == 1 else -bias
    else:
        raise ValueError("Expected a binary linear classifier")
    current_margin = x @ direction + offset
    needed = (margin - current_margin).clamp_min(0.0)
    if feature_weights is None:
        inverse_metric = torch.ones_like(direction)
    else:
        inverse_metric = 1.0 / feature_weights.square().clamp_min(1e-12)
    denominator = (direction.square() * inverse_metric).sum().clamp_min(1e-12)
    delta = needed.unsqueeze(1) * (direction * inverse_metric).unsqueeze(0) / denominator
    return delta


class RobustPGDRecourseSolver(PGDRecourseSolver):
    """Recourse solver robust to a finite uncertainty set of deployment models."""

    def solve_ensemble(self, models: list[nn.Module], x: torch.Tensor,
                       target: torch.Tensor | int, action_set: ActionSet,
                       weights: torch.Tensor | None = None,
                       worst_case: bool = True) -> RecourseResult:
        return self._search(models, x, target, action_set, weights, worst_case=worst_case)
