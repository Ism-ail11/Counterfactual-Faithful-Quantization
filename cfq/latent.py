from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LatentRecourseResult:
    delta: torch.Tensor
    counterfactual: torch.Tensor
    success: torch.Tensor
    cost: torch.Tensor
    target_loss: torch.Tensor


class LatentRecourseSolver:
    def __init__(
        self,
        steps: int = 50,
        step_size: float = 0.08,
        restarts: int = 2,
        radius: float = 4.0,
        pixel_budget: float = 120.0,
        lambda_pixel: float = 0.02,
        cost_weight: float = 0.01,
    ) -> None:
        if steps < 0 or step_size <= 0 or restarts < 1 or min(radius, pixel_budget, lambda_pixel, cost_weight) < 0:
            raise ValueError("Invalid latent solver steps, restarts, step size, or budget")
        self.steps = steps
        self.step_size = step_size
        self.restarts = restarts
        self.radius = radius
        self.pixel_budget = pixel_budget
        self.lambda_pixel = lambda_pixel
        self.cost_weight = cost_weight

    def _project(self, delta: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.minimum(torch.ones_like(norm), torch.tensor(self.radius, device=delta.device) / norm)
        return delta * scale

    @torch.enable_grad()
    def solve(
        self,
        classifier: nn.Module,
        autoencoder,
        x: torch.Tensor,
        target: torch.Tensor | int,
    ) -> LatentRecourseResult:
        if isinstance(target, int):
            target = torch.full((x.shape[0],), target, dtype=torch.long, device=x.device)
        else:
            target = target.to(device=x.device, dtype=torch.long)
        with torch.no_grad():
            z = autoencoder.encode(x)
        best_delta = torch.zeros_like(z)
        best_cf = x.clone()
        best_success = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        best_cost = torch.full((x.shape[0],), float("inf"), device=x.device)
        best_loss = torch.full((x.shape[0],), float("inf"), device=x.device)
        if len(x) == 0:
            return LatentRecourseResult(best_delta, best_cf, best_success, best_cost, best_loss)

        for restart in range(max(self.restarts, 1)):
            delta = torch.zeros_like(z) if restart == 0 else self._project(torch.randn_like(z) * 0.05)
            delta.requires_grad_(True)
            for step in range(self.steps + 1):
                decoded = autoencoder.decode(z + delta).clamp(0, 1)
                logits = classifier(decoded)
                ce = F.cross_entropy(logits, target, reduction="none")
                latent_cost = torch.linalg.vector_norm(delta, dim=-1)
                pixel_cost = (decoded - x).abs().flatten(1).sum(dim=1)
                cost = latent_cost + self.lambda_pixel * pixel_cost
                success = logits.argmax(dim=1).eq(target) & (pixel_cost <= self.pixel_budget + 1e-5)
                better = (success & (~best_success | (cost < best_cost))) | (~success & ~best_success & (ce < best_loss))
                best_delta = torch.where(better[:, None], delta.detach(), best_delta)
                view = better.view(-1, *([1] * (x.ndim - 1)))
                best_cf = torch.where(view, decoded.detach(), best_cf)
                best_success = torch.where(better, success, best_success)
                best_cost = torch.where(better, cost.detach(), best_cost)
                best_loss = torch.where(better, ce.detach(), best_loss)
                if step == self.steps:
                    break
                manifold_penalty = F.relu(pixel_cost - self.pixel_budget)
                objective = ce + self.cost_weight * (latent_cost + self.lambda_pixel * pixel_cost) + 0.01 * manifold_penalty
                gradient = torch.autograd.grad(objective.sum(), delta)[0]
                delta = self._project(delta - self.step_size * gradient).detach().requires_grad_(True)
        return LatentRecourseResult(
            best_delta.detach(), best_cf.detach(), best_success.detach(), best_cost.detach(), best_loss.detach()
        )
