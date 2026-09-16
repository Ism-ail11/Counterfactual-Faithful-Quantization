from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch


@dataclass
class ActionSet:
    """Actionability constraints in preprocessed feature space.

    Bounds apply to the post-action point ``x + delta``. Categorical groups are
    lists of one-hot coordinates. Ordinal domains map a feature index to its
    allowed scalar values. ``immutable`` always overrides other constraints.
    """

    lower: torch.Tensor
    upper: torch.Tensor
    immutable: tuple[int, ...] = ()
    sparsity: int | None = None
    categorical_groups: tuple[tuple[int, ...], ...] = ()
    ordinal_domains: dict[int, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lower.ndim != 1 or self.upper.shape != self.lower.shape:
            raise ValueError("Bounds must be one-dimensional tensors with matching shapes")
        if torch.isnan(self.lower).any() or torch.isnan(self.upper).any() or (self.lower > self.upper).any():
            raise ValueError("Bounds must be ordered and cannot contain NaN")
        indices = [*self.immutable, *self.ordinal_domains]
        grouped = [index for group in self.categorical_groups for index in group]
        indices.extend(grouped)
        if any(index < 0 or index >= self.dimension for index in indices):
            raise ValueError("Constraint feature index is outside the feature dimension")
        if len(set(grouped)) != len(grouped) or set(grouped) & set(self.ordinal_domains):
            raise ValueError("Categorical groups must be disjoint and cannot contain ordinal features")
        if any(not domain for domain in self.ordinal_domains.values()):
            raise ValueError("Ordinal domains cannot be empty")
        if self.sparsity is not None and (not isinstance(self.sparsity, int) or self.sparsity < 0):
            raise ValueError("sparsity must be a nonnegative integer or None")

    def to(self, device: torch.device | str) -> "ActionSet":
        return ActionSet(
            lower=self.lower.to(device),
            upper=self.upper.to(device),
            immutable=self.immutable,
            sparsity=self.sparsity,
            categorical_groups=self.categorical_groups,
            ordinal_domains=self.ordinal_domains,
        )

    @property
    def dimension(self) -> int:
        return int(self.lower.numel())

    def actionable_mask(self, device: torch.device | str | None = None) -> torch.Tensor:
        target_device = device or self.lower.device
        mask = torch.ones(self.dimension, dtype=torch.bool, device=target_device)
        if self.immutable:
            mask[list(self.immutable)] = False
        return mask

    def project(self, x: torch.Tensor, delta: torch.Tensor, ste: bool = False, relax_discrete: bool = False) -> torch.Tensor:
        if x.shape != delta.shape:
            raise ValueError(f"x and delta must have the same shape, got {x.shape} and {delta.shape}")
        if x.shape[-1] != self.dimension:
            raise ValueError("Input feature dimension does not match the action set")
        lower, upper = self.lower.to(x), self.upper.to(x)
        projected = delta
        if self.immutable:
            mask = self.actionable_mask(delta.device).to(delta.dtype)
            projected = projected * mask

        x_new = torch.maximum(torch.minimum(x + projected, upper), lower)

        for group in self.categorical_groups:
            if not group or relax_discrete or all(i in self.immutable for i in group):
                continue
            index = torch.as_tensor(group, device=x.device, dtype=torch.long)
            values = x_new.index_select(-1, index)
            choices = torch.eye(len(group), device=x.device, dtype=x.dtype)
            allowed = ((choices >= lower[index]) & (choices <= upper[index])).all(dim=-1)
            allowed = allowed.expand(*values.shape[:-1], len(group))
            frozen = [position for position, i in enumerate(group) if i in self.immutable]
            if frozen:
                original = x.index_select(-1, index)
                matches = (choices[:, frozen] == original[..., frozen].unsqueeze(-2)).all(dim=-1)
                allowed = allowed & matches
            selected = values.masked_fill(~allowed, float("-inf")).argmax(dim=-1)
            hard = torch.nn.functional.one_hot(selected, num_classes=len(group)).to(values.dtype)
            hard = torch.where(allowed.any(dim=-1, keepdim=True), hard, x[..., index])
            if ste:
                hard = values + (hard - values).detach()
            x_new = x_new.clone()
            x_new[..., index] = hard

        for feature_index, domain in self.ordinal_domains.items():
            if relax_discrete or feature_index in self.immutable:
                continue
            values = torch.as_tensor(domain, device=x.device, dtype=x.dtype)
            values = values[(values >= lower[feature_index]) & (values <= upper[feature_index])]
            if not values.numel():
                raise ValueError(f"Ordinal domain for feature {feature_index} has no value within its bounds")
            current = x_new[..., feature_index].unsqueeze(-1)
            nearest = values[(current - values).abs().argmin(dim=-1)]
            if ste:
                nearest = x_new[..., feature_index] + (nearest - x_new[..., feature_index]).detach()
            x_new = x_new.clone()
            x_new[..., feature_index] = nearest

        projected = x_new - x
        if self.immutable:
            projected = projected * self.actionable_mask(delta.device).to(delta.dtype)

        if self.sparsity is not None:
            actionable = self.actionable_mask(delta.device)
            grouped_indices = {index for group in self.categorical_groups for index in group}
            units: list[tuple[int, ...]] = []
            for group in self.categorical_groups:
                active_group = tuple(index for index in group if bool(actionable[index].item()))
                if active_group:
                    units.append(active_group)
            for index in range(self.dimension):
                if index not in grouped_indices and bool(actionable[index].item()):
                    units.append((index,))
            k = max(0, min(int(self.sparsity), len(units)))
            if k == 0:
                projected = projected * 0.0
            elif k < len(units):
                unit_scores = torch.stack(
                    [projected[..., list(unit)].abs().sum(dim=-1) for unit in units], dim=-1
                )
                selected_units = unit_scores.topk(k, dim=-1).indices
                unit_mask = torch.zeros_like(unit_scores).scatter_(-1, selected_units, 1.0)
                sparse_mask = torch.zeros_like(projected)
                for unit_index, unit in enumerate(units):
                    sparse_mask[..., list(unit)] = unit_mask[..., unit_index].unsqueeze(-1)
                projected = projected * sparse_mask
        return projected

    def is_feasible(self, x: torch.Tensor, delta: torch.Tensor, atol: float = 1e-5) -> torch.Tensor:
        """Check every action constraint; immutable values override box/domain rules."""
        point = x + delta
        active = self.actionable_mask(x.device)
        valid = torch.isfinite(point).all(dim=-1)
        valid &= ((point[..., active] >= self.lower.to(x)[active] - atol)
                  & (point[..., active] <= self.upper.to(x)[active] + atol)).all(dim=-1)
        valid &= (delta[..., ~active].abs() <= atol).all(dim=-1)
        grouped = set()
        units = []
        for group in self.categorical_groups:
            grouped.update(group)
            if not group or all(i in self.immutable for i in group):
                continue
            values = point[..., list(group)]
            valid &= (values.sum(dim=-1) - 1).abs() <= atol
            valid &= ((values.abs() <= atol) | ((values - 1).abs() <= atol)).all(dim=-1)
            units.append(delta[..., list(group)].abs().amax(dim=-1) > atol)
        for index, domain in self.ordinal_domains.items():
            if index not in self.immutable:
                values = point.new_tensor(domain)
                valid &= (point[..., index, None] - values).abs().amin(dim=-1) <= atol
        for index in range(self.dimension):
            if index not in grouped and index not in self.immutable:
                units.append(delta[..., index].abs() > atol)
        if self.sparsity is not None and units:
            valid &= torch.stack(units, dim=-1).sum(dim=-1) <= self.sparsity
        return valid

    @classmethod
    def unconstrained(cls, dimension: int, bound: float = 5.0) -> "ActionSet":
        return cls(
            lower=torch.full((dimension,), -bound),
            upper=torch.full((dimension,), bound),
            immutable=(),
            sparsity=None,
        )


def groups_to_tuples(groups: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(index) for index in group) for group in groups)
