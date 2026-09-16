from __future__ import annotations

import copy

import torch

from .constraints import ActionSet
from .quantization import iter_quant_layers


def feature_noise_shift(
    x: torch.Tensor,
    action_set: ActionSet,
    sigma: float = 0.1,
    seed: int = 42,
) -> torch.Tensor:
    generator = torch.Generator(device=x.device).manual_seed(seed)
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) * sigma
    noise = noise * action_set.actionable_mask(x.device).to(x.dtype)
    shifted_delta = action_set.to(x.device).project(x, noise)
    return x + shifted_delta


def reweighted_indices(
    group: torch.Tensor,
    desired_group_one_fraction: float,
    n: int | None = None,
    seed: int = 42,
) -> torch.Tensor:
    n = group.numel() if n is None else n
    if not 0 <= desired_group_one_fraction <= 1 or n < 0:
        raise ValueError("The group fraction must be in [0, 1] and n must be nonnegative")
    generator = torch.Generator(device=group.device).manual_seed(seed)
    group_zero = torch.where(group == 0)[0]
    group_one = torch.where(group != 0)[0]
    n_one = int(round(n * desired_group_one_fraction))
    n_zero = n - n_one
    if (n_zero and not len(group_zero)) or (n_one and not len(group_one)):
        raise ValueError("Cannot sample a requested group that is absent from the input")
    selected_zero = group_zero[torch.randint(len(group_zero), (n_zero,), generator=generator, device=group.device)] if n_zero else group_zero[:0]
    selected_one = group_one[torch.randint(len(group_one), (n_one,), generator=generator, device=group.device)] if n_one else group_one[:0]
    return torch.cat([selected_zero, selected_one])[torch.randperm(n, generator=generator, device=group.device)]


def target_imbalance_indices(
    y: torch.Tensor,
    positive_fraction: float,
    n: int | None = None,
    seed: int = 42,
) -> torch.Tensor:
    return reweighted_indices(y, positive_fraction, n=n, seed=seed)


def constraint_variant(action_set: ActionSet, mode: str) -> ActionSet:
    mode = mode.lower()
    span = action_set.upper - action_set.lower
    center = (action_set.upper + action_set.lower) / 2
    if mode == "restrictive":
        lower = center - 0.35 * span
        upper = center + 0.35 * span
        sparsity = max(1, (action_set.sparsity or action_set.dimension) - 2)
    elif mode == "moderate":
        lower, upper = action_set.lower.clone(), action_set.upper.clone()
        sparsity = action_set.sparsity
    elif mode == "permissive":
        lower = center - 0.75 * span
        upper = center + 0.75 * span
        sparsity = min(action_set.dimension, (action_set.sparsity or action_set.dimension) + 2)
    else:
        raise ValueError("mode must be restrictive, moderate, or permissive")
    for group in action_set.categorical_groups:
        lower[list(group)] = 0.0
        upper[list(group)] = 1.0
    return ActionSet(
        lower=lower,
        upper=upper,
        immutable=action_set.immutable,
        sparsity=sparsity,
        categorical_groups=action_set.categorical_groups,
        ordinal_domains=action_set.ordinal_domains,
    )


def sampled_quantization_variants(model: torch.nn.Module, count: int = 5, scale_sigma: float = 0.03):
    variants = []
    for _ in range(count):
        # Forward-pass budget caches can be non-leaf tensors. They are diagnostic
        # state, not part of a checkpoint, and must not retain an autograd graph.
        memo = {
            id(module.last_expected_bit): module.last_expected_bit.detach().clone()
            for module in model.modules() if hasattr(module, "last_expected_bit")
        }
        variant = copy.deepcopy(model, memo)
        with torch.no_grad():
            for layer in iter_quant_layers(variant):
                layer.quantizer.raw_steps.add_(torch.randn_like(layer.quantizer.raw_steps) * scale_sigma)
        variants.append(variant)
    return variants
