from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


def positive(raw: torch.Tensor, minimum: float = 1e-8) -> torch.Tensor:
    return F.softplus(raw) + minimum


def symmetric_quantize(x: torch.Tensor, bit: int, step: torch.Tensor) -> torch.Tensor:
    qmin = -(2 ** (bit - 1))
    qmax = 2 ** (bit - 1) - 1
    scaled = x / step
    clipped = scaled.clamp(qmin, qmax)
    return round_ste(clipped) * step


def unsigned_quantize(x: torch.Tensor, bit: int, clip_value: torch.Tensor) -> torch.Tensor:
    qmax = 2**bit - 1
    clipped = torch.minimum(torch.maximum(x, torch.zeros_like(x)), clip_value)
    step = clip_value / max(qmax, 1)
    return round_ste(clipped / step) * step


@dataclass
class BitSelection:
    weights: torch.Tensor
    expected_bit: torch.Tensor
    hard_bit: int


class BitPolicy(nn.Module):
    def __init__(self, bits: Iterable[int], init_bit: int | None = None) -> None:
        super().__init__()
        bits_tuple = tuple(int(bit) for bit in bits)
        if not bits_tuple:
            raise ValueError("At least one bit candidate is required")
        if len(set(bits_tuple)) != len(bits_tuple) or any(bit < 2 for bit in bits_tuple):
            raise ValueError("Bit candidates must be distinct integers of at least 2")
        if init_bit is not None and init_bit not in bits_tuple:
            raise ValueError("init_bit must be one of the bit candidates")
        self.register_buffer("bits", torch.tensor(bits_tuple, dtype=torch.float32))
        logits = torch.zeros(len(bits_tuple))
        if init_bit is not None and init_bit in bits_tuple:
            logits[bits_tuple.index(init_bit)] = 2.0
        self.logits = nn.Parameter(logits)

    def select(self, temperature: float, hard: bool, stochastic: bool) -> BitSelection:
        if self.training and stochastic:
            weights = F.gumbel_softmax(self.logits, tau=max(temperature, 1e-4), hard=hard, dim=0)
        else:
            soft = F.softmax(self.logits / max(temperature, 1e-4), dim=0)
            if hard:
                one_hot = F.one_hot(soft.argmax(), num_classes=soft.numel()).to(soft.dtype)
                weights = soft + (one_hot - soft).detach()
            else:
                weights = soft
        expected = (weights * self.bits).sum()
        hard_bit = int(self.bits[weights.argmax()].item())
        return BitSelection(weights=weights, expected_bit=expected, hard_bit=hard_bit)


class MixedPrecisionWeight(nn.Module):
    def __init__(self, shape: torch.Size, bits: Iterable[int], init_bit: int = 4) -> None:
        super().__init__()
        self.bits = tuple(int(bit) for bit in bits)
        self.policy = BitPolicy(self.bits, init_bit=init_bit)
        self.raw_steps = nn.Parameter(torch.full((len(self.bits),), -3.0))
        self.register_buffer("initialized", torch.tensor(False))
        self.shape = tuple(shape)

    @torch.no_grad()
    def initialize(self, weight: torch.Tensor) -> None:
        mean_abs = weight.detach().abs().mean().clamp_min(1e-6)
        for index, bit in enumerate(self.bits):
            qmax = max(2 ** (bit - 1) - 1, 1)
            desired = 2.0 * mean_abs / math.sqrt(qmax)
            desired = desired.clamp_min(1e-6)
            self.raw_steps[index].copy_(desired + torch.log(-torch.expm1(-desired)))
        self.initialized.fill_(True)

    def forward(
        self,
        weight: torch.Tensor,
        temperature: float,
        hard: bool = True,
        stochastic: bool = True,
        fixed_bit: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if not bool(self.initialized.item()):
            self.initialize(weight)
        if fixed_bit is not None:
            if fixed_bit not in self.bits:
                raise ValueError(f"fixed_bit={fixed_bit} not in {self.bits}")
            index = self.bits.index(fixed_bit)
            step = positive(self.raw_steps[index])
            return symmetric_quantize(weight, fixed_bit, step), weight.new_tensor(float(fixed_bit)), fixed_bit
        selection = self.policy.select(temperature, hard=hard, stochastic=stochastic)
        candidates = []
        for index, bit in enumerate(self.bits):
            candidates.append(symmetric_quantize(weight, bit, positive(self.raw_steps[index])))
        stacked = torch.stack(candidates, dim=0)
        view_shape = (len(self.bits),) + (1,) * weight.dim()
        quantized = (stacked * selection.weights.view(view_shape)).sum(dim=0)
        return quantized, selection.expected_bit, selection.hard_bit


class MixedPrecisionActivation(nn.Module):
    def __init__(self, bits: Iterable[int], init_bit: int = 4, init_clip: float = 6.0) -> None:
        super().__init__()
        self.bits = tuple(int(bit) for bit in bits)
        self.policy = BitPolicy(self.bits, init_bit=init_bit)
        if not math.isfinite(init_clip) or init_clip <= 0:
            raise ValueError("init_clip must be finite and positive")
        initial = torch.tensor(float(init_clip))
        raw = initial + torch.log(-torch.expm1(-initial))
        self.raw_clips = nn.Parameter(raw.repeat(len(self.bits)))
        self.last_expected_bit = torch.tensor(float(init_bit))
        self.last_hard_bit = init_bit

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = True,
        stochastic: bool = True,
        fixed_bit: int | None = None,
    ) -> torch.Tensor:
        if fixed_bit is not None:
            index = self.bits.index(fixed_bit)
            self.last_expected_bit = x.new_tensor(float(fixed_bit))
            self.last_hard_bit = fixed_bit
            return unsigned_quantize(x, fixed_bit, positive(self.raw_clips[index]))
        selection = self.policy.select(temperature, hard=hard, stochastic=stochastic)
        candidates = [
            unsigned_quantize(x, bit, positive(self.raw_clips[index])) for index, bit in enumerate(self.bits)
        ]
        stacked = torch.stack(candidates, dim=0)
        view_shape = (len(self.bits),) + (1,) * x.dim()
        self.last_expected_bit = selection.expected_bit
        self.last_hard_bit = selection.hard_bit
        return (stacked * selection.weights.view(view_shape)).sum(dim=0)


class QuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: Iterable[int] = (2, 3, 4, 8),
        bias: bool = True,
        init_bit: int = 4,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.quantizer = MixedPrecisionWeight(self.weight.shape, bits=bits, init_bit=init_bit)
        self.temperature = 1.0
        self.hard = True
        self.stochastic = True
        self.fixed_bit: int | None = None
        self.last_expected_bit = torch.tensor(float(init_bit))
        self.last_hard_bit = init_bit
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized, expected, hard_bit = self.quantizer(
            self.weight,
            temperature=self.temperature,
            hard=self.hard,
            stochastic=self.stochastic,
            fixed_bit=self.fixed_bit,
        )
        self.last_expected_bit = expected
        self.last_hard_bit = hard_bit
        return F.linear(x, quantized, self.bias)

    @property
    def parameter_count_for_budget(self) -> int:
        return self.weight.numel()


class QuantConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bits: Iterable[int] = (2, 3, 4, 8),
        bias: bool = True,
        init_bit: int = 4,
    ) -> None:
        super().__init__()
        kernel = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.quantizer = MixedPrecisionWeight(self.weight.shape, bits=bits, init_bit=init_bit)
        self.temperature = 1.0
        self.hard = True
        self.stochastic = True
        self.fixed_bit: int | None = None
        self.last_expected_bit = torch.tensor(float(init_bit))
        self.last_hard_bit = init_bit
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized, expected, hard_bit = self.quantizer(
            self.weight,
            temperature=self.temperature,
            hard=self.hard,
            stochastic=self.stochastic,
            fixed_bit=self.fixed_bit,
        )
        self.last_expected_bit = expected
        self.last_hard_bit = hard_bit
        return F.conv2d(x, quantized, self.bias, self.stride, self.padding)

    @property
    def parameter_count_for_budget(self) -> int:
        return self.weight.numel()


def iter_quant_layers(model: nn.Module):
    for module in model.modules():
        if isinstance(module, (QuantLinear, QuantConv2d)):
            yield module


def iter_activation_quantizers(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MixedPrecisionActivation):
            yield module


def configure_quantization(
    model: nn.Module,
    temperature: float,
    hard: bool = True,
    stochastic: bool = True,
    fixed_bit: int | None = None,
) -> None:
    # The enclosing model forwards these settings to its children on every call.
    # Configuring just the children is otherwise immediately overwritten.
    for name, value in (("temperature", float(temperature)), ("hard", hard),
                        ("stochastic", stochastic), ("fixed_bit", fixed_bit)):
        setattr(model, name, value)
    for layer in iter_quant_layers(model):
        layer.temperature = float(temperature)
        layer.hard = hard
        layer.stochastic = stochastic
        layer.fixed_bit = fixed_bit
    for activation in iter_activation_quantizers(model):
        activation.last_expected_bit = torch.tensor(float(fixed_bit or 4), device=activation.raw_clips.device)


def bit_cost(model: nn.Module, include_activations: bool = False) -> torch.Tensor:
    numerator: torch.Tensor | None = None
    denominator = 0
    for layer in iter_quant_layers(model):
        value = layer.last_expected_bit.to(layer.weight.device) * layer.parameter_count_for_budget
        numerator = value if numerator is None else numerator + value
        denominator += layer.parameter_count_for_budget
    if numerator is None or denominator == 0:
        parameter = next(model.parameters(), None)
        return torch.tensor(32.0) if parameter is None else parameter.new_tensor(32.0)
    result = numerator / denominator
    if include_activations:
        activation_bits = [module.last_expected_bit.to(result.device) for module in iter_activation_quantizers(model)]
        if activation_bits:
            result = 0.5 * result + 0.5 * torch.stack(activation_bits).mean()
    return result


def hard_bit_allocation(model: nn.Module) -> list[int]:
    return [layer.last_hard_bit for layer in iter_quant_layers(model)]
