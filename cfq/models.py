from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .quantization import MixedPrecisionActivation, QuantConv2d, QuantLinear


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        num_classes: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class QuantTabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        num_classes: int = 2,
        dropout: float = 0.0,
        bits: Iterable[int] = (2, 3, 4, 8),
        init_bit: int = 4,
        quantize_activations: bool = True,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        self.blocks = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            block = nn.ModuleDict(
                {
                    "linear": QuantLinear(in_dim, out_dim, bits=bits, init_bit=init_bit),
                    "activation": MixedPrecisionActivation(bits=bits, init_bit=init_bit),
                    "dropout": nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                }
            )
            self.blocks.append(block)
        self.classifier = QuantLinear(dims[-1], num_classes, bits=bits, init_bit=init_bit)
        self.quantize_activations = quantize_activations
        self.temperature = 1.0
        self.hard = True
        self.stochastic = True
        self.fixed_bit: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            linear: QuantLinear = block["linear"]  # type: ignore[assignment]
            linear.temperature = self.temperature
            linear.hard = self.hard
            linear.stochastic = self.stochastic
            linear.fixed_bit = self.fixed_bit
            x = F.relu(linear(x))
            if self.quantize_activations:
                activation: MixedPrecisionActivation = block["activation"]  # type: ignore[assignment]
                x = activation(
                    x,
                    temperature=self.temperature,
                    hard=self.hard,
                    stochastic=self.stochastic,
                    fixed_bit=self.fixed_bit,
                )
            x = block["dropout"](x)
        self.classifier.temperature = self.temperature
        self.classifier.hard = self.hard
        self.classifier.stochastic = self.stochastic
        self.classifier.fixed_bit = self.fixed_bit
        return self.classifier(x)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class QuantLogisticRegression(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2, bits=(2, 3, 4, 8), init_bit: int = 4) -> None:
        super().__init__()
        self.linear = QuantLinear(input_dim, num_classes, bits=bits, init_bit=init_bit)
        self.temperature = 1.0
        self.hard = True
        self.stochastic = True
        self.fixed_bit: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.linear.temperature = self.temperature
        self.linear.hard = self.hard
        self.linear.stochastic = self.stochastic
        self.linear.fixed_bit = self.fixed_bit
        return self.linear(x)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, latent_width: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, latent_width), nn.ReLU(), nn.Linear(latent_width, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class QuantSmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, bits=(2, 3, 4, 8), init_bit: int = 4,
                 quantize_activations: bool = True) -> None:
        super().__init__()
        self.conv1 = QuantConv2d(1, 32, 3, padding=1, bits=bits, init_bit=init_bit)
        self.conv2 = QuantConv2d(32, 64, 3, padding=1, bits=bits, init_bit=init_bit)
        self.fc1 = QuantLinear(64 * 7 * 7, 64, bits=bits, init_bit=init_bit)
        self.fc2 = QuantLinear(64, num_classes, bits=bits, init_bit=init_bit)
        self.act1 = MixedPrecisionActivation(bits, init_bit)
        self.act2 = MixedPrecisionActivation(bits, init_bit)
        self.act3 = MixedPrecisionActivation(bits, init_bit)
        self.temperature = 1.0
        self.hard = True
        self.stochastic = True
        self.fixed_bit: int | None = None
        self.quantize_activations = quantize_activations

    def _configure(self) -> None:
        for layer in (self.conv1, self.conv2, self.fc1, self.fc2):
            layer.temperature = self.temperature
            layer.hard = self.hard
            layer.stochastic = self.stochastic
            layer.fixed_bit = self.fixed_bit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._configure()
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        if self.quantize_activations:
            x = self.act1(x, self.temperature, self.hard, self.stochastic, self.fixed_bit)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        if self.quantize_activations:
            x = self.act2(x, self.temperature, self.hard, self.stochastic, self.fixed_bit)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        if self.quantize_activations:
            x = self.act3(x, self.temperature, self.hard, self.stochastic, self.fixed_bit)
        return self.fc2(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder_fc = nn.Linear(32 * 7 * 7, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder_conv(x)
        return self.encoder_fc(hidden.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_fc(z).view(-1, 32, 7, 7)
        return self.decoder_conv(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def make_tabular_model(backbone: str, input_dim: int, hidden_dims=(128, 64), num_classes: int = 2, dropout: float = 0.0):
    if backbone == "logreg":
        return LogisticRegression(input_dim, num_classes)
    if backbone == "mlp":
        return TabularMLP(input_dim, hidden_dims, num_classes, dropout=dropout)
    if backbone == "deep_mlp":
        return TabularMLP(input_dim, hidden_dims=(256, 128, 64, 32), num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown backbone: {backbone}")


def make_quant_tabular_model(
    backbone: str,
    input_dim: int,
    hidden_dims=(128, 64),
    num_classes: int = 2,
    bits=(2, 3, 4, 8),
    init_bit: int = 4,
    quantize_activations: bool = True,
    dropout: float = 0.0,
):
    if backbone == "logreg":
        return QuantLogisticRegression(input_dim, num_classes, bits=bits, init_bit=init_bit)
    if backbone == "mlp":
        return QuantTabularMLP(
            input_dim,
            hidden_dims,
            num_classes,
            bits=bits,
            init_bit=init_bit,
            quantize_activations=quantize_activations,
            dropout=dropout,
        )
    if backbone == "deep_mlp":
        return QuantTabularMLP(
            input_dim,
            (256, 128, 64, 32),
            num_classes,
            bits=bits,
            init_bit=init_bit,
            quantize_activations=quantize_activations,
            dropout=dropout,
        )
    raise ValueError(f"Unknown backbone: {backbone}")


@torch.no_grad()
def copy_fp_to_quantized(fp_model: nn.Module, quantized_model: nn.Module) -> None:
    fp_linear = [module for module in fp_model.modules() if isinstance(module, nn.Linear)]
    q_linear = [module for module in quantized_model.modules() if isinstance(module, QuantLinear)]
    if len(fp_linear) != len(q_linear):
        raise ValueError(f"Linear layer mismatch: {len(fp_linear)} FP vs {len(q_linear)} quantized")
    for source, target in zip(fp_linear, q_linear):
        target.weight.copy_(source.weight)
        if source.bias is not None and target.bias is not None:
            target.bias.copy_(source.bias)
        target.quantizer.initialize(target.weight)

    fp_conv = [module for module in fp_model.modules() if isinstance(module, nn.Conv2d)]
    q_conv = [module for module in quantized_model.modules() if isinstance(module, QuantConv2d)]
    if fp_conv or q_conv:
        if len(fp_conv) != len(q_conv):
            raise ValueError(f"Conv layer mismatch: {len(fp_conv)} FP vs {len(q_conv)} quantized")
        for source, target in zip(fp_conv, q_conv):
            target.weight.copy_(source.weight)
            if source.bias is not None and target.bias is not None:
                target.bias.copy_(source.bias)
            target.quantizer.initialize(target.weight)
