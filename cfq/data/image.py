from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision import datasets, transforms


@dataclass
class ImageDatasetBundle:
    name: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    num_classes: int


def _to_tensors(dataset, max_samples: int | None, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset)
    if n == 0 or (max_samples is not None and max_samples < 1):
        raise ValueError("Image datasets and sample limits must be nonempty/positive")
    indices = torch.arange(n)
    if max_samples is not None and max_samples < n:
        generator = torch.Generator().manual_seed(seed)
        indices = indices[torch.randperm(n, generator=generator)[:max_samples]]
    images, labels = [], []
    for index in indices.tolist():
        image, label = dataset[index]
        images.append(image)
        labels.append(int(label))
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def load_image_dataset(
    name: str,
    root: str | Path = "data",
    max_train: int | None = None,
    max_test: int | None = None,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> ImageDatasetBundle:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    key = name.lower().replace("_", "-")
    transform = transforms.ToTensor()
    root = str(root)
    if key in {"mnist", "mnist-recourse"}:
        train_set = datasets.MNIST(root, train=True, transform=transform, download=True)
        test_set = datasets.MNIST(root, train=False, transform=transform, download=True)
        canonical = "mnist"
    elif key in {"fashion-mnist", "fashion", "fashion-mnist-recourse"}:
        train_set = datasets.FashionMNIST(root, train=True, transform=transform, download=True)
        test_set = datasets.FashionMNIST(root, train=False, transform=transform, download=True)
        canonical = "fashion-mnist"
    else:
        raise ValueError(
            "The executable latent pipeline supports MNIST and Fashion-MNIST. "
            "CelebA requires a user-supplied semantic editor; see cfq.semantic."
        )
    x_all, y_all = _to_tensors(train_set, max_train, seed)
    x_test, y_test = _to_tensors(test_set, max_test, seed + 1)
    if len(x_all) < 2:
        raise ValueError("At least two training images are needed for a train/validation split")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(x_all.shape[0], generator=generator)
    n_val = min(len(x_all) - 1, max(1, int(round(validation_fraction * x_all.shape[0]))))
    val_indices, train_indices = permutation[:n_val], permutation[n_val:]
    return ImageDatasetBundle(
        name=canonical,
        x_train=x_all[train_indices],
        y_train=y_all[train_indices],
        x_val=x_all[val_indices],
        y_val=y_all[val_indices],
        x_test=x_test,
        y_test=y_test,
        num_classes=10,
    )
