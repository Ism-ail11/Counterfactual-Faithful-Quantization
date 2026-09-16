from __future__ import annotations

from typing import Protocol

import torch


class SemanticEditor(Protocol):
    """Required adapter for the paper's CelebA semantic-recourse experiment.

    The manuscript does not identify a trained editor, latent representation,
    attribute directions, protected-attribute list, or checkpoints. A concrete
    public reproduction must provide this interface rather than pretending that
    arbitrary pixel perturbations reproduce semantic edits.
    """

    def encode_attributes(self, images: torch.Tensor) -> torch.Tensor: ...

    def apply_edit(self, images: torch.Tensor, delta_semantic: torch.Tensor) -> torch.Tensor: ...

    def project_edit(self, delta_semantic: torch.Tensor) -> torch.Tensor: ...

    def edit_cost(self, delta_semantic: torch.Tensor) -> torch.Tensor: ...
