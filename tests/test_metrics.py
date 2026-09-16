import math
import torch

from cfq.metrics import action_overlap, direction_similarity


def test_direction_similarity_and_overlap():
    a = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    b = torch.tensor([[2.0, 0.0], [1.0, -1.0]])
    similarity = direction_similarity(a, b)
    overlap = action_overlap(a, b)
    assert math.isclose(similarity[0].item(), 1.0, rel_tol=1e-5)
    assert math.isclose(overlap[0].item(), 1.0, rel_tol=1e-5)
    assert abs(similarity[1].item()) < 1e-5
