import torch

from cfq.constraints import ActionSet


def test_projection_enforces_immutability_bounds_and_group_sparsity():
    action_set = ActionSet(
        lower=torch.tensor([-1.0, 0.0, 0.0, -2.0, -2.0]),
        upper=torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0]),
        immutable=(0,),
        sparsity=2,
        categorical_groups=((1, 2),),
    )
    x = torch.tensor([[0.5, 1.0, 0.0, 0.0, 0.0]])
    delta = torch.tensor([[9.0, -0.2, 2.0, 5.0, -4.0]])
    projected = action_set.project(x, delta)
    x_new = x + projected
    assert projected[0, 0].item() == 0.0
    assert torch.all(x_new >= action_set.lower)
    assert torch.all(x_new <= action_set.upper)
    assert torch.isclose(x_new[0, 1:3].sum(), torch.tensor(1.0))
    assert set(x_new[0, 1:3].tolist()).issubset({0.0, 1.0})
    # A category switch counts as one unit, regardless of its two changed bits.
    changed_scalar_units = int((projected[0, 3:].abs() > 1e-8).sum().item())
    changed_category_units = int((projected[0, 1:3].abs() > 1e-8).any().item())
    assert changed_scalar_units + changed_category_units <= 2


def test_unconstrained_factory():
    action_set = ActionSet.unconstrained(3, bound=2.0)
    x = torch.zeros(2, 3)
    delta = torch.full_like(x, 4.0)
    projected = action_set.project(x, delta)
    assert torch.allclose(projected, torch.full_like(x, 2.0))
