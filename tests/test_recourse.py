import torch

from cfq.constraints import ActionSet
from cfq.recourse import PGDRecourseSolver, solve_linear_l2


def test_pgd_recourse_flips_simple_linear_model():
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
        model.bias.zero_()
    x = torch.tensor([[-1.0, 0.0], [-0.5, 0.2]])
    action_set = ActionSet(
        lower=torch.tensor([-2.0, -2.0]),
        upper=torch.tensor([2.0, 2.0]),
        immutable=(1,),
        sparsity=1,
    )
    solver = PGDRecourseSolver(steps=40, step_size=0.1, restarts=1, cost_weight=0.001)
    result = solver.solve(model, x, 1, action_set)
    assert result.success.all()
    assert torch.allclose(result.delta[:, 1], torch.zeros(2))


def test_closed_form_linear_l2_points_toward_target():
    weight = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    bias = torch.zeros(2)
    x = torch.tensor([[-1.0, 0.0]])
    delta = solve_linear_l2(weight, bias, x, target_class=1)
    assert delta[0, 0] > 0
