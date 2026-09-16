"""Regression tests for reproduced numerical, constraint and PTQ failures."""
import json
import math

import pytest
import torch

from cfq.constraints import ActionSet
from cfq.costs import recourse_cost
from cfq.metrics import evaluate_recourse
from cfq.models import QuantTabularMLP, TabularMLP, copy_fp_to_quantized
from cfq.ptq import calibrate_ptq, layer_sensitivity
from cfq.quantization import configure_quantization, hard_bit_allocation
from cfq.recourse import PGDRecourseSolver, RobustPGDRecourseSolver, solve_linear_l2
from cfq.shifts import reweighted_indices, sampled_quantization_variants
from cfq.theory import margin_diagnostics
from cfq.utils import save_json


def linear_model():
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[-1., 0.], [1., 0.]]))
        model.bias.zero_()
    return model


def test_partial_immutable_category_stays_one_hot():
    action = ActionSet(torch.zeros(3), torch.ones(3), immutable=(2,), categorical_groups=((0, 1, 2),))
    x = torch.tensor([[0., 0., 1.]])
    delta = action.project(x, torch.tensor([[2., 0., 0.]]))
    assert torch.equal(x + delta, x)
    assert action.is_feasible(x, delta).all()


def test_ordinal_projection_respects_box():
    action = ActionSet(torch.tensor([.4]), torch.tensor([1.1]), ordinal_domains={0: (0., 1., 2.)})
    x = torch.tensor([[1.]])
    result = action.project(x, torch.tensor([[-10.]]))
    assert torch.equal(result, torch.zeros_like(x))
    assert action.is_feasible(x, result).all()


def test_already_target_has_zero_recourse_cost():
    x = torch.tensor([[1., 0.]])
    result = PGDRecourseSolver(steps=5, restarts=2).solve(linear_model(), x, 1, ActionSet.unconstrained(2))
    assert result.success.all()
    assert torch.equal(result.delta, torch.zeros_like(x))
    assert result.cost.item() == 0


def test_solver_keeps_cheaper_earlier_success_and_enforces_margin():
    x = torch.tensor([[-.2, 0.]])
    model = linear_model()
    action = ActionSet.unconstrained(2)
    short = PGDRecourseSolver(steps=5, step_size=.1, restarts=1, margin=.1).solve(model, x, 1, action)
    long = PGDRecourseSolver(steps=30, step_size=.1, restarts=1, margin=.1).solve(model, x, 1, action)
    assert short.success.all() and long.success.all()
    assert long.cost.item() <= short.cost.item() + 1e-6
    logits = model(x + long.delta)
    assert (logits[:, 1] - logits[:, 0] >= .1).all()


def test_discrete_search_accumulates_small_steps():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    x = torch.tensor([[1., 0.]])
    action = ActionSet(torch.zeros(2), torch.ones(2), categorical_groups=((0, 1),), sparsity=1)
    result = PGDRecourseSolver(steps=30, step_size=.1, restarts=1, cost_weight=0).solve(model, x, 1, action)
    assert result.success.all()
    assert torch.equal(x + result.delta, torch.tensor([[0., 1.]]))


def test_box_violation_cannot_be_reported_success():
    x = torch.tensor([[2., 0.]])
    action = ActionSet(torch.tensor([-1., -1.]), torch.tensor([1., 1.]), sparsity=0)
    result = PGDRecourseSolver(steps=1, restarts=1).solve(linear_model(), x, 1, action)
    assert not result.success.any()


def test_robust_cost_uses_custom_mixed_coefficients():
    model = linear_model()
    x = torch.tensor([[-.5, 0.]])
    solver = RobustPGDRecourseSolver(steps=8, step_size=.1, restarts=1, cost_kind='mixed', mixed_l1=.2, mixed_l2=1.7)
    result = solver.solve_ensemble([model, model], x, 1, ActionSet.unconstrained(2))
    assert torch.allclose(result.cost, recourse_cost(result.delta, kind='mixed', mixed_l1=.2, mixed_l2=1.7))


def test_differentiable_l2_has_finite_second_order_gradients():
    model = linear_model()
    solver = PGDRecourseSolver(steps=2, restarts=1, cost_kind='l2')
    result = solver.solve(model, torch.tensor([[-1., 0.]]), 1, ActionSet.unconstrained(2), create_graph=True, detach_result=False)
    result.delta.square().sum().backward()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all()


def test_linear_solution_crosses_argmax_tie():
    model = linear_model()
    x = torch.tensor([[-1., 0.]])
    delta = solve_linear_l2(model.weight, model.bias, x)
    assert model(x + delta).argmax(1).item() == 1


def test_frozen_sensitivity_preserves_requires_grad_and_existing_gradients():
    model = linear_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = torch.ones_like(p)
    scores = layer_sensitivity(model, torch.tensor([[-1., 1.], [1., 0.]]), torch.tensor([1, 0]))
    assert scores[0] > 0
    assert all(not p.requires_grad for p in model.parameters())
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    assert not model.training


@pytest.mark.parametrize('bit', [4, 8])
def test_uniform_ptq_calibrates_weights_and_activations(bit):
    torch.manual_seed(7)
    fp = TabularMLP(2, (4,)).eval()
    q = QuantTabularMLP(2, (4,))
    copy_fp_to_quantized(fp, q)
    x = torch.randn(12, 2)
    y = torch.arange(12) % 2
    history = calibrate_ptq(q, fp, x, y, ActionSet.unconstrained(2), torch.ones(2), 1,
                            (bit,), float(bit), torch.device('cpu'), calibration_epochs=1)
    with torch.no_grad():
        q(x)
    assert history.allocation == [bit, bit]
    assert hard_bit_allocation(q) == [bit, bit]
    assert q.blocks[0]['activation'].last_hard_bit == bit
    assert math.isfinite(history.losses[0])


def test_configure_parent_and_clone_after_gradient_forward():
    model = QuantTabularMLP(2, (3,))
    configure_quantization(model, .25, stochastic=False, fixed_bit=8)
    model(torch.ones(1, 2))
    assert hard_bit_allocation(model) == [8, 8]
    # Mixed precision caches a tensor attached to the policy's graph.
    configure_quantization(model, .25, stochastic=False)
    model(torch.ones(1, 2))
    variants = sampled_quantization_variants(model, count=2)
    assert len(variants) == 2
    assert torch.isfinite(variants[0](torch.ones(1, 2))).all()


def test_empty_conditioning_is_undefined_and_json_is_standard(tmp_path):
    model = linear_model()
    x = torch.tensor([[-1., 0.]])
    action = ActionSet.unconstrained(2)
    action.immutable = (0, 1)
    metrics, _, _ = evaluate_recourse(model, model, x, torch.tensor([0]), 1, action,
                                      PGDRecourseSolver(steps=1, restarts=1))
    assert math.isnan(metrics.validity_drop)
    path = tmp_path / 'metrics.json'
    save_json({'metrics': metrics.to_dict(), 'tensor': torch.tensor([float('inf')])}, path)
    result = json.loads(path.read_text(), parse_constant=lambda text: pytest.fail(text))
    assert result['metrics']['validity_drop'] is None
    assert result['tensor'] == [None]


def test_empty_theory_diagnostics_do_not_crash():
    model = linear_model()
    result = margin_diagnostics(model, model, torch.empty(0, 2), torch.empty(0, dtype=torch.long))
    assert math.isnan(result.empirical_epsilon)


def test_reweighting_endpoints_and_missing_group():
    group = torch.ones(5, dtype=torch.long)
    assert len(reweighted_indices(group, 1., n=8)) == 8
    assert len(reweighted_indices(group, .5, n=0)) == 0
    with pytest.raises(ValueError, match='absent'):
        reweighted_indices(group, .5, n=8)
