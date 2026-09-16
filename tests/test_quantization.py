import torch

from cfq.quantization import QuantLinear, bit_cost


def test_mixed_precision_policy_receives_gradient():
    layer = QuantLinear(4, 2, bits=(2, 4, 8), init_bit=4)
    layer.train()
    layer.temperature = 1.0
    layer.hard = True
    layer.stochastic = True
    x = torch.randn(8, 4)
    loss = layer(x).square().mean()
    loss.backward()
    assert layer.quantizer.policy.logits.grad is not None
    assert torch.isfinite(layer.quantizer.policy.logits.grad).all()
    assert layer.quantizer.raw_steps.grad is not None


def test_fixed_bit_is_deterministic():
    layer = QuantLinear(3, 2, bits=(2, 4, 8), init_bit=4)
    layer.eval()
    layer.fixed_bit = 4
    x = torch.randn(5, 3)
    first = layer(x)
    second = layer(x)
    assert torch.allclose(first, second)
    assert layer.last_hard_bit == 4


def test_bit_cost_reports_average():
    model = torch.nn.Sequential(QuantLinear(3, 4, bits=(2, 4, 8)), QuantLinear(4, 2, bits=(2, 4, 8)))
    for layer in model:
        layer.fixed_bit = 4
    _ = model(torch.randn(2, 3))
    assert torch.isclose(bit_cost(model), torch.tensor(4.0))
