"""Each supported tabular method must train, evaluate, serialize and reload."""
import json
import math

import pytest
import torch

from cfq.config import ExperimentConfig, TABULAR_METHODS
from cfq.experiments import run_tabular_experiment
from cfq.metrics import accuracy
from cfq.quantization import bit_cost
from scripts._load import load_models


@pytest.mark.parametrize('method', sorted(TABULAR_METHODS))
def test_method_roundtrip(method, tmp_path):
    config = ExperimentConfig(method=method, output_dir=str(tmp_path / method), device='cpu')
    config.model.hidden_dims = (8, 4)
    config.train.epochs_fp = 1
    config.train.epochs_qat = 1
    config.train.batch_size = 64
    config.recourse.train_steps = 1
    config.recourse.eval_steps = 3
    config.recourse.eval_restarts = 1
    config.quant.bits = (2, 4, 8)
    if method == 'cfq_match':
        config.train.match_alpha1 = .02
        config.train.match_alpha2 = .02
        config.recourse.cost_kind = 'l2'
    result = run_tabular_experiment(config, max_samples=100, max_eval_examples=8)
    bundle, fp, q = load_models(config, config.output_dir, max_samples=100)
    with torch.no_grad():
        logits = q(bundle.x_test)
    assert torch.isfinite(logits).all()
    assert accuracy(q, bundle.x_test, bundle.y_test) == result['metrics']['accuracy']
    assert accuracy(fp, bundle.x_test, bundle.y_test) == result['fp_accuracy']
    assert math.isclose(bit_cost(q).item(), result['quantized_average_bits'])
    if method in {'ptq4', 'ptq8'}:
        assert q.blocks[0]['activation'].last_hard_bit == int(method[-1])
    if method == 'lsq':
        assert not q.quantize_activations
    if method == 'fp32':
        assert result['metrics']['validity_drop'] == 0 or math.isnan(result['metrics']['validity_drop'])
    if method == 'prune_quant':
        from cfq.quantization import iter_quant_layers
        layers = list(iter_quant_layers(q))
        zeros = sum(int(layer.weight.eq(0).sum()) for layer in layers)
        assert zeros / sum(layer.weight.numel() for layer in layers) >= .29
    data = json.loads((tmp_path / method / 'metrics.json').read_text(), parse_constant=lambda x: pytest.fail(x))
    assert data['method'] == method
    assert (tmp_path / method / 'preprocessor.joblib').exists()
    for history in result['history'].values():
        if history:
            assert all(math.isfinite(loss) for loss in history['losses'])


def test_disabled_mixed_precision_and_offline_reload(tmp_path, monkeypatch):
    config = ExperimentConfig(method='cfq', output_dir=str(tmp_path), device='cpu')
    config.model.hidden_dims = (4,)
    config.quant.mixed_precision = False
    config.quant.uniform_bit = 8
    config.train.epochs_fp = config.train.epochs_qat = 1
    config.recourse.train_steps = config.recourse.eval_steps = 1
    config.recourse.eval_restarts = 1
    run_tabular_experiment(config, max_samples=100, max_eval_examples=4)
    def unexpected_download(*args, **kwargs):
        pytest.fail('Reload should use the persisted dataset')
    monkeypatch.setattr('scripts._load.load_tabular_dataset', unexpected_download)
    bundle, _, q = load_models(config, tmp_path)
    with torch.no_grad():
        q(bundle.x_test)
    assert bit_cost(q).item() == 8.
    config.train.seed += 1
    with pytest.raises(ValueError, match='configuration differs'):
        load_models(config, tmp_path)
