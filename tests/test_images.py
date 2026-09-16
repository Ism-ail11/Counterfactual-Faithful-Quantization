import json
import math

import pytest
import torch

from cfq.data.image import ImageDatasetBundle
from cfq.experiments.image import _latent_metrics, run_image_experiment
from cfq.latent import LatentRecourseSolver
from cfq.models import ConvAutoencoder, SmallCNN


class IdentityAutoencoder:
    def encode(self, x):
        return x.flatten(1)

    def decode(self, z):
        return z.view(-1, 1, 2, 2)


def image_classifier():
    classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    with torch.no_grad():
        classifier[1].weight[0].fill_(-1.)
        classifier[1].weight[1].fill_(1.)
        classifier[1].bias.copy_(torch.tensor([2., -2.]))
    return classifier


def test_latent_success_requires_pixel_budget():
    x = torch.zeros(1, 1, 2, 2)
    result = LatentRecourseSolver(steps=8, step_size=1., restarts=1, pixel_budget=.1).solve(
        image_classifier(), IdentityAutoencoder(), x, 1)
    assert not result.success.any()


def test_latent_noop_retains_zero_cost():
    x = torch.ones(1, 1, 2, 2)
    result = LatentRecourseSolver(steps=4, restarts=1).solve(image_classifier(), IdentityAutoencoder(), x, 1)
    assert result.success.all()
    assert result.cost.item() == 0


def test_image_metrics_handles_all_examples_in_target_class():
    model = SmallCNN().eval()
    x = torch.rand(2, 1, 28, 28)
    with torch.no_grad():
        expected = model(x).argmax(1).eq(0).float().mean().item()
    metrics = _latent_metrics(model, model, ConvAutoencoder().eval(), x, torch.zeros(2, dtype=torch.long), 0)
    assert math.isnan(metrics['validity_drop'])
    assert metrics['accuracy'] == expected


@pytest.mark.parametrize('method', ['fp32', 'ptq8', 'cfptq', 'lsq', 'cfq', 'prune_quant'])
def test_image_pipeline_on_local_fixture(method, monkeypatch, tmp_path):
    generator = torch.Generator().manual_seed(44)
    x = torch.rand(8, 1, 28, 28, generator=generator)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    bundle = ImageDatasetBundle('fixture', x[:4], y[:4], x[4:6], y[4:6], x[6:], y[6:], 10)
    monkeypatch.setattr('cfq.experiments.image.load_image_dataset', lambda *a, **kw: bundle)
    result = run_image_experiment(method=method, output_dir=tmp_path, classifier_epochs=1,
                                  autoencoder_epochs=1, qat_epochs=1, device='cpu')
    assert 0 <= result['metrics']['accuracy'] <= 1
    json.loads((tmp_path / 'metrics.json').read_text(), parse_constant=lambda x: pytest.fail(x))
    assert (tmp_path / 'autoencoder.pt').exists()
    settings = json.loads((tmp_path / 'quantization.json').read_text())
    if method == 'ptq8':
        assert settings['fixed_bit'] == 8
    if method == 'lsq':
        assert settings['quantize_activations'] is False
