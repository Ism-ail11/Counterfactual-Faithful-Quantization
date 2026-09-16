from cfq.config import ExperimentConfig
from cfq.experiments import run_tabular_experiment


def test_end_to_end_synthetic(tmp_path):
    config = ExperimentConfig(dataset="synthetic", method="cfq", output_dir=str(tmp_path / "run"))
    config.model.hidden_dims = (16, 8)
    config.train.epochs_fp = 2
    config.train.epochs_qat = 1
    config.train.batch_size = 64
    config.train.early_stopping_patience = 2
    config.recourse.train_steps = 1
    config.recourse.eval_steps = 5
    config.recourse.eval_restarts = 1
    config.quant.bits = (2, 4, 8)
    result = run_tabular_experiment(config, max_samples=300, max_eval_examples=16)
    assert result["dataset"] == "synthetic"
    assert "validity_drop" in result["metrics"]
    assert (tmp_path / "run" / "metrics.json").exists()
    assert (tmp_path / "run" / "quantized_model.pt").exists()
