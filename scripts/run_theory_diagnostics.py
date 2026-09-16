from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

if __package__ in {None, ""}:
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch

from cfq.experiments import run_tabular_experiment
from cfq.recourse import PGDRecourseSolver
from cfq.theory import counterfactual_layer_sensitivity, margin_diagnostics, quantization_error_by_layer
from cfq.utils import resolve_device, save_json
from scripts._common import base_config, clone_config
from scripts._load import load_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate epsilon, margins, layer sensitivity, and quantization errors")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dataset", default="adult")
    parser.add_argument("--method", default="cfq")
    parser.add_argument("--output", default="results/theory")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-eval-examples", type=int, default=256)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = clone_config(base_config(args.config))
    config.dataset = args.dataset
    config.method = args.method
    config.device = args.device
    config.output_dir = args.output
    run_dir = Path(args.output)
    if not (run_dir / "quantized_model.pt").exists():
        run_tabular_experiment(config, max_samples=args.max_samples, max_eval_examples=args.max_eval_examples)
    bundle, fp_model, q_model = load_models(config, run_dir, max_samples=args.max_samples)
    device = resolve_device(args.device)
    fp_model.to(device).eval()
    q_model.to(device).eval()
    x = bundle.x_test[: args.max_eval_examples].to(device)
    target = torch.full((len(x),), bundle.target_label, device=device, dtype=torch.long)
    solver = PGDRecourseSolver(
        steps=config.recourse.eval_steps,
        step_size=config.recourse.eval_step_size,
        restarts=config.recourse.eval_restarts,
        cost_kind=config.recourse.cost_kind,
        cost_weight=config.recourse.cost_weight,
        mixed_l1=config.recourse.mixed_l1,
        mixed_l2=config.recourse.mixed_l2,
        margin=config.recourse.margin,
    )
    with torch.enable_grad():
        recourse = solver.solve(fp_model, x, target, bundle.action_set, bundle.feature_weights.to(device))
    points = x + recourse.delta
    valid = recourse.success
    points = points[valid]
    target = target[valid]
    diagnostics = margin_diagnostics(fp_model, q_model, points, target)
    sensitivity = counterfactual_layer_sensitivity(fp_model, points, target)
    errors = quantization_error_by_layer(q_model)
    save_json(
        {
            "margin_diagnostics": diagnostics.to_dict(),
            "counterfactual_layer_sensitivity": sensitivity,
            "quantization_error_by_layer": errors,
        },
        run_dir / "theory_diagnostics.json",
    )


if __name__ == "__main__":
    main()
