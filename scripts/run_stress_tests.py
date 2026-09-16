from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

if __package__ in {None, ""}:
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch

from cfq.experiments import run_tabular_experiment
from cfq.metrics import evaluate_recourse, subgroup_metrics
from cfq.recourse import PGDRecourseSolver, RobustPGDRecourseSolver
from cfq.reporting import write_rows
from cfq.shifts import (
    constraint_variant,
    feature_noise_shift,
    reweighted_indices,
    sampled_quantization_variants,
    target_imbalance_indices,
)
from cfq.utils import resolve_device, save_json
from scripts._common import base_config, clone_config
from scripts._load import load_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run constraints, shifts, subgroup, and robust-solver appendix tests")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dataset", default="adult")
    parser.add_argument("--methods", nargs="+", default=["mixedprec", "cfq"])
    parser.add_argument("--output", default="results/stress")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-eval-examples", type=int, default=384)
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base = base_config(args.config)
    rows_constraints, rows_shifts, rows_groups, rows_robust = [], [], [], []
    for method in args.methods:
        config = clone_config(base)
        config.dataset = args.dataset
        config.method = method
        config.device = args.device
        run_dir = Path(args.output) / "models" / method
        config.output_dir = str(run_dir)
        if args.force or not (run_dir / "quantized_model.pt").exists():
            run_tabular_experiment(
                config,
                max_samples=args.max_samples,
                max_eval_examples=args.max_eval_examples,
            )
        bundle, fp_model, q_model = load_models(config, run_dir, max_samples=args.max_samples)
        device = resolve_device(args.device)
        fp_model.to(device).eval()
        q_model.to(device).eval()
        x = bundle.x_test[: args.max_eval_examples].to(device)
        y = bundle.y_test[: args.max_eval_examples].to(device)
        weights = bundle.feature_weights.to(device)
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

        for mode in ["restrictive", "moderate", "permissive"]:
            action = constraint_variant(bundle.action_set, mode)
            metrics, _, _ = evaluate_recourse(
                fp_model, q_model, x, y, bundle.target_label, action, solver, weights
            )
            rows_constraints.append({"method": method, "constraint": mode, **metrics.to_dict()})
            write_rows(rows_constraints, Path(args.output) / "constraint_sensitivity.csv")

        shifted_sets: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = {
            "original": (x, y, bundle.group_test[: len(x)].to(device) if bundle.group_test is not None else None),
            "feature_noise": (
                feature_noise_shift(x, bundle.action_set.to(device), sigma=0.10, seed=config.train.seed),
                y,
                bundle.group_test[: len(x)].to(device) if bundle.group_test is not None else None,
            ),
        }
        if bundle.group_test is not None:
            group = bundle.group_test[: len(x)].to(device)
            indices = reweighted_indices(group, desired_group_one_fraction=0.75, n=len(x), seed=config.train.seed)
            shifted_sets["group_reweight"] = (x[indices], y[indices], group[indices])
        indices = target_imbalance_indices(y, positive_fraction=0.20, n=len(x), seed=config.train.seed)
        shifted_sets["target_imbalance"] = (x[indices], y[indices], None)

        for shift_name, (shift_x, shift_y, shift_group) in shifted_sets.items():
            metrics, fp_result, q_result = evaluate_recourse(
                fp_model,
                q_model,
                shift_x,
                shift_y,
                bundle.target_label,
                bundle.action_set,
                solver,
                weights,
            )
            rows_shifts.append({"method": method, "shift": shift_name, **metrics.to_dict()})
            write_rows(rows_shifts, Path(args.output) / "distribution_shift.csv")
            if shift_name == "original":
                if shift_group is not None:
                    target = torch.full((len(shift_x),), bundle.target_label, device=device, dtype=torch.long)
                    groups = subgroup_metrics(
                        shift_group,
                        fp_result,
                        q_result,
                        q_model,
                        shift_x,
                        target,
                        weights,
                        config.recourse.cost_kind,
                    )
                    for group_value, values in groups.items():
                        rows_groups.append({"method": method, "group": group_value, **values})
                    write_rows(rows_groups, Path(args.output) / "subgroups.csv")

        variants = [variant.to(device).eval() for variant in sampled_quantization_variants(q_model, count=5)]
        target = torch.full((len(x),), bundle.target_label, device=device, dtype=torch.long)
        standard_result = solver.solve(q_model, x, target, bundle.action_set, weights)
        robust_solver = RobustPGDRecourseSolver(
            steps=config.recourse.eval_steps,
            step_size=config.recourse.eval_step_size,
            restarts=max(1, config.recourse.eval_restarts),
            cost_kind=config.recourse.cost_kind,
            cost_weight=config.recourse.cost_weight,
            mixed_l1=config.recourse.mixed_l1,
            mixed_l2=config.recourse.mixed_l2,
            margin=config.recourse.margin,
        )
        robust_result = robust_solver.solve_ensemble([q_model, *variants], x, target, bundle.action_set, weights)
        with torch.no_grad():
            standard_nominal = q_model(x + standard_result.delta).argmax(1).eq(target)
            robust_nominal = q_model(x + robust_result.delta).argmax(1).eq(target)
            standard_variant_success = torch.stack(
                [variant(x + standard_result.delta).argmax(1).eq(target) for variant in variants]
            ).float().mean(0)
            robust_variant_success = torch.stack(
                [variant(x + robust_result.delta).argmax(1).eq(target) for variant in variants]
            ).float().mean(0)
            standard_cost = standard_result.cost
            robust_cost = robust_result.cost
            rows_robust.extend(
                [
                    {
                        "method": method,
                        "solver": "standard",
                        "nominal_success": float(standard_nominal.float().mean().item()),
                        "robust_success": float(standard_variant_success.mean().item()),
                        "mean_cost": float(standard_cost.mean().item()),
                    },
                    {
                        "method": method,
                        "solver": "robust",
                        "nominal_success": float(robust_nominal.float().mean().item()),
                        "robust_success": float(robust_variant_success.mean().item()),
                        "mean_cost": float(robust_cost.mean().item()),
                    },
                ]
            )
            write_rows(rows_robust, Path(args.output) / "robust_solver.csv")

    save_json(
        {
            "constraint_rows": len(rows_constraints),
            "shift_rows": len(rows_shifts),
            "subgroup_rows": len(rows_groups),
            "robust_rows": len(rows_robust),
        },
        Path(args.output) / "manifest.json",
    )


if __name__ == "__main__":
    main()
