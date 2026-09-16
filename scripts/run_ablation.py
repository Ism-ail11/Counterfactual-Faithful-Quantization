from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

if __package__ in {None, ""}:
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from cfq.experiments import run_tabular_experiment
from cfq.reporting import write_rows
from scripts._common import base_config, clone_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CFQ ablations from the main text and appendix")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dataset", default="adult")
    parser.add_argument("--output", default="results/ablations")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-eval-examples", type=int, default=512)
    args = parser.parse_args()
    base = base_config(args.config)
    variants = [
        ("cfq_full", "cfq", 3, 1.0, 0.0, 0.0),
        ("without_cf_loss", "mixedprec", 3, 0.0, 0.0, 0.0),
        ("uniform_bits", "cfq_uniform", 3, 1.0, 0.0, 0.0),
        ("teacher_k1", "cfq", 1, 1.0, 0.0, 0.0),
        ("teacher_k3", "cfq", 3, 1.0, 0.0, 0.0),
        ("hinge", "cfq", 3, 1.0, 0.25, 0.0),
        ("student_match", "cfq_match", 2, 1.0, 0.0, 0.02),
    ]
    rows = []
    for name, method, steps, eta, hinge_beta, match_alpha in variants:
        config = clone_config(base)
        config.dataset = args.dataset
        config.method = method
        config.recourse.train_steps = steps
        config.train.eta = eta
        config.train.hinge_beta = hinge_beta
        config.train.match_alpha1 = match_alpha
        config.train.match_alpha2 = match_alpha
        config.output_dir = str(Path(args.output) / name)
        result = run_tabular_experiment(config, max_samples=args.max_samples, max_eval_examples=args.max_eval_examples)
        rows.append({"variant": name, **result["metrics"], "bits": result["quantized_average_bits"]})
        write_rows(rows, Path(args.output) / "summary.csv")


if __name__ == "__main__":
    main()
