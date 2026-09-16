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
    parser = argparse.ArgumentParser(description="Run low-K and noisy-teacher diagnostics")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dataset", default="adult")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.10])
    parser.add_argument("--output", default="results/teacher_quality")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-eval-examples", type=int, default=512)
    args = parser.parse_args()
    base = base_config(args.config)
    rows = []
    for steps in args.steps:
        for noise in args.noise:
            config = clone_config(base)
            config.dataset = args.dataset
            config.method = "cfq"
            config.recourse.train_steps = steps
            config.output_dir = str(Path(args.output) / f"k_{steps}" / f"noise_{noise:g}")
            result = run_tabular_experiment(
                config,
                max_samples=args.max_samples,
                max_eval_examples=args.max_eval_examples,
                teacher_noise=noise,
            )
            rows.append({"K": steps, "teacher_noise": noise, **result["metrics"]})
            write_rows(rows, Path(args.output) / "summary.csv")


if __name__ == "__main__":
    main()
