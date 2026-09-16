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
    parser = argparse.ArgumentParser(description="Run backbone and recourse-cost robustness experiments")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dataset", default="adult")
    parser.add_argument("--output", default="results/backbones_costs")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-eval-examples", type=int, default=512)
    args = parser.parse_args()
    base = base_config(args.config)
    rows = []
    for backbone in ["logreg", "mlp", "deep_mlp"]:
        for cost in ["l1", "l2", "mixed"]:
            for method in ["mixedprec", "cfq"]:
                config = clone_config(base)
                config.dataset = args.dataset
                config.method = method
                config.model.backbone = backbone
                config.recourse.cost_kind = cost
                config.output_dir = str(Path(args.output) / backbone / cost / method)
                result = run_tabular_experiment(config, max_samples=args.max_samples, max_eval_examples=args.max_eval_examples)
                rows.append({"backbone": backbone, "cost": cost, "method": method, **result["metrics"]})
                write_rows(rows, Path(args.output) / "summary.csv")


if __name__ == "__main__":
    main()
