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
    parser = argparse.ArgumentParser(description="Run the main and extended tabular tables")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--datasets", nargs="+", default=["adult", "german", "compas", "bank", "default"])
    parser.add_argument("--methods", nargs="+", default=["lsq", "pact", "mixedprec", "cfq"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--output", default="results/main")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-eval-examples", type=int, default=512)
    args = parser.parse_args()

    base = base_config(args.config)
    rows = []
    for dataset in args.datasets:
        for method in args.methods:
            for seed in args.seeds:
                config = clone_config(base)
                config.dataset = dataset
                config.method = method
                config.train.seed = seed
                config.quant.quantize_activations = method != "lsq"
                config.output_dir = str(Path(args.output) / dataset / method / f"seed_{seed}")
                result = run_tabular_experiment(
                    config,
                    max_samples=args.max_samples,
                    max_eval_examples=args.max_eval_examples,
                )
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "fp_accuracy": result["fp_accuracy"],
                    "bits": result["quantized_average_bits"],
                    **result["metrics"],
                })
                write_rows(rows, Path(args.output) / "summary.csv")


if __name__ == "__main__":
    main()
