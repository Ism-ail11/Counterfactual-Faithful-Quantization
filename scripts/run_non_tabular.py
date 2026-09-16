from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

if __package__ in {None, ""}:
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from cfq.experiments.image import run_image_experiment
from cfq.reporting import write_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run latent MNIST/Fashion-MNIST recourse experiments")
    parser.add_argument("--datasets", nargs="+", default=["mnist", "fashion-mnist"])
    parser.add_argument("--methods", nargs="+", default=["ptq8", "ptq4", "mixedprec", "prune_quant", "cfptq", "cfq"])
    parser.add_argument("--output", default="results/non_tabular")
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--max-test", type=int, default=1000)
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--device")
    args = parser.parse_args()
    rows = []
    for dataset in args.datasets:
        for method in args.methods:
            output = Path(args.output) / dataset / method
            result = run_image_experiment(
                dataset=dataset,
                method=method,
                output_dir=output,
                target_class=args.target_class,
                max_train=args.max_train,
                max_test=args.max_test,
                device=args.device,
            )
            rows.append({"dataset": dataset, "method": method, **result["metrics"]})
            write_rows(rows, Path(args.output) / "summary.csv")


if __name__ == "__main__":
    main()
