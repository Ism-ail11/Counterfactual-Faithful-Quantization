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
    parser = argparse.ArgumentParser(description="Measure CFQ overhead as teacher K changes")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dataset", default="adult")
    parser.add_argument("--output", default="results/runtime")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-eval-examples", type=int, default=128)
    args = parser.parse_args()
    base = base_config(args.config)
    rows = []
    for method, steps in [("mixedprec", 0), ("cfq", 1), ("cfq", 2), ("cfq", 3)]:
        config = clone_config(base)
        config.dataset = args.dataset
        config.method = method
        if steps:
            config.recourse.train_steps = steps
        label = "qat" if method == "mixedprec" else f"cfq_k{steps}"
        config.output_dir = str(Path(args.output) / label)
        result = run_tabular_experiment(config, max_samples=args.max_samples, max_eval_examples=args.max_eval_examples)
        elapsed = result["history"]["quantized"]["elapsed_seconds"]
        rows.append({"variant": label, "K": steps, "elapsed_seconds": elapsed, **result["metrics"]})
        baseline = rows[0]["elapsed_seconds"]
        for row in rows:
            row["relative_time"] = row["elapsed_seconds"] / max(baseline, 1e-12)
        write_rows(rows, Path(args.output) / "summary.csv")


if __name__ == "__main__":
    main()
