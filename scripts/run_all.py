from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

if __package__ in {None, ""}:
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every paper experiment family")
    parser.add_argument("--fast", action="store_true", help="Use synthetic/small subsets for code validation")
    args = parser.parse_args()
    python = sys.executable
    if args.fast:
        commands = [
            [python, "-m", "cfq.cli", "smoke", "--output-dir", "results/smoke"],
            [python, "scripts/run_ablation.py", "--config", "configs/fast.yaml", "--dataset", "synthetic", "--max-samples", "600", "--max-eval-examples", "48"],
            [python, "scripts/run_teacher_quality.py", "--config", "configs/fast.yaml", "--dataset", "synthetic", "--max-samples", "600", "--max-eval-examples", "48"],
        ]
    else:
        commands = [
            [python, "scripts/run_main.py"],
            [python, "scripts/run_ablation.py"],
            [python, "scripts/run_budget_curves.py"],
            [python, "scripts/run_backbones_costs.py"],
            [python, "scripts/run_more_baselines.py"],
            [python, "scripts/run_cfptq.py"],
            [python, "scripts/run_teacher_quality.py"],
            [python, "scripts/run_runtime.py"],
            [python, "scripts/run_stress_tests.py"],
            [python, "scripts/run_theory_diagnostics.py"],
            [python, "scripts/run_non_tabular.py"],
        ]
    for command in commands:
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=_BootstrapPath(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
