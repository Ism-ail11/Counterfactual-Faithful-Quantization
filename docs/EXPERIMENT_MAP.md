# Paper-to-code experiment map

| Paper component | Executable entry point | Main outputs |
|---|---|---|
| Main Adult/German/COMPAS results | `scripts/run_main.py` | `results/main/summary.csv` |
| Extended Bank/Default results | `scripts/run_main.py` | same CSV |
| LSQ-QAT and PACT-QAT | `scripts/run_main.py` | per-run `metrics.json` |
| Mixed-precision baseline and CFQ | `scripts/run_main.py` | metrics, checkpoints, bit allocations |
| Main ablations: no CF loss, uniform bits, K=1/K=3 | `scripts/run_ablation.py` | `summary.csv` |
| Hinge and student matching add-ons | `scripts/run_ablation.py` | `summary.csv` |
| Budget/VD/CRG curves | `scripts/run_budget_curves.py` | `summary.csv` |
| Logistic/shallow/deep backbone robustness | `scripts/run_backbones_costs.py` | `summary.csv` |
| Weighted L1/L2/mixed cost sensitivity | `scripts/run_backbones_costs.py` | `summary.csv` |
| R-Margin and R-Consistency | `scripts/run_more_baselines.py` | `summary.csv` |
| Prune+Quant and KD+Quant | `scripts/run_more_baselines.py` | `summary.csv` |
| CF-PTQ and sensitivity allocation | `scripts/run_cfptq.py` | `summary.csv` |
| Low-K and noisy teacher diagnostics | `scripts/run_teacher_quality.py` | `summary.csv` |
| Runtime overhead versus K | `scripts/run_runtime.py` | `summary.csv` |
| Restrictive/moderate/permissive constraints | `scripts/run_stress_tests.py` | `constraint_sensitivity.csv` |
| Feature, group, and target imbalance shifts | `scripts/run_stress_tests.py` | `distribution_shift.csv` |
| Subgroup VD/CRG | `scripts/run_stress_tests.py` | `subgroups.csv` |
| Robust recourse under quantization variants | `scripts/run_stress_tests.py` | `robust_solver.csv` |
| Empirical epsilon and margin condition | `scripts/run_theory_diagnostics.py` | `theory_diagnostics.json` |
| Counterfactual layer sensitivity/bit errors | `scripts/run_theory_diagnostics.py` | same JSON |
| MNIST/Fashion latent recourse | `scripts/run_non_tabular.py` | `summary.csv`, autoencoder/model checkpoints |
