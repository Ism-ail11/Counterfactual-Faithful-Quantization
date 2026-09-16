# CFQ code repair report

Source: https://github.com/Ism-ail11/When-Bits-Break-Recourse-Counterfactual-Faithful-Quantization

Reviewed base commit: `aa2b85ca05cff4cdbc54bf4c5f66cedc21587de2`.

This distribution contains the complete source project with corrections, regression
tests, installation instructions and an automated CPU test workflow. The original
license, citation, experiment families and configuration files are retained. The
GitHub repository was not modified.

## Confirmed issues corrected

1. **PTQ crashed while computing sensitivity.** Its teacher had already been
   frozen, so `loss.backward()` had no gradient graph. Sensitivity now temporarily
   enables the required weight derivatives, uses `autograd.grad`, and restores
   training flags and `requires_grad` without destroying existing gradients.
2. **Uniform PTQ was not uniformly quantized.** Allocation cleared the fixed
   bitwidth; an 8-bit weight run could use 4-bit activations. Uniform calibration
   now configures both, and runtime settings are persisted for reloads. Mixed PTQ
   preserves its allocated weight policies while calibrating scales/clips.
3. **Bank/Default labels could leak into model inputs.** Creating a target alias
   on OpenML's combined `.frame` left the original target among the predictors.
   Loaders now start with `.data` and add exactly one target column.
4. **Bank's target polarity and dataset field names were wrong.** OpenML 1461
   encodes no/yes as 1/2. The favorable class now maps to yes. Bank's `V1`–`V16`
   and Default's `x1`–`x23` are resolved to their documented names, restoring
   immutable features and subgroup reporting. Default's category codes use
   one-hot groups; its repayment statuses use ordinal domains. The Default YAML
   uses the documented L2 cost. References: [AIF360's Bank loader](https://github.com/Trusted-AI/AIF360/blob/master/aif360/sklearn/datasets/openml_datasets.py)
   and [OpenML's Default dataset](https://www.openml.org/d/42477).
5. **Numerical imputation leaked held-out information.** Medians now fit on
   training rows only, including handling completely missing columns. Missing
   categorical values become an explicit category. Subsampling is stratified
   and retains the source test proportion. Numeric float labels are normalized;
   unknown target semantics raise an error instead of silently reversing labels.
6. **Recourse discarded good intermediate solutions and moved already-favorable
   examples.** Solvers now compare the starting point and every iterate, retain
   the cheapest successful action found, enforce the requested target margin,
   and check feasibility. Zero-action solutions are preserved when feasible.
7. **Discrete projection could invalidate categories.** Partly immutable one-hot
   groups could contain two active entries; ordinal projection could select a
   value outside its box. Candidate selection now respects both constraints.
   Continuous proposals accumulate between hard projections, allowing small PGD
   steps to change a category or ordinal value. The zero-sparsity path preserves
   a valid differentiation graph.
8. **Robust/matching costs ignored custom mixed-cost weights.** The configured
   L1/L2 coefficients now carry through the solvers, metrics and diagnostics.
   Differentiable inner L2 objectives are smoothed at zero to avoid NaN second
   derivatives; ordinary evaluation still uses the exact norm.
9. **Several reported metrics were misleading.** Experiment accuracy now covers
   the full test split, independent of the recourse cap. Reference accuracy is
   recomputed after pruning/distillation. CRG excludes zero FP cost, and empty
   conditional aggregates remain undefined. JSON represents undefined values
   as `null` rather than emitting non-standard `NaN`/`Infinity` tokens.
10. **Configuration options were ignored.** Dropout and disabled mixed precision
    now take effect; LSQ consistently disables activation quantization. Missing
    explicit YAML files, misspelled keys, unsupported methods and invalid core
    settings produce errors. Quantization configuration reaches both parent
    models and their layers. Deterministic seeding now enables deterministic
    algorithms instead of disabling them.
11. **Model reloads were incomplete.** Runs now save quantization settings,
    exact preprocessed data/constraints and the fitted preprocessor. Reloads
    support FP32, preserve uniform/mixed precision, use saved data offline, and
    reject incompatible model configurations. Quantizer scales are initialized
    from copied full-precision weights. Stable inverse-softplus initialization
    avoids unnecessary overflow.
12. **Stress/theory edge cases crashed.** Sampling quantization variants now
    handles non-leaf budget-cache tensors. Empty theory subsets no longer reduce
    an empty tensor with `amax`; subgroup sampling handles zero-size draws and
    gives a clear error for an absent requested group. Constraint variants avoid
    modifying the original bounds.
13. **Image recourse could claim success outside its pixel budget.** Success now
    requires the budget, and the solver retains useful intermediate solutions.
    Empty target-excluded subsets are supported. Image FP32/LSQ dispatch,
    full-test accuracy, valid-only CF calibration, autoencoder evaluation mode,
    and saved runtime settings were corrected.
14. **Pruning was undone by QAT.** Pruned coordinates now remain zero during
    subsequent tabular/image QAT. Distillation history corresponds to the saved
    reference student. Noisy teacher actions have their validity rechecked.
15. **Validation and launch support were incomplete.** Added regression and
    end-to-end method tests, real lint checks, source/output ignore rules, a
    GitHub Actions CPU workflow, and setup/reload documentation. `run_all.py`
    resolves its project directory and includes theory diagnostics.

## Verification

The exact environment, test counts and completed script checks are recorded in
`VALIDATION_REPORT.json`. Validation includes the original tests plus regression
coverage, each supported tabular method on synthetic data, six image methods on
local image fixtures, actual tabular dataset downloads and reduced real-data
training/evaluation/reloads. The command-line families were also executed with
small synthetic configurations, including `run_all.py --fast`.

The final ZIP is checked for integrity and extracted into a clean directory for
its test run. It contains all original tracked project files and the new source,
tests and documentation. Generated datasets, model weights, caches, environments,
and Git history are excluded; the code downloads datasets and creates trained
weights when you run the experiments.

## Scope and remaining limits

- This is a tested correction pass, not a claim that every possible execution
  or future dependency combination is error-free. Only the reported environment
  was executed; CUDA, Windows, macOS and Docker execution were not tested.
- Real-data checks use small stratified samples and short training. They do not
  establish research accuracy, reproduce paper tables, or replace full runs.
  MNIST/Fashion download servers and full image experiments were not exercised;
  their training/calibration/evaluation code was tested with local 28x28 fixtures.
- The solvers remain heuristic. A failed search does not prove that no feasible
  action exists. Global minimum cost is not guaranteed, particularly with
  discrete features, sparsity and tight bounds.
- Old experiment results must be regenerated. The target-leakage fixes change
  feature dimensions and invalidate affected old checkpoints/results. Metric
  conventions and discrete search behavior also change comparisons.
- Quantizers simulate low precision using floating-point tensors. The project
  does not export packed integer kernels or demonstrate deployment speedups.
  The bit-budget training term is a penalty, not a hard guarantee that a trained
  QAT model meets the requested budget. Inspect the actual reported allocation.
- CelebA remains an interface requiring a supplied semantic editor, as in the
  original project. No editor, pretrained weights or paper results are invented.

Start with the setup commands and smoke test in `README.md`.
