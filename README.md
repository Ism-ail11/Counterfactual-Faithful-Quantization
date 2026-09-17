# Counterfactual-Faithful Quantization (CFQ)

A clean PyTorch reference implementation for the paper **“When Bits Break Recourse: Counterfactual-Faithful Quantization.”** The repository implements the method, metrics, actionable projections, quantizers, mixed-precision allocation, post-training variant, baselines, and experiment families described in the main text and appendix.


## Implemented components

- Validity Drop (VD), Counterfactual Recourse Gap (CRG), direction similarity, action overlap, feasible recourse rate, and target-margin diagnostics.
- Action sets with immutable features, box constraints, top-k sparsity, one-hot projection, ordinal projection, and straight-through gradients.
- Training and evaluation PGD recourse solvers, a binary linear L2 sanity-check solver, and a robust solver over quantization variants.
- LSQ-style learned weight steps, PACT-style activation clipping, hard/soft Gumbel mixed precision, bit-specific quantizer parameters, and budget penalties.
- CFQ-QAT, uniform QAT, accuracy-centric mixed precision, CF-PTQ, counterfactual sensitivity allocation, R-Margin, R-Consistency, pruning+quantization, and distillation+quantization.
- Adult, German Credit, COMPAS, Bank Marketing, Default of Credit Card Clients, and synthetic tabular loaders.
- MNIST and Fashion-MNIST latent-recourse experiments with a convolutional autoencoder.
- Constraint tightness, distribution shift, subgroup reporting, teacher quality/noise, budget curves, runtime, robust-solver, and theory diagnostics.


## Metric conventions

- **VD:** target failure under the quantized model among examples for which FP recourse was feasible.
- **CRG:** relative cost change on examples for which both FP and quantized recourse were feasible.
- **Infeasible recourse:** reported separately through FP/Q feasible rates rather than hidden inside an arbitrary cost penalty.


## Setup

Use Python 3.10 or newer. The local verification for this corrected distribution
used Python 3.12 on Linux with CPU PyTorch.

From the extracted project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pytest
python -m cfq.cli smoke --output-dir results/smoke
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
For CUDA, install a compatible PyTorch/torchvision CUDA pair for your system in
place of the CPU installation command. CUDA execution was not tested in this
repair environment. For small CPU runs, setting `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1` can reduce thread overhead.

## Run experiments

The smoke test and synthetic experiments do not download datasets.

```bash
python -m cfq.cli tabular --config configs/fast.yaml --dataset synthetic --method cfq --max-samples 600 --max-eval-examples 48 --output-dir results/example
python scripts/run_all.py --fast
python scripts/run_cfptq.py --config configs/fast.yaml --datasets synthetic --max-samples 600 --max-eval-examples 48
```

Real tabular experiments download data from OpenML or the COMPAS source. Image
experiments download MNIST/Fashion-MNIST through torchvision. These runs require
network access and have substantially longer training times. Downloaded data is
not part of the source archive. OpenML datasets use version 1 and the selected
cache directory. `docs/EXPERIMENT_MAP.md` lists the available script families.

The `cfq tabular` entry point works after installation. Without `--config`, it
uses the dataclass defaults. An explicitly supplied missing configuration file,
unknown method, or misspelled configuration key fails with an error.

Supported tabular methods are `fp32`, `lsq`, `pact`, `mixedprec`, `cfq`,
`cfq_uniform`, `cfq_match`, `r_margin`, `r_consistency`, `prune_quant`, `kd_quant`,
`ptq4`, `ptq8`, `mixedptq`, `cfptq`, and `cfptq_sensitivity`. Set
`train.match_alpha1` and/or `train.match_alpha2` above zero to enable a nonzero
matching penalty with `cfq_match`; the ablation script provides an example.

## Outputs and reloads

Each tabular run writes:

- `metrics.json` and `config.json`: measured results and resolved settings.
- `fp_model.pt`, `quantized_model.pt`, and `recourse.pt`: weights and evaluated actions.
- `quantization.json`: runtime bitwidth and activation settings, which are not
  included in a plain PyTorch state dictionary.
- `dataset.pt`: the exact preprocessed splits, labels, feature names, weights,
  and action constraints used by the run.
- `preprocessor.joblib`: fitted preprocessing for transforming new raw rows.

The stress/theory scripts use `scripts._load.load_models` to restore saved data
and quantization settings. To inspect a run interactively from the project root:

```python
import json
from pathlib import Path
from cfq.cli import config_from_mapping
from scripts._load import load_models

run = Path("results/example")
config = config_from_mapping(json.loads((run / "config.json").read_text()))
bundle, fp_model, quantized_model = load_models(config, run)
```

Image runs also save the autoencoder, run configuration and quantization settings.
The `load_models` helper is for tabular runs. Construct `SmallCNN`,
`QuantSmallCNN` (or `SmallCNN` for the `fp32` comparison), and `ConvAutoencoder`
for image checkpoints; restore the quantization settings with
`cfq.checkpoints.load_quantization_settings` after loading weights.

## Evaluation details

- Classification accuracy in an experiment result uses the entire test split;
  `max_eval_examples` limits the recourse calculation only.
- The recourse solvers retain the cheapest successful iterate they find,
  including zero action when the starting example already satisfies the target.
  Success also requires action feasibility and any configured target margin.
- Categorical/ordinal searches accumulate continuous updates, evaluate hard
  projected actions, and retain only actions satisfying the constraints as
  successful results. This remains a heuristic search, not a guarantee of a
  global minimum or a complete infeasibility proof.
- CRG additionally requires positive FP action cost, because relative cost
  change is undefined at zero. Undefined aggregates are `null` in JSON and NaN
  in the in-memory metrics/CSV output. Empty conditioning sets do not imply zero
  validity drop.
- The reported bit budget is parameter-count-weighted **weight precision**.
  Activation policies remain separate; uniform PTQ uses the requested bitwidth
  for both weights and activations. These are floating-point fake-quantization
  models, not packed integer inference binaries or measured hardware speedups.
- Real experiment results must be regenerated after the corrections. Passing
  code tests does not reproduce or verify the paper's numerical conclusions.
