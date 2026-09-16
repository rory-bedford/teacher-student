# Noisy Weights

Training with noisy or uninformative synaptic weights. Tests whether the student can recover correct scaling factors when individual synaptic weights are perturbed.

## Method

Cell-type-pair specific weight noise is applied: for each connection type (e.g. exc-to-exc, inh-to-exc), Gaussian noise proportional to each weight's magnitude is added, then rescaled to preserve the mean and standard deviation of that connection type. This maintains E/I balance while perturbing individual synaptic weights.

Scaling factors are optimised via gradient descent with Van Rossum loss.

**Scripts:**
- `*/train.py` — training with noisy weights for each sub-experiment
- `inference/inference.py` — no training; runs inference with correct scaling factors to establish baseline performance at each noise level

## Sub-experiments

| Config | Description |
|---|---|
| `varying-noise/experiment.toml` | Grid search over noise levels 0.05–0.50 |
| `convergence-check/experiment.toml` | 10 random seeds at noise=0.4 |
| `inference/experiment.toml` | Baseline with correct scaling factors (SF=1.0), no training |
| `shuffled-control/experiment.toml` | Control with shuffled weights |
| `stability-check/experiment.toml` | No perturbation, tests stability at the optimum |

## Parameters

- `noise_level` — fraction of each weight's magnitude used as noise standard deviation (grid search variable: 0.05–0.50)

## Usage

```bash
# Single run
./run experiments/teacher-student/noisy-weights/varying-noise/experiment.toml

# Grid search over noise levels
./run --grid experiments/teacher-student/noisy-weights/varying-noise/experiment.toml
```

## Analysis

`analysis.ipynb` produces convergence plots, scaling factor recovery by noise level, spike train comparisons, and summary statistics.
