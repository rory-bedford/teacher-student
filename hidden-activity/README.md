# Hidden Activity

Training a student network when some neurons are hidden (unobserved). Only visible neurons can be compared to the teacher. The goal is to recover the correct connectivity scaling factors despite missing observations.

## Strategies

- **`em-algorithm`** — E-step clamps visible neurons to teacher spikes and infers hidden activity. M-step updates scaling factors via surrogate gradient backprop. Loss on visible neurons only.
- **`fully-recurrent`** — Full recurrent network simulated in one forward pass; scaling factors optimised by backpropagating through recurrent dynamics. No E-step.
- **`visible-driven`** — Two-layer architecture: recurrent hidden layer + feedforward visible layer, with teacher visible spikes injected as feedforward input. Gradients flow from visible loss to FF→hidden weights. Loss on visible neurons.
- **`ignore-hidden`** — Feedforward model on visible neurons only; hidden neurons absent from the model entirely.

Training scripts live in `scripts/` and are shared across sub-experiments. Each sub-experiment folder contains only config (`.toml`) and parameter files.

## Sub-experiments

| Folder | Description |
|---|---|
| `testing-strategies/` | Main benchmark: 50% hidden neurons, perturbed scaling factors — tests whether each strategy can recover the correct values |
| `stability-check/` | Scaling factors start at the correct values (no perturbation) — tests whether training stays stable at the optimum |
| `increasing-hidden-fraction/` | Grid search over hidden fraction (0.1–0.9) with clamped E-step |
| `convergence-check/` | Same setup as testing-strategies but with multiple random seeds — tests convergence reliability |

## Parameters

- `hidden_cell_fraction` — fraction of neurons hidden from the student (grid search variable: 0.1–0.9)
- `em.total_iterations` — number of E-step/M-step cycles (default: 50)
- `em.epochs_per_update` — M-step gradient epochs per EM iteration (default: 1)

## Usage

```bash
# Run a single strategy
./run experiments/teacher-student/hidden-activity/testing-strategies/em-algorithm.toml

# Run the hidden-fraction grid search
./run --grid experiments/teacher-student/hidden-activity/increasing-hidden-fraction/experiment.toml
```

## Analysis

Each sub-experiment has its own `analysis.ipynb` with scaling factor trajectories, convergence plots, firing rate comparisons, and spike rasters.
