# Hidden Units

Training with structural errors in the connectome: a fraction of recurrent neurons are completely removed from the student's model. Unlike the hidden-activity experiment (where neurons exist but are unobserved), here the student's weight matrix is physically smaller — it lacks all connections to and from hidden neurons.

## Method

A fraction of recurrent neurons are randomly designated as "hidden" and removed from the student. The student's weight matrix contains only visible-to-visible recurrent connections and feedforward-to-visible connections. Training targets are the visible neurons' spike trains only.

Scaling factors are optimised via gradient descent with Van Rossum loss. The question is how well they can be recovered despite the mis-specified model.

**Scripts:**
- `train.py` — training with the reduced (visible-only) weight matrix
- `inference/inference.py` — no training; runs inference with correct scaling factors to establish baseline performance at each hidden fraction

## Sub-experiments

| Config | Description |
|---|---|
| `increasing-hidden-fraction/experiment.toml` | Grid search over hidden fractions (0.05 to 0.50) |
| `convergence-check/experiment.toml` | 10 random seeds at fixed hidden fraction (0.1) |
| `stability-check/experiment.toml` | No perturbation (SF=1.0), tests stability at the optimum |
| `inference/experiment.toml` | Baseline inference with correct scaling factors, no training |

## Parameters

- `hidden_unit_fraction` — fraction of recurrent neurons removed from the student (grid search variable: 0.05–0.50)

## Usage

```bash
# Single run
./run experiments/teacher-student/hidden-units/increasing-hidden-fraction/experiment.toml

# Grid search over hidden fractions
./run --grid experiments/teacher-student/hidden-units/increasing-hidden-fraction/experiment.toml
```

## Analysis

`analysis.ipynb` (in `increasing-hidden-fraction/`) produces:
- Scaling factor trajectories and final values by hidden fraction
- Loss curves showing increasing difficulty with more hidden neurons
- Per-neuron firing rate scatter plots (student vs teacher)
- R² metrics for firing rate accuracy and temporal fluctuation fidelity
- Spike raster comparisons (teacher vs student interleaved)
