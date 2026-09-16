# Inferring Inputs

Recovering perturbed synaptic scaling factors when the true feedforward input is unknown. The student receives exact recurrent spike trains from the teacher, but the feedforward signal must be inferred or replaced with a proxy.

## Method

Two strategies are compared for handling the missing input signal:

- **OU-rates** — feedforward neurons emit Poisson spikes sampled from saved OU process rates, capturing structured temporal dynamics. An estimate of the input statistics, but not the exact spike times.
- **Uniform inputs** — feedforward neurons emit Poisson spikes at a constant baseline rate (6 Hz). The most degenerate case: no structured input at all.

Both strategies use gradient descent to recover 6 cell-type-specific scaling factors.

## Sub-experiments

| Config | Description |
|---|---|
| `ou-rates/experiment.toml` | OU-rate inputs, grid search over recurrent smoothing tau |
| `uniform-inputs/experiment.toml` | Constant-rate Poisson inputs |
| `learn-latent-input/experiment.toml` | Learn linear mapping from raw OU latents + recover scaling factors |

The OU-rates strategy includes a grid search over `recurrent_smoothing_tau` (50, 100, 200, 500, 1000, 2000 ms) to test sensitivity to temporal smoothing assumptions.

## Parameters

- `recurrent_smoothing_tau` — temporal smoothing kernel applied to recurrent spike inputs (OU-rates only, grid search: 50–2000 ms)
- `firing_rate_override` — constant Poisson rate replacing structured input (uniform-inputs only, default: 6.0 Hz)

## Usage

```bash
./run experiments/teacher-student/inferring-inputs/ou-rates/experiment.toml
./run experiments/teacher-student/inferring-inputs/uniform-inputs/experiment.toml
```

## Analysis

`analysis.ipynb` (top-level) compares both strategies:
- Scaling factor recovery (bar charts vs target)
- Firing rate scatter plots by cell type
- Spike raster comparisons
- R² for activity and temporal fluctuations
- Feedforward input comparison (OU trajectories vs constant rate)

`ou-rates/analysis.ipynb` analyses the smoothing tau grid search:
- Loss curves and scaling factor trajectories across tau values
- R² vs tau (log scale)
- Generalisation test: do recovered scaling factors work with exact (unsmoothed) inputs?
