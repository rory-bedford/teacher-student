# Dimensionality Reduction

Low-dimensional analysis of population activity. Compresses the high-dimensional feedforward spike trains (1,500 mitral cells) into compact representations useful for downstream analyses and visualisation.

## Method

Binary spike trains are Gaussian-smoothed along the time axis to obtain continuous firing rate estimates, then reduced via PCA or NMF. The fitted model is applied to each trial independently, producing low-dimensional timeseries that capture the dominant patterns of input variability.

## Sub-experiments

| Config | Description |
|---|---|
| `compute-pcas/experiment.toml` | PCA on smoothed spike trains |
| `compute-log-pcas/experiment.toml` | PCA on log-transformed smoothed spike trains |
| `compute-nmfs/experiment.toml` | Non-negative matrix factorisation on smoothed spike trains |

## Parameters

- `gaussian_sigma_ms` — smoothing kernel width in ms
- `n_components` — number of components to retain

## Usage

```bash
./run experiments/teacher-student/dimensionality-reduction/compute-pcas/experiment.toml
```

## Analysis

Outputs are saved as zarr arrays (`pc_timeseries`, `components`, `mean`, `explained_variance_ratio`) for use by other experiments.
