# Learn FF Connectivity

Learn the feedforward connectivity matrix from scratch using actual mitral cell spike trains as input, with a low-rank constraint on the learned FF weight matrix.

## Method

Unlike `learn-latent-input` which maps continuous OU latent trajectories to recurrent neurons, this experiment uses the actual mitral cell feedforward spike trains as input. The FF weight matrix is learned from scratch (no known FF weights) and constrained to be low-rank via a U @ V parametrisation during Phase 3. The same 3-phase training pipeline is used: CMA-ES for coarse search, gradient on scaling factors only, then gradient on low-rank FF weights + recurrent scaling factors.

## Parameters

| Parameter | Description |
|---|---|
| `ff_rank` | Rank of the low-rank decomposition for the FF weight matrix (default: 10) |
| `weight_perturbation_variance` | Variance for log-normal perturbation of recurrent scaling factors |

## Usage

```bash
./run experiments/teacher-student/inferring-inputs/learn-ff-connectivity/experiment.toml
```

## Analysis

Key metrics tracked during training:
- Correlation between learned and teacher FF weights
- Effective rank of the FF weight matrix (SVD participation ratio)
- Feedforward vs recurrent drive fractions
- Per-cell-type firing rates and scaling factor recovery
