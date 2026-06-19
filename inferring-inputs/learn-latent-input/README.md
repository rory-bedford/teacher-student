# Learn Latent Input

Learn a linear mapping from continuous OU latent trajectories to neurons, while simultaneously recovering perturbed recurrent scaling factors.

## Method

Instead of sampling spikes from the OU latent process (as in ou-rates), this experiment feeds the raw continuous OU mixing weights directly as feedforward input. The network learns feedforward weights (20 latents → 5000 neurons) while also recovering the 2×2 recurrent scaling factor matrix.

**Phase 1 (CMA-ES):** Gradient-free search over 6 parameters — 4 recurrent scaling factors (2×2 matrix) and 2 FF weight constants (one per output cell type: latent→excitatory, latent→inhibitory). All parameters are searched in log-space.

**Phase 2 (Gradient):** Adam optimisation of both feedforward weights and recurrent scaling factors simultaneously, starting from the CMA-ES solution. Separate learning rates and cosine annealing per parameter group.

## Parameters

- `lr_weights` — learning rate for feedforward weight parameters (default: 1e-3)
- `lr_scaling` — learning rate for recurrent scaling factor parameters (default: 1e-2)
- `recurrent_smoothing_tau` — temporal smoothing kernel for recurrent spike inputs (ms)
- `training.optimisable` — dict with per-group modes: `{feedforward = "weights", recurrent = "scaling_factors"}`

## Usage

```bash
./run experiments/teacher-student/inferring-inputs/learn-latent-input/experiment.toml
```

Same data dependencies as ou-rates (teacher-activity network_structure.npz and spike_data.zarr).

## Analysis

Planned: comparison of learned latent→neuron weights against true teacher FF weights projected onto OU basis, scaling factor recovery, firing rate matching.
