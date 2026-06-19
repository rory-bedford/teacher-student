# teacher-student

Synthetic teacher-student experiments on conductance-based spiking networks: a
teacher network generates activity, and a student is trained to recover the
teacher's perturbed parameters (scaling factors, weights, hidden activity, latent
inputs) under varying observability. Used to validate the inference machinery
before applying it to real connectome data.

This repo holds the **experiment code**; it consumes the **connectome-snns**
library (network simulators, snn_runners, dataloaders, analysis, visualization)
as an editable dependency. Outputs are **not** stored here — they live in the
`dp-simulations/` tree and are surfaced via a `runs` symlink for convenience.

## Sub-areas

| Dir | What it covers |
|---|---|
| `generate-teacher-activity/` | Produce the teacher network + spike data (inputs the students consume). |
| `fully-observed/` | Recover scaling factors with all neurons observed, no hidden units. |
| `noisy-weights/` | Recover perturbed weights; convergence/stability/shuffle controls. |
| `hidden-activity/` | Inference with unobserved (hidden) activity. |
| `hidden-units/` | Inference with hidden units; convergence/stability/inference. |
| `inferring-inputs/` | Recover latent / feedforward inputs (OU rates, uniform inputs). |
| `full-inference/` | Joint inference (hidden units + connectivity), with controls. |
| `dimensionality-reduction/` | PCA / log-PCA / NMF of teacher & student activity. |

## Setup

Almost everything trains spiking networks, so torch is required — sync with the
extra matching the machine:

```bash
uv sync --extra cu129   # GPU
uv sync --extra cpu     # CPU
```

`connectome-snns` is an editable path dependency (`../connectome-snns`); pin it to
a git rev in `pyproject.toml` for archival reproducibility.

## Running

```bash
./run noisy-weights/convergence-check/experiment.toml          # single run
./run --grid noisy-weights/convergence-check/experiment.toml   # grid search
```

The library's run framework reads the `experiment.toml`, makes a timestamped dir
under `output_dir` (in `dp-simulations/`), snapshots params + commit, symlinks
inputs, and calls the script's `main()`.

## Outputs

Results live in `dp-simulations/` (325 GB of training runs / checkpoints,
read-only provenance) — browse them via `./runs/`. Each experiment's
`analysis.ipynb` reads its run via `load_experiment_config("experiment.toml")`.
