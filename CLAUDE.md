# Project Instructions — teacher-student

Synthetic teacher-student experiments on conductance-based spiking networks. This
is a **project repo that depends on the `connectome-snns` library** (editable path
dependency at `../connectome-snns`) for network simulators, snn_runners,
dataloaders, analysis, and visualization. It is not a library itself.

## Python Environment

Always use `uv run python`, never bare `python`/`python3`. Experiments import from
the library with flat names (`from network_simulators...`, `from snn_runners...`,
`from utils.reproducibility import ...`, `from visualization import ...`). Torch is
needed for nearly everything — sync with `--extra cu129` (GPU) or `--extra cpu`.

## Repository Structure

```
generate-teacher-activity/   # makes the teacher network + spike data (inputs)
fully-observed/  noisy-weights/  hidden-activity/  hidden-units/
inferring-inputs/  full-inference/  dimensionality-reduction/
runs                         # symlink -> dp-simulations (results; read-only)
run                          # wrapper over the connectome-snns run framework
```

Sub-areas nest further (e.g. `noisy-weights/convergence-check/`). Each leaf has
`experiment.toml`, `parameters.toml`, a script (`train.py` / `compute_*.py`),
`analysis.ipynb`, and often `run_grid_search.py`.

## Running Experiments

```bash
./run noisy-weights/convergence-check/experiment.toml          # single run
./run --grid noisy-weights/convergence-check/experiment.toml   # grid search
./run --resume runs/<dir> <config.toml>                        # resume
./run --no-commit <config.toml>                                # skip git check (dev)
```

The framework reads `experiment.toml`, makes a timestamped dir under `output_dir`,
snapshots params + commit hash, symlinks inputs, and calls the script's
`main(input_dir, output_dir, params_file, wandb_config=None, resume_from=None)`.
Scripts never construct their own paths.

## What you can and cannot run

**Never run** experiments or anything that writes outputs: `./run`, training
scripts, grid searches. The reproducibility framework runs these.

**OK to run** for verification: small import checks, syntax checks, tiny smoke
tests.

## Results are read-only

Outputs live in the `dp-simulations/` tree (surfaced via `runs/`). It holds
provenance (param snapshot, commit hash, symlinked inputs) and is read-only —
never edit files under `runs/` / `dp-simulations/`. Patch only with explicit user
confirmation.

## Conventions

- **No analysis in scripts.** Training/compute scripts save raw outputs only;
  metrics/plots live in `analysis.ipynb`.
- **Imports at the top** (first notebook cell; module level in scripts);
  capitalised constants right after.
- **All plotting via `visualization`** (from connectome-snns) — never hardcode
  colours.
- **No hardcoded paths in notebooks** — read paths from
  `load_experiment_config("experiment.toml")`.
- After editing `.py`/notebooks: `uv run ruff check --fix <file>` then
  `uv run ruff format <file>`.
- Commits: one-line gitmoji subject, no Co-Authored-By.

## Editing the library

`connectome-snns` is an editable dependency — edits to `../connectome-snns/src`
take effect immediately and affect every dependent project. Prefer additive
changes; flag breaking ones.
