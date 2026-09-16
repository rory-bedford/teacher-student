# Project Instructions — teacher-student

Synthetic teacher-student experiments on conductance-based spiking networks, organised
as the figures of the Bernstein talk. This is a **project repo that depends on the
`connectome-snns` library** (editable path dependency at `../connectome-snns`) for
network simulators, snn_runners, dataloaders, analysis, and visualization. It is not a
library itself.

## Python Environment

Always use `uv run python`, never bare `python`/`python3`. Import the library by its
namespace (`from connectome_snns.network_simulators...`,
`from connectome_snns.utils.reproducibility import ...`). Torch is needed for nearly
everything — sync with `--extra cu129` (GPU) or `--extra cpu`. `ruff` is not in the
environment; use `uvx ruff`.

## Repository Structure

```
METHODS.md  PRIORITY.md  TODO.md   # shared methods, run priority, deferred work — read first
generate-teacher-activity/         # makes the teacher network + spike data (the inputs)
common/                            # student, training, evaluation, plotting shared by all figures
fig01-full-reconstruction/  fig02-controls/  fig03-observed-fraction/
fig04-reconstruction-errors/  fig05-weight-noise/  fig06-learnt-feedforward/
archive/                           # previous experiments and figures — reference only
run                                # wrapper over the connectome-snns run framework
```

Each figure folder has `README.md`, `experiment.toml`, `parameters.toml`, `train.py`
(a thin wrapper around `common.training`), `run_grid_search.py`, `analysis.py` and
`figures.py`. What a figure trains is set entirely by `parameters.toml` — the
`[student]` table selects the manipulation (see `common/structure.py`).

`archive/` is not runnable as-is: its configs contain absolute paths to where the
scripts used to live and point at the old `dp-simulations/` tree.

## Data

**All data lives in `../bernstein`** (`/tachyon/groups/scratch/gzenke/bedfrory/bernstein`),
never in this repo and with no symlink to it. The teacher is `bernstein/teacher-activity/`
(its config is `generate-teacher-activity/experiment.toml`); each figure writes to
`bernstein/<figure-folder>/<run>/`. The old results in `dp-simulations/` belong to the
archive. W&B project: `bernstein`, one group per figure.

## Workflow for a figure

```bash
./run --grid fig01-full-reconstruction/experiment.toml     # train (commit first)
uv run python fig01-full-reconstruction/analysis.py         # held-out evaluation -> CSVs
uv run python fig01-full-reconstruction/figures.py          # CSVs -> figNN.svg
```

- Grid searches run from a git worktree snapshot, so code must be committed (`.toml` and
  `run_grid_search.py` changes are allowed dirty).
- `analysis.py` evaluates every completed run on a held-out teacher trial (cached per
  run as `evaluation.npz`) and writes only the CSVs the figure needs, next to itself.
  Figures 2–5 read Figure 1's runs as their baseline condition.
- `figures.py` reads only the CSVs. No notebooks.

## What you can and cannot run

**Exception (granted 2026-09-16):** Claude may launch and babysit `./run` and
`./run --grid` for the `fig*` folders when the user asks — including pilots — and run
their `analysis.py` / `figures.py` on finished runs. Report failures rather than editing
code mid-grid; code changes need the user's agreement. Anything else that writes to
`../bernstein` (teacher regeneration, deleting or overwriting runs) still needs explicit
permission.

**OK to run** for verification: import/syntax checks and tiny smoke tests that write
to the scratchpad (e.g. training on a few chunks of a sliced teacher).

## Results are read-only

`../bernstein` (and the archived `dp-simulations/`) hold provenance — param snapshot,
commit hash, symlinked inputs — and are read-only. Never edit files there; patch only
with explicit user confirmation.

## Conventions

- **No analysis in training scripts.** Training saves raw outputs only; metrics live in
  `analysis.py`, plotting in `figures.py`.
- **Naming:** observed / unobserved and reconstructed / unreconstructed. Never "hidden",
  "full inference" or "unreconstructed fraction" in names, titles or labels.
- **Imports at the top** (module level); capitalised constants right after.
- **All plotting via `visualization`** (from connectome-snns) — never hardcode
  colours. Fixed per-condition colours (`FULL_CONNECTOME_COLOR`, `OBSERVED_COLOR`, …)
  live in `connectome_snns.visualization.colors`.
- **No hardcoded data paths in analysis/figure scripts** — read them from
  `load_experiment_config("experiment.toml")`.
- After editing `.py`: `uvx ruff check --fix <file>` then `uvx ruff format <file>`.
- Commits: one-line gitmoji subject, no Co-Authored-By.

## Editing the library

`connectome-snns` is an editable dependency — edits to `../connectome-snns/src`
take effect immediately and affect every dependent project. Prefer additive
changes; flag breaking ones.
