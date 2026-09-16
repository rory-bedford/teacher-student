# teacher-student

Synthetic teacher-student experiments on conductance-based spiking networks, organised as
the figures of the Bernstein talk. A teacher network generates activity; a student with a
(partially) reconstructed connectome is trained on part of that activity and asked to
predict the rest on held-out stimuli.

This repo holds the **experiment code**; it consumes the **connectome-snns** library
(simulators, run framework, dataloaders, analysis, visualization) as an editable dependency.
Data is **not** stored here — it lives in `../bernstein`.

Read `METHODS.md` first, then `PRIORITY.md`.

| Dir | What it covers |
|---|---|
| `generate-teacher-activity/` | The teacher network + spike data → `bernstein/teacher-activity/` |
| `common/` | Student construction, training, held-out evaluation and plotting shared by all figures |
| `fig01-full-reconstruction/` | Full reconstruction, 10% observed (the baseline for Figs 2–5) |
| `fig02-controls/` | Learnt recurrence, shuffled weights, configuration-model rewire |
| `fig03-observed-fraction/` | Observed-fraction sweep, 50% → 0.5% |
| `fig04-reconstruction-errors/` | Neuron removal vs synapse dropout |
| `fig05-weight-noise/` | Weight-noise sweep |
| `fig06-learnt-feedforward/` | Unreconstructed inputs with learnt weights, reconstructed fraction 100% → 10% |
| `archive/` | Previous experiments and figures (reference only, not runnable as-is) |

## Setup

```bash
uv sync --extra cu129   # GPU
uv sync --extra cpu     # CPU
```

## Running a figure

```bash
./run --grid fig01-full-reconstruction/experiment.toml   # train (code must be committed)
uv run python fig01-full-reconstruction/analysis.py       # held-out evaluation -> CSVs
uv run python fig01-full-reconstruction/figures.py        # CSVs -> fig01.svg
```

Each figure's `README.md` records its configuration, the archived settings it reuses, and
where it departs from them.
