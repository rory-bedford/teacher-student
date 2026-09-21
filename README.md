# teacher-student

Synthetic teacher-student experiments on conductance-based spiking networks, organised as
the figures of the Bernstein talk. A teacher network generates activity; a student with a
(partially) reconstructed connectome is trained on part of that activity and asked to
predict the rest on held-out stimuli.

This repo holds the **experiment code**; it consumes the **connectome-snns** library
(simulators, run framework, dataloaders, analysis, visualization) as an editable dependency.
Data is **not** stored here — it lives in `../bernstein`.

Read [`METHODS.md`](METHODS.md) first: it defines the model, the teacher forcing, the
metrics and the naming used everywhere. [`COLORSCHEME.txt`](COLORSCHEME.txt) defines what
each colour means across the talk.

| Dir | What it covers |
|---|---|
| `generate-teacher-activity/` | The teacher network + spike data → `bernstein/teacher-activity/` |
| `common/` | Student construction, training, held-out evaluation, perturbation and plotting shared by all figures |
| `fig01-full-reconstruction/` | Full reconstruction, 50% observed (the baseline for Figs 2–5) |
| `fig02-controls/` | Learnt recurrence, shuffled weights, shuffled topology |
| `fig03-observed-fraction/` | Observed-fraction sweep, 50% → 1% |
| `fig04-reconstruction-errors/` | Neuron removal vs synapse dropout |
| `fig05-weight-noise/` | Weight-noise sweep, 0 → 0.5 |
| `fig06-learnt-feedforward/` | Unreconstructed inputs with learnt weights, reconstructed fraction 100% → 10% |
| `slurm/` | Cluster submission: code-snapshot arrays, diagnostic probes, run status |

## Setup

```bash
uv sync --extra cu129   # GPU
uv sync --extra cpu     # CPU
```

## Running a figure

```bash
./run --grid fig01-full-reconstruction/experiment.toml   # train (code must be committed)
uv run python fig01-full-reconstruction/analysis.py       # held-out evaluation -> CSVs
uv run python fig01-full-reconstruction/figures.py        # CSVs -> one SVG per panel
```

`analysis.py` evaluates every completed run on a held-out teacher trial and on the
perturbation, caching both per run, and writes only the CSVs its figure needs.
`figures.py` reads those CSVs and nothing else. Figures 2–5 read Figure 1's runs as their
baseline condition.

Each figure's `README.md` records its claim, its configuration, its evaluation protocol,
its panels and the training recipe as run.
