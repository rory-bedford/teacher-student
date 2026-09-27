# Connectome-constrained spiking network models of functional activity

Code and figures for my talk at the Bernstein Conference 2026 satellite workshop *Advances
in optimization of biologically constrained models and how to use them* (28 September 2026).

Dynamical connectomics pairs a circuit's wiring diagram with recordings of its activity,
which makes it possible to fit highly constrained mechanistic models to data. Such fits are
otherwise notoriously degenerate: many parameter sets reproduce the same activity. The talk
asks whether the connectome resolves that degeneracy when only part of the circuit is
recorded, and only part of it is reconstructed — and how much of that survives the errors a
real electron-microscopy reconstruction brings.

## What we did

We use a teacher-student paradigm on a conductance-based spiking network modelled on
zebrafish Dp: 5000 excitatory and inhibitory neurons driven by 1500 odour-responsive mitral
cells. The **teacher** generates activity in response to odours. The **student** has the same
neuron model and is given the connectome, so the only things it has to learn from activity
are six scaling factors, one per pair of cell types. It is trained with surrogate gradients
to reproduce the spikes of a fraction of the neurons (the *observed* neurons), and then
tested on odour stimuli it has never seen, on:

- the neurons it was trained on, and — the real test — the **unobserved** neurons it never saw;
- its response to a **perturbation**: silencing part of the inhibitory population, as in an
  optogenetic experiment.

Every score is shown against a noise ceiling, the score of a perfectly specified student,
since the teacher is chaotic and not even the true model tracks it spike for spike.

We then degrade the student the way real data would — fewer recorded neurons, reconstruction
errors, noisy synaptic weights, an incomplete reconstruction, a mismatched neuron model — and
compare it with controls that are given no connectome, or a scrambled one. Shared methods
are described in [`METHODS.md`](METHODS.md), and each figure's folder has a README with its
configuration and results.

## Experiments

| Figure | Folder | What it tests |
|---|---|---|
| 0 | [`fig00-teacher-activity`](fig00-teacher-activity) | The teacher network and its odour-evoked activity: the data every other figure fits |
| 1 | [`fig01-full-reconstruction`](fig01-full-reconstruction) | Full connectome, 50% of neurons observed: the student predicts the unobserved half. The baseline for everything else |
| 2 | [`fig02-controls`](fig02-controls) | Controls: the true topology with every weight learnt, no connectome at all (25 M free weights), shuffled weights, shuffled topology |
| 3 | [`fig03-observed-fraction`](fig03-observed-fraction) | Fewer recorded neurons: 50% down to 1% observed |
| 4 | [`fig04-reconstruction-errors`](fig04-reconstruction-errors) | Reconstruction errors: missing neurons versus missing synapses, up to half the recurrent input lost |
| 5 | [`fig05-weight-noise`](fig05-weight-noise) | Imprecise synaptic weights: multiplicative weight noise from 0 to 0.8 |
| 6 | [`fig06-learnt-feedforward`](fig06-learnt-feedforward) | The recurrent connectome given but the feedforward (mitral) weights learnt |
| 7 | [`fig07-partial-reconstruction`](fig07-partial-reconstruction) | Incomplete reconstruction: 100% down to 10% of the circuit reconstructed, the rest reaching it through learnt weights |
| 8 | [`fig08-model-mismatch`](fig08-model-mismatch) | A mismatched neuron model: student spike thresholds scattered by 1–4 mV around the teacher's |

Shared code for the student, training, evaluation and plotting is in [`common/`](common).

## Running the code

### Installation

You need [uv](https://docs.astral.sh/uv/) and, realistically, a GPU: one training run takes
about 3.6 hours on a Quadro RTX 5000, and each figure is a grid of runs over three seeds.

```bash
git clone https://github.com/rory-bedford/teacher-student.git
cd teacher-student
uv sync --extra cu129     # NVIDIA GPU (CUDA 12.9)
uv sync --extra cpu       # or CPU only
```

The simulators, training loop, data loading, run framework and plotting all live in a
separate library, [connectome-snns](https://github.com/rory-bedford/connectome-snns), which
`uv sync` installs from GitHub at the commit these experiments were run with (see
`pyproject.toml`). To work on the library alongside the experiments, clone it next to this
repository and point `pyproject.toml` at it instead:

```toml
[tool.uv.sources]
connectome-snns = { path = "../connectome-snns", editable = true }
```

### Choose where the data goes

Every experiment is described by an `experiment.toml` holding absolute paths: the script to
run, its parameters, where its runs are written and which teacher data it reads. Set them
all at once for your machine:

```bash
uv run python configure.py /path/to/data                   # W&B logging off
uv run python configure.py /path/to/data --wandb my-project
```

The teacher is then written to `/path/to/data/teacher-activity` and each figure's runs to
`/path/to/data/<figure folder>`. Re-run it if you move the repository or the data.

### Reproduce a figure

First generate the teacher, which every other figure reads, then train Figure 1, whose runs
the other figures use as their baseline condition:

```bash
./run fig00-teacher-activity/experiment.toml             # the teacher network and its activity
./run --grid fig01-full-reconstruction/experiment.toml   # train every run of the figure
```

Then for any figure, train, evaluate and plot:

```bash
./run --grid fig05-weight-noise/experiment.toml     # train every run of the grid
uv run python fig05-weight-noise/analysis.py        # evaluate the runs -> CSVs
uv run python fig05-weight-noise/figures.py         # CSVs -> one SVG per panel
```

`analysis.py` scores every finished run on held-out stimuli and on the perturbation, caches
the scores inside each run's folder, and writes the CSVs its figure needs next to itself.
`figures.py` reads only those CSVs, so panels can be rebuilt without a GPU. What a figure
varies, and at which levels, is set in its `parameters.toml` and `run_grid_search.py`.

### Provenance: always launch runs with `./run`

`./run` launches experiments through the connectome-snns run framework, which ties every
result to the exact code that produced it:

- it refuses to start from uncommitted code (edits to `.toml` files and
  `run_grid_search.py` are allowed), so commit before you run;
- each run folder gets a `metadata.json` recording the commit of this repository and of
  connectome-snns (and whether the library had uncommitted changes), start and end times and
  its status, alongside copies of the `experiment.toml` and `parameters.toml` it ran with and
  links to its input data;
- a grid runs from a snapshot of the committed code, so you can keep working while it trains,
  and skips runs that have already finished, so an interrupted grid can simply be relaunched;
- `./run --resume <run folder> <experiment.toml>` continues a run from its last checkpoint.

Running `train.py` directly works, but produces results nobody can trace back to their code.
