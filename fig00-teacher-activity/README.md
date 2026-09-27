# Figure 0 — The teacher network

The teacher is the conductance-based spiking network whose activity every other figure
fits. This figure describes it: its odour input, how its assemblies respond, what a single
neuron's inputs look like, and how high-dimensional the resulting activity is.

Shared methods (model, training, evaluation, noise ceiling): [`../METHODS.md`](../METHODS.md).

## Experiment

The teacher is a conductance-based leaky integrate-and-fire network with assembly
structure, adapted from Meissner-Bernard et al. (2025) and tuned to zebrafish Dp. Its
recurrent neurons are organised in 20 assemblies, connected more densely within an
assembly than between assemblies, and receive input from 1500 mitral cells.

The odour input drifts smoothly rather than switching between discrete stimuli. An
Ornstein-Uhlenbeck process drives a softmax over 20 odourant patterns, one per assembly,
and the mixture sets the mitral cells' Poisson rates. Each odourant raises 10% of the
mitral cells above baseline and lowers the rest, so an odourant cannot be identified from
the total input drive. `generate.py` builds the connectome and simulates 50 independent
trials; all 50 are training data, and each figure's `analysis.py` scores students on a
fresh odour trajectory never used in training.

| Setting | Value |
|---|---|
| Recurrent neurons | 5000: 4000 excitatory, 1000 inhibitory |
| Assemblies | 20 |
| Connection probability | 0.3–0.4 within an assembly, 0.05 between |
| Feedforward (mitral) units | 1500, connection probability 0.05 |
| Synapses | AMPA + NMDA (mitral and excitatory), GABA_A (inhibitory) |
| Mitral rates | 6 Hz baseline; an odourant lifts 10% of cells by 9 Hz |
| Odour mixing | Ornstein-Uhlenbeck, τ = 700 s, σ = 5, softmax temperature 0.2 |
| Simulation | dt = 1 ms, 50 trials of 15 s, seed 43 |

## Results

| Measure | Value |
|---|---|
| Share of excitatory drive: feedforward / recurrent | 0.29 / 0.71 (SD 0.02 over trials) |
| Participation ratio | 41.4 |
| Principal components for 50% / 90% of variance | 19 / 236 |

The network is recurrence-dominated: recurrent excitation supplies about 2.4 times the
drive the mitral input does. Its activity is nonetheless far from low-dimensional, which
is what makes predicting unobserved neurons a non-trivial test. Dimensionality is computed
over all 5000 neurons and 50 trials, smoothed with the 50 ms Gaussian used for Fluctuation
R², with the first 2 s of each trial discarded.

## Panels

- `fig00-a-ou-trajectories.svg` — the mixing coefficient of each of the 20 odourants over one trial.
- `fig00-b-assembly-rates.svg` — each assembly's excitatory rate over the same trial, relative to its own mean across all trials.
- `fig00-c-neuron-traces.svg` — one excitatory neuron's membrane potential, spikes, input currents and synaptic conductances over a 2 s window.
- `fig00-d-synaptic-drive.svg` — feedforward against recurrent excitatory share of synaptic drive, mean and SD over trials.
- `fig00-e-variance-spectrum.svg` — variance fraction of the first 100 principal components.
- `fig00-f-assembly-rastermap.svg` — z-scored rates of the neurons in the trial's two leading assemblies, sorted by assembly and rate.
- `fig00-g-loss-kernel.svg` — the van Rossum loss kernel used in training, and the difference between two filtered spike trains that the loss squares.

## Running

```bash
./run fig00-teacher-activity/experiment.toml
uv run python fig00-teacher-activity/analysis.py
uv run python fig00-teacher-activity/figures.py
```

The teacher must be generated before any other figure is trained, and regenerating it
invalidates every trained student. `analysis.py` re-simulates the short windows the
panels need and writes small CSV and NPZ files next to itself, so `figures.py` needs
neither the spike data nor a GPU. Its dimensionality step is the slow one (tens of
minutes); steps whose outputs exist are skipped unless `--force` is given.

## Notes

- Assemblies differ in intrinsic rate by more than a stimulus moves them, so panel (b)
  plots each assembly's rate relative to its own mean across all trials; raw rates would
  show which assembly is fastest rather than which odourant is on.
- The trial shown in (a), (b) and (f) is chosen for a clear switch between two
  odourants, and the traced neuron in (c) is a typical excitatory cell (rate close to its
  population mean, no unusually strong mitral synapse). Both are illustrations; the
  across-trial numbers are above.
- Currents in (c) are drawn inward-positive, and spikes as vertical lines, since the
  simulator resets the voltage at threshold.
- The weight distributions are heavy-tailed: the library treats `w_sigma` in
  `parameters.toml` as a variance rather than a standard deviation, so every block has a
  weight SD of about 0.22. Teacher and student share the connectome, so no comparison is
  affected.
