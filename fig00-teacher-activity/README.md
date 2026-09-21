# Figure 0 — the teacher network

> Shared methods — model, teacher forcing, metrics, the noise ceiling, naming: see [`../README.md`](../README.md).

**What it is:** the conductance-based spiking network that generates the activity every
other figure is trained against, and the panels that characterise it. This is not a
result — it is the ground truth. Nothing downstream can be interpreted without knowing
what this network does.

It must be run before any other figure; they all read its outputs.

## The network

A conductance-based LIF network with assembly structure, adapted from Meissner-Bernard et
al. (2025), receives odour-modulated feedforward input.

| | |
|---|---|
| recurrent neurons | 5000 — 4000 excitatory / 1000 inhibitory (80/20) |
| assemblies | 20 |
| feedforward (mitral) neurons | 1500 |
| connection probability | 0.3–0.4 within an assembly, 0.05 between |
| recurrent synapses | AMPA + NMDA (E sources), GABA_A (I sources) |
| feedforward synapses | AMPA + NMDA |
| simulation | dt = 1 ms, 15 s per trial, 50 independent trials, seed 43 |

## The input

Feedforward rates are driven by an Ornstein-Uhlenbeck process controlling a softmax over
20 assembly-specific odourant patterns, so the input drifts smoothly between odourants
rather than switching between discrete stimuli. Mitral cells fire at 6 Hz baseline; each
odourant lifts 10% of them by 9 Hz, and the rest are depressed so the population mean
rate is unchanged — the network cannot identify an odourant from its total input drive.

| Parameter | Value |
|---|---|
| `tau` | 700 s — OU mean-reversion time constant |
| `temperature` | 0.2 — softmax sharpness over odourant patterns |
| `sigma` | 5.0 — OU noise amplitude |
| `baseline_rate` | 6.0 Hz |
| `modulation_rate` | 9.0 Hz above baseline for active cells |
| `modulation_fraction` | 0.1 of mitral cells per odourant |
| `batch_size` | 50 independent OU trajectories (trials) |

All 50 trials are training data. Evaluation does not hold one of them back: every
figure's `analysis.py` simulates the teacher again on a **fresh odour trajectory** never
seen in training (cached per run as `test_inputs.zarr`) and scores the student on that.

## Dimensionality

Computed by `analysis.py` over all 5000 neurons and all 50 trials, each smoothed with the
same 50 ms Gaussian used for Fluctuation R² and with the first 2 s discarded, from the
eigenspectrum of the pooled neuron × neuron covariance:

| | |
|---|---|
| participation ratio | **41.4** (0.8% of the population) |
| PCs for 50% of variance | 19 |
| PCs for 80% / 90% / 95% | 104 / 236 / 432 |

Figure 3 marks the 90%-variance count on its observed-neuron axis; the participation
ratio is deliberately not marked there, since at 41 neurons it sits inside the region
where the fit has already collapsed.

## Panels

One SVG each, in narrative order: what the input is, how the network answers it, what the
dynamics look like, and how high-dimensional the result is.

| | Panel | What it shows |
|---|---|---|
| **(a)** | `coding-schematic` | how an odourant is constructed: one assembly's input lifted to 15 Hz, the rest depressed so the 6 Hz mean is unchanged |
| **(b)** | `input-rates` | distribution of feedforward input rates, all 1500 neurons across all 20 odourants |
| **(c)** | `rates-odour-baseline` | per-neuron firing rate, one odourant vs homogeneous-Poisson baseline |
| **(d)** | `rates-odour-repeat` | the same odourant twice with different Poisson noise — the network's own trial-to-trial variability, the scale against which (c) should be read |
| **(e)** | `raster` | 300 neurons of one trial over 10 s, ordered by assembly |
| **(f)** | `assemblies` | the OU mixing weight of each odourant and each assembly's firing rate over one trial |
| **(g)** | `neuron-traces` | one neuron's membrane potential and its recurrent-E, recurrent-I, feedforward and leak currents |
| **(h)** | `conductances` | the same neuron's conductance per synapse type |
| **(i)** | `integrated-conductance` | integrated conductance per synapse type across sampled neurons |
| **(j)** | `synaptic-drive` | feedforward vs recurrent-excitatory share of excitatory drive, over the whole dataset (ratio 2.4) |
| **(k)** | `variance-explained` | cumulative variance explained, with the participation ratio and the 90% count marked |
| **(l)** | `variance-spectrum` | variance fraction per component, log-log |

Synapse pathways take their presynaptic population's colour — red from excitatory, blue
from inhibitory, grey from mitral — with AMPA and NMDA separated by line style. The two
assembly heatmaps use a sequential ramp built from the scheme's own steel blue, since
20 assemblies cannot be carried by a palette of six colours.

## Files

```
fig00-teacher-activity/
  generate.py        the run script — simulates the teacher and writes the spike data
  analysis.py        everything measured from it -> CSVs and NPZs next to this script
  figures.py         those files -> fig00-<letter>-<slug>.svg
  experiment.toml    paths, W&B
  parameters.toml    network, input and simulation parameters
  README.md
  fig00_coding_schematic.csv  fig00_input_rates.npz  fig00_condition_rates.csv
  fig00_raster.npz  fig00_traces.npz  fig00_conductance_integral.csv
  fig00_assemblies.npz  fig00_drive.csv
  fig00_pca_spectrum.csv  fig00_dimensionality.csv
```

`analysis.py` runs in steps (`--steps`, default all) and skips any whose outputs already
exist unless `--force`; it defaults to `--device cpu` and only takes a GPU that is
mostly free. The dimensionality step is the expensive one — it smooths and pools all 50
trials — so it is not recomputed casually. Everything it writes is small enough to commit,
because `figures.py` must be able to rebuild every panel without the spike data.

## How to run

```bash
./run fig00-teacher-activity/experiment.toml      # simulate (writes ../bernstein/teacher-activity)
uv run python fig00-teacher-activity/analysis.py   # measurements -> CSVs
uv run python fig00-teacher-activity/figures.py    # CSVs -> panel SVGs
```

The output folder is `../bernstein/teacher-activity/`, named in `experiment.toml`. It
keeps the historical name rather than `fig00-…` because every figure's `experiment.toml`
symlinks its inputs from that path, and the runs on disk record it in their provenance.

**Regenerating the teacher invalidates every figure**, which is why it needs explicit
agreement first: the students are trained against these exact spike trains.
