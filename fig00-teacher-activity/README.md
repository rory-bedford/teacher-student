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
| **(a)** | `ou-trajectories` | the Ornstein-Uhlenbeck mixing coefficient of each of the 20 inputs over one trial, the two leaders drawn heavy |
| **(b)** | `assembly-rates` | each assembly's excitatory population rate over the same trial, **against its own mean across all 50 trials**, Gaussian σ = 500 ms |
| **(c)** | `neuron-traces` | one excitatory neuron: membrane potential with its spikes, input currents, and every synapse type's conductance on one shared axes, over a 2 s window chosen for spike density and re-zeroed |
| **(d)** | `synaptic-drive` | feedforward vs recurrent-excitatory share of excitatory drive, mean over the 50 trials with an SD whisker — **0.29 ± 0.02 against 0.71 ± 0.02**, a 2.4:1 split that barely varies between trials |
| **(e)** | `variance-spectrum` | variance fraction per component, first 100 components, linear axes |

These follow the library's own dashboards (`create_activity_dashboard`,
`create_assembly_activity_dashboard`, `plot_synaptic_conductances`,
`plot_spike_trains`) in layout and convention, in house colours:

- **Currents are inward-positive**, so excitatory input goes up. The simulator stores
  g(V − E_syn), which is negative for an excitatory synapse, so the panel flips the sign.
- **Spikes are drawn as vertical lines from threshold to 0 mV.** The simulator resets the
  voltage at threshold, so the trace itself has no spike peak.
- **Conductance axes are capped at the 98th percentile** (inhibition at ten times it), the
  library's rule. Rare transients are therefore clipped rather than allowed to flatten
  every other trace.
- The 20 assemblies use the dashboard's own categorical map, since the talk palette has
  six colours. Everything else uses the semantic names from `common/style.py`.
- Panels (b) and (c) are the assembly dashboard's pair, from the same trial with the same
  colour per assembly, so they can be shown side by side. Odourant *k* drives assembly
  *k*: the library generates one input pattern per assembly, so column *k* means the same
  assembly in both. Smoothing is 50 ms rather than the dashboard's 200 ms, matching
  Fluctuation R²'s kernel.

Five panels were built and cut on 2026-09-21 — the input-coding schematic, the input-rate
histogram, the odourant-vs-baseline scatter, an integrated-conductance comparison and a
per-assembly feedforward-drive trace. They are in the git history.

**Reading (a) against (b): the stimulus is decodable from the deviation, not from the
rate.** Assemblies differ in intrinsic rate by far more than a stimulus moves them — the
spread across assemblies has SD 1.13 Hz, a stimulus-driven deviation SD 0.41 Hz — so raw
rates show which assembly is fastest (always assembly 18, in all 50 trials, profile
correlation 0.93 between them) rather than which odourant is on. Panel (c) therefore plots
each assembly's rate against **its own mean over all 50 trials**, which removes the
ordering and leaves the response.

Two further choices make it legible, both of which the panel states:

- **The trial is chosen, not fixed:** the one with the cleanest **switch** between two
  odourants, scored by how long each of its two leading odourants leads and how strongly
  it is mixed while leading. That currently selects **trial 48**, where odourant 18 holds
  a mixing weight near 1.0 from 2 to 6 s and then hands over to odourant 9. A trial whose
  stimulus is an even blend has nothing to show, and one held on a single odourant shows a
  level rather than a change. The panels are an illustration of the mechanism; its general
  strength is the across-trial number below.
- **Smoothing is 500 ms**, not the dashboard's 200 ms or Fluctuation R²'s 50 ms. The
  response to an odourant is a sustained offset across the whole trial, because the OU
  process has τ = 700 s against a 15 s trial, so what has to be averaged away is the
  recurrent fluctuation. At 500 ms the driven assembly sits 1.3 times the other
  assemblies' SD above its own mean; at 50 ms, 0.7 times.

**Across all 50 trials**, an odourant's mixing coefficient correlates **+0.51** with its
own assembly's deviation from that assembly's mean, and the dominant odourant's assembly
is the largest deviator in **62%** of trials and in the top three in **82%**. Held static
instead of mixed, odourant 1 raises assembly 0 by 2.7 Hz, the largest response of the
twenty.

**This teacher is recurrence-dominated, and an earlier one was not.** The teacher in
`dp-simulations/teacher-activity-feedforward` (9 March 2026) shows the same two panels
with the stimulus unmistakable: assemblies at 1.6-3.6 Hz (SD 0.54) with the driven one at
6-9 Hz. It differs in that its odourant patterns are perfectly whitened (every pattern
mean 6.0 Hz, where this teacher's `baseline_variance = 1.5` spreads them 3.5-8.6 Hz), and
in being more feedforward-driven: mitral `w_mu` 0.03 against 0.02 here, mitral NMDA
`g_bar` 0.4 against 0.1, recurrent E->E `w_mu` 0.01 against 0.02, E->I connectivity 0.3
against 0.4. This teacher measures 2.4:1 recurrent-to-feedforward drive (panel (h)).

## The traced neuron is chosen, not pinned

The notebook pinned neuron 13 for its traces. That neuron carries a single mitral synapse
of weight **2.32 — the 96th percentile of the network** — so one presynaptic spike injects
about 14 nS and its currents are dominated by that one synapse, with transients near
1 nA. It is not a representative cell, and the dashboards never showed this because their
axes are capped at the 98th percentile.

`analysis.py` now *chooses* the traced neuron: among cells of `TRACE_CELL_TYPE` whose
strongest mitral synapse is no larger than the network median, the one firing closest to
its cell type's mean rate (the library's own "typical cell" rule). That currently selects
**neuron 2947**, an excitatory cell at 4.2 Hz against a population mean of 4.2 Hz, whose
strongest mitral synapse is 0.180 against a median of 0.288. Its currents show the
balanced excitation and inhibition the dashboard shows, ±150 pA against a leak that
tracks the mean.

The panel draws **2 s** of the 5 s that `analysis.py` stores: spike lines over a longer
window read as bands rather than as spikes.

## Feedforward weights are heavy-tailed, by a wider margin than the config implies

Worth knowing before quoting any weight statistic. `assign_weights_lognormal` treats
`w_sigma` as the **variance** of the weight distribution and solves for the log-normal
matching that mean and variance, while `parameters.toml` calls the field a standard
deviation. With `w_sigma = 0.05` every block therefore ends up with SD ≈ 0.224 regardless
of its mean:

| block | mean | `w_sigma` | log-normal σ | median | SD |
|---|---|---|---|---|---|
| mitral→E | 0.02 | 0.05 | 2.20 | 0.0018 | 0.224 (11× the mean) |
| mitral→I | 0.01 | 0.05 | 2.49 | 0.0005 | 0.224 (22× the mean) |
| E→E | 0.02 | 0.05 | 2.20 | 0.0018 | 0.224 (11× the mean) |
| I→E | 0.08 | 0.05 | 1.48 | 0.0270 | 0.224 (3× the mean) |

The mitral weights that result have median 0.0014 and a maximum of **65.5**, and 12.7% of
neurons carry at least one mitral synapse above 1.0. Nothing downstream is invalidated —
teacher and student share the connectome, so every fit is on equal terms — but do not
describe these weights as having an SD of 0.05. If SD 0.05 was intended, `w_sigma` should
be 0.0025.

## Files

```
fig00-teacher-activity/
  generate.py        the run script — simulates the teacher and writes the spike data
  analysis.py        everything measured from it -> CSVs and NPZs next to this script
  figures.py         those files -> fig00-<letter>-<slug>.svg
  experiment.toml    paths, W&B
  parameters.toml    network, input and simulation parameters
  README.md
  fig00_condition_rates.csv  fig00_assemblies.npz  fig00_raster.npz  fig00_traces.npz  fig00_drive.csv
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
