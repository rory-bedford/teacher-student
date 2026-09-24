# Figure 5 — Weight precision is not the binding constraint

> Shared methods — model, teacher forcing, metrics, the noise ceiling, naming: see [`../README.md`](../README.md).

**Claim:** the model tolerates substantial error in synaptic weights. Compared against Figure 4, imprecise weights cost far less than missing connections — so the limiting factor is what you have reconstructed, not how accurately you have measured it.

This figure exists mainly to set up that contrast. On its own it is a robustness check; beside Figure 4 it is an argument about where effort should go.

## Configuration

Identical to Figure 1 except for the swept degradation:

| Parameter | Value |
|---|---|
| feedforward connections | reconstructed |
| recurrent reconstruction | 100% |
| observed fraction | **50%** (matches Figures 1–4; 2026-09-18, was 10%) |
| trained parameters | 6 scaling factors |
| **weight noise** | **swept**, 0 → 0.8 |
| seeds | 3 per point |

## What weight noise is

The noisy-weights implementation of the archived runs, unchanged. For each (source type,
target type) block of the concatenated [mitral; recurrent] weight matrix — mitral→E,
mitral→I, E→E, E→I, I→E, I→I:

1. multiply each non-zero weight by exp(σ·N(0,1) − σ²/2), σ = weight_noise (mean-1 log-normal);
2. affinely rescale the non-zero weights back to the block's original mean and SD;
3. clip at zero.

So the noise is **multiplicative log-normal**, not additive Gaussian, the block's original
**mean and SD are preserved**, and it reaches the **mitral weights** as well as the recurrent
ones. No synapses are created or deleted — topology is untouched, only the values move,
except that clipping sets a weight to exactly zero.

Step 2 can push weights below zero; step 3 clips them, so **no synapse changes sign** and
Dale's law holds. The fraction clipped is recorded per run as `noise_clipped_fraction` in
`fig05_summary.csv`.

**Measured clipping (2026-09-21, three seeds):** 3.3% at noise 0.1, 4.4% at noise 0.2, 4.5%
at noise 0.3, 3.4% at noise 0.4, 1.5% at noise 0.5. These were printed under panel (a)'s
title until they crowded it out; the panel now carries the title alone.

Put the one-line version in the caption. It matters for interpretation: this is measurement
error on weights, not a change in connectivity, which is what makes the contrast with
Figure 4 meaningful.

## Evaluation

- Held-out test set of new stimuli.
- **Fluctuation R²** primary, Activity R² secondary.
- Evaluated on **unobserved** neurons — the discriminative group, comparable with Figures 1–4.
- **Noise ceiling** (the perfectly specified student under the same forcing and flips); no floor is plotted -- the shuffled-identity floor was dropped on 2026-09-17.

## Panels

As built (2026-09-21), one SVG each:

- **(a)** `fig05-a-weight-perturbation` — what the noise does to a synapse: the perturbed
  weight against the teacher's, side by side at noise 0.1 and 0.8, with the identity
  dashed. A recreation of the old repository's `weight_perturbation.svg`. The box carries r, the correlation
  between the two weight sets — not R², which everywhere else means variance explained —
  falling **0.993 → 0.770**. Noise **0.7** is the level to talk about: there the
  teacher's and student's weights correlate at r = 0.81, the weight-against-synapse-volume
  correlation measured in Holler et al., *Structure and function of a neocortical synapse*.
  The panel shows **0.8** (r = 0.77), a shade past that precision; the sweep runs both. The population mean and SD are not shown (2026-09-23): the noise
  preserves them exactly by construction — 0.0325 and 0.2085 at both levels — so they are
  recorded in `fig05_weight_perturbation.csv` rather than on the panel.
  Needs no trained run: `analysis.py` applies the same function training applies.
- **(b)** `fig05-b-curve` — Fluctuation R² vs weight noise, observed and unobserved,
  individual seeds, no error bars.
- **(c)** `fig05-c-delta-fluctuation` — perturbation ΔFluctuation R², cell types pooled,
  shared y range with (b).

Panel (a)'s axes stop at the 99th percentile of the weights: a handful of synapses run two
orders of magnitude further out (see the teacher's heavy-tailed weights in
`../fig00-teacher-activity/README.md`), and plotting the full range puts every point in
one corner. Its statistics are the whole non-zero population's, not the plotted sample's.

The contrast panel (weight noise beside Figure 4's neuron removal) was removed: it
duplicated Figure 4's own curve, and comparing the two error types is a job for the slide
deck. The sentence the pair still earns: *at 50% weight noise the model still works; at
50% of input missing it does not.*

## Files

```
fig05-weight-noise/
  analysis.py
  figures.py
  run_grid_search.py
  train.py
  experiment.toml
  parameters.toml
  README.md
  fig05_rates.csv
  fig05_summary.csv
  fig05_weight_perturbation.csv
  fig05-a-weight-perturbation.svg
  fig05-b-curve.svg
  fig05-c-delta-fluctuation.svg
```

---

## Implementation (recorded settings)

*Added when the code was written.*

### How to run

```bash
./run --grid fig05-weight-noise/experiment.toml   # 5 levels x 3 seeds = 15 runs
uv run python fig05-weight-noise/analysis.py       # reads Figure 1's runs as weight noise 0
uv run python fig05-weight-noise/figures.py        # panel SVGs, from this figure's CSVs
```

Identical to Figure 1 except `[student].weight_noise` ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
(`NOISE_LEVELS` in `run_grid_search.py`); noise 0 is Figure 1.
