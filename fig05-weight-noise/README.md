# Figure 5 — Weight precision is not the binding constraint

> **Read `../METHODS.md` first** — model, teacher forcing, metrics, floor, seeds, naming, and the instruction to make only minimal edits to the existing code.

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
| **weight noise** | **swept**, 0 → 0.5 |
| seeds | ≥3 per point |

## Evaluation

- Held-out test set of new stimuli.
- **Fluctuation R²** primary, Activity R² secondary.
- Evaluated on **unobserved** neurons — the discriminative group, comparable with Figures 1–4.
- Shuffled-identity floor.

## Panels

- **(a)** Fluctuation R² vs weight noise fraction, error bars over seeds, floor dashed.
- **(b)** **The contrast panel.** Side by side with Figure 4's neuron-removal curve, sharing a y-axis: weight noise on the left, input volume lost on the right. The two x-axes are different quantities and should not be forced onto one axis, but the shared y makes the comparison immediate.

The sentence this figure earns: *at 50% weight noise the model still works; at 50% of input missing it does not.*

## Files

```
fig05-weight-noise/
  README.md
  fig05_summary.csv     weight_noise, seed, group{observed,unobserved}, metric, value, floor_value
  fig05_rates.csv       weight_noise, neuron_id, cell_type, observed{0,1}, seed, teacher_rate_hz, student_rate_hz
  plot_fig05.py
  fig05.svg
  config.yaml
```

## Status

**Needs rerunning at obs-10.** The archived `sweep-wn__ff-known__obs-100__recur-100__curve.svg` is clean — one factor varies — but sits at obs-100, so its metric is computed on neurons the student was trained on. Not comparable with the rest of the sequence.

Its shape (Activity R² ≈ 0.99 at wn = 0.05, ≈ 0.83 at wn = 0.5) is a good prior for sweep levels. Expect the obs-10 version to sit lower throughout, since it is measured on unobserved neurons.

## Notes

- Levels of 0, 0.1, 0.2, 0.3, 0.4, 0.5 are enough; the archived curve is smooth and monotone, so density buys little.
- If the curve is still high at 0.5, extend to 0.75 or 1.0. "Still works at 100% weight noise" would be a stronger and more surprising claim than stopping at half.

## What weight noise is

Gaussian noise **added** to existing synaptic weights, then rescaled so that the **mean and variance of the weight distribution are preserved**. No synapses are created or deleted — topology is untouched, only the values move.

Put this in the caption in one line. It matters for interpretation: this is measurement error on weights, not a change in connectivity, which is what makes the contrast with Figure 4 meaningful.

## Open question

1. Does the added noise ever **flip the sign** of a weight? If so, some synapses change from excitatory to inhibitory and the student violates Dale's law, which is a modelling artefact rather than a realistic measurement error — and an easy thing to be asked about. If signs do flip, worth reporting what fraction, or clipping at zero and saying so.

---

## Implementation (recorded settings)

*Added when the code was written.*

### How to run

```bash
./run --grid fig05-weight-noise/experiment.toml   # 5 levels x 3 seeds = 15 runs
uv run python fig05-weight-noise/analysis.py       # reads Figure 1's runs as weight noise 0
uv run python fig05-weight-noise/figures.py        # fig05.svg; panel (b) needs fig04_summary.csv
```

Identical to Figure 1 except `[student].weight_noise` ∈ {0.1, 0.2, 0.3, 0.4, 0.5}; 0 is
Figure 1. To extend to 0.75 / 1.0, append to `NOISE_LEVELS`. `PRIORITY.md` budgets 3 levels ×
2 seeds.

### What weight noise is, as implemented

The archived noisy-weights implementation (`noisy-weights/varying-noise/train.py`), unchanged.
For each (source type, target type) block of the concatenated [mitral; recurrent] weight
matrix — mitral→E, mitral→I, E→E, E→I, I→E, I→I:

1. multiply each non-zero weight by exp(σ·N(0,1) − σ²/2), σ = weight_noise (mean-1 log-normal);
2. affinely rescale the non-zero weights back to the block's original mean and SD;
3. clip at zero.

So the noise is multiplicative log-normal, not additive Gaussian as the text above says, and it
reaches the mitral weights too. Topology is untouched, except that clipping sets a weight to
exactly zero.

**Measured clipping (2026-09-21, three seeds):** 3.3% at noise 0.1, 4.4% at noise 0.2, 4.5% at noise 0.3, 3.4% at noise 0.4, 1.5% at noise 0.5. These were printed under panel
(a)'s title until they crowded it out; the panel now carries the title alone.

**Sign flips (open question 1):** step 2 can push weights below zero; step 3 clips them, so no
synapse changes sign and Dale's law holds. The fraction clipped is recorded per run as
`noise_clipped_fraction` in `fig05_summary.csv` and printed on panel (a).

### Panels as implemented

(a) Fluctuation R² vs weight noise, observed and unobserved (mean ± SD, seeds as dots),
ceilings, unobserved floor, clipped fraction per level. (b) Unobserved Fluctuation R² vs weight
noise (left) beside Figure 4's neuron-removal curve vs κ_lost (right), shared y axis.
