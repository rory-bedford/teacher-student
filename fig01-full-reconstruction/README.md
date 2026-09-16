# Figure 1 — Full reconstruction recovers the teacher

> **Read `../METHODS.md` first** — model, teacher forcing, metrics, floor, seeds, naming, and the instruction to make only minimal edits to the existing code.

**Claim:** with the feedforward inputs recorded and reconstructed and the recurrent connectome reconstructed, the student reproduces the teacher's activity — including for neurons it never observed — on stimuli it was never trained on.

The foundation of the talk, and the recapitulation of the known result (Beiran & Litwin-Kumar) in a spiking network with only a handful of free parameters. Everything later is a degradation of this condition.

## Configuration

| Parameter | Value |
|---|---|
| feedforward connections | **reconstructed** (given to student) |
| feedforward activity | recorded |
| recurrent reconstruction | **100%** |
| observed fraction | **10%** |
| weight noise | **0** |
| trained parameters | cell-type × cell-type scaling factors only ⟨n_params = ?⟩ |
| seeds | ≥3 for summary numbers; one seed shown in raster/scatter |

Exactly one factor departs from the ideal case — 90% of neurons unobserved — and that departure is the point: the circuit is recovered anyway.

## Evaluation

- **Held-out test set of new stimuli.** ✔ confirmed for the existing runs.
- **Fluctuation R²** (primary): spikes smoothed with a 50 ms Gaussian, then R². This is a close drop-in for the calcium trace we match against (exponential filter, τ = 100 ms), so it answers "how well would these spike trains agree once seen through calcium". **Say this in the talk** — it makes the metric a property of the real experiment rather than an arbitrary choice.
- **Activity R²** (secondary): firing rates.
- Reported separately for **observed** and **unobserved** neurons.
- **Shuffled-identity floor**: recompute after permuting the teacher↔student neuron mapping, so 0.995 can be read against chance.

## Panels

- **(a)** Spike raster, teacher vs student, ~5 example neurons, held-out stimulus window. **At least one must be an unobserved neuron, labelled as such** — a neuron the model never saw, reproduced spike for spike on a new stimulus, is the single most striking thing in the talk.
- **(b)** Rate scatter, teacher vs student, coloured by E/I, two panels (observed / unobserved), Fluctuation R² and Activity R² in each title, floor annotated.
- **(c)** *optional* — per-neuron R² histogram, split E/I, floor marked.

## Files

```
fig01-full-reconstruction/
  README.md             this file
  fig01_rates.csv       neuron_id, cell_type, observed{0,1}, seed, teacher_rate_hz, student_rate_hz
  fig01_spikes.csv      neuron_id, observed{0,1}, seed, source{teacher,student}, time_s
  fig01_summary.csv     seed, group{observed,unobserved}, metric{activity_r2,fluctuation_r2}, value, floor_value
  plot_fig01.py         reads the three CSVs, writes the figure
  fig01.svg             output
  config.yaml           copy of, or path to, the run config that produced the CSVs
```

## Status

**Reusable — no retraining.** The archived `ff-known__obs-10__recur-100__wn-0__scatter.svg` matches this configuration (observed R² = 0.997, unobserved R² = 0.995) and was evaluated on held-out stimuli.

To do:
1. Export the three CSVs from the existing run.
2. Add the shuffled-identity floor.
3. Add seed spread (≥3) to the summary numbers.
4. **Redo the raster**, labelling observed vs unobserved neurons. The archived raster's provenance is unclear and an unlabelled raster can't carry the claim.

## Notes

- Figure 3 (observed-fraction sweep) will contain this configuration as one of its points. Intended: Figure 1 is the picture, Figure 3 is the curve.
- An obs-100 run is not needed. It would show only the fit to neurons the student was trained on, which is a weaker claim.

## Open questions

1. n_params — how many cell-type × cell-type scaling factors, and are inhibitory types included as both pre and post?

---

## Implementation (recorded settings)

*Added when the code was written. Values are read off the archived implementation; where
this code departs from it, the reason is given.*

### How to run

```bash
./run --grid fig01-full-reconstruction/experiment.toml   # 3 seeds -> bernstein/fig01-full-reconstruction/seed-*
uv run python fig01-full-reconstruction/analysis.py       # fig01_summary.csv, fig01_rates.csv, fig01_spikes.csv
uv run python fig01-full-reconstruction/figures.py        # fig01.svg
```

`train.py` is a thin wrapper around `common/training.py`; the student, evaluation and
plot helpers are shared with every figure (`common/`). Figures 2–5 read these runs as
their baseline condition, so this grid runs first.

### Student

- **Teacher:** `bernstein/teacher-activity` — 5000 recurrent neurons (4000 E / 1000 I, 20
  assemblies), 1500 mitral inputs, 50 training trials × 15 s at dt = 1 ms.
- **Architecture:** the archived two-layer teacher-forced model
  (`hidden-activity/scripts/train_visible_driven.py`, `full-inference/hidden-units`).
  Layer 1 = unobserved neurons, recurrent among themselves, receiving mitral input and
  the observed neurons' teacher spikes. Layer 2 = observed neurons, no recurrence, receiving
  mitral input, layer-1 spikes and the observed neurons' teacher spikes. Verified: at the
  correct scaling factors the student reproduces the teacher spike for spike (0 mismatches
  over 2 s, observed and unobserved).
- **Observed neurons:** 500 of 5000, drawn at random per seed.
- **Trained parameters (open question 1): 6** — mitral→E, mitral→I, E→E, E→I, I→E, I→I;
  inhibitory types are both pre- and postsynaptic. Each is one parameter for the whole
  network. *The archived model had 10:* its unobserved→unobserved block had its own four
  scaling factors, separate from the same pathways arriving from observed neurons. They
  are tied here because the two layers implement one network.
- **What training has to recover:** every pathway's weights are multiplied by a log-normal
  factor (variance 0.5, mean 1); the scaling factors start at 1, so the correct values
  are the inverse factors. Per-seed draw.

### Training (archived visible-driven recipe)

| Setting | Value | Note |
|---|---|---|
| chunk size | 160 ms | |
| epochs | 50 (93 chunks each) | archived run converged to within ~5% by epoch 12 |
| burn-in | 20 chunks per epoch, no loss | |
| chunks per update | 6 | |
| loss | van Rossum, τ_rise 10 ms, τ_decay 100 ms | |
| optimiser | Adam, β = (0.95, 0.999), eps = 1e-8 | archived eps = 1e-4 was silently dropped by the config class, so never applied |
| learning rate | 8e-3 → 5e-4, cosine over epochs | the archived cosine was given the chunk count, so the rate barely decayed; fixed |
| gradient clip | 5.0 | archived FF-group clip 3.5 was silently dropped too |
| surrogate gradient scale | 5.0 | |
| precision | fp32 | archived fp16; fp16 does not reproduce the teacher at the correct scaling factors |
| physiology | teacher's, verbatim | the archived visible-driven run used mitral g_bar [4.0, 0.4] and NMDA τ_decay 70, not the teacher's [6.0, 0.1] and 60 |

Measured cost: 3.05 s per training chunk, 9.3 GB (Quadro RTX 5000, fp32) → ≈ 4.3 min per
epoch → **≈ 3.6 h per run, ≈ 11 GPU-h for the 3 seeds**.

### Evaluation (shared by all figures, `common/evaluation.py`)

- **Held-out stimulus:** one new 15 s teacher trial on an unseen odour trajectory
  (`connectome_snns.analysis.ensure_test_inputs`: teacher seed + 10 = 53, cached per run as
  `test_inputs.zarr`). The student runs on it with exactly the training teacher forcing.
  The first 2 s are discarded; 12.9 s are scored.
- **Fluctuation R²:** Gaussian σ = 50 ms, R² over neurons × time. **Activity R²:** R² of
  per-neuron rates. Both per group (observed / unobserved).
- **Floor:** the same metric with the teacher↔student neuron identities permuted within the
  group, mean of 5 permutations.
- **Ceiling (added):** the same metric for a *perfectly specified* student — teacher weights,
  correct scaling factors, identical teacher forcing. **It is not 1.** The teacher network is
  chaotic: a perfectly specified student matches it spike for spike for seconds, but float32
  rounding eventually flips one spike in the simulated population, and the trajectories
  then decorrelate within ~1 s. On a test trial this happened after 9–10 s, giving a ceiling of
  Fluctuation R² ≈ 0.92–0.95 and Activity R² ≈ 0.998. The METHODS statement that "a perfectly
  specified student reaches ~0 error" therefore holds only for short windows; every panel
  shows the ceiling.
- **Correction to Status above:** the archived 0.997 / 0.995 were rates computed on *training*
  trial 0 (`compute_per_neuron_rates` reads the training `spike_data.zarr`), not held-out
  stimuli, and came from the EM-clamped model rather than this one. Fig 1 is retrained here.

### Panels as implemented

(a) raster: 2 observed + 3 unobserved neurons with teacher rates 2–20 Hz, first 3 s after
burn-in, labelled. (b) rate scatters with Fluctuation/Activity R², floor and ceiling in each
title (mean over seeds; points from the first seed). (c) per-neuron Fluctuation R² histograms,
E vs I, floor and ceiling marked.
