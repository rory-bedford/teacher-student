# Figure 1 — Full reconstruction recovers the teacher

> Model, teacher forcing, metrics, the noise ceiling and naming: see [`../METHODS.md`](../METHODS.md).

**Claim:** with the feedforward inputs recorded and reconstructed and the recurrent connectome reconstructed, the student reproduces the teacher's activity — including for neurons it never observed — on stimuli it was never trained on.

The foundation of the talk, and the recapitulation of the known result (Beiran & Litwin-Kumar) in a spiking network with only a handful of free parameters. Everything later is a degradation of this condition.

## Configuration

| Parameter | Value |
|---|---|
| feedforward connections | **reconstructed** (given to student) |
| feedforward activity | recorded |
| recurrent reconstruction | **100%** |
| observed fraction | **50%** (2026-09-18; was 10%, roughly the fraction of reconstructed neurons expected to have activity recorded) |
| weight noise | **0** |
| trained parameters | **6** cell-type × cell-type scaling factors |
| seeds | 3 (44, 45, 46); one seed shown in the raster and the scatters |

Exactly one factor departs from the ideal case — half of the neurons unobserved — and that
departure is the point: the circuit is recovered anyway.

## Evaluation

- **Held-out test set of new stimuli.**
- **Fluctuation R²** (primary): spikes smoothed with a 50 ms Gaussian, then R². This is a close drop-in for the calcium trace we match against (exponential filter, τ = 100 ms), so it answers "how well would these spike trains agree once seen through calcium". **Say this in the talk** — it makes the metric a property of the real experiment rather than an arbitrary choice.
- **Activity R²** (secondary): firing rates.
- Reported separately for **observed** and **unobserved** neurons.
- **As built** (three seeds, mean ± SD over seeds, ceiling in brackets): held-out
  Fluctuation R² 0.974 ± 0.015 observed [0.997] and 0.968 ± 0.025 unobserved [0.997];
  Activity R² 0.986 ± 0.011 and 0.982 ± 0.019 [1.000].
- **Noise ceiling**: the perfectly specified student under the same teacher forcing and the
  same spike-flip draws, so the score can be read against what is achievable rather than
  against chance. The shuffled-identity floor of the original design was dropped on
  2026-09-17 (it sits near -1, not 0: pooled R² against the identity line punishes
  mismatched means), and no figure plots a floor.

## Panels

One SVG each:

- **(a)** `fig01-a-raster` — teacher and student spikes, 3 observed and 3 unobserved neurons, held-out stimulus.
- **(b)** `fig01-b-scatter` — firing rates, student vs teacher, observed beside unobserved, linear 0–40 Hz (the few faster cells are left off the axis rather than piled on its edge; say the clip in the caption).
- **(c)** `fig01-c-delta-scatter` — the perturbation's per-neuron Δrate, teacher vs student, linear ±25 Hz, targeted cells marked.
- **(d)** `fig01-d-delta-means` — mean Δrate per population (targeted I, non-targeted I, non-targeted E), teacher beside student.
- **(e)** `fig01-e-scaling-factors` — the six tied scaling factors, learnt / true, this figure's runs beside the fully observed runs from Figure 2's grid.

The per-neuron R² histogram of the original design was dropped: the raster and the scatters
show the same thing more directly.

## Files

```
fig01-full-reconstruction/
  analysis.py
  figures.py
  run_grid_search.py
  train.py
  experiment.toml
  parameters.toml
  README.md
  fig01_perturbation.csv
  fig01_rates.csv
  fig01_scaling_factors.csv
  fig01_spikes.csv
  fig01_summary.csv
  fig01-a-raster.svg
  fig01-b-scatter.svg
  fig01-c-delta-scatter.svg
  fig01-d-delta-means.svg
  fig01-e-scaling-factors.svg
```

## Notes

- Figure 3 (observed-fraction sweep) contains this configuration as one of its points:
  Figure 1 is the picture, Figure 3 is the curve.
- An obs-100 run is not part of this figure. It would show only the fit to neurons the
  student was trained on, which is a weaker claim; it is kept as the identifiability
  check behind panel (e).

## Implementation (recorded settings)

### How to run

```bash
./run --grid fig01-full-reconstruction/experiment.toml   # 3 seeds -> bernstein/fig01-full-reconstruction/seed-*
uv run python fig01-full-reconstruction/analysis.py       # fig01_summary.csv, fig01_rates.csv, fig01_spikes.csv, fig01_scaling_factors.csv, fig01_perturbation.csv
uv run python fig01-full-reconstruction/figures.py        # fig01-a … fig01-e SVGs
```

`train.py` is a thin wrapper around `common/training.py`; the student, evaluation and
plot helpers are shared with every figure (`common/`). Figures 2–5 read these runs as
their baseline condition, so this grid runs first.

### Student

- **Teacher:** `bernstein/teacher-activity` — 5000 recurrent neurons (4000 E / 1000 I, 20
  assemblies), 1500 mitral inputs, 50 training trials × 15 s at dt = 1 ms.
- **Architecture:** the archived two-layer teacher-forced model. Layer 1 = unobserved
  neurons, recurrent among themselves, receiving mitral input and the observed neurons'
  teacher spikes. Layer 2 = observed neurons, no recurrence, receiving mitral input,
  layer-1 spikes and the observed neurons' teacher spikes. Verified: at the correct
  scaling factors the student reproduces the teacher spike for spike (0 mismatches over
  2 s, observed and unobserved).
- **Observed neurons:** 2500 of 5000, drawn at random per seed.
- **Trained parameters: 6** — mitral→E, mitral→I, E→E, E→I, I→E, I→I; inhibitory types are
  both pre- and postsynaptic. Each is one parameter for the whole network. *The archived
  model had 10:* its unobserved→unobserved block had its own four scaling factors, separate
  from the same pathways arriving from observed neurons. They are tied here because the two
  layers implement one network.
- **What training has to recover:** every pathway's weights are multiplied by a log-normal
  factor (variance 0.5, mean 1); the scaling factors start at 1, so the correct values
  are the inverse factors. Per-seed draw.

### Training (archived visible-driven recipe + unobserved-rate penalties)

| Setting | Value | Note |
|---|---|---|
| chunk size | 160 ms | |
| epochs | 50 (93 chunks each) | archived run converged to within ~5% by epoch 12 |
| burn-in | 20 chunks per epoch, no loss | |
| chunks per update | 6 | |
| loss | van Rossum, τ_rise 10 ms, τ_decay 100 ms, + unobserved-rate penalties | mean and SD of each cell type's unobserved rates vs the observed teacher neurons, weight 0.5 each (as the archived full-inference runs with structural mismatch, and Figure 6). Added 2026-09-17: without them, 10% synapse dropout silenced the unobserved inhibitory population (6 vs 16 Hz by mid-training, E→I scaling factor 0.24× target) |
| optimiser | Adam, β = (0.95, 0.999), eps = 1e-8 | archived eps = 1e-4 was silently dropped by the config class, so never applied |
| learning rate | 8e-3 → 5e-4, cosine over epochs | as archived. Tried 4e-3 → 4e-4 with clip 2 (2026-09-17): at epoch 31 held-out Fluctuation R² 0.79 / 0.85 and observed Activity R² 0.89, vs 0.86 / 0.90 and 0.98 for this recipe, so reverted. The archived cosine was given the chunk count, so the rate barely decayed; fixed |
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
- **Perturbation protocol (2026-09-17):** every simulated model is run 20 times, each with one
  extra spike injected at t = 0 into a different unobserved neuron (same draws and identical
  teacher forcing for the trained and the perfect student). Smoothed traces are averaged over
  the 20 draws and the average is scored (activity R² on draw-averaged rates). Per-neuron R²
  uses the draw average. No floor is reported.
- **Ceiling:** the same metric for a *perfectly specified* student — teacher weights,
  correct scaling factors, identical teacher forcing and perturbations. **It is not 1.** The
  teacher network is chaotic: a perfectly specified student matches it spike for spike until
  float32 rounding flips one spike in the simulated population, at an arbitrary time, and
  timing then decorrelates within ~1 s. The t = 0 flips replace that arbitrary moment with a
  controlled one. Test on seed 44 (earlier recipe, K = 20): ceiling Fluctuation R² 0.92 / 0.94
  (observed / unobserved), Activity R² 0.998; trained student 0.86 / 0.90 and 0.98. Scoring
  single draws instead gave noisy ceilings that could fall below the student.
- **Observed vs unobserved is not comparable within one seed (2026-09-18).** In the seed-44 runs
  of that earlier recipe the unobserved group scored higher than the observed one on Fluctuation R²
  (0.844 vs 0.798, and the same ordering in the *ceiling*, 0.940 vs 0.920, where no training is
  involved). This is a property of the neuron draw, not of the model:
  - Decomposing the pooled metric for that run: residual and within-neuron variance are equal
    between the groups to within 4%; the whole difference is the denominator. The unobserved pool
    has 1.5x the between-neuron variance, i.e. a wider spread of mean rates, and pooled R² counts
    that spread as explainable variance.
  - Firing rates are very heavy-tailed, so the spread of a group is set by its few fastest cells.
    Seed 44's observed sample misses the tail (fastest observed cell 176 Hz vs 268 Hz unobserved).
    Across seeds the ratio of observed to complement spread is 0.66 (seed 44), 2.09 (45), 1.05 (46),
    1.60 (47), 0.82 (48) — seed 44 is simply an unlucky draw, and every figure inherits it because
    the observed sets are nested prefixes of one permutation per seed.
  - Size-matching the groups does not remove it (matched 500-neuron subsets of the unobserved pool
    still give 0.842); averaging over seeds does.
  - Scoring each neuron against its own mean (per-neuron R², equal weight per neuron, cells >= 1 Hz)
    makes the groups equivalent — observed sits at the 52nd–55th percentile of matched subsets — but
    it discards the between-neuron variance, so values drop a lot (0.55 against a ceiling of 0.81)
    and our stored ceilings do not apply to it.
  - **Read the unobserved group as the headline** and treat single-seed observed/unobserved
    differences as noise.

The archived scatter for this configuration quoted 0.997 / 0.995, but those were rates computed on
*training* trial 0 (`compute_per_neuron_rates` reads the training `spike_data.zarr`), not held-out
stimuli, and came from the EM-clamped model rather than this one. Figure 1 is retrained here.
