# Figure 5 — Weight noise

The connectome gives the student its synaptic weights, but a real reconstruction measures
them imprecisely. How much does that cost? Prediction degrades steadily rather than
suddenly: at weight noise 0.3 the student still explains 0.78 of the unobserved neurons'
fluctuations, but at 0.7, the precision we expect of a real reconstruction, it falls to
0.38, about as much as losing 30% of the recurrent input in Figure 4.

Shared methods (model, training, evaluation, noise ceiling): [`../METHODS.md`](../METHODS.md).

## Experiment

Everything is as in Figure 1 except that the student's weights are noisy copies of the
teacher's. For each (source type, target type) block of the feedforward and recurrent
weights (mitral→E, mitral→I, E→E, E→I, I→E, I→I), the noise:

1. multiplies each existing weight by an independent log-normal factor
   exp(σ·N(0,1) − σ²/2), with mean 1, where σ is the weight noise;
2. rescales the block's weights so that they keep their original mean and SD;
3. clips at zero.

No synapses are created and none changes sign. Clipping can delete synapses, though, and
how many depends strongly on the draw: the rescaling matches each block's SD, which is set
by a few very large weights, so when those draw small multipliers the rescale stretches the
rest below zero. Seed 44 loses 3–14% of its synapses across the sweep (at noise 0.3, half
of its mitral→I synapses), while seeds 45 and 46 lose none. Seed 44's scores sit between the
other two seeds at every level, so the curve is not driven by it. The fraction clipped is
recorded per run as `noise_clipped_fraction` in `fig05_summary.csv`.

The grey band at noise 0.7 is where the teacher's and student's recurrent weights
correlate at r = 0.81, the correlation between synaptic weight and synapse volume measured
by Holler et al., *Structure and function of a neocortical synapse*.

| Setting | Value |
|---|---|
| Weight noise σ | 0.1, 0.2, ..., 0.8; 0 is Figure 1's runs |
| Noisy weights | feedforward and recurrent, per cell-type block |
| Observed fraction | 50% |
| Trained parameters | 6 scaling factors |
| Epochs | 50 |
| Seeds | 3 (44, 45, 46) |

## Results

Fluctuation R², mean over three seeds (noise ceiling 1.00 held-out, 0.99 perturbation):

| Weight noise | Observed | Unobserved | ΔFluct. R² perturbation, observed | ΔFluct. R² perturbation, unobserved |
|---|---|---|---|---|
| 0 (Figure 1) | 0.97 | 0.97 | 0.94 | 0.94 |
| 0.1 | 0.95 | 0.93 | 0.89 | 0.88 |
| 0.2 | 0.89 | 0.86 | 0.79 | 0.75 |
| 0.3 | 0.82 | 0.78 | 0.66 | 0.60 |
| 0.4 | 0.73 | 0.67 | 0.54 | 0.46 |
| 0.5 | 0.63 | 0.57 | 0.44 | 0.35 |
| 0.6 | 0.53 | 0.47 | 0.35 | 0.31 |
| 0.7 | 0.45 | 0.38 | 0.28 | 0.26 |
| 0.8 | 0.38 | 0.31 | 0.23 | 0.23 |

The observed scores are nearly identical across seeds, while the unobserved ones spread as
the noise grows (0.28–0.53 at 0.7).

## Panels

- `fig05-a-weight-perturbation.svg` — single recurrent synapses, noisy weight against
  teacher weight, at noise 0.1 and 0.8, identity dashed, with the correlation r over all
  synapses (0.993 and 0.770). Axes stop at the 99th percentile of the weights.
- `fig05-b-curve.svg` — Fluctuation R² on held-out stimuli against weight noise, observed
  and unobserved, individual seeds, with the grey band marking the estimated real-dataset
  level (noise about 0.7).
- `fig05-c-delta-fluctuation.svg` — perturbation ΔFluctuation R² against weight noise,
  observed and unobserved (E and I pooled, targeted cells excluded), with the same band.

## Running

```bash
./run --grid fig05-weight-noise/experiment.toml
uv run python fig05-weight-noise/analysis.py
uv run python fig05-weight-noise/figures.py
```

`analysis.py` reads Figure 1's runs as noise 0, and builds panel (a) directly from the
teacher's weights with the function training uses. The grid is 24 runs (8 levels × 3
seeds), each about 3.6 h on a Quadro RTX 5000, as in Figure 1.

## Notes

- Keeping each block's mean and SD is deliberate: the noise changes which synapses are
  strong, not a pathway's overall strength, which the six scaling factors could absorb.
- Clipping at zero keeps every synapse's sign, at the cost of the seed-dependent deletions
  described above.
- The noise panel (a) shows r, a correlation between weights, not R², which on every other
  panel means variance explained.
