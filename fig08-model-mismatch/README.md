# Figure 8 — Model mismatch

Every other figure degrades the connectome; this one degrades the neuron model. How much does
the student lose when its spike thresholds are not the teacher's? Degradation is graceful:
1 mV of threshold heterogeneity costs 0.02–0.03 in Fluctuation R², and even at 4 mV the
student explains two thirds of the unobserved neurons' fluctuations, although its prediction
of the perturbation response falls faster and varies more between seeds.

Shared methods (model, training, evaluation, noise ceiling): [`../METHODS.md`](../METHODS.md).

## Experiment

Each student neuron's spike threshold is offset from the teacher's by an independent Gaussian
draw, with the same absolute SD for excitatory and inhibitory neurons, then centred within
each cell type so every population's mean threshold is exactly the teacher's. The mismatch is
therefore pure per-neuron heterogeneity, which six shared scaling factors cannot absorb. The
offset applies to every simulated neuron, observed and unobserved; every other parameter,
including the weights, is the teacher's. Everything else is as in Figure 1: full connectome,
50% of neurons observed.

The teacher's thresholds are −38 mV (E) and −45 mV (I), with rest and reset at −60 mV, so an
SD of 4 mV is 18% (E) and 27% (I) of the gap between rest and threshold.

| Setting | Value |
|---|---|
| Threshold heterogeneity (SD) | 1, 2, 4 mV; 0 is Figure 1's runs |
| Observed fraction | 50% |
| Trained parameters | 6 scaling factors |
| Epochs | 50 |
| Seeds | 3 (44, 45, 46) |

## Results

Fluctuation R², mean over three seeds (noise ceiling 1.00 held-out, 0.99 perturbation):

| SD (mV) | Observed | Unobserved | ΔFluct. R² perturbation, observed | ΔFluct. R² perturbation, unobserved |
|---|---|---|---|---|
| 0 (Figure 1) | 0.97 | 0.97 | 0.94 | 0.94 |
| 1 | 0.95 | 0.94 | 0.91 | 0.90 |
| 2 | 0.88 | 0.84 | 0.81 | 0.74 |
| 4 | 0.70 | 0.66 | 0.57 | 0.52 |

The perturbation response on unobserved neurons spreads widely across seeds at 2–4 mV
(SD 0.16–0.20). The learnt scaling factors move little up to 2 mV (0.90–0.99 of the truth, against
0.94–1.07 in Figure 1); at 4 mV they fall to 0.73–0.90 (I→I lowest), so the student weakens
its inputs slightly rather than compensating neuron by neuron, which six shared parameters
cannot do.

## Panels

- `fig08-a-curve.svg` — Fluctuation R² on held-out stimuli against threshold heterogeneity,
  observed and unobserved.
- `fig08-b-delta-fluctuation.svg` — perturbation ΔFluctuation R² against threshold
  heterogeneity, observed and unobserved (E and I pooled, targeted cells excluded).
- `fig08-c-scaling-factors.svg` — learnt / true scaling factor against threshold
  heterogeneity; 1.0 is the true model at every level.

## Running

```bash
./run --grid fig08-model-mismatch/experiment.toml
uv run python fig08-model-mismatch/analysis.py
uv run python fig08-model-mismatch/figures.py
```

`analysis.py` reads Figure 1's runs as the zero-heterogeneity point. The grid is 9 runs
(3 levels × 3 seeds), each about 3.6 h on a Quadro RTX 5000, as in Figure 1.

## Notes

- The offsets are set by `[student].threshold_heterogeneity` and drawn in
  `common/structure.py` from their own random stream, so they are independent of every
  other random draw for the seed. The physiology block in `parameters.toml` stays the
  teacher's.
- The noise ceiling is the perfectly specified student, which has the teacher's thresholds,
  so it does not depend on the mismatch: it is Figure 1's, drawn as one constant line.
- Only heterogeneity is tested; a uniform shift of every threshold is not.
