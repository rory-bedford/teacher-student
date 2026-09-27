# Figure 6 — Learnt feedforward input

Is the recurrent connectome enough if the input to the circuit is not reconstructed? We give
the student the whole recurrent connectome and every neuron, but make it learn every
feedforward (mitral→E/I) weight. It still fits the neurons it observes, but its prediction of
the unobserved neurons falls well below that of the student given the input: fitting and
predicting come apart even though no recurrent connectivity is missing.

Shared methods (model, training, evaluation, noise ceiling): [`../METHODS.md`](../METHODS.md).

## Experiment

The student is Figure 1's in every respect except the feedforward weights. The 1500 mitral
units keep their recorded activity, but their weights onto the excitatory and inhibitory
neurons are not given: each block (mitral→E, mitral→I) is learnt as `exp(U V)` at full rank,
so a failure to predict cannot be put down to limited capacity. The recurrent weights are the
teacher's with the four recurrent scaling factors (E→E, E→I, I→E, I→I) learnt as in Figure 1,
and 50% of neurons are observed.

This isolates one half of Figure 7's manipulation: an unreconstructed unit is
indistinguishable from external drive, and here the only unreconstructed units are the
inputs. The comparison condition is Figure 1's seed-matched runs, where the same student at
the same observation level is given the feedforward weights.

| Setting | Value |
|---|---|
| Feedforward weights | learnt, full rank (`low_rank = 1500`), 12,000,000 weights |
| Recurrent connectome | given, 4 learnt scaling factors |
| Observed fraction | 50% |
| Learning rate (learnt weights) | 8e-3 → 5e-4, cosine schedule |
| L1 penalty on learnt weights | 5000 × mean(`exp(U V)`) per block |
| Epochs | 200 |
| Seeds | 3 (44, 45, 46) |

## Results

Fluctuation R², mean over three seeds (noise ceiling 1.00 for every entry, 0.99 for the
perturbation):

| Condition | Observed | Unobserved | ΔFluct. R² perturbation, observed | ΔFluct. R² perturbation, unobserved |
|---|---|---|---|---|
| Input given (Figure 1) | 0.97 | 0.97 | 0.94 | 0.94 |
| Input learnt | 0.89 | 0.73 | 0.80 | 0.74 |

With the input learnt, the observed neurons are still fitted well (0.89, consistent across
seeds), while the unobserved neurons drop to 0.73 and vary more between seeds (0.66–0.79).
The response to the perturbation degrades in the same way.

## Panels

- `fig06-a-bars-held-out.svg` — Fluctuation R² on held-out stimuli, observed | unobserved,
  input given beside input learnt, per-seed dots and the noise ceiling.
- `fig06-b-bars-perturbation.svg` — perturbation ΔFluctuation R², observed and unobserved
  (E and I pooled), same two conditions.
- `fig06-c-legend.svg` — the shared legend, as its own file.
- `fig06-d-scatter.svg` — teacher vs student firing rate, observed | unobserved, input-learnt
  condition.

## Running

```bash
./run --grid fig06-learnt-feedforward/experiment.toml
uv run python fig06-learnt-feedforward/analysis.py
uv run python fig06-learnt-feedforward/figures.py
```

`analysis.py` reads Figure 1's runs as the input-given condition, so Figure 1 must be trained
first. A run is 30,000 chunks (200 epochs); at the measured 1.4 chunks/s that is about 6 h.

## Notes

- Full rank costs no extra time: `exp(U V)` is built once per chunk, not per timestep.
- The learning rate of 8e-3 for the learnt weights is deliberate; at 1e-3 the learnt
  feedforward matrix stays at chance similarity to the teacher's.
- 200 epochs is enough for the fit on observed neurons, which is what the figure needs.
  Recovering the true feedforward matrix would take far longer and is not the question.
- The mitral→E and mitral→I scaling factors remain in the parameter set but act on no given
  weights, so they receive no gradient. `n_free_params` in the CSVs (12,000,006) counts
  them; they are not a recovery failure, and no scaling-factor panel is drawn for this figure.
