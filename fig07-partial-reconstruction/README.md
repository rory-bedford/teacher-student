# Figure 7 — Partial reconstruction

What happens when only part of the circuit is reconstructed and the influence of the rest
has to be learnt? The student keeps fitting the neurons it observes, but its prediction of
the unobserved neurons collapses quickly as reconstruction falls: already at 70% reconstructed
it is 0.42, and below 50% it is near zero. At the roughly 10% we expect for the real dataset,
fitting and predicting have come apart entirely.

Shared methods (model, training, evaluation, noise ceiling): [`../METHODS.md`](../METHODS.md).

## Experiment

A random segment of the 6500 pooled units (1500 feedforward + 5000 recurrent) is
reconstructed, with no distinction between feedforward and recurrent units: an
unreconstructed unit is indistinguishable from external drive, wherever it sits. Within the
segment the connectome is known, with the six scaling factors of Figure 1, and its recurrent
units are simulated. Every unit outside it keeps its recorded activity and reaches the
modelled neurons through learnt weights: one rank-1 block `exp(U V)` per source type (mitral,
E, I) and target type (E, I), each initialised at the mean teacher weight of its block.

Which recurrent neurons can be observed is fixed once per seed: a pool of 50% of the 5000
(`recorded_pool_fraction = 0.5`), the same at every level. Observed neurons are the
reconstructed neurons in that pool; unobserved neurons are the reconstructed neurons outside
it, simulated but never in the loss. So the number of neurons in the loss falls with
reconstruction (about 2500 at 100%, 240 at 10%). This is intrinsic rather than a confound: an
unreconstructed neuron's activity is used as input and cannot also be predicted.

Learnt weights reach every modelled neuron, observed and unobserved alike. The block onto
unobserved neurons is constrained only through their influence on observed ones, which is
the mechanism of the degeneracy (see [`../METHODS.md`](../METHODS.md#teacher-forcing)).

| Setting | Value |
|---|---|
| Reconstructed fraction | 0.1, 0.2, ..., 1.0 |
| Recorded pool | 50% of recurrent neurons, fixed per seed |
| Trained parameters | 6 scaling factors + 6 rank-1 learnt blocks (about 13,000–15,000; 6 at 100%) |
| Epochs | 100 |
| Seeds | 3 (44, 45, 46), plus one fully observed control run |

## Results

Fluctuation R², mean over three seeds:

| Reconstructed | Observed | Unobserved | ΔFluct. R² perturbation, observed | ΔFluct. R² perturbation, unobserved |
|---|---|---|---|---|
| 100% | 0.97 | 0.97 | 0.93 | 0.93 |
| 90% | 0.86 | 0.72 | 0.67 | 0.60 |
| 70% | 0.75 | 0.42 | 0.40 | 0.31 |
| 50% | 0.68 | 0.14 | 0.24 | 0.07 |
| 30% | 0.61 | −0.05 | 0.17 | 0.07 |
| 10% | 0.51 | −0.15 | 0.08 | −0.11 |

The noise ceiling is 1.00 throughout (0.997 at 100%). The observed fit declines gently, while
prediction of the unobserved neurons and of the perturbation response collapses within the
first 30% of units left unreconstructed. The fraction of each neuron's input that is known
(κ, a column of `fig07_summary.csv`) matches the reconstructed fraction to within 0.01.

A fully observed control at 10% reconstructed (every reconstructed neuron in the loss, one
seed) reaches 0.71 on observed neurons, against 0.97 at full reconstruction: recording more
neurons does not substitute for reconstructing them. It is scored in the CSV
(`recorded_pool_fraction` 1.0) but not plotted.

## Panels

- `fig07-a-curve.svg` — Fluctuation R² against reconstructed fraction, observed and
  unobserved, individual seeds, with the grey band marking the estimated real-dataset level
  (about 10% reconstructed).
- `fig07-b-scatter-50pct.svg` — teacher vs student firing rate at 50% reconstructed,
  observed | unobserved.
- `fig07-c-delta-fluctuation.svg` — perturbation ΔFluctuation R² against reconstructed
  fraction, observed and unobserved (E and I pooled), with the same band.

## Running

```bash
./run --grid fig07-partial-reconstruction/experiment.toml
uv run python fig07-partial-reconstruction/analysis.py
uv run python fig07-partial-reconstruction/figures.py
```

The grid is 31 runs (10 levels × 3 seeds, plus the fully observed control). A run takes
about 7 h at 100% reconstructed and 4.5 h at 10% on a Quadro RTX 5000.

## Notes

- The 100% level is not Figure 1: it has no learnt weights but uses this figure's training
  recipe (100 epochs, smaller learning rates for the scaling factors), so it is the
  consistency check at the end of the curve.
- Unreconstructed E and I sources get their own learnt blocks, separate from the mitral
  ones, because they drive different synapse types.
- The CSVs key unobserved neurons as `heldout`; the figures label them unobserved.
- Every learnt block carries an L1 penalty, 5000 × mean(`exp(U V)`).
