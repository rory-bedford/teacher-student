# Figure 3 — Observed fraction

How many neurons have to be observed for the student to predict the rest? Down to 25%
observed a run still predicts the unobserved neurons well (0.94 for two of three seeds),
but below that training becomes unreliable: a run either fits normally or its excitatory
population falls silent. At the roughly 15% we expect for the real dataset, the runs that
train reach 0.82–0.84 on the unobserved neurons.

Shared methods (model, training, evaluation, noise ceiling): [`../METHODS.md`](../METHODS.md).

## Experiment

Everything is as in Figure 1 (full connectome, six scaling factors, no weight noise)
except the fraction of recurrent neurons observed, which falls from 50% to 1%, that is
from 2500 to 50 of the 5000 neurons. Which neurons are observed is drawn at random and
re-drawn per seed. The x axis is logarithmic and decreases left to right, with a second
axis giving the number of observed neurons.

| Setting | Value |
|---|---|
| Observed fraction | 25, 10, 5, 2, 1%; 50% is Figure 1's runs |
| Observed neurons | 1250, 500, 250, 100, 50 (2500 at 50%) |
| Connectome | feedforward and recurrent, 100% reconstructed, no weight noise |
| Trained parameters | 6 scaling factors |
| Epochs | 50 |
| Seeds | 3 (44, 45, 46), each drawing its own observed set |

## Results

Fluctuation R², mean over three seeds:

| Observed | Observed | Unobserved | ΔFluct. R² perturbation, observed | ΔFluct. R² perturbation, unobserved |
|---|---|---|---|---|
| 50% (Figure 1) | 0.97 | 0.97 | 0.94 | 0.94 |
| 25% | 0.83 | 0.76 | 0.70 | 0.68 |
| 10% | 0.56 | 0.46 | 0.33 | 0.35 |
| 5% | 0.51 | 0.38 | 0.23 | 0.17 |
| 2% | 0.23 | 0.20 | 0.04 | 0.09 |
| 1% | 0.13 | 0.15 | −0.13 | −0.07 |

The means hide a bimodal outcome. At 10% observed, seeds 44 and 46 reach 0.84 and 0.82 on
the unobserved neurons while seed 45 scores −0.28: its excitatory neurons fire at 0.3–0.5 Hz
against the teacher's 4 Hz. Yet the same seed trains normally at 2% (0.67), and seed 44,
which collapses at 1% and 2%, trains at 5% and 10%. Failure does not follow the amount
observed, so it is an optimisation failure rather than a limit on the information in the
data. This is why the panels show individual seeds rather than means.

The noise ceiling itself falls as fewer neurons are observed: 0.98 at 25%, 0.92 at 10% and
0.66–0.80 at 1% held-out (0.99 down to 0.28–0.34 for the perturbation).

## Panels

- `fig03-a-curve.svg` — Fluctuation R² on held-out stimuli against observed fraction,
  observed and unobserved, individual seeds, with the grey band marking the estimated
  real-dataset level (about 15% observed).
- `fig03-b-scatter.svg` — teacher vs student firing rate of the unobserved neurons at 50%,
  10% and 1% observed, one seed.
- `fig03-c-delta-fluctuation.svg` — perturbation ΔFluctuation R² against observed fraction,
  observed and unobserved (E and I pooled, targeted cells excluded), with the same band.

## Running

```bash
./run --grid fig03-observed-fraction/experiment.toml
uv run python fig03-observed-fraction/analysis.py
uv run python fig03-observed-fraction/figures.py
```

`analysis.py` reads Figure 1's runs as the 50% level. The grid is 15 runs (5 fractions ×
3 seeds), each about 3.6 h on a Quadro RTX 5000, as in Figure 1. Panel (b)'s fractions can
be chosen with `figures.py --scatter-fractions`.

## Notes

- In a collapsed run the observed excitatory neurons fall silent along with the unobserved
  ones. Changing the targets of the unobserved-rate penalties or the learning rate did not
  prevent it, so the failing levels are kept to show where the fit stops being reliable.
- The ceiling falls because fewer neurons receive recorded input: more of the student runs
  free, and the teacher's chaos decorrelates it sooner (see teacher forcing in METHODS).
- Observed neurons are a uniform random sample. Real recordings are not random, which this
  figure does not test.
