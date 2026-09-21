# Figure 3 — How few neurons do you need to observe?

> Model, teacher forcing, metrics, the noise ceiling and naming: see [`../METHODS.md`](../METHODS.md).

**Claim:** with the connectome fully reconstructed, unobserved neurons are recovered even when the great majority of the network is never observed — and there is a threshold below which this fails.

This is the spiking-network version of Beiran & Litwin-Kumar's central result. Their theory predicts the required number of observed neurons tracks the **dimensionality of the activity**, not network size. Finding the break point turns this figure from a robustness check into a direct test of that prediction.

## Configuration

Identical to Figure 1 except for the swept factor:

| Parameter | Value |
|---|---|
| feedforward connections | reconstructed |
| recurrent reconstruction | 100% |
| weight noise | 0 |
| trained parameters | 6 scaling factors |
| **observed fraction** | **swept** |
| seeds | 3 per point (44, 45, 46) |

### Sweep

`obs ∈ {50, 25, 10, 5, 2, 1}%` — i.e. 2500 → 50 observed neurons out of 5000.

Figure 1's baseline sits at obs = 50%, so the interesting region is **below** it: a clean fit at
50% observed says nothing about where the limit is, and the point of the figure is to go well
past 90% unobserved.

The x axis is plotted in percentages, decreasing left to right, with a second x-axis in
**number of observed neurons**, since that is the quantity the theory is stated in and the
quantity an experimentalist plans around.

## Evaluation

- Held-out test set of new stimuli.
- **Fluctuation R²** primary, Activity R² secondary.
- Two series: **observed** and **unobserved** neurons.
- **Noise ceiling** (the perfectly specified student under the same forcing and flips); no floor is plotted -- the shuffled-identity floor was dropped on 2026-09-17.

## Panels

As built (2026-09-21), one SVG each:

- **(a)** `fig03-a-curve` — Fluctuation R² vs observed fraction, observed and unobserved, individual seeds, no error bars, 90%-variance marker, x decreasing left to right in percentages.
- **(b)** `fig03-b-scatter` — unobserved firing rates at three observed fractions, percentages in the titles.
- **(c)** `fig03-c-delta-fluctuation` — perturbation ΔFluctuation R², same axis treatment as (a), cell types pooled.

Panel (a) carries no title: the slide does. The marker on its observed-neuron axis is the
90%-variance PC count, not the participation ratio: at 41 neurons (0.8%) the participation
ratio sits below everything tested and would imply the opposite of what the sweep shows.

Panel (b)'s three fractions are chosen automatically (comfortably above the break, near it,
and the lowest run) and can be overridden with `--scatter-fractions`.

## Files

```
fig03-observed-fraction/
  analysis.py
  figures.py
  run_grid_search.py
  train.py
  experiment.toml
  parameters.toml
  README.md
  fig03_rates.csv
  fig03_summary.csv
  fig03-a-curve.svg
  fig03-b-scatter.svg
  fig03-c-delta-fluctuation.svg
```

## Notes

- Which neurons are observed is **random** and re-drawn per seed, so the curve is not an artefact of one lucky subset. Worth stating on the slide, because non-random observation is the realistic case and a natural question.
- If the break is sharp, that is the more striking result and deserves its own sentence: "below ⟨N⟩ observed neurons, the model no longer recovers the rest".

---

## Implementation (recorded settings)

### How to run

```bash
./run --grid fig03-observed-fraction/experiment.toml   # 5 fractions x 3 seeds = 15 runs
uv run python fig03-observed-fraction/analysis.py       # reads Figure 1's runs for the 50% point
uv run python fig03-observed-fraction/figures.py        # panel SVGs (--scatter-fractions to choose panel b)
```

Identical to Figure 1 (student, recipe, evaluation — see its README) except
`[student].observed_fraction`. The 50% point **is** Figure 1's runs, so it is not trained again
here.

| obs fraction | 0.50 (Fig 1) | 0.25 | 0.10 | 0.05 | 0.02 | 0.01 |
|---|---|---|---|---|---|---|
| observed neurons | 2500 | 1250 | 500 | 250 | 100 | 50 |

- Observed neurons are drawn at random and re-drawn per seed (independent random stream).
- Runs are ordered seed by seed, and within a seed from 1% upward — the break is expected
  at the low end, so a truncated grid still covers it.
- 1% and 2% are the unreliable end: below ~25% observed a run either trains or collapses
  (the unobserved *and* observed excitatory populations fall silent, whatever the
  rate-penalty targets or learning rate — probes on 2026-09-18). They are kept
  deliberately, to show where the fit stops being reliable, and the per-seed points
  therefore matter more than the mean.
- 0.5% observed finished for seed 44 only and is not reported; its run folder is still on
  disk, so adding the fraction back to `REPORTED_FRACTIONS` in `analysis.py` would include
  it again.

### Dimensionality

Computed once for the teacher by `generate-teacher-activity/dimensionality.py` (not by this
figure's analysis): all 5000 neurons, all 50 training trials with the first 2 s of each
discarded, smoothed with the same 50 ms Gaussian as Fluctuation R², then the participation
ratio (Σλ)²/Σλ² of the pooled neuron × neuron covariance: **41.4** (90% of variance in 236
PCs). `figures.py` reads `generate-teacher-activity/teacher_dimensionality.csv` and marks the
90%-variance count on panel (a)'s observed-neuron axis rather than in a separate panel.

The pooled training-trial estimate is used in preference to a held-out one (2026-09-17): the
single 13 s held-out trial gives 25.9, a noisier, lower estimate from fewer activity
patterns.
