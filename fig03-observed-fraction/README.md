# Figure 3 — How few neurons do you need to observe?

> **Read `../METHODS.md` first** — model, teacher forcing, metrics, the noise ceiling, seeds, naming, and the instruction to make only minimal edits to the existing code.

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
| seeds | ≥3 per point |

### Sweep

`obs ∈ {50, 25, 10, 5, 2, 1, 0.5}%` — i.e. 2500 → 25 observed neurons out of 5000.

The existing clean run sits at obs = 10% (2 s.f.: observed 0.997, unobserved 0.995), so the interesting region is **below** it. Going well past 90% unobserved is the point of the figure: 0.995 at 10% observed says nothing about where the limit is.

Plot a second x-axis in **number of observed neurons**, since that is the quantity the theory is stated in and the quantity an experimentalist plans around.

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

Panel (a) carries no title: the slide does. The participation ratio is *not* marked --
at 41 neurons (0.8%) it sits below everything tested and would imply the opposite of what
the sweep shows.

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

## Status

**Needs running.** One point exists (obs = 10%, reusable from Figure 1). The rest of the sweep, and especially the sub-10% region where the break should appear, does not.

## Notes

- Which neurons are observed should be **random** and re-drawn per seed, so the curve is not an artefact of one lucky subset. Worth stating on the slide, because non-random observation is the realistic case and a natural question.
- If the break is sharp, that is the more striking result and deserves its own sentence: "below ⟨N⟩ observed neurons, the model no longer recovers the rest".

## Open questions

1. Is the participation ratio of the teacher's activity already computed anywhere? If not it is a few lines and makes panel (c) possible.

---

## Implementation (recorded settings)

*Added when the code was written.*

### How to run

```bash
./run --grid fig03-observed-fraction/experiment.toml   # 6 fractions x 3 seeds = 18 runs
uv run python fig03-observed-fraction/analysis.py       # reads Figure 1's runs for the 10% point
uv run python fig03-observed-fraction/figures.py        # fig03.svg (--scatter-fractions to choose panel b)
```

Identical to Figure 1 (student, recipe, evaluation — see its README) except
`[student].observed_fraction`. The 10% point **is** Figure 1's runs.

| obs fraction | 0.5 | 0.25 | 0.10 (Fig 1) | 0.05 | 0.02 | 0.01 | 0.005 |
|---|---|---|---|---|---|---|---|
| observed neurons | 2500 | 1250 | 500 | 250 | 100 | 50 | 25 |

- Observed neurons are drawn at random and re-drawn per seed (independent random stream).
- Runs are ordered seed by seed, and within a seed from 0.5% upward — the break is expected
  below 10%, so a truncated grid still covers it.
- `PRIORITY.md` budgets 5 levels; the README's 7 are implemented. Drop entries from
  `OBSERVED_FRACTIONS` in `run_grid_search.py` to match the budget.

### Dimensionality (panel c, open question 1)

Computed once for the teacher by `generate-teacher-activity/dimensionality.py` (not by this
figure's analysis): all 5000 neurons, all 50 training trials with the first 2 s of each
discarded, smoothed with the same 50 ms Gaussian as Fluctuation R², then the participation
ratio (Σλ)²/Σλ² of the pooled neuron × neuron covariance: **41.4** (90% of variance in 236
PCs). `figures.py` reads `generate-teacher-activity/teacher_dimensionality.csv` and marks the
participation ratio on panel (a)'s observed-neuron axis rather than a separate panel (c).

Departure from the spec above (2026-09-17): the spec asks for held-out stimuli; the single
13 s held-out trial gives 25.9, a noisier, lower estimate from fewer activity patterns, so the
pooled training-trial value is used.

### Panels as implemented

See **Panels** above. Superseded details from the original spec: no floor, individual seeds
instead of error bars, the x axis decreases left to right in percentages, and the marker is
the 90%-variance count rather than the participation ratio. Panel (b)'s three fractions are
still chosen automatically (comfortably above the break, near it, and the lowest run) and can
be overridden with `--scatter-fractions`.
