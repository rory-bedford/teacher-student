# Figure 6 — Learnt input fits the cells it sees and fails on the rest

> Shared methods — model, teacher forcing, metrics, the noise ceiling, naming: see [`../README.md`](../README.md).

**Claim:** give the student the whole recurrent connectome but make it *learn* the
feedforward input, and it still fits the neurons in its loss while losing the ability to
predict the ones that are not. Fitting and predicting come apart, with no connectivity
missing from the recurrent network at all.

This is the cleanest form of the project's unifying idea — **an unreconstructed unit is
indistinguishable from external drive** — because only one thing is unknown. Figure 7
makes feedforward and recurrent units unknown together and sweeps how many; this isolates
the feedforward half at full strength.

## Configuration

Identical to Figure 1 except that the feedforward weights are learnt instead of given:

| Parameter | Value |
|---|---|
| recurrent connectome | **given**, with 4 learnt scaling factors (see below) |
| feedforward connectome | **learnt**, every weight |
| observed fraction | 50% of modelled neurons, as every other figure |
| weight noise | 0 |
| seeds | 3 (44, 45, 46) |

### What is learnt

| | |
|---|---|
| feedforward weights | **12,000,000** — `exp(U V)` at **full rank** for each of the two blocks: mitral→E (U 1500×1500, V 1500×4000) and mitral→I (U 1500×1500, V 1500×1000) |
| scaling factors | **4 live** — E→E, E→I, I→E, I→I |
| | **2 dead** — mitral→E and mitral→I exist in the parameter dictionary but scale the *known* mitral rows, of which there are none. They receive exactly zero gradient and never move from their initialisation (verified over a full epoch, 2026-09-23). `n_free_params` in the CSVs therefore over-reports by two, and Figure 1's scaling-factor panel must never be drawn for this figure — those two values would look like a recovery failure for a pathway that has no scaling factor at all. |

**Full rank is free.** `MixedProjection.forward` builds the weight matrix once per chunk,
not per timestep, so rank costs no time: measured 1.41 chunk/s at rank 1500 against 1.37
at rank 20. Running at full rank removes "you did not give it enough capacity" as an
objection to the result.

### Why 200 epochs

Twice the budget that trained Figure 2's 25-million-parameter learnt recurrence to 0.887
on observed neurons, and twice Figure 7's. The archived rank-20 run's cosine similarity to
the teacher's feedforward matrix only starts climbing past ~150 epochs (0.02 at 100, 0.11
at 200, 0.30 at 300, 0.38 at 432) — but that measures **recovery of the true input**,
which this figure does not need and is not helped by: a model that recovered the input
would predict well everywhere and there would be no result. What the figure needs is a
good fit on observed cells, and every comparable run here reaches that within 100 epochs.

`lr_weights = 8e-3` is the one learning rate ever observed to learn the input here; every
rank-1 run at 1e-3, archived and ours, left the feedforward matrix at chance (cosine ~0.01).

## Evaluation

- Held-out test set of new stimuli, and the perturbation, both as everywhere else.
- **Fluctuation R²** primary, reported separately for observed and unobserved neurons.
- The control is **Figure 1's seed-matched runs**, not retrained here: the same student at
  the same 50% observed with the feedforward input given. Both conditions must sit at the
  same observation level or the comparison is not like for like.

## Panels

- **(a)** `fig06-a-bars-held-out` — Fluctuation R², Observed | Unobserved, input given
  beside input learnt, per-seed dots, dotted noise ceiling.
- **(b)** `fig06-b-bars-perturbation` — ΔFluctuation R² on the unobserved population, cell
  types pooled, sharing (a)'s y range.
- **(c)** `fig06-c-legend` — the shared legend, its own file.
- **(d)** `fig06-d-scatter` — teacher vs student firing rate, Observed | Unobserved, learnt
  condition, 400 dpi PNG.

The claim is the gap between the two bars *within* the unobserved panel of (a), against
Figure 1 where both populations sit near the ceiling.

## Files

```
fig06-learnt-feedforward/
  train.py  run_grid_search.py  analysis.py  figures.py
  experiment.toml  parameters.toml  README.md
  fig06_summary.csv  fig06_rates.csv
  fig06-a-bars-held-out.svg  fig06-b-bars-perturbation.svg
  fig06-c-legend.svg  fig06-d-scatter.png
```

## How to run

```bash
./run --grid fig06-learnt-feedforward/experiment.toml   # 3 seeds, ~5 h each on an A40
uv run python fig06-learnt-feedforward/analysis.py       # reads Figure 1's runs as the control
uv run python fig06-learnt-feedforward/figures.py        # panels, from this figure's CSVs
```

Runs write to `bernstein/fig06-learnt-input/`. Note that Figure 7's data directory is
still `bernstein/fig06-learnt-feedforward/` — it was Figure 6 until 2026-09-23 and its 31
finished runs record that path in their provenance, so the directory keeps the old name.

## Status

**Running (2026-09-23).** Seeds 44 and 45 on A40s (`bernstein/slurm/20260923-133124`),
seed 46 locally. Measured rates: 2.10, 1.22 and 0.64 chunk/s, so ~4 h, ~7 h and ~13 h for
the 30,000 chunks. `figures.py` was exercised end to end against a synthetic CSV of the
same schema before the runs finished.
