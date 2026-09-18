# Figure 4 — Reconstruction errors: does it matter *how* you lose input?

> **Read `../METHODS.md` first** — model, teacher forcing, metrics, floor, seeds, naming, and the instruction to make only minimal edits to the existing code.

**Claim:** prediction degrades with the amount of synaptic input missing from the model — but two error models that remove the same amount of input do not cost the same. Losing a whole presynaptic *source* is worse than losing scattered synapses.

If that holds, it is the result that motivates everything later about where reconstruction effort should go. If both models collapse onto one curve, the story simplifies to "only the amount matters" — also worth knowing, and worth saying plainly.

## Configuration

Identical to Figure 1 except for the swept degradation:

| Parameter | Value |
|---|---|
| feedforward connections | reconstructed |
| weight noise | 0 |
| trained parameters | 6 scaling factors |
| observed fraction | **10% of retained neurons** (matches Figure 1) |
| seeds | ≥3 per point |

### Error models

| Model | What is removed | Note |
|---|---|---|
| **(a) Neuron removal** | a random subset of neurons is deleted from the student entirely — all their synapses go with them | matches the real situation: an unproofread cell is absent from the model. Removed neurons are necessarily unobserved. |
| **(b) Synapse dropout** | a random subset of *synapses* is deleted; every neuron remains in the model | removes uncorrelated fragments of many sources rather than whole sources |

Sweep each to cover the same range of input volume lost, roughly 0 → 50%.

## The shared axis

For each neuron *i*, and each degradation:

```
κ_lost(i) = Σ|w_ij| over removed synapses  /  Σ|w_ij| over all of i's teacher synapses
```

Report both the **mean over neurons** (the x-axis of panel a) and the **per-neuron value** (the covariate in panel b). Computed from the connectivity matrix and the removal mask — no simulation.

This is the same quantity as input completeness in the real Dp connectome, so this figure and the connectome-cost figure later in the talk share an axis.

## Evaluation

- Held-out test set of new stimuli.
- **Fluctuation R²** primary, Activity R² secondary.
- Evaluated on **unobserved retained** neurons (the discriminative group, comparable to Figures 1–3).
- Shuffled-identity floor.

## Panels

- **(a)** Fluctuation R² vs **mean input volume lost**, two series (neuron removal, synapse dropout), error bars over seeds, floor dashed. Separation between the series is the result.
- **(b)** **Per-neuron** Fluctuation R² vs **per-neuron κ_lost**, both models pooled, coloured by model. Do they lie on one curve? Under neuron removal the spread in κ_lost across cells should be much wider, which is itself informative.
- **(c)** *optional* — same, split E vs I.

## Files

```
fig04-reconstruction-errors/
  README.md
  fig04_summary.csv      error_model{neuron_removal,synapse_dropout}, level, mean_kappa_lost, seed, metric, value, floor_value
  fig04_per_neuron.csv   error_model, level, seed, neuron_id, cell_type, kappa_lost, fluctuation_r2, activity_r2
  plot_fig04.py
  fig04.svg
  config.yaml
```

## Status

**Needs running.**

The archived `sweep-recur__ff-known__obs-100__wn-0__curve.svg` is error model (a) and is clean in itself, but sits at **obs-100**, so its metric is computed on observed neurons and it is not comparable with Figures 1–3. Rerun at obs-10 of retained neurons so the whole sequence shares a configuration. Its shape (R² 0.97 at 5% removed → 0.53 at 50% removed) is a useful prior for choosing sweep levels.

Error model (b) does not exist yet.

## Notes

- Removed neurons cannot be observed — removal implies unobserved. Define the observed fraction relative to **retained** neurons so it stays 10% at every degradation level, otherwise observation and reconstruction vary together.
- Random removal loses roughly the same mean volume as its neuron fraction, so panel (a)'s x-axis will look similar to "fraction removed". The value of the volume axis is that it makes model (b) comparable and gives panel (b) its covariate.
- If the two models separate, the one-sentence version for the talk is: *what you are missing matters more than how much.*

---

## Implementation (recorded settings)

*Added when the code was written.*

### How to run

```bash
./run --grid fig04-reconstruction-errors/experiment.toml   # 2 models x 5 levels x 3 seeds = 30 runs
uv run python fig04-reconstruction-errors/analysis.py       # reads Figure 1's runs as level 0
uv run python fig04-reconstruction-errors/figures.py        # fig04.svg
```

Identical to Figure 1 except one of two `[student]` entries per run. Level 0 of both models
is Figure 1's runs. Synapse dropout (the new arm) is ordered before neuron removal, and runs
are ordered seed by seed. `PRIORITY.md` budgets 2 seeds: set `SEEDS = [44, 45]`.

### Error models (`common/structure.py`)

- **Neuron removal** (`neuron_removal_fraction` ∈ {0.1, 0.2, 0.3, 0.5}): a random subset of
  recurrent neurons is deleted from the student — it neither simulates them nor receives
  their spikes (the archived hidden-units semantics). Observed neurons are 10% of the
  **retained** neurons, drawn after removal.
- **Synapse dropout** (`synapse_dropout_fraction` ∈ {0.1, 0.2, 0.3, 0.5}): each recurrent
  synapse is deleted independently with that probability; every neuron stays.
- Feedforward (mitral) input is always fully reconstructed.

### κ_lost

Per retained neuron i: Σ teacher recurrent weights onto i from removed neurons / dropped
synapses, divided by Σ all teacher recurrent weights onto i. The denominator is recurrent
input only, so that neuron removal at fraction p gives mean κ ≈ p as the README expects;
including the always-reconstructed mitral input would compress both axes. Stored per neuron
in `student_structure.npz` and exported to both CSVs.

### Departure from the README file spec

`fig04_per_neuron.csv` has no `activity_r2`: R² needs many samples, and a single neuron has
one rate. It carries `teacher_rate_hz` and `student_rate_hz` instead, alongside the per-neuron
`fluctuation_r2` (over time) and `observed` (the figure uses unobserved neurons).

### Panels as implemented

(a) Unobserved Fluctuation R² vs mean κ_lost, both models (mean ± SD, seeds as dots), each
model's ceiling (dotted) and floor (dashed). (b) Per-neuron Fluctuation R² vs per-neuron κ_lost,
unobserved neurons at all non-zero levels and seeds pooled (subsampled to 20k points), coloured
by model, with binned medians. The optional E/I split (c) is not drawn; `cell_type` is in the CSV.
