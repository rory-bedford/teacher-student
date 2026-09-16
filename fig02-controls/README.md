# Figure 2 — The connectome is what's doing the work

> **Read `../METHODS.md` first** — model, teacher forcing, metrics, floor, seeds, naming, and the instruction to make only minimal edits to the existing code.

**Claim:** the recovery in Figure 1 comes from the measured connectome, not from the flexibility of the model. Controls that discard or scramble connectivity fail, at the same training budget and on the same held-out stimuli.

**The headline contrast:** 6 parameters with the connectome beat 25 million parameters without it.

## Configuration

Identical to Figure 1 in every respect except the connectivity given to the student:

| Parameter | Value |
|---|---|
| feedforward connections | reconstructed |
| recurrent reconstruction | 100% |
| observed fraction | **10%** (same as Figure 1) |
| weight noise | **0** — for every variant |
| seeds | **≥3 per variant** |
| training budget | identical across variants |

### Variants

| Variant | What the student gets | Free parameters |
|---|---|---|
| **Full connectome** | true topology and weights | **6** (FF→E, FF→I, E→E, E→I, I→E, I→I scalings) |
| **Learnt recurrence** | no connectome; all recurrent weights free | ~5000² = **25 M** |
| **Shuffle weights** | true topology, weights permuted among existing synapses | 6 |
| **Configuration-model rewire** | rewired **within each block** (E→E, E→I, …) preserving in/out degree sequences and the within-block weight distribution | 6 |

**Why the configuration model rather than a random within-block rewire.** A fully random rewire destroys degree heterogeneity as well as specific wiring, so a failure is attributable to either. The configuration model preserves block statistics *and* degree and destroys only the specific wiring — the stringent null. If the full connectome still wins against it, that is the strong claim.

**Why shuffle-weights stays.** It tests whether the *weight values* carry information given correct topology — which in the real pipeline is exactly the question of whether synapse volumes are informative. It is the control closest to the assumption the Dp model rests on.

The configuration model **replaces** the naive random within-block rewire; it is not an extra bar. Four variants total.

## Evaluation

- Held-out test set of new stimuli.
- **Fluctuation R²** primary (50 ms Gaussian smoothing of spikes; stand-in for the τ = 100 ms calcium filter we match against), Activity R² secondary.
- Reported separately for **observed** and **unobserved** neurons. Unobserved is the discriminative group — every variant can fit what it is shown.
- Shuffled-identity floor per variant.

## Panels

- **(a)** Grouped bars: variant × {observed, unobserved}, **Fluctuation R²**, error bars over seeds, floor as a dashed line.
- **(b)** Activity R², same layout, smaller or inset. The variants are expected to separate far less here — that contrast is worth showing, since it is *why* Fluctuation R² is primary.
- **(c)** Schematic of the connectivity variants (archived `perturbations.svg` does this; relabel each panel with its variant name).

## Files

```
fig02-controls/
  README.md
  fig02_summary.csv     variant, seed, group{observed,unobserved}, metric{activity_r2,fluctuation_r2}, value, floor_value
  fig02_rates.csv       variant, neuron_id, cell_type, observed{0,1}, seed, teacher_rate_hz, student_rate_hz
  plot_fig02.py
  fig02.svg
  config.yaml
```

## Status

**Needs rerunning — confirmed.** The archived controls bar chart ran at 20% or 30% weight noise depending on the bar, and at obs-90 / recur-81. Bars at different weight noise are not comparable to each other, and none of them is comparable to Figure 1.

Four variants (five with the configuration-model shuffle) × ≥3 seeds, at the Figure 1 configuration with **wn = 0 throughout**.

## Notes

- In the archived version, Activity R² barely separated Full Connectome (~0.97) from Learnt Recurrence (~0.90) and Shuffle Weights (~0.90), while Fluctuation R² separated them clearly (~0.90 / ~0.60 / ~0.55) and Shuffle Connections went negative. If that survives the rerun, state it: mean rates are easy to match, so the connectome earns its place on the fluctuations.
- Learnt recurrence is the control people will ask about — it is the standard data-constrained RNN. Put its parameter count beside the constrained model's 6 on the slide.

---

## Implementation (recorded settings)

*Added when the code was written.*

### How to run

```bash
./run --grid fig02-controls/experiment.toml   # 3 controls x 3 seeds = 9 runs -> bernstein/fig02-controls/
uv run python fig02-controls/analysis.py       # reads Figure 1's runs too; fig02_summary.csv, fig02_rates.csv
uv run python fig02-controls/figures.py        # fig02.svg
```

**Full connectome is not retrained**: it is Figure 1's three seeds, read by `analysis.py`.
Everything else — student, observed split per seed, perturbation, recipe, evaluation — is
Figure 1's (see its README); only `[student].recurrent_model` changes. Runs are ordered seed
by seed, so all controls get one seed before any gets a second. Cost ≈ 3.6 h per run as
Figure 1 for the two six-parameter controls (learnt recurrence: see below).

### Variants as implemented (`common/structure.py`, `common/model.py`)

| Variant | Implementation | Free parameters |
|---|---|---|
| Full connectome | Figure 1 | 6 |
| Learnt recurrence | every recurrent block replaced by a full-rank, fully connected log-weight matrix, shared across the two layers; mitral connectome and its 2 scaling factors kept | **25,000,002** (5000² + 2) |
| Shuffle weights | non-zero weights permuted within each E/I block; topology untouched (archived function) | 6 |
| Configuration-model rewire | within each block, in-stubs randomly re-paired with out-stubs; self-connections and duplicate synapses repaired by swapping targets with random distinct edges; the block's weights then randomly reassigned | 6 |

Verified on the full 5000-neuron matrix (seed 44): the configuration model preserves every
in- and out-degree and the weight multiset in all four blocks, has no self-connections, and
shares 6–7% of the teacher's synapses (= the connection density, i.e. chance). Shuffle
weights keeps the topology identical.

### Learnt recurrence — settings and a concern

Archived no-connectome control (`full-inference/no-hidden-units`, phase 2): initialised
**fully connected at each block's mean non-zero teacher weight** (times the perturbation);
Adam lr 1e-3 → 5e-4, gradient clip 100, trained alongside the mitral scaling factors (lr
8e-3 as Figure 1). Same epochs and data as every other variant.

Flag before running: that initialisation gives each neuron roughly 1/density ≈ 16× the
teacher's total recurrent drive. With ~600 optimiser updates at lr ≈ 1e-3, log-weights can
move by less than the ≈2.8 needed to undo it, so this control may fail from its initialisation
rather than from lacking the connectome. The archived run had 250 phase-2 epochs to recover.
If that matters for the claim, a density-matched initialisation (block mean *including* zeros)
is a one-line change in `common/model.py` (`learnt_recurrence_block`).

### Panels as implemented

(a) Fluctuation R², bars = variant × {observed (light), unobserved (dark)}, mean ± SD with
seeds as dots, floor (dashed) and ceiling (dotted) per bar. (b) Activity R², same layout.
(c) The archived `perturbations.svg` does not exist anywhere in the repo or in
`dp-simulations/`, so the schematic is generated: a toy 20-neuron E/I connectome passed
through the real shuffle and configuration-model functions, plus the dense learnt matrix,
each titled with its free-parameter count.
