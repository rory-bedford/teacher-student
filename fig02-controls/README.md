# Figure 2 — The connectome is what's doing the work

> **Read `../METHODS.md` first** — model, teacher forcing, metrics, floor, seeds, naming, and the instruction to make only minimal edits to the existing code.

**Claim:** the recovery in Figure 1 comes from the measured connectome, not from the flexibility of the model. Controls that discard or scramble connectivity fail, at the same training budget and on the same held-out stimuli.

**The headline contrast:** 6 parameters with the connectome beat 25 million parameters without it.

## Observed fraction: 50%, not 10% (2026-09-18)

At 10% observed both wrong-connectome controls (shuffled weights, configuration model) lost their
unobserved excitatory population entirely (0.36-0.38 Hz vs the teacher's 4.5) while still lowering
the loss through inhibition alone; the configuration model did the same under the earlier VR-only
recipe (0.02 Hz), so this is the regime rather than the recipe. With 90% of each neuron's recurrent
input coming from the student's own spikes, a wrong connectome has nothing holding it near the
teacher's operating point. The archived controls figure
(`archive/figures/data/controls__ff-learnt__obs-90__recur-81__wn-mixed__bar.csv`) was *fully
observed* and gave Full Connectome 0.91 / Learnt-Recurrence 0.61 / Shuffle-Connections 0.55 /
Shuffle-Weights -0.26 Fluctuation R².

So the controls now run at **50% observed**, which matches the reconstruction budget of the real
dataset and is the healthiest point of Figure 3's sweep. The full-connectome baseline for the bars
is Figure 3's `obs-0.5` runs (same recipe and seeds), not Figure 1. The 10% runs are kept in
`bernstein/_superseded/fig02-controls-obs-0.1/` as the collapse observation.

## Configuration

Identical to Figure 1 in every respect except the connectivity given to the student:

| Parameter | Value |
|---|---|
| feedforward connections | reconstructed |
| recurrent reconstruction | 100% |
| observed fraction | **50%** (2026-09-18; was 10%, see below) |
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
| Learnt recurrence | every recurrent block replaced by a dense, full-rank matrix of free log-weights, shared across the two layers; true feedforward pattern and its 2 scaling factors kept (see below) | **25,000,002** (5000² + 2) |
| Shuffle weights | non-zero weights permuted within each E/I block; topology untouched (archived function) | 6 |
| Configuration-model rewire | within each block, in-stubs randomly re-paired with out-stubs; self-connections and duplicate synapses repaired by swapping targets with random distinct edges; the block's weights then randomly reassigned | 6 |

Verified on the full 5000-neuron matrix (seed 44): the configuration model preserves every
in- and out-degree and the weight multiset in all four blocks, has no self-connections, and
shares 6–7% of the teacher's synapses (= the connection density, i.e. chance). Shuffle
weights keeps the topology identical.

### What the learnt-recurrence control is — state this on the slide

It isolates the **recurrent** connectome and nothing else. Exactly like every other variant
(and Figure 1), the student is given:

- the **true feedforward weight pattern** (mitral -> E/I), which is perturbed by an unknown
  per-pathway factor and rescaled by **2 learnt feedforward scaling factors** (mitral->E,
  mitral->I). So it knows *which* inputs each neuron receives, not their absolute scale;
- the true recorded activity of the feedforward units and of the 10% observed neurons
  (teacher forcing), identical held-out evaluation, identical training budget (50 epochs).

What it does **not** get is the recurrent connectome: every recurrent block (E->E, E->I, I->E,
I->I) is a **dense, fully connected, full-rank matrix of free weights** — 5000² = 25,000,000
parameters, shared between the simulated unobserved population and the observed neurons — with no
scaling factors on top. Total free parameters: **25,000,002** (vs 6 with the connectome). Suggested
slide wording: *"learnt recurrence: all 25M recurrent weights free; feedforward pattern and
recordings as for the connectome model"*.

Because it is handed the feedforward pattern, this control is **conservative** — it starts with
more of the true circuit than a generic data-constrained RNN would. A version that also learns the
feedforward weights was considered and not included.

### Learnt recurrence — settings

| Setting | Value | Why |
|---|---|---|
| parametrisation | log-weights, `exp(W)`, all pairs, per E/I block | archived no-connectome control |
| initialisation | each block at its **mean weight including absent synapses** × perturbation | density-matched: every neuron starts with the teacher's total recurrent drive per block |
| learning rate | Adam, **5e-3 → 5e-4** cosine over 50 epochs | weights move ~lr per update in log space; ~600 updates |
| gradient clip | 100 | archived |
| feedforward scaling factors | lr 8e-3 → 5e-4, clip 5 | as Figure 1 |

**History (smoke run, 2026-09-17).** The first version used the archived initialisation
(fully connected at the mean *non-zero* weight, ≈16× the teacher's drive at ~6% density) and lr
1e-3 → 5e-4. In 50 epochs it did not fit even the observed neurons: van Rossum loss 433 -> 269
(Figure 1: 337 -> 88), observed E rate 2.8 vs 4.5 Hz, mitral->E scaling factor compensating to 4.7×.
Held-out unobserved Fluctuation R² was 0.04 — but that could not be separated from bad
initialisation, so the initialisation and learning rate were changed as above before running the
grid. The superseded run is kept at
`bernstein/fig02-controls/_superseded-learnt-meannonzero-init__seed-44/`.

### Panels as implemented

(a) Fluctuation R², bars = variant × {observed (light), unobserved (dark)}, mean ± SD with
seeds as dots, floor (dashed) and ceiling (dotted) per bar. (b) Activity R², same layout.
(c) The archived `perturbations.svg` does not exist anywhere in the repo or in
`dp-simulations/`, so the schematic is generated: a toy 20-neuron E/I connectome passed
through the real shuffle and configuration-model functions, plus the dense learnt matrix,
each titled with its free-parameter count.
