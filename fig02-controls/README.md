# Figure 2 — The connectome is what's doing the work

> Shared methods — model, teacher forcing, metrics, the noise ceiling, naming: see [`../README.md`](../README.md).

**Claim:** the recovery in Figure 1 comes from the measured connectome, not from the flexibility of the model. Controls that discard or scramble connectivity fail, at the same training budget and on the same held-out stimuli.

**The headline contrast:** 6 parameters with the connectome beat 25 million parameters without it.

## Observed fraction: 50%, not 10% (2026-09-18)

At 10% observed both wrong-connectome controls (shuffled weights, configuration model) lost their
unobserved excitatory population entirely (0.36-0.38 Hz vs the teacher's 4.5) while still lowering
the loss through inhibition alone; the configuration model did the same under the earlier VR-only
recipe (0.02 Hz), so this is the regime rather than the recipe. With 90% of each neuron's recurrent
input coming from the student's own spikes, a wrong connectome has nothing holding it near the
teacher's operating point. The archived controls figure was *fully observed* and gave Full
Connectome 0.91 / Learnt-Recurrence 0.61 / Shuffle-Connections 0.55 / Shuffle-Weights -0.26
Fluctuation R².

So the controls run at **50% observed**, which matches the reconstruction budget of the real
dataset and is the healthiest point of Figure 3's sweep. The full-connectome baseline for the bars
is Figure 1's seed-matched runs: every figure runs at 50% observed as of 2026-09-18. The 10% runs are
kept in `bernstein/_superseded/fig02-controls-obs-0.1/` as the collapse observation.

## Configuration

Identical to Figure 1 in every respect except the connectivity given to the student:

| Parameter | Value |
|---|---|
| feedforward connections | reconstructed |
| recurrent reconstruction | 100% |
| observed fraction | **50%** (2026-09-18; was 10%, see above) |
| weight noise | **0** — for every variant |
| seeds | **3 per variant** (44, 45, 46) |
| training budget | identical across the plotted variants (50 epochs; see the epoch-budget note) |

### Variants

| Variant | What the student gets | Free parameters |
|---|---|---|
| **Full connectome** | true topology and weights | **6** (FF→E, FF→I, E→E, E→I, I→E, I→I scalings) |
| **Learnt recurrence** | no connectome; all recurrent weights free | ~5000² = **25 M** |
| **Shuffled weights** | true topology, each neuron's input weights permuted among its own presynaptic partners, so its total input weight is preserved exactly | 6 |
| **Shuffled topology** (configuration-model rewire) | rewired **within each block** (E→E, E→I, …) preserving in/out degree sequences and the within-block weight distribution | 6 |

**Why the configuration model rather than a random within-block rewire.** A fully random rewire destroys degree heterogeneity as well as specific wiring, so a failure is attributable to either. The configuration model preserves block statistics *and* degree and destroys only the specific wiring — the stringent null. If the full connectome still wins against it, that is the strong claim.

**Why the weight shuffle stays.** It tests whether the *weight values* carry information given correct topology — which in the real pipeline is exactly the question of whether synapse volumes are informative. It is the control closest to the assumption the Dp model rests on.

The configuration model **replaces** the naive random within-block rewire; it is not an extra bar. Four variants total.

## Evaluation

- Held-out test set of new stimuli.
- **Fluctuation R²** primary (50 ms Gaussian smoothing of spikes; stand-in for the τ = 100 ms calcium filter we match against), Activity R² secondary.
- Reported separately for **observed** and **unobserved** neurons. Unobserved is the discriminative group — every variant can fit what it is shown.
- **Noise ceiling** (the perfectly specified student under the same forcing and flips); no floor is plotted — the shuffled-identity floor was dropped on 2026-09-17.
- **As built** (three seeds, held-out Fluctuation R², observed / unobserved): full
  connectome 0.97 / 0.97, learnt recurrence 0.73 / -0.30, shuffled weights -0.15 / -0.18,
  shuffled topology -0.19 / -0.20.

## Panels

One SVG each, subpanels the same size in both figures so they tile on one slide:

- **(a)** `fig02-a-bars-held-out` — held-out Fluctuation R², one subpanel for observed and one for unobserved, one bar per variant.
- **(b)** `fig02-b-bars-perturbation` — perturbation ΔFluctuation R², one subpanel: the unobserved E and I cells pooled, with the targeted cells excluded — the same population every other figure's perturbation panel reports. The targeted cells and the separate cell types are still scored, in `fig02_summary.csv`.
- **(c)** `fig02-c-legend` — the shared legend, stacked vertically, on its own.

Bars are mean ± SD over seeds with the individual seeds as dots and the noise ceiling dotted
per bar. Neither bar figure carries a legend, and both share one y range. The Activity R²
panel of the original design was dropped (rates are reported by the scatter panels of
Figures 1 and 3, and Activity R² stays in the CSVs), as was the connectivity schematic — it
belongs on a slide of its own. The weight shuffle is trained and scored; whether it is
plotted is `PLOTTED_VARIANTS` in `figures.py`.

## Files

```
fig02-controls/
  analysis.py
  figures.py
  run_grid_search.py
  train.py
  experiment.toml
  parameters.toml
  README.md
  fig02_rates.csv
  fig02_summary.csv
  fig02-a-bars-held-out.svg
  fig02-b-bars-perturbation.svg
  fig02-c-legend.svg
```

## Notes

- Activity R² barely separates the variants on the **observed** neurons (full connectome
  0.99, learnt recurrence 0.96): mean rates of neurons the student is shown are easy to
  match. It is the unobserved population, and the fluctuations, on which the connectome
  earns its place — learnt recurrence goes from 0.96 observed to -0.46 unobserved on the
  same metric.
- Learnt recurrence is the control people will ask about — it is the standard
  data-constrained RNN. Put its parameter count beside the constrained model's 6 on the
  slide.
- **There is no graded version of the wrong-connectome controls, and that is a property of
  shuffling, not an omission.** Untrained damage scans: at equal per-neuron weight
  correlation (0.99), weight noise costs 0.22 Fluctuation R² while a shuffle costs 0.78.
  Only shuffling 1–2% of synapses lands anywhere in between, and dynamically that is the
  same manipulation as mild weight noise — which is Figure 5's sweep. So the controls here
  are all-or-nothing by construction, and the graded axis is Figure 5's.

## Implementation (recorded settings)

### How to run

```bash
./run --grid fig02-controls/experiment.toml   # 3 controls x 3 seeds, plus the fully observed check -> bernstein/fig02-controls/
uv run python fig02-controls/analysis.py       # reads Figure 1's runs too; fig02_summary.csv, fig02_rates.csv
uv run python fig02-controls/figures.py        # fig02-a … fig02-c SVGs
```

**Full connectome is not retrained**: it is Figure 1's three seeds, read by `analysis.py`.
Everything else — student, observed split per seed, perturbation, recipe, evaluation — is
Figure 1's (see its README); only `[student].recurrent_model` changes. Runs are ordered seed
by seed, so all controls get one seed before any gets a second. Cost ≈ 3.6 h per run as
Figure 1 for the two six-parameter controls (learnt recurrence: see below).

The grid also trains one **fully observed** full-connectome run per seed
(`connectome-fully-observed__seed-*`). It is no bar in this figure: with every neuron
teacher-forced there is no unobserved population, so the rate penalties do not exist as
loss terms and the six scaling factors are recovered exactly. It is the identifiability
check, read by Figure 1's `analysis.py` for its scaling-factor panel.

### Variants as implemented (`common/structure.py`, `common/model.py`)

| Variant | Implementation | Free parameters |
|---|---|---|
| Full connectome | Figure 1 | 6 |
| Learnt recurrence | every recurrent block replaced by a dense, full-rank matrix of free log-weights, shared across the two layers; true feedforward pattern and its 2 scaling factors kept (see below) | **25,000,002** (5000² + 2) |
| Shuffled weights | each neuron's non-zero input weights permuted among its own presynaptic partners, within cell type; topology and total input untouched (`recurrent_model = "shuffle_inputs"`) | 6 |
| Shuffled topology | within each block, in-stubs randomly re-paired with out-stubs; self-connections and duplicate synapses repaired by swapping targets with random distinct edges; the block's weights then randomly reassigned | 6 |

Verified on the full 5000-neuron matrix (seed 44): the configuration model preserves every
in- and out-degree and the weight multiset in all four blocks, has no self-connections, and
shares 6–7% of the teacher's synapses (= the connection density, i.e. chance). The weight
shuffle keeps the topology identical.

A superseded whole-connectome weight shuffle (weights permuted within type-pair blocks
across the whole matrix, one seed) is kept under the distinct name `shuffle_weights_global`
in the CSVs so the two can never be confused; the per-neuron shuffle is the weight control
(2026-09-21).

### What the learnt-recurrence control is — state this on the slide

It isolates the **recurrent** connectome and nothing else. Exactly like every other variant
(and Figure 1), the student is given:

- the **true feedforward weight pattern** (mitral -> E/I), which is perturbed by an unknown
  per-pathway factor and rescaled by **2 learnt feedforward scaling factors** (mitral->E,
  mitral->I). So it knows *which* inputs each neuron receives, not their absolute scale;
- the true recorded activity of the feedforward units and of the observed neurons
  (teacher forcing), identical held-out evaluation, identical training budget.

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
| learning rate | Adam, **5e-3 → 5e-4** cosine over the epoch budget | weights move ~lr per update in log space |
| gradient clip | 100 | archived |
| feedforward scaling factors | lr 8e-3 → 5e-4, clip 5 | as Figure 1 |

The first version of this control used the archived initialisation (fully connected at the
mean *non-zero* weight, ≈16× the teacher's drive at ~6% density) and lr 1e-3 → 5e-4. In 50
epochs it did not fit even the observed neurons: van Rossum loss 433 -> 269 (Figure 1:
337 -> 88), observed E rate 2.8 vs 4.5 Hz, mitral->E scaling factor compensating to 4.7×,
held-out unobserved Fluctuation R² 0.04 — which could not be separated from bad
initialisation, so the initialisation and learning rate were changed as above before the
grid was run. That run is kept at
`bernstein/_superseded/lr8e-3-vr-only/fig02-controls__learnt-meannonzero-init__seed-44/`.

**Epoch budget.** The six-parameter controls run 50 epochs, where they have long since
converged. Learnt recurrence fits 25M weights and was still descending at 50 epochs (van
Rossum 145 -> 141 over the final tenth), so `run_grid_search.py` also submits it at 100
epochs — the budget every other learnt-weights model in this project gets — into its own
`learnt-100ep__seed-*` directories. The plotted bars are the 50-epoch runs; `LEARNT_EPOCHS`
in `figures.py` switches the figure over once the longer runs finish.
