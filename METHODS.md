# Shared methods

Conventions common to every figure in this directory. Individual figure READMEs specify
only what differs. Hyperparameters are recorded in each figure's README, as run.

---

## Observation level: 50% observed everywhere (2026-09-18)

Every figure with a fixed observation level runs at **50% of modelled neurons observed**:
roughly the fraction of reconstructed neurons expected to have activity recorded. Figure
3 sweeps the level instead, and takes its 50% point from Figure 1's runs; Figure 6 fixes
`recorded_pool_fraction = 0.5`. Earlier runs at 10% observed are in
`bernstein/_superseded/obs-0.1/` (a move, not a deletion) and stay available if a figure
needs the harder regime to show an effect.

---

## What the fit recovers, and what it does not

The student's six tied scaling factors are the only free parameters in the fully
reconstructed case, and the true model is scaling factor 1.0 for all six (the student's
physiology is the teacher's). What training recovers depends on the observation level:

| observed | learnt / true | note |
|---|---|---|
| 100% | 1.00 (7 d.p.) | no unobserved population, so the rate penalties do not exist as loss terms; the van Rossum term alone has a zero-loss optimum at the truth, reached from a log-normal initialisation up to 2x off |
| 50% | 0.79-1.01 | typical error 6% |
| 10% | 0.57-1.19 | |
| <=2% | 0.23-11.1 | unreliable: succeeds on some observed draws, collapses on others |

Two things make partial observation different, and neither is "the same loss, harder":
the rate penalties are **absent** when nothing is unobserved, and at 50% observed the SD
penalty is ~18% of the total loss; and the van Rossum term's own optimum moves, because
a partly free-running network cannot match the teacher spike for spike.

**Below ~25% observed the outcome is bimodal, not graded** (three seeds, 2026-09-19): a run
either trains normally or loses its excitatory population entirely, and which happens depends
on the observed draw rather than the fraction. Unobserved Fluctuation R² per seed: 2% observed
-0.03 / 0.67 / -0.03; 5% observed 0.82 / -0.45 / 0.76; 25% observed 0.94 / 0.39 / 0.94; 50%
observed 0.98 / 0.94 / 0.99. Report the individual seeds, not just mean +- SD, and describe the
low end as unreliable rather than impossible -- it is an optimisation failure, not an
information limit.

So state results as **prediction**, not parameter recovery: at 50% observed the student
predicts unobserved activity well while its parameters drift, and the perfectly specified
student (scaling factor 1.0) scores *higher* on held-out data than the trained one. Part
of every figure's gap to its ceiling is therefore our own regulariser, not missing
information. Do not claim recovered parameters outside the fully observed case.

**Neuron model caveat:** `tau_ref` appears in the physiology but is never applied by the
simulator, so there is no refractory period -- the fastest teacher cells fire with 2 ms
inter-spike intervals, up to ~270 Hz. Teacher and student share the model, so every
comparison holds, but do not describe the network as having an 8 ms refractory period.

---

## Model

Spiking teacher–student. Teacher: **5000 recurrent neurons** (E/I assemblies, tuned to zebrafish Dp) driven by **1500 feedforward units**.

Three cell types — feedforward, E, I — giving **6 scaling factors** as the only free parameters in the fully reconstructed case: FF→E, FF→I, E→E, E→I, I→E, I→I.

Where part of the network is not reconstructed, units outside the reconstructed set keep their **known activity** and contribute **learnt weights** onto the modelled neurons. Activity is cheap; reconstruction is the bottleneck. This is the regime the whole project is about.

---

## Teacher forcing — what "two layer" / "three layer" means

**Everything in this project is teacher forced.** Wherever a presynaptic neuron's true activity is known, the ground truth is injected instead of the simulated value. This reduces variance and prevents error accumulation through the recurrent loop; it is a training and stability device, not a different model.

Concretely, the implementation has three parts:

1. **Feedforward units** — recorded activity, injected. Never simulated.
2. **Unobserved recurrent neurons** — genuinely recurrent among themselves, because their activity is unknown and must be simulated. They additionally receive the **recorded** neurons' true activities, injected through the correct connectome weights and scaling factors.
3. **Recorded recurrent neurons** — no simulated recurrence among themselves, since that input arrives as injected ground truth. They receive injected recorded activities and the simulated unobserved-layer activities.

This is equivalent to the single recurrent network it describes: FF units and recurrent units as one simulated population with fixed connectome weights among reconstructed units and learnt weights from unreconstructed sources. The layer decomposition is how teacher forcing is implemented, not a feedforward cascade.

**Learnt feedforward weights go to every modelled neuron, observed and unobserved alike.** The weights onto *unobserved* neurons are constrained by no data directly — only indirectly, through those neurons' recurrent influence on observed ones. That block of loosely tethered parameters is the mechanism behind the degeneracy in Figure 6, and is worth stating as a mechanism rather than describing the failure phenomenologically.

**Teacher forcing is the standard throughout. Do not add a free-running evaluation variant.**

One caveat to be aware of for questions, not to act on: the two groups are not assessed on identical terms, since observed neurons are predicted from ground-truth input while unobserved neurons are simulated. **Figure 1 is the control that answers it** — at full reconstruction the unobserved population is simulated in exactly the same way and still reaches R² = 0.995, so the gap in Figure 6 cannot be attributed to teacher forcing. Keep that comparison explicit in the talk.

---

## Evaluation

- **Held-out test set of new stimuli** for every figure. Odour trajectories never used in training.
- **Fluctuation R² (primary)**: spike trains smoothed with a **50 ms Gaussian**, then R². A close stand-in for the calcium trace the real pipeline matches against (exponential filter, τ = 100 ms), so it answers "how well would these spike trains agree once seen through calcium". Say this in the talk — it makes the metric a property of the experiment rather than an arbitrary choice.
- **Activity R² (secondary)**: firing rates.
- Reported **separately for observed and unobserved** neurons. Unobserved is the discriminative group; every model can fit what it is shown.
- ~~Shuffled-identity floor~~ — **dropped 2026-09-17**: not reported on any figure (overrides the per-figure READMEs). The reference is instead a **ceiling**: a perfectly specified student under the same teacher forcing and spike-flip perturbations (see `fig01-full-reconstruction/README.md`, Evaluation).
- **≥3 seeds** per condition, with spread shown.
- **Do not compare observed with unobserved within one seed.** Pooled R² counts the spread of mean rates across the group as explainable variance, and with heavy-tailed rates that spread is set by a handful of fast cells, so which neurons land in the observed sample shifts the value (seed 44's observed draw misses the tail, which is why unobserved looks better there). It averages out over seeds; see `fig01-full-reconstruction/README.md`.

---

## Weight noise

Gaussian noise **added** to existing weights, then rescaled so the **mean and variance of the weight distribution are preserved**. Topology untouched — no synapses created or deleted. This is measurement error on weights, which is what makes the contrast with missing connections meaningful.

*Open:* does the added noise ever flip a weight's sign? If so the student violates Dale's law, which is a modelling artefact rather than a realistic measurement error. Report the fraction, or clip at zero and say so.

---

## Figure conventions

- Fixed colours across all figures, never reassigned: see `COLORSCHEME.txt` for what each one means and `common/style.py` for the semantic names figures import.
- Titles state the finding, not the topic.
- Vector output at final size; axis fonts legible at 12 cm wide on a projected slide.
- **One SVG per panel**, `figNN-<letter>-<slug>.svg`, at the size it is inserted into the talk at 100%. A rebuild deletes that figure's existing panels first (`common.style.clear_panels`), so an orphan from an earlier build cannot end up on a slide.
- Style is the paper-figure style of the earlier work, shared in `common/style.py`; the same panel functions serve `placeholder_figures/`, which only adds a watermark and fake CSVs.
- **Fluctuation R² only** outside the rate scatters (2026-09-18): Activity R² is still scored and kept in every CSV, but the scatters are where rates are reported.
- Rate scatters are linear over **0-40 Hz**; cells beyond are counted in the axis label, not plotted.
- Each figure folder carries its own `README.md`, named CSVs, `analysis.py`, `figures.py`, the panel SVGs, and the run config.

---

## Naming

```
ff-<known|learnt>__obs-<pct>__recur-<pct>__wn-<pct>__<panel>
```

Retire "hidden", "full inference" and "unreconstructed fraction" — each meant two different things in different archived figures, which caused real confusion. Use **observed / unobserved** and **reconstructed / unreconstructed** throughout.
