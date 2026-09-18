# Shared methods — read before any figure README

Conventions common to every figure in this directory. Individual figure READMEs specify only what differs.

---

## Implementation: minimal edits to existing code

**The code for all of this already exists.** Every figure below is a small variation on runs that have already been done. Do not rebuild anything.

Start by reading the archived figures in `archive-old-figures/` and the scripts and configs that produced them. They establish the model, the teacher-forcing scheme, the training loop, the metrics and every hyperparameter.

**All hyperparameters are already determined and should be reused unchanged** — regularisation on the learnt feedforward weights, their initialisation, optimiser and schedule, number of epochs, how weight noise is constructed, the stimulus set and the train/test split. Do not re-tune them.

For each figure, **read the relevant values off the existing implementation and record them in that figure's README**, so the settings are documented rather than rediscovered. Where a figure README leaves something unspecified, the existing code is the authority.

---

## Model

Spiking teacher–student. Teacher: **5000 recurrent neurons** (E/I assemblies, tuned to zebrafish Dp) driven by **1500 feedforward units**.

Three cell types — feedforward, E, I — giving **6 scaling factors** as the only free parameters in the fully reconstructed case: FF→E, FF→I, E→E, E→I, I→E, I→I.

Where part of the network is not reconstructed, units outside the reconstructed set keep their **known activity** and contribute **learnt weights** onto the modelled neurons. Activity is cheap; reconstruction is the bottleneck. This is the regime the whole project is about.

---

## Teacher forcing — what "two layer" / "three layer" means in the archive

**Everything in this project is teacher forced.** Wherever a presynaptic neuron's true activity is known, the ground truth is injected instead of the simulated value. This reduces variance and prevents error accumulation through the recurrent loop; it is a training and stability device, not a different model.

Concretely, in the archived ff-learnt run the implementation has three parts:

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

- Fixed colours across all figures for: full connectome, learnt recurrence, shuffle weights, configuration-model rewire, observed, unobserved, floor. Never reassigned.
- Titles state the finding, not the topic.
- Vector output at final size; axis fonts legible at 12 cm wide on a projected slide.
- Each figure folder carries its own `README.md`, named CSVs, `plot_figNN.py`, the output, and the run config.

---

## Naming

```
ff-<known|learnt>__obs-<pct>__recur-<pct>__wn-<pct>__<panel>
```

Retire "hidden", "full inference" and "unreconstructed fraction" — each meant two different things in different archived figures, which caused real confusion. Use **observed / unobserved** and **reconstructed / unreconstructed** throughout.
