# Shared methods

Conventions common to every figure. Each figure's own README describes only what differs
from these, and records its hyperparameters as run.

## Model

The teacher is a conductance-based spiking network of **5000 recurrent neurons** (excitatory
and inhibitory, organised in assemblies and tuned to zebrafish Dp) driven by **1500
feedforward (mitral) units**, which carry odour-evoked input.

The student has the same neuron and synapse model. It is given the connectome — which
neurons connect and with what weight — and its only free parameters are one **scaling
factor per pair of cell types**: mitral→E, mitral→I, E→E, E→I, I→E, I→I, so six in the fully
reconstructed case. The student's weights are the teacher's multiplied by an unknown
log-normal factor per pathway, so the correct scaling factors are not 1 at the start of
training and have to be learnt.

Where part of the network is not reconstructed, the units outside the reconstructed set
keep their recorded activity and reach the modelled neurons through **learnt weights**.
Recording activity is cheap; reconstruction is the bottleneck, and this is the regime the
project is about.

Training minimises a van Rossum distance between the student's and the teacher's spike
trains on the observed neurons, plus penalties keeping the unobserved population's mean
rate and rate spread near those of the observed neurons, using surrogate gradients.

**Neuron model caveat:** `tau_ref` appears in the physiology but the simulator does not
apply it, so there is no refractory period; the fastest teacher cells fire at up to ~270 Hz.
Teacher and student share the model, so every comparison holds, but the network should not
be described as having an 8 ms refractory period.

## Teacher forcing

Wherever a neuron's true activity is known, the student receives it instead of its own
simulated spikes. This keeps errors from accumulating through the recurrent loop; it is
how the known activity enters the model, not a different model.

1. **Feedforward units** — recorded activity, injected. Never simulated.
2. **Unobserved recurrent neurons** — simulated and genuinely recurrent among themselves.
   They also receive the observed neurons' recorded activity through the connectome.
3. **Observed recurrent neurons** — simulated from their inputs (recorded activity plus the
   simulated unobserved neurons) and compared with the teacher in the loss; their own
   recurrent input arrives as recorded activity.

This is the single recurrent network it describes, split into two layers only to implement
teacher forcing. Observed and unobserved neurons are therefore not assessed on identical
terms — the observed neurons are predicted from recorded input — which is why every figure
reports the two separately, and why Figure 1 matters: at full reconstruction the unobserved
population is simulated in exactly the same way and still reaches a Fluctuation R² of 0.97.

Learnt weights reach every modelled neuron, observed and unobserved alike. The weights onto
unobserved neurons are constrained by no data directly, only through those neurons'
influence on observed ones — the mechanism behind the degeneracy in Figure 7.

## Observation level

Every figure with a fixed observation level runs at **50% of modelled neurons observed**,
roughly the fraction of reconstructed neurons we expect to have recorded activity for.
Figure 3 sweeps the level instead, and Figure 7 fixes `recorded_pool_fraction = 0.5`.

## Evaluation

- **Held-out stimuli**: every run is scored on odour trajectories never used in training.
- **Fluctuation R² (primary)**: spike trains smoothed with a 50 ms Gaussian, then R² over
  neurons and time. This stands in for the calcium signal the real data provides
  (exponential filter, τ = 100 ms): it measures how well the spike trains would agree once
  seen through calcium. Plotted as "R²".
- **Activity R² (secondary)**: R² of firing rates. Scored and kept in every CSV; rates are
  shown in the scatter panels.
- **Observed and unobserved neurons are reported separately.** The unobserved neurons are
  the discriminating test: any model can fit what it is shown.
- **Noise ceiling**: the teacher network is chaotic, so even a perfectly specified student
  drifts from it after a single flipped spike. Every figure plots the score of that perfect
  student — teacher weights, correct scaling factors, the same teacher forcing and the same
  20 spike-flip draws — as its reference (see `fig01-full-reconstruction/README.md`).
- **Perturbation**: an optogenetic-style constant hyperpolarising current applied to 25% of
  the unobserved inhibitory cells, in teacher and student alike, calibrated to halve the
  targeted cells' rates. The student is teacher-forced with the perturbed recording and
  scored on the change it predicts (perturbed minus unperturbed) for the observed neurons,
  the non-targeted unobserved neurons and the targets.
- **Three seeds** per condition, shown individually.
- Observed and unobserved scores should not be compared within one seed. Pooled R² counts
  the spread of mean rates across the group as explainable variance, and with heavy-tailed
  rates that spread depends on which fast cells land in the observed sample. It averages
  out over seeds.

## What the fit recovers

In the fully reconstructed case the six scaling factors are the only free parameters and the
true model is 1.0 for all six. What training recovers depends on the observation level:

| Observed | Learnt / true | Note |
|---|---|---|
| 100% | 1.00 (7 d.p.) | no unobserved population, so the rate penalties vanish and the van Rossum term alone has its optimum at the truth |
| 50% | 0.79–1.01 | typical error 6% |
| 10% | 0.57–1.19 | |
| ≤2% | 0.23–11.1 | unreliable: succeeds for some observed draws, collapses for others |

Below ~25% observed the outcome is bimodal rather than graded: a run either trains normally
or loses its excitatory population, depending on which neurons were observed rather than how
many. This is an optimisation failure, not an information limit, so Figure 3 shows the
individual seeds.

Results are therefore stated as **prediction**, not parameter recovery: at 50% observed the
student predicts unobserved activity well while its parameters drift, and the perfectly
specified student scores higher on held-out data than the trained one.

## Weight noise

Multiplicative log-normal noise on existing weights, applied per (source type, target type)
block, then rescaled so each block keeps its **mean and SD**, then clipped at zero. No
synapses are created or deleted and none changes sign, so this is measurement error on the
weights alone. The fraction clipped is recorded per run as `noise_clipped_fraction`.

## Figure conventions

- One SVG per panel, `figNN-<letter>-<slug>.svg`, sized to go into a slide at 100%. Dense
  panels (rasters, scatters, traces) embed their data as a 400 dpi image inside the SVG;
  axes and labels stay vector.
- Colours are fixed across all figures: meanings in [`COLORSCHEME.txt`](COLORSCHEME.txt),
  names in `common/style.py`.
- Grey bands on the degradation sweeps mark where we estimate the real dataset to be.
- Training scripts save raw outputs only; `analysis.py` computes metrics and `figures.py`
  plots from the CSVs. No notebooks.

## Naming

**observed / unobserved** for whether a neuron's activity is recorded, and **reconstructed /
unreconstructed** for whether its connectivity is known, throughout the code, CSVs and
figures. Run folders are named for what they vary: `seed-44`, `wn-0.3__seed-45`,
`recon-0.5__seed-44`.
