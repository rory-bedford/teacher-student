# Shared methods

Conventions common to every figure. Each figure's README describes only what differs.

## Model

The teacher is a conductance-based spiking network of 5000 recurrent neurons (excitatory
and inhibitory, organised in assemblies and tuned to zebrafish Dp), driven by 1500
feedforward (mitral) units that carry odour-evoked input.

The student has the same neuron and synapse model and is given the connectome. Its free
parameters are one **scaling factor per pair of cell types** (mitral→E, mitral→I, E→E,
E→I, I→E, I→I): six in the fully reconstructed case. Each pathway's weights start
multiplied by an unknown log-normal factor, so the correct scaling factors have to be
learnt. Where part of the network is not reconstructed, units outside the reconstructed
set keep their recorded activity and reach the modelled neurons through learnt weights.

The student is trained with surrogate gradients to minimise a van Rossum distance between
its spike trains and the teacher's on the observed neurons, plus penalties that keep the
unobserved population's mean rate and rate spread near those of the observed neurons.

**Teacher forcing.** To reduce variance during training and evaluation, wherever a neuron's
true activity is known the student receives it in place of its own simulated spikes: the
mitral units and the observed neurons provide recorded input, while the unobserved neurons
are simulated and recurrent among themselves. Because the observed neurons are predicted
from recorded input, observed and unobserved neurons are always reported separately.

The simulator does not apply the `tau_ref` parameter, so the neurons have no refractory
period. Teacher and student share this model, so comparisons are unaffected.

## Evaluation

- **Held-out stimuli**: every run is scored on odour trajectories never used in training.
- **Fluctuation R²** (primary, labelled "R²" in the panels): spike trains smoothed with a
  50 ms Gaussian, then R² over neurons and time — a stand-in for the calcium signal
  available in real data. **Activity R²** (firing rates) is kept in the CSVs.
- **Observed and unobserved** neurons are scored separately. The unobserved neurons are the
  real test: any model can fit what it is shown.
- **Noise ceiling**: the teacher is chaotic, so even a perfectly specified student drifts
  from it after one differing spike. Every panel shows the score of that perfect student —
  teacher weights, correct scaling factors, the same conditions — as its reference.
- **Perturbation**: a constant hyperpolarising current applied to 25% of the unobserved
  inhibitory cells in teacher and student, calibrated to halve their rates, as in an
  optogenetic experiment. The student is scored on the change it predicts (perturbed minus
  unperturbed), for observed and non-targeted unobserved neurons.
- **Three seeds** per condition, each drawing its own observed set and initial scaling
  factors, shown individually.

Every figure runs at **50% of neurons observed**, roughly the fraction of reconstructed
neurons we expect to have activity for, except Figure 3, which varies it.

## Figures and naming

- One SVG per panel, `figNN-<letter>-<slug>.svg`, sized for a slide. Dense panels embed their
  data as an image inside the SVG; axes and labels stay vector.
- Colours are fixed across figures ([`COLORSCHEME.txt`](COLORSCHEME.txt)). Grey bands on the
  degradation sweeps mark where we estimate the real dataset to be.
- Training scripts save raw outputs only; `analysis.py` computes metrics into CSVs and
  `figures.py` plots from them.
- **Observed / unobserved** refers to whether a neuron's activity is recorded,
  **reconstructed / unreconstructed** to whether its connectivity is known.
