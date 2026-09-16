# Generate Teacher Activity

Generates synthetic spike train data from a teacher network to serve as training targets for all downstream teacher-student experiments.

## Method

A conductance-based LIF network with assembly structure receives odour-modulated feedforward input. Sensory inputs are driven by an Ornstein-Uhlenbeck process that smoothly transitions between odourant patterns over time, modulating the firing rates of feedforward (mitral cell) neurons. The network produces structured population activity that encodes different sensory stimuli.

The OU process controls a softmax over assembly-specific odourant patterns, creating smooth, naturalistic input trajectories across 50 independent trials.

## Parameters

- `ou_tau` — OU time constant controlling how quickly odourant patterns transition (default: 700s)
- `ou_temperature` — softmax sharpness for pattern selectivity (default: 0.2)
- `odour_modulated_rate` — firing rate increase for active odourant neurons above baseline (default: 9.0 Hz)
- `modulation_fraction` — fraction of feedforward neurons modulated by each odourant (default: 0.1)
- `batch_size` — number of independent OU trajectories / trials (default: 50)

## Usage

```bash
./run generate-teacher-activity/experiment.toml
```

This must be run before any other teacher-student experiment, as they all depend on its output.

## Analysis

`analysis.ipynb` produces:
- Odourant response comparisons (odour vs baseline, odour vs different noise realisations)
- Input pattern visualisations showing per-assembly modulation
- Feedforward vs recurrent synaptic drive analysis
- Assembly-level activity dynamics over time
