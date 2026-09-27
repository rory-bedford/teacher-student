# Figure 4 — Reconstruction errors

Does it matter how a reconstruction loses input — whole neurons missing, or synapses
missing throughout? Barely: neuron removal and synapse dropout give nearly the same curve,
so what counts is how much recurrent input is lost. And it counts heavily: losing 10% of it
brings unobserved Fluctuation R² from 0.97 to 0.71–0.74, and at the roughly 15% we estimate
for the real dataset the student lies between the 0.1 and 0.2 levels (0.50–0.74).

Shared methods (model, training, evaluation, noise ceiling): [`../METHODS.md`](../METHODS.md).

## Experiment

Everything is as in Figure 1 except that the student's recurrent connectome is degraded by
one of two error models:

- **Neuron removal**: a random subset of recurrent neurons is deleted from the student. It
  neither simulates them nor receives their spikes, as for a cell that was never
  proofread. Observed neurons are 50% of the retained neurons.
- **Synapse dropout**: each recurrent synapse is deleted independently with the given
  probability. Every neuron stays in the model.

The feedforward (mitral) input is always complete. Both models are plotted against the
same quantity, the fraction of recurrent input lost: for each retained neuron, the summed
teacher weight of its missing recurrent synapses divided by the summed weight of all its
recurrent synapses, averaged over neurons. The realised values lie within 0.015 of the
nominal level and are snapped to the 0.1 grid, so the two models align.

| Setting | Value |
|---|---|
| Error models | neuron removal, synapse dropout |
| Level (fraction removed) | 0.1, 0.2, ..., 0.5; 0 is Figure 1's runs |
| Observed fraction | 50% of retained neurons |
| Trained parameters | 6 scaling factors |
| Epochs | 50 |
| Seeds | 3 (44, 45, 46) |

## Results

Fluctuation R², mean over three seeds:

| Input lost | Error model | Observed | Unobserved | ΔFluct. R² perturbation, observed | ΔFluct. R² perturbation, unobserved |
|---|---|---|---|---|---|
| 0 (Figure 1) | — | 0.97 | 0.97 | 0.94 | 0.94 |
| 0.1 | neuron removal | 0.78 | 0.74 | 0.63 | 0.67 |
| 0.1 | synapse dropout | 0.75 | 0.71 | 0.64 | 0.63 |
| 0.2 | neuron removal | 0.59 | 0.63 | 0.47 | 0.47 |
| 0.2 | synapse dropout | 0.60 | 0.50 | 0.45 | 0.42 |
| 0.3 | neuron removal | 0.41 | 0.36 | 0.34 | 0.29 |
| 0.3 | synapse dropout | 0.45 | 0.37 | 0.34 | 0.32 |
| 0.4 | neuron removal | 0.23 | 0.22 | 0.21 | 0.21 |
| 0.4 | synapse dropout | 0.34 | 0.27 | 0.26 | 0.25 |
| 0.5 | neuron removal | 0.20 | 0.14 | 0.13 | 0.15 |
| 0.5 | synapse dropout | 0.22 | 0.20 | 0.17 | 0.15 |

The differences between the two models are within the spread across seeds. The noise
ceiling is 0.97 or above held-out at every level; for the perturbation it is 0.99 for
synapse dropout and falls to 0.90 for neuron removal at half the input lost.

## Panels

- `fig04-a-curve.svg` — unobserved Fluctuation R² on held-out stimuli against the fraction
  of recurrent input lost, one series per error model, individual seeds, with the grey band
  marking the estimated real-dataset level (about 15% of input lost).
- `fig04-b-delta-fluctuation.svg` — perturbation ΔFluctuation R² of the unobserved neurons
  against input lost, both error models (E and I pooled, targeted cells excluded).
- `fig04-c-delta-fluctuation-observed.svg` — the same for the observed neurons.

## Running

```bash
./run --grid fig04-reconstruction-errors/experiment.toml
uv run python fig04-reconstruction-errors/analysis.py
uv run python fig04-reconstruction-errors/figures.py
```

`analysis.py` reads Figure 1's runs as level 0 of both error models. The grid is 30 runs
(2 models × 5 levels × 3 seeds), each about 3.6 h on a Quadro RTX 5000, as in Figure 1.

## Notes

- A removed neuron cannot be observed, so the observed fraction is taken of the retained
  neurons. It stays 50% at every level, rather than observation and reconstruction falling
  together; the number of observed neurons does fall, to 1250 at half the input lost.
- The input-lost fraction counts recurrent input only, so neuron removal at fraction p
  loses about p of it; including the always-complete mitral input would compress both axes.
- Observed Fluctuation R² is in the CSV but panel (a) shows only the unobserved neurons,
  the ones the model has to predict.
