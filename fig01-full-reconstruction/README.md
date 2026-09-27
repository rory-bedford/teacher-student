# Figure 1 — Full reconstruction

Given the full connectome and the activity of half of the neurons, can the student predict
the other half? It can: on odour stimuli it never saw in training, the student reaches a
Fluctuation R² of 0.97 on the unobserved neurons, close to the noise ceiling, and predicts
how the whole network responds to silencing part of its inhibitory population.

Shared methods (model, training, evaluation, noise ceiling): [`../METHODS.md`](../METHODS.md).

## Experiment

The student is given the complete connectome, feedforward and recurrent, and the recorded
activity of the mitral inputs and of 50% of the recurrent neurons, drawn at random per
seed. Its only free parameters are the six scaling factors, one per pair of cell types.
Before training every pathway's weights are multiplied by an unknown log-normal factor, so
the student starts away from the true model and has to recover it from the activity of the
observed neurons alone.

This is the baseline condition of the project: every later figure changes one thing about
it. The student is trained against the observed neurons with the van Rossum loss and the
unobserved-rate penalties described in METHODS, and scored on a held-out trial and on the
perturbation.

| Setting | Value |
|---|---|
| Connectome | feedforward and recurrent, 100% reconstructed, no weight noise |
| Observed fraction | 50% (2500 of 5000 neurons) |
| Free parameters | 6 scaling factors (mitral→E, mitral→I, E→E, E→I, I→E, I→I) |
| Initial perturbation | log-normal factor per pathway, variance 0.5 |
| Seeds | 44, 45, 46 (each draws the perturbation and the observed set) |
| Epochs | 50 |

## Results

Mean over three seeds; noise ceiling in brackets.

| | Observed | Unobserved |
|---|---|---|
| Fluctuation R², held-out stimuli | 0.97 [1.00] | 0.97 [1.00] |
| Activity R², held-out stimuli | 0.99 [1.00] | 0.98 [1.00] |
| ΔFluctuation R², perturbation | 0.94 [0.99] | 0.94 [0.99] |

The student predicts the unobserved neurons as well as the observed ones, on new stimuli
and under a perturbation it was never trained on. The perturbation scores pool excitatory
and inhibitory cells and exclude the targeted inhibitory cells.

The fit does not, however, recover the parameters exactly. At 50% observed the learnt
scaling factors lie between 0.79 and 1.28 of the truth (mean absolute error 9%), while in
fully observed runs they are recovered to within 10⁻⁵. The result is prediction, not
parameter identification: many nearby parameter sets reproduce the activity equally well.

## Panels

- `fig01-a-raster.svg` — teacher and student spikes on a held-out stimulus, three observed and three unobserved neurons.
- `fig01-b-scatter.svg` — firing rate per neuron, student against teacher, observed beside unobserved (axis clipped at 40 Hz).
- `fig01-c-delta-scatter.svg` — the change in each neuron's rate caused by the perturbation, student against teacher, targeted cells marked.
- `fig01-d-delta-means.svg` — mean rate change per population (targeted inhibitory, non-targeted inhibitory, excitatory), teacher beside student.
- `fig01-e-scaling-factors.svg` — the six learnt scaling factors relative to the truth, for these runs and for fully observed runs.

## Running

```bash
./run --grid fig01-full-reconstruction/experiment.toml
uv run python fig01-full-reconstruction/analysis.py
uv run python fig01-full-reconstruction/figures.py
```

Train this figure first: Figures 2–6 and 8 read its runs as their baseline condition. The
fully observed runs in panel (e) are trained in Figure 2's grid, so that grid must have
finished for the panel to be complete. One run takes about 3.6 hours on a Quadro RTX 5000.

## Notes

- The noise ceiling is the score of a perfectly specified student (teacher weights,
  correct scaling factors) under the same conditions. It is not 1 because the teacher is
  chaotic: once a single spike differs, spike timing decorrelates within about a second.
- In the fully observed runs there are no unobserved neurons, so the rate penalties vanish
  and the loss has its optimum exactly at the true scaling factors. This is the
  identifiability check behind panel (e).
- Observed and unobserved scores should not be compared within a single seed: pooled R²
  counts the spread of mean rates as explainable variance, and with heavy-tailed rates that
  depends on which fast cells land in the observed sample. It averages out over seeds.
- The student runs in fp32: at the correct scaling factors it then reproduces the teacher
  spike for spike before any chaotic divergence.
