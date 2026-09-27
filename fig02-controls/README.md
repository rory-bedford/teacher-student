# Figure 2 — Controls

Is it the connectome that lets the student predict unobserved neurons, or just a flexible
enough model? We compare the full connectome with students given only its topology, no
connectome at all, or a scrambled one. Students with learnt weights fit the observed
neurons, and their response to the perturbation, but fail on the unobserved ones; the
scrambled connectomes fail on both. Only the full connectome does both, with six free
parameters against 1.6 M or 25 M.

Shared methods (model, training, evaluation, noise ceiling): [`../METHODS.md`](../METHODS.md).

## Experiment

Everything is as in Figure 1 (50% of neurons observed, feedforward connectome and recorded
inputs given, the same held-out stimuli and perturbation) except the recurrent
connectivity the student is given, set by `[student].recurrent_model`:

| Variant | What the student is given | Free parameters |
|---|---|---|
| Full connectome | true topology and weights (Figure 1's runs) | 6 |
| Fixed topology | true synapses; one free weight per synapse | 1,582,452 |
| Unconstrained | no recurrent connectome; every recurrent weight free, dense | 25,000,002 |
| Shuffled weights | true topology; each neuron's input weights permuted among its presynaptic partners of the same cell type | 6 |
| Shuffled topology | each cell-type block rewired by the configuration model, preserving in- and out-degrees and the weight distribution | 6 |

Fixed topology and unconstrained (`recurrent_model = "learnt"`) learn log-weights for the
recurrent blocks and keep the two feedforward scaling factors, so they know which inputs
each neuron receives but not their scale. Unconstrained is initialised at each block's
mean weight including absent synapses, and fixed topology at the mean non-zero weight, so
both start with the teacher's total recurrent drive per block. The two shuffles keep the
six scaling factors and change only which weights go where.

| Setting | Value |
|---|---|
| Observed fraction | 50% |
| Seeds | 44, 45, 46 |
| Epochs | 100 for fixed topology and unconstrained, 50 for the others |
| Learnt-weight optimiser | Adam, learning rate 5e-3 → 5e-4 (cosine), gradient clip 100 |

## Results

Mean over three seeds; the noise ceiling is 1.00 for held-out Fluctuation R² and 0.99 for
the perturbation.

| Variant | Fluctuation R², observed | Fluctuation R², unobserved | ΔFluctuation R², observed | ΔFluctuation R², unobserved |
|---|---|---|---|---|
| Full connectome | 0.97 | 0.97 | 0.94 | 0.94 |
| Fixed topology | 0.92 | -0.46 | 0.65 | -0.58 |
| Unconstrained | 0.89 | -0.38 | 0.67 | -0.24 |
| Shuffled weights | -0.15 | -0.18 | -0.10 | -0.08 |
| Shuffled topology | -0.19 | -0.20 | -0.09 | -0.14 |

With weights learnt from data, fixed topology and unconstrained fit the neurons they are
shown and partly predict how those neurons respond to the perturbation, and they match
every population's mean rate, but their Fluctuation R² on unobserved neurons is
negative. With the wrong connectome, the six scaling factors cannot fit even the
observed neurons: under both shuffles the excitatory population falls to about 1 Hz,
against the teacher's 4 Hz. Only the measured connectome predicts the unobserved neurons.

## Panels

- `fig02-a-bars-held-out.svg` — held-out Fluctuation R² per variant, observed and unobserved subpanels.
- `fig02-b-bars-perturbation.svg` — perturbation ΔFluctuation R² per variant, observed and unobserved (non-targeted, E and I pooled) subpanels.
- `fig02-c-legend.svg` — the shared legend.

Bars are mean ± SD over seeds, with individual seeds as dots and the noise ceiling marked
per bar.

## Running

```bash
./run --grid fig02-controls/experiment.toml
uv run python fig02-controls/analysis.py
uv run python fig02-controls/figures.py
```

The full-connectome bars are Figure 1's runs, read by `analysis.py` and not retrained, so
Figure 1 must be trained first. The grid is 15 runs: the four controls and a fully
observed full-connectome run for each seed. A six-parameter run takes about 3.6 hours on
a Quadro RTX 5000; the learnt-weight variants train for twice as many epochs.

## Notes

- The fully observed full-connectome runs are not a bar here. They are the
  identifiability check shown in Figure 1's scaling-factor panel.
- Fixed topology fails on unobserved neurons because its learnt weights settle on a
  different solution: per block they correlate with the teacher's at only r = 0.05–0.29,
  although each block's total weight stays within about 0.9–1.5× of the teacher's. With
  1.6 M free weights fitted to half the neurons, the topology alone does not pin them down.
- Shuffled topology uses the configuration model rather than a random rewire, so that
  block statistics, degree sequences and the weight distribution are all preserved and
  only the specific wiring is destroyed.
- Shuffled weights asks whether the weight values carry information once the topology is
  right, which in real data is the question of whether synapse sizes are informative.
- Unconstrained is given the true feedforward pattern, so it starts with more of the
  circuit than a generic data-constrained RNN would; the comparison is conservative.
- The learnt-weight variants train for 100 epochs because unconstrained was still
  improving at 50. `fig02_summary.csv` also holds 50-epoch unconstrained runs and a
  single-seed whole-connectome weight shuffle, which are not plotted.
