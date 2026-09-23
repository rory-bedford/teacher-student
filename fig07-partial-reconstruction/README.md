# Figure 7 — Unreconstructed inputs break prediction of unobserved neurons

> Shared methods — model, teacher forcing, metrics, the noise ceiling, naming: see [`../README.md`](../README.md).

**The climax of the talk.**

**Claim:** when part of the network is not reconstructed and its influence has to be learnt instead, the student still fits the neurons it observes — but loses the ability to predict the ones it doesn't. Fitting and predicting come apart.

The unification this rests on: **an unreconstructed unit is indistinguishable from external drive**, whether it sits in the olfactory bulb or next door in the recurrent network. "Partial reconstruction" and "learnt feedforward input" are the same knob, and this figure turns it.

## Configuration

| Parameter | Value |
|---|---|
| recorded pool | **2500 of 5000 recurrent neurons (50%)**, fixed for the whole sweep |
| held-out pool | the other 2500, fixed |
| weight noise | 0 |
| recurrent scaling parameters | 6 shared scaling factors, as in Figures 1–5 |
| **reconstructed fraction** | **swept, 100% → 10%** |
| seeds | 3 per level |

**The 10% endpoint is the real operating point.** ~500 proofread cells in a ~5000-neuron circuit is 10%, so the sweep runs from the ideal case to the dataset actually in hand. Panel (a) shades it and labels it *Our Dataset*.

## The swept factor — the reconstructed segment

Pick a **reconstructed segment S** from the 6500 units (5000 recurrent + 1500 feedforward), drawn at **random** from the pooled set, with no distinction between feedforward and recurrent. Within S the full connectivity submatrix is known. Everything outside S has unknown connectivity but **known activity**, and is injected with **learnt weights** onto the modelled neurons.

The student simulates the **recurrent units in S**. At S = everything the learnt bucket is
empty, which is the no-learnt-weights consistency check on the pipeline and the endpoint of
the curve. It is **not** Figure 1: with a 50% recorded pool, S = everything observes 2500
neurons (Figure 3's 50% point), and it is trained with this figure's recipe rather than
Figure 1's.

**Learnt weights reach every modelled neuron, observed and unobserved alike.** The block landing on *unobserved* neurons is constrained by no data directly — only indirectly, through those neurons' recurrent influence on observed ones. **That loosely tethered block is the mechanism of the degeneracy**, and the talk should state it as a mechanism rather than describe the failure phenomenologically.

## Observation: fixed pools, not a fixed count

The recorded/held-out split is **50/50 and fixed**, set once over the 5000 recurrent neurons and independent of S. What the sweep changes is whether a given neuron falls inside S — simulated, with its recorded activity entering the loss — or outside S, injected as a source.

So the number of neurons actually in the loss is ~f × 2500, falling with reconstruction. **This co-variation is intrinsic, not a confound:** a neuron outside S cannot be constrained, because its activity is being *used as input* rather than *predicted*. Reconstruction caps how many recordings can be used at all, which is the claim itself.

State it on the slide in those terms: *at 10% reconstruction only a tenth of our recordings can be used — that is the cost of partial reconstruction.*

**Evaluation group:** held-out-pool neurons that lie inside S, so they are simulated but never in the loss. That is ~f × 2500 neurons, ~250 at the endpoint.

**Restricted-loss control: not implemented.** The design's one remaining open question was whether, at full reconstruction, artificially restricting the loss to the number of neurons the 10% level affords would also cost prediction — which would separate missing connectivity from merely having fewer constraints. That run was never made.

**Fully observed at 10% reconstruction, one run — trained, scored, not plotted.** Every
neuron in S enters the loss, so the only test left is generalisation to held-out stimuli.
It reaches **0.71** on observed neurons, against ~0.97 at full reconstruction, which is the
answer to "why not just record more neurons instead of reconstructing them?" — observation
does not substitute for reconstruction.

It was dropped from panel (a) on 2026-09-21: an unlabelled cross needed a sentence of
setup that the slide cannot spare, and it does not cleanly separate missing connectivity
from missing constraints — going from 219 to 489 neurons in the loss lifted the observed
fit from 0.57 to 0.71, so observation helps without closing the gap. The run stays in
`fig07_summary.csv` (`recorded_pool_fraction` 1.0) as a backup slide or a verbal answer.
The matched control that would separate the two — restricting the loss to 219 neurons at
*full* reconstruction — is still not implemented.

## Axes

Report against both the **fraction of units reconstructed** and **κ**, the fraction of each modelled neuron's input volume that is known. For a random S these coincide in expectation, but κ is the invariant that lets this figure be compared with anything else. Both are columns of `fig07_summary.csv`; panel (a) plots the reconstructed fraction.

## Panels

As built (2026-09-21), one SVG each:

- **(a)** `fig07-a-curve` — Fluctuation R² vs reconstructed fraction, observed and unobserved, individual seeds, the 10% operating point shaded.
- **(b)** `fig07-b-scatter-50pct` — firing rates at 50% reconstructed, observed beside unobserved, with Activity R² in each title.
- **(c)** `fig07-c-delta-fluctuation` — perturbation ΔFluctuation R², cell types pooled, shared y range with (a).

The free-parameter counts were removed from panel (a) (they hardly vary across the sweep)
and the raster was dropped. Group naming follows the project convention -- observed and
unobserved -- although the CSVs still key the unobserved group as `heldout`.

## Files

```
fig07-partial-reconstruction/
  analysis.py
  figures.py
  run_grid_search.py
  train.py
  experiment.toml
  parameters.toml
  README.md
  fig07_rates.csv
  fig07_spikes.csv
  fig07_summary.csv
  fig07-a-curve.svg
  fig07-b-scatter-50pct.svg
  fig07-c-delta-fluctuation.svg
```

---

## Implementation (recorded settings)

*Added when the code was written.*

### How to run

```bash
./run --grid fig07-partial-reconstruction/experiment.toml   # 10 levels x 3 seeds + 1 fully observed = 31 runs
uv run python fig07-partial-reconstruction/analysis.py       # fig07_summary.csv, fig07_rates.csv, fig07_spikes.csv
uv run python fig07-partial-reconstruction/figures.py        # panel SVGs, from this figure's CSVs
```

Levels: reconstructed fraction ∈ {0.1, 0.2, ..., 1.0} (`RECONSTRUCTED_FRACTIONS` in
`run_grid_search.py`), ordered from 0.1 upward, seed by seed. The fully observed control
(every modelled neuron in the loss, 10% reconstructed, seed 44) follows the first seed's
sweep. The levels were evened out to steps of 0.1 on 2026-09-21, from
{0.1, 0.2, 0.3, 0.5, 0.7, 1.0}: the collapse is at the *top* of the sweep — held-out
Fluctuation R² falls 0.97 → 0.42 between full reconstruction and 70%, then is flat and
negative below 30% — so it was the range from 0.7 to 1.0 that needed resolving, not the
bottom end.

### Construction (`common/structure.py`, `common/model.py`)

- **Segment S:** round(f × 6500) units drawn uniformly from the pooled 1500 mitral + 5000
  recurrent units. Recurrent units in S are modelled (simulated). Connectivity from S onto
  modelled neurons is the teacher's (perturbed, with the 6 shared scaling factors as in
  Figures 1–5).
- **Unreconstructed units** (everything outside S, mitral or recurrent): their teacher spikes
  are injected, through learnt weights, onto **every** modelled neuron.
- **Recorded pool:** 2500 of 5000 recurrent neurons, drawn once per seed from its own random
  stream, so it is identical at every level of that seed. Observed (layer 2, in the loss) =
  S ∩ pool; held-out (layer 1, simulated, never in the loss) = S \ pool.
- **Learnt weights:** one rank-1 log-space block per (source type, target type) — mitral, E
  and I unreconstructed sources onto E and I modelled neurons; exp(U V), fully connected,
  shared across both layers. *Extension:* the archived model learnt only the mitral block
  (one per target type); unreconstructed E and I sources are new here and use the same form.
  They must be separate blocks, since E and I sources drive different synapse types.
- **Initialisation (archived "constant"):** each block starts at its mean teacher weight
  (including zeros) times the perturbation factor — the exact rank-1 solution for a constant
  matrix.
- **Verified:** with S = everything the student reproduces the teacher spike for spike; with
  S = 40% and the learnt rows replaced by the true teacher weights, 0 mismatches in either
  group over 2 s, and every learnt block's low-rank slicing matches a direct computation.

### Training (the archived ff-learnt recipe of run `surrgrad-10`)

| Setting | Value | Note |
|---|---|---|
| chunk size / epochs / burn-in | 100 ms / **100** / 50 chunks | archived 500 — see Cost |
| chunks per update | 10 | |
| learnt weights | Adam lr 1e-3 → 5e-4, clip 0.8, rank 1 | |
| scaling factors | Adam lr 1e-3 → 1e-4, clip 0.8 | archived had 12 untied recurrent SFs; 6 tied here as in Fig 1 |
| β / eps | (0.95, 0.999) / 1e-8 | archived eps = 1e-4 never applied |
| schedule | cosine over epochs | archived cosine was given chunks, so it barely decayed |
| surrogate gradient scale | 10 | |
| loss | van Rossum (10/100 ms) + 0.5 × held-out population mean-rate MSE + 0.5 × rate-SD MSE (targets from observed teacher neurons) + 5000 × mean(exp(U V)) per learnt block | L1 weight applied to every learnt block, including the new E/I ones |
| precision | fp32 | archived fp16 |
| feedforward smoothing | **off** | archived: σ = 100 ms Gaussian smoothing + Bernoulli resampling of mitral spikes |

**Why smoothing is off:** it would make the "known activity" of mitral units inexact but not of
unreconstructed recurrent units (the dataset only smooths mitral input), and it would stop the
S = 100% endpoint from being the exactly specified model. Re-enabling it needs a dataset change.

### Cost

Measured here (Quadro RTX 5000, fp32, 1 epoch = 150 chunks): **4.2 min/epoch at 100%
reconstructed** (5.6 GB) and **2.7 min/epoch at 10%** (1.5 GB).

**Epochs reduced from the archived 500 to 100.** In the archived run (`surrgrad-10`) the van
Rossum loss had made 90% of its total improvement by epoch ~20, 98% by ~90 and 99% by ~170.
The scaling factors, however, were still drifting at 500 — E→I and I→I rose almost linearly
throughout and E→E / I→E settled near 0.7 of their targets — so the model moves along a flat
valley long after the loss stops improving. 100 epochs was chosen for cost; if held-out
prediction turns out to depend on that late drift, rerun with more epochs. The cosine learning-
rate decay now completes within the 100 epochs.

At 100 epochs a run is **≈ 7 h at 100% reconstructed and ≈ 4.5 h at 10%**. The first
six-level grid of 19 runs came to **≈ 100 GPU-h** (≈ 500 GPU-h at the archived 500 epochs);
the resolved ten-level sweep is 31 runs.
