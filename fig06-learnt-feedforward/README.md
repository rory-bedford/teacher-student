# Figure 6 — Unreconstructed inputs break prediction of unobserved neurons

> **Read `../METHODS.md` first** — model, teacher forcing, metrics, floor, seeds, naming, and the instruction to make only minimal edits to the existing code.

**The climax of the talk.**

**Claim:** when part of the network is not reconstructed and its influence has to be learnt instead, the student still fits the neurons it observes — but loses the ability to predict the ones it doesn't. Fitting and predicting come apart.

The unification this rests on: **an unreconstructed unit is indistinguishable from external drive**, whether it sits in the olfactory bulb or next door in the recurrent network. "Partial reconstruction" and "learnt feedforward input" are the same knob, and this figure turns it.

## Configuration

| Parameter | Value |
|---|---|
| recorded pool | **2500 of 5000 recurrent neurons (50%)**, fixed for the whole sweep |
| held-out pool | the other 2500, fixed |
| weight noise | 0 |
| recurrent scaling parameters | as in the existing implementation |
| **reconstructed fraction** | **swept, 100% → 10%** |
| seeds | 3 per level |

**The 10% endpoint is the real operating point.** ~500 proofread cells in a ~5000-neuron circuit is 10%, so the sweep runs from the ideal case to the dataset actually in hand. Mark it on the figure.

## The swept factor — the reconstructed segment

Pick a **reconstructed segment S** from the 6500 units (5000 recurrent + 1500 feedforward), drawn at **random** from the pooled set, with no distinction between feedforward and recurrent. Within S the full connectivity submatrix is known. Everything outside S has unknown connectivity but **known activity**, and is injected with **learnt weights** onto the modelled neurons.

The student simulates the **recurrent units in S**. At S = everything the learnt bucket is empty and this reduces exactly to **Figure 1** — the endpoint of the curve and a consistency check on the pipeline.

**Learnt weights reach every modelled neuron, observed and unobserved alike.** The block landing on *unobserved* neurons is constrained by no data directly — only indirectly, through those neurons' recurrent influence on observed ones. **That loosely tethered block is the mechanism of the degeneracy**, and the talk should state it as a mechanism rather than describe the failure phenomenologically.

## Observation: fixed pools, not a fixed count

The recorded/held-out split is **50/50 and fixed**, set once over the 5000 recurrent neurons and independent of S. What the sweep changes is whether a given neuron falls inside S — simulated, with its recorded activity entering the loss — or outside S, injected as a source.

So the number of neurons actually in the loss is ~f × 2500, falling with reconstruction. **This co-variation is intrinsic, not a confound:** a neuron outside S cannot be constrained, because its activity is being *used as input* rather than *predicted*. Reconstruction caps how many recordings can be used at all, which is the claim itself.

State it on the slide in those terms: *at 10% reconstruction only a tenth of our recordings can be used — that is the cost of partial reconstruction.*

**Evaluation group:** held-out-pool neurons that lie inside S, so they are simulated but never in the loss. That is ~f × 2500 neurons, ~250 at the endpoint.

**Optional control, one run.** At full reconstruction, artificially restrict the loss to the number of neurons the 10% level affords. If performance stays high, the damage at low reconstruction comes from missing connectivity rather than from having fewer constraints. This is the only question the design leaves open, and it closes it.

**Also worth one run: fully observed at 10% reconstruction.** Every neuron in S enters the loss and the only test left is generalisation to held-out stimuli. If prediction is still poor there, observation demonstrably cannot substitute for reconstruction — the claim that most cleanly separates this work from Beiran & Litwin-Kumar. Plot as a single annotated point.

## Axes

Report against both the **fraction of units reconstructed** and **κ**, the fraction of each modelled neuron's input volume that is known. For a random S these coincide in expectation, but κ is the invariant that lets this figure be compared with anything else.

## Panels

- **(a)** **The scatter.** Rate scatter, observed vs unobserved, teacher against student, coloured by E/I, R² in each title, floor annotated. **No extra runs needed** — export per-neuron rates at *every* sweep level and choose the display level when plotting. Likely two: one mid-sweep where the effect is partial, and the 10% endpoint.
- **(b)** **The sweep.** Fluctuation R² vs reconstructed fraction, observed and held-out series, error bars over seeds, floor dashed, Figure 1's point at 100%. Annotate the free-parameter count at a few levels — it is the mechanism, and it pre-empts the objection that the learnt bucket can fit anything.
- **(c)** *optional* — raster for one observed and one held-out neuron at the same level.

## Files

```
fig06-learnt-feedforward/
  README.md
  fig06_summary.csv     reconstructed_fraction, kappa, n_free_params, n_in_loss, seed, group{observed,heldout}, metric, value, floor_value
  fig06_rates.csv       reconstructed_fraction, neuron_id, cell_type, group{observed,heldout}, seed, teacher_rate_hz, student_rate_hz
  fig06_spikes.csv      reconstructed_fraction, neuron_id, group, seed, source{teacher,student}, time_s
  plot_fig06.py
  fig06.svg
  config.yaml           the resolved config actually used, written by the run
```

## Status

**Needs running.** The archived `ff-learnt__obs-45__recur-81__wn-30__scatter.svg` (observed 0.953, unobserved 0.243) shows the phenomenon and is the best visual currently available, but varies four factors at once, so it cannot attribute the failure to unreconstructed input. See `../PRIORITY.md` for how it can and cannot be used as a fallback.

Its numbers are a good prior for what to expect mid-sweep.

---

## Implementation (recorded settings)

*Added when the code was written.*

### How to run

```bash
./run --grid fig06-learnt-feedforward/experiment.toml   # 6 levels x 3 seeds + 1 fully observed = 19 runs
uv run python fig06-learnt-feedforward/analysis.py       # fig06_summary.csv, fig06_rates.csv, fig06_spikes.csv
uv run python fig06-learnt-feedforward/figures.py        # fig06.svg (--scatter-fractions for panel a)
```

Levels: reconstructed fraction ∈ {1.0, 0.7, 0.5, 0.3, 0.2, 0.1}, ordered from 0.1 upward, seed
by seed. The fully observed control (every modelled neuron in the loss, 10% reconstructed,
seed 44) follows the first seed's sweep.

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

### Training (archived ff-learnt recipe, `full-inference/hidden-units`, run `surrgrad-10`)

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

### Corrections to the design text above

- **The 100% endpoint is not Figure 1.** With a 50% recorded pool, S = everything observes 2500
  neurons (Figure 3's 50% point), and it is trained with this recipe rather than Figure 1's.
  It is still the no-learnt-weights consistency check, just not at 10% observed.
- **Optional restricted-loss control:** not implemented.

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

At 100 epochs a run is **≈ 7 h at 100% reconstructed and ≈ 4.5 h at 10%**, so the 19-run grid is
**≈ 100 GPU-h** (≈ 500 GPU-h at the archived 500 epochs).

### Panels as implemented

(b) Fluctuation R² vs reconstructed fraction (x decreasing), observed and held-out series
(mean ± SD, seeds as dots), ceilings dotted, held-out floor dashed, free-parameter count
annotated at each level, top axis κ (mean known fraction of input volume), 10% operating point
shaded, fully observed run as ×. (a) Rate scatters, observed and held-out, at the middle level
and the 10% endpoint. (c) Raster, one observed and one held-out neuron, 10% endpoint.
