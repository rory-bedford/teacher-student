# Figure 8 — Model mismatch: the student's thresholds are wrong

> Shared methods — model, teacher forcing, metrics, the noise ceiling, naming: see [`../README.md`](../README.md).

**Question:** the connectome student assumes the teacher's single-neuron physiology exactly.
How much does it lose when that assumption is wrong? Every other figure degrades the
*connectome*; this one degrades the *neuron model*, with the connectome perfect.

The manipulation is **heterogeneity in the spike threshold θ**: each student neuron's
threshold is offset from the teacher's by an independent Gaussian draw, centred within each
cell type so every population's mean threshold stays exactly the teacher's. It is
per-neuron error, which six *shared* scaling factors cannot absorb.

A homogeneous shift (every threshold moved by the same amount) was considered and dropped
(2026-09-26): mean zero only, so the manipulation is purely heterogeneity.

## Configuration

Identical to Figure 1 except for the swept mismatch:

| Parameter | Value |
|---|---|
| feedforward connections | reconstructed |
| recurrent reconstruction | 100%, teacher weights, no noise |
| observed fraction | 50% |
| trained parameters | 6 scaling factors |
| **threshold heterogeneity** | **swept**, SD 1, 2, 4 mV |
| seeds | 3 per point (44, 45, 46) |

Heterogeneity 0 is Figure 1's runs, read by `analysis.py`. The teacher's thresholds are
−38 mV (E) and −45 mV (I), with rest and reset at −60 mV, so the threshold sits 22 mV (E) /
15 mV (I) above rest: an SD of 4 mV is 18% / 27% of that gap.

## Implementation

`[student].threshold_heterogeneity` (SD, mV) is drawn into a per-neuron `theta_offset` in
`student_structure.npz` (`common/structure.py`, its own random stream, so the draw per seed
is independent of every other manipulation), then centred within each cell type.
`common/model.py` adds it to each simulated neuron's threshold — the unobserved neurons of
layer 1 and the observed neurons of layer 2, whose spikes the loss compares with the
teacher's. The physiology block in `parameters.toml` stays the teacher's.

The **noise ceiling** is the perfectly specified student, and that has the teacher's
thresholds: `perfect_structure` zeroes the offset. So the ceiling does not depend on the
mismatch, equals Figure 1's per seed, and is drawn as **one constant line** from Figure 1's
runs.

Structures saved before 2026-09-26 have no `theta_offset`; the model treats that as no
mismatch, so every earlier run and cached evaluation is unchanged.

## Panels

- **(a)** `fig08-a-curve` — held-out Fluctuation R² vs threshold heterogeneity, observed /
  unobserved.
- **(b)** `fig08-b-delta-fluctuation` — perturbation ΔFluctuation R², unobserved E and I
  pooled, targeted cells excluded.
- **(c)** `fig08-c-scaling-factors` — learnt / true scaling factor vs threshold
  heterogeneity: what the six parameters do under the mismatch. With the teacher's
  weights, 1.0 is the true model at every level.

## How to run

```bash
./run --grid fig08-model-mismatch/experiment.toml   # 3 levels x 3 seeds = 9 runs -> bernstein/fig08-model-mismatch/
uv run python fig08-model-mismatch/analysis.py       # reads Figure 1's runs too
uv run python fig08-model-mismatch/figures.py        # fig08-a … fig08-c SVGs
```

Cost as Figure 1, ≈ 3.6 h per run on a Quadro RTX 5000 (50 epochs).

## Files

```
fig08-model-mismatch/
  analysis.py  figures.py  run_grid_search.py  train.py
  experiment.toml  parameters.toml  README.md
  fig08_summary.csv  fig08_rates.csv  fig08_scaling_factors.csv
  fig08-a … fig08-c SVGs
```
