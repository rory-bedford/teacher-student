# Full Inference

Combined experiment that simultaneously tests all teacher-student inference
challenges. The student must recover perturbed scaling factors despite
multiple sources of model mismatch.

This folder is split into two sub-experiments:

```
full-inference/
├── hidden-units/      # partially-observed student (two-layer visible-driven)
│   ├── train.py
│   ├── experiment.toml
│   ├── parameters.toml
│   └── analysis.ipynb
├── no-hidden-units/   # fully-observed student (learn-ff-connectivity pipeline)
│   ├── train.py
│   ├── experiment.toml         # main run
│   ├── parameters.toml
│   ├── experiment-control.toml # shuffled-connectome control
│   ├── parameters-control.toml
│   └── analysis.ipynb
└── bias-check/        # inference only: is the learnt scaling-factor shrinkage
    ├── compute_inference_losses.py   #  where the loss actually wants to be?
    ├── experiment.toml
    ├── parameters.toml
    └── analysis.ipynb
```

Both sides share the same teacher data, scaling-factor perturbation scheme,
and `[student_mismatch]` knobs where they overlap.

## hidden-units/

A fraction of neurons are unobserved. Uses a two-layer architecture
(recurrent hidden + feedforward visible) with teacher visible spikes
injected as feedforward input. Gradients flow from visible loss to
FF→hidden weights; loss is computed on visible neurons only.

Additional simultaneous challenges:

1. **Missing units**: a fraction of neurons are structurally absent from
   the student model — removed from the weight matrix entirely.
2. **Weight noise**: multiplicative log-normal noise applied per
   cell-type pair to recurrent weights, preserving mean and std.
3. **Learned FF mapping**: the student does not know the true FF weights.

Training optimises FF weights via a shared rank-`ff_rank` factor `U @ V`
over the mitral block (split column-wise across layers) plus full-rank
non-mitral rows, and recurrent scaling factors. Recurrent weights
themselves are never trained.

```bash
./run experiments/teacher-student/full-inference/hidden-units/experiment.toml
```

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `student_mismatch.hidden_fraction` | 0.0 | Fraction of (remaining) neurons that are unobserved |
| `student_mismatch.missing_unit_fraction` | 0.0 | Fraction of neurons structurally removed |
| `student_mismatch.noise_frac` | 0.0 | Magnitude of multiplicative weight noise |
| `student_mismatch.input_type` | `"constant"` | FF input signal: `"constant"`, `"latents"`, or `"exact_spikes"` |
| `student_mismatch.ff_init` | `"constant"` | FF weight init: `"constant"` or `"teacher"` (oracle) |
| `training.ff_rank` | 20 | Rank of the shared mitral-block low-rank factor `U @ V` |
| `training.ff_smoothing_tau` | 100.0 | Gaussian sigma (ms) for FF input smoothing + Bernoulli resampling |
| `optimiser.lr_weights` / `lr_scaling` | 8e-3 / 1e-3 | LR for weight vs scaling-factor groups |
| `optimiser.grad_clip` | 0.4 | Single grad-norm clip applied to every parameter group |

## no-hidden-units/

Fully-observed case — the entire network is visible. Two-phase pipeline
(scaling-factor gradient → low-rank FF gradient) adapted from
`inferring-inputs/learn-ff-connectivity`, plus the `[student_mismatch]`
perturbations:

| Knob | Description |
|---|---|
| `input_type` | `"exact_spikes"` (default), `"constant"`, or `"latents"` |
| `firing_rate_hz` | Poisson rate when `input_type="constant"` |
| `noise_frac` | Per-cell-type-pair multiplicative log-normal noise on recurrent weights |
| `missing_unit_fraction` | Fraction of neurons structurally removed from the student |
| `shuffle_connectome` | Per-cell-type-pair shuffle of the recurrent matrix (E-E, E-I, I-E, I-I permuted independently). Preserves per-pair distribution and sparsity; destroys the specific wiring. |
| `shuffle_weights` | Per-cell-type-pair shuffle of nonzero weight values within the existing connectome. Preserves the binary topology exactly; destroys only the pairing between weight values and specific connections. |
| `no_connectome` | Replace the recurrent matrix with a fully-connected, per-cell-type-pair uniform matrix. Phase 2 then learns the recurrent weights directly (full-rank, no scaling factors). |

```bash
./run experiments/teacher-student/full-inference/no-hidden-units/experiment.toml
./run experiments/teacher-student/full-inference/no-hidden-units/experiment-shuffle-control.toml
./run experiments/teacher-student/full-inference/no-hidden-units/experiment-shuffle-weights-control.toml
./run experiments/teacher-student/full-inference/no-hidden-units/experiment-no-connectome-control.toml
```

The `experiment-shuffle-control.toml` variant flips
`shuffle_connectome = true` and leaves everything else identical. It's
the per-pair-statistics null: if inference recovers scaling factors just
as well on the shuffled connectome, specific wiring isn't what training
was exploiting.

The `experiment-shuffle-weights-control.toml` variant is a finer null:
the binary connectome (which neurons connect to which) is held fixed,
and only the nonzero weight VALUES are permuted within each
cell-type-pair block. If inference still works, then the *topology*
alone — independent of how strong each specific connection is —
carries the relevant information.

The `experiment-no-connectome-control.toml` variant goes further: it
replaces the recurrent matrix entirely with a fully-connected uniform
matrix and learns the recurrent weights from scratch in phase 2. Tests
how much the connectome helps over learning weights from per-pair means
on full connectivity.

## Analysis

See each sub-experiment's `analysis.ipynb`. Both notebooks read their
sibling `experiment.toml` via `load_experiment_config("experiment.toml")`.

## bias-check/

Diagnostic, no training. Learnt scaling factors come out systematically below
their targets. This runs a completed reference run forward at its learnt scaling
factors and again with only the recurrent scaling factors replaced by their
targets, holding the learnt low-rank feedforward block fixed in both.

If `correct` gives the lower loss the shrinkage is a training bias; if `learnt`
does, the objective's minimum genuinely sits below the targets — which is the
expected outcome whenever `noise_frac` or `missing_unit_fraction` are non-zero,
since no scaling factor can undo those.

Weights are read from the reference run's own `initial_state/`, `final_state/`
and `targets/` snapshots rather than re-derived from the teacher. That matters:
input symlinks resolve by path, so regenerating `teacher-activity` silently
changes what a historical run appears to have trained on.

```bash
./run full-inference/bias-check/experiment.toml
```
