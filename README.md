# teacher-student

Code and figures for the talk **"Connectome-constrained spiking network models of
functional activity"**, given on **28 September 2026** at the Bernstein satellite
workshop *Advances in optimization of biologically constrained models and how to use
them*.

Synthetic teacher-student experiments on conductance-based spiking networks, organised as
the figures of that talk. A teacher network generates activity; a student with a
(partially) reconstructed connectome is trained on part of that activity and asked to
predict the rest on held-out stimuli.

This repo holds the **experiment code**; it consumes the **connectome-snns** library
(simulators, run framework, dataloaders, analysis, visualization) as an editable
dependency. Data is **not** stored here — it lives in `../bernstein`.

| Dir | What it covers |
|---|---|
| `fig00-teacher-activity/` | The teacher network + spike data → `bernstein/teacher-activity/` |
| `common/` | Student construction, training, held-out evaluation, perturbation and plotting shared by all figures |
| `fig01-full-reconstruction/` | Full reconstruction, 50% observed (the baseline for Figs 2–5) |
| `fig02-controls/` | Learnt recurrence, shuffled weights, shuffled topology |
| `fig03-observed-fraction/` | Observed-fraction sweep, 50% → 1% |
| `fig04-reconstruction-errors/` | Neuron removal vs synapse dropout |
| `fig05-weight-noise/` | Weight-noise sweep, 0 → 0.5 |
| `fig06-learnt-feedforward/` | Unreconstructed inputs with learnt weights, reconstructed fraction 100% → 10% |

Each figure folder holds `README.md` (its claim, configuration and recipe as run),
`experiment.toml`, `parameters.toml`, `train.py`, `run_grid_search.py`, `analysis.py`,
`figures.py`, its CSVs and its panel SVGs. [`COLORSCHEME.txt`](COLORSCHEME.txt) defines
what each colour means across the talk.

---

# Running it

```bash
uv sync --extra cu129   # GPU
uv sync --extra cpu     # CPU
```

Always `uv run python`, never bare `python`. `ruff` is not in the environment — use
`uvx ruff`.

## The three steps

```bash
./run --grid fig05-weight-noise/experiment.toml     # train every run of the grid
uv run python fig05-weight-noise/analysis.py        # evaluate -> CSVs next to the script
uv run python fig05-weight-noise/figures.py         # CSVs -> one SVG per panel
```

`analysis.py` evaluates every completed run on a held-out teacher trial and on the
perturbation, caches both inside the run folder (`evaluation.npz`, `perturbation.npz`),
and writes only the CSVs its figure needs. `figures.py` reads those CSVs and nothing
else, so panels can be rebuilt on a laptop. Figures 2–5 read Figure 1's runs as their
baseline condition, so Figure 1 must be trained and analysed first.

`./run` is a thin wrapper over the library's run framework: `./run <config>` for one run,
`./run --grid <config>` for a grid, `./run --resume <run dir> <config>` to continue from a
checkpoint (model, optimiser, scaler, scheduler, simulation state and RNG are all
restored), `--no-commit` to skip the git check. A grid search runs from a git worktree
snapshot, so **code must be committed first**; `.toml` and `run_grid_search.py` may be
dirty.

**The teacher must exist before anything else.** `./run fig00-teacher-activity/experiment.toml`
writes `../bernstein/teacher-activity/`, which every figure symlinks as its input.
Regenerating it invalidates every trained run.

## What to edit

| To change | Edit |
|---|---|
| Where code, data and logs live | `experiment.toml` — `script`, `output_dir`, `parameters_file`, `log_file` and the `[[data.inputs]]` paths are **absolute**, so they all need updating if the repo or `../bernstein` moves |
| W&B project, group, tags | `experiment.toml` `[wandb]` — project `bernstein`, one group per figure |
| What the student is (the manipulation) | `parameters.toml` `[student]` — one entry selects it: `observed_fraction`, `weight_noise`, `neuron_removal_fraction`, `synapse_dropout_fraction`, `reconstructed_fraction`, `recurrent_model`. The dispatch is `common/structure.py` |
| Training recipe | `parameters.toml` `[training]`, `[optimiser]`, `[evaluation]` |
| Which levels and seeds a grid runs | the constants at the top of `run_grid_search.py` (`SEEDS`, `LEVELS` / `NOISE_LEVELS` / `OBSERVED_FRACTIONS` / `RECONSTRUCTED_FRACTIONS` / `VARIANTS`) |
| Which GPUs a local grid uses | `CUDA_VISIBLE_DEVICES` at the top of `run_grid_search.py` |
| Which runs a figure plots | the constants at the top of `figures.py` (e.g. `PLOTTED_VARIANTS`, `SCATTER_LEVEL`) |

A grid skips runs that have already completed, so an interrupted grid can simply be
re-launched. Cluster submission is done by scripts kept outside version control
(`slurm/`, local to this machine); they snapshot the repo per submission and run
`.venv/bin/python` directly rather than `uv run`.

## Results are read-only

`../bernstein` holds the provenance of everything: each run folder carries its own
`metadata.json` and `README.md` (this repo's commit, the library's commit and whether it
was dirty, start/end, status), snapshots of the `experiment.toml` and `parameters.toml`
it ran with, and `inputs/` symlinked to the teacher data. Never edit files there.

---

# Shared methods

Conventions common to every figure. Individual figure READMEs specify only what differs,
and record their hyperparameters as run.

## Model

Spiking teacher–student. Teacher: **5000 recurrent neurons** (E/I assemblies, tuned to
zebrafish Dp) driven by **1500 feedforward units**.

Three cell types — feedforward, E, I — giving **6 scaling factors** as the only free
parameters in the fully reconstructed case: FF→E, FF→I, E→E, E→I, I→E, I→I.

Where part of the network is not reconstructed, units outside the reconstructed set keep
their **known activity** and contribute **learnt weights** onto the modelled neurons.
Activity is cheap; reconstruction is the bottleneck. This is the regime the whole project
is about.

**Neuron model caveat:** `tau_ref` appears in the physiology but is never applied by the
simulator, so there is no refractory period — the fastest teacher cells fire with 2 ms
inter-spike intervals, up to ~270 Hz. Teacher and student share the model, so every
comparison holds, but do not describe the network as having an 8 ms refractory period.

## Teacher forcing

**Everything in this project is teacher forced.** Wherever a presynaptic neuron's true
activity is known, the ground truth is injected instead of the simulated value. This
reduces variance and prevents error accumulation through the recurrent loop; it is a
training and stability device, not a different model.

Concretely, the implementation has three parts:

1. **Feedforward units** — recorded activity, injected. Never simulated.
2. **Unobserved recurrent neurons** — genuinely recurrent among themselves, because their
   activity is unknown and must be simulated. They additionally receive the **recorded**
   neurons' true activities, injected through the correct connectome weights and scaling
   factors.
3. **Recorded recurrent neurons** — no simulated recurrence among themselves, since that
   input arrives as injected ground truth. They receive injected recorded activities and
   the simulated unobserved-layer activities.

This is equivalent to the single recurrent network it describes: FF units and recurrent
units as one simulated population, with fixed connectome weights among reconstructed
units and learnt weights from unreconstructed sources. The layer decomposition is how
teacher forcing is implemented, not a feedforward cascade.

**Learnt weights reach every modelled neuron, observed and unobserved alike.** The
weights onto *unobserved* neurons are constrained by no data directly — only indirectly,
through those neurons' recurrent influence on observed ones. That block of loosely
tethered parameters is the mechanism behind the degeneracy in Figure 6, and is worth
stating as a mechanism rather than describing the failure phenomenologically.

**Teacher forcing is the standard throughout. There is no free-running evaluation
variant.**

One caveat to be aware of for questions, not to act on: the two groups are not assessed
on identical terms, since observed neurons are predicted from ground-truth input while
unobserved neurons are simulated. **Figure 1 is the control that answers it** — at full
reconstruction the unobserved population is simulated in exactly the same way and still
reaches R² = 0.995, so the gap in Figure 6 cannot be attributed to teacher forcing. Keep
that comparison explicit in the talk.

## Observation level: 50% observed everywhere

Every figure with a fixed observation level runs at **50% of modelled neurons observed**:
roughly the fraction of reconstructed neurons expected to have activity recorded. Figure
3 sweeps the level instead and takes its 50% point from Figure 1's runs; Figure 6 fixes
`recorded_pool_fraction = 0.5`. Earlier runs at 10% observed are in
`bernstein/_superseded/obs-0.1/` (a move, not a deletion) and stay available if a figure
needs the harder regime to show an effect.

## Evaluation

- **Held-out test set of new stimuli** for every figure. Odour trajectories never used in
  training.
- **Fluctuation R² (primary)**: spike trains smoothed with a **50 ms Gaussian**, then R².
  A close stand-in for the calcium trace the real pipeline matches against (exponential
  filter, τ = 100 ms), so it answers "how well would these spike trains agree once seen
  through calcium". Say this in the talk — it makes the metric a property of the
  experiment rather than an arbitrary choice.
- **Activity R² (secondary)**: firing rates.
- Reported **separately for observed and unobserved** neurons. Unobserved is the
  discriminative group; every model can fit what it is shown.
- The reference on every figure is a **noise ceiling**: a perfectly specified student
  under the same teacher forcing and the same 20 spike-flip draws (see
  `fig01-full-reconstruction/README.md`). No floor is plotted — the shuffled-identity
  floor was dropped on 2026-09-17.
- **Perturbation evaluation**, on every figure as its own panel: an optogenetic-style
  E_L shift (−99.5 mV, ≈ −90 pA) applied to 25% of unobserved inhibitory cells, scored on
  Δ = on − off. The perturbed teacher is shared between runs with the same target set.
- **≥3 seeds** per condition, with the spread shown.
- **Do not compare observed with unobserved within one seed.** Pooled R² counts the
  spread of mean rates across the group as explainable variance, and with heavy-tailed
  rates that spread is set by a handful of fast cells, so which neurons land in the
  observed sample shifts the value (seed 44's observed draw misses the tail, which is why
  unobserved looks better there). It averages out over seeds.

## What the fit recovers, and what it does not

The student's six tied scaling factors are the only free parameters in the fully
reconstructed case, and the true model is scaling factor 1.0 for all six (the student's
physiology is the teacher's). What training recovers depends on the observation level:

| observed | learnt / true | note |
|---|---|---|
| 100% | 1.00 (7 d.p.) | no unobserved population, so the rate penalties do not exist as loss terms; the van Rossum term alone has a zero-loss optimum at the truth, reached from a log-normal initialisation up to 2× off |
| 50% | 0.79–1.01 | typical error 6% |
| 10% | 0.57–1.19 | |
| ≤2% | 0.23–11.1 | unreliable: succeeds on some observed draws, collapses on others |

Two things make partial observation different, and neither is "the same loss, harder": the
rate penalties are **absent** when nothing is unobserved, and at 50% observed the SD
penalty is ~18% of the total loss; and the van Rossum term's own optimum moves, because a
partly free-running network cannot match the teacher spike for spike.

**Below ~25% observed the outcome is bimodal, not graded** (three seeds, 2026-09-19): a
run either trains normally or loses its excitatory population entirely, and which happens
depends on the observed draw rather than the fraction. Unobserved Fluctuation R² per
seed: 2% observed −0.03 / 0.67 / −0.03; 5% observed 0.82 / −0.45 / 0.76; 25% observed
0.94 / 0.39 / 0.94; 50% observed 0.98 / 0.94 / 0.99. Report the individual seeds, not just
mean ± SD, and describe the low end as unreliable rather than impossible — it is an
optimisation failure, not an information limit.

So state results as **prediction**, not parameter recovery: at 50% observed the student
predicts unobserved activity well while its parameters drift, and the perfectly specified
student (scaling factor 1.0) scores *higher* on held-out data than the trained one. Part
of every figure's gap to its ceiling is therefore our own regulariser, not missing
information. Do not claim recovered parameters outside the fully observed case.

## Weight noise

Multiplicative log-normal noise on existing weights, applied per (source type, target
type) block and then rescaled so the **mean and SD of each block are preserved**, finally
clipped at zero. Topology is untouched — no synapses created or deleted, and clipping
means no synapse changes sign, so Dale's law holds. This is measurement error on weights,
which is what makes the contrast with missing connections meaningful. The clipped
fraction is recorded per run as `noise_clipped_fraction`.

## Figure conventions

- **One SVG per panel**, `figNN-<letter>-<slug>.svg`, at the size it is inserted into the
  talk at 100%. A rebuild deletes that figure's existing panels first
  (`common.style.clear_panels`), so an orphan from an earlier build cannot end up on a
  slide.
- Panel titles say **what is plotted**, not what it proves — the slide carries the claim.
- Fixed colours across all figures, never reassigned: `COLORSCHEME.txt` for the meanings,
  `common/style.py` for the semantic names figures import (`TRUTH`, `MODEL`, `OBSERVED`,
  `UNOBSERVED`, `EXCITATORY`, `INHIBITORY`, `REFERENCE_GREY`).
- **Fluctuation R² only** outside the rate scatters: Activity R² is still scored and kept
  in every CSV, but the scatters are where rates are reported.
- Rate scatters are linear over **0–40 Hz** and perturbation Δ scatters over **±25 Hz**;
  cells beyond the range are not plotted and not counted in the labels.
- Legends are framed, with one marker scale across every panel.
- Vector output at final size; axis fonts legible at 12 cm wide on a projected slide.
- **No analysis in training scripts**: training saves raw outputs, `analysis.py` computes
  metrics, `figures.py` plots. No notebooks.

## Naming

**observed / unobserved** and **reconstructed / unreconstructed**, throughout — in code,
CSV columns, titles and labels. "Hidden", "full inference" and "unreconstructed fraction"
are retired: each meant two different things in different earlier figures, which caused
real confusion.

Run folders are named for what they vary — `seed-44`, `wn-0.3__seed-45`,
`recon-0.5__seed-44`, `learnt-100ep__seed-46` — never by date. A rerun of a condition
replaces its folder rather than accumulating copies.

---

# Not done

- **The perturbation has no free-running variant** (it is teacher forced like everything
  else), and it has not been applied to Figure 6's trained networks.
- **Merge errors** (above), and **non-random choice of the reconstructed segment** —
  Figure 6 with a connected subpopulation instead of a random draw, at matched budget.
- **Figure 7, model mismatch:** the student's physiological parameters differing from the
  teacher's. Not started.
