# TODO

## In progress

### Perturbation evaluation (Figures 1 and 2)

Teacher-forced, as specified in `fig01-full-reconstruction/SMOKETEST.md` (sections 1–2). A constant
hyperpolarising current on a random 25% of each run's unobserved I cells, in teacher and student;
the student is forced with the teacher's perturbed observed activity; scored on Δ = perturbed −
unperturbed for unobserved non-targeted neurons (E / I), targeted cells reported separately.

Done (2026-09-17):
- Smoke test on the old-recipe Figure 1 seed-44 run (`bernstein/_tests/perturbation-smoketest-oldrecipe-seed44/`,
  NOTES.md there). Non-targeted unobserved ΔActivity R² 0.93 (E) / 0.92 (I), ceilings 0.96;
  ΔFluctuation R² 0.66 / 0.67, ceilings 0.77 / 0.76. Teacher: targeted I −11 Hz, other I +3.7 Hz,
  E +0.7 Hz (disinhibition; mean I rate rises slightly).
- Current calibrated once on the teacher: rest-potential shift −99.5 mV ≈ −90 pA for a 50% rate
  loss (`bernstein/_evaluation/perturbation-calibration/current.toml`).
- Staged SLURM pipeline (`slurm/submit_perturbation.sh`) and GPU scoring (38 s).

Still to do:
- Move the smoke-test logic into the shared analysis (`common/perturbation.py`, cached per run) so
  every figure's `analysis.py` computes it; runs sharing a seed share the perturbed teacher.
- Figure 1: Δrate scatter panel (teacher vs student Δrate per neuron, targeted cells marked, inset of
  mean Δ per population); draft at `.../perturbation-smoketest-oldrecipe-seed44/fig01_perturbation_draft.png`.
  Consider a symlog axis: a few neurons change by ±100 Hz, most by < 20 Hz.
- Figure 2: second row of bars per variant — ΔActivity R² as headline, ΔFluctuation R² alongside.
- Rerun on the new-recipe Figure 1 seed-44 run once it finishes, then all Figure 1–2 seeds.
- **The claim to test:** the connectome's advantage over the controls is larger on perturbations
  than on held-out stimuli. No perturbation claim in the talk until Figure 2's controls are scored.
- Optional: SMOKETEST.md section 3 (free running) — not implemented.

## Planned

### Figure 7 — model mismatch

Test robustness to mismatch in the neuron model: the student's physiological parameters differ from
the teacher's. Details to come.

## Deferred — not for the Bernstein talk

### Tidy the READMEs once everything has run

These READMEs are written as instructions to an agent. Once the runs are done and the figures exist, strip out everything that was scaffolding: the "needs running / reusable" status sections, the pointers to the archive, the minimal-edits instructions, open questions that have been answered, and the rationale for design choices already made.

What should remain is a record of what each experiment *was* — configuration, evaluation protocol, panels, file contents — so the folders read as documentation of the work rather than as a task list.

### Further perturbations

Other interventions (stimulate one assembly, ablate a cell type) and applying the perturbation
evaluation to Figure 6's trained networks.

### Maybe: an extra slide on the real dataset

Optional, only if the talk has room after rehearsal. The 10% endpoint of Figure 6 already carries the connection — it is our actual reconstruction budget — so a dedicated slide is a bonus, not a gap to fill.

### Real Dp connectome figure

Dropped from the talk. The material exists in `archive-old-figures/input_cumulative_volume.*`: for each proofread cell, how much of its input volume is captured as presynaptic partners are resolved, split spiny (E) / smooth (I).

Headline numbers, if ever wanted: spiny cells have a median of 700 presynaptic partners and need ~456 of them for 90% of input volume; smooth cells have ~48 and need ~24. At a 10% budget, resolving partners at random captures ~10% of input volume.

Caveat before any use: the smooth-cell partner count may be detection-limited on smooth dendrites, which would undercut the E/I asymmetry.

### Allocation / active connectomics

Deferred entirely. Would be the same experiment as Figure 6 with a non-random rule for choosing the reconstructed segment S — a connected subpopulation rather than a random draw — at matched budget.

## Decisions log (2026-09-17)

- **Figures 1–5 training recipe:** archived visible-driven recipe (lr 8e-3 → 5e-4, clip 5) plus
  unobserved-rate penalties (mean and SD, 0.5 each). The VR-only recipe collapsed under 10% synapse
  dropout; lr 4e-3 / clip 2 converged worse on held-out data. Earlier runs are in `bernstein/_superseded/`.
- **Evaluation:** 20 spike-flip draws per model (one extra spike at t = 0 in an unobserved neuron),
  traces averaged across draws before scoring; ceiling = perfectly specified student under the same
  forcing and flips. No shuffled floor and no Poisson noise ceiling on the figures.
- **Weight noise (Figure 5):** the mean/SD-preserving perturbation and its clipping are intentional.
- **Figure 3:** marks the teacher participation ratio from `generate-teacher-activity/dimensionality.py`
  (all training trials, PR 41.4), not the held-out trial.
- **Cluster:** SLURM arrays run from per-submission code snapshots; nodes call `.venv/bin/python`
  directly (never `uv run`); W&B key from `/tachyon/.../.netrc-wandb`.
