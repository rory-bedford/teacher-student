# TODO

## In progress

### Nothing

The perturbation evaluation is **done** (2026-09-18): `common.perturbation.evaluate_perturbation`
is cached per run as `perturbation.npz`, every figure's `analysis.py` writes its delta rows, and
every figure has a perturbation panel. The perturbed teacher is shared between runs with the same
target set. Figure 1 also carries the Δrate scatter and the mean-Δ-per-population panel.

The claim to test with it, once all seeds are in: the connectome's advantage over the controls is
larger on perturbations than on held-out stimuli. Do not make that claim in the talk until Figure
2's controls are scored on more than seed 44 -- on the old-recipe smoke test only missing
connections (Figure 4's dropout) showed the asymmetry, and the wrong-connectome controls were bad
at both.

Not implemented: SMOKETEST.md section 3 (free running).

## Planned

### Merge and split errors, in connectomics terms (2026-09-21)

Not for this talk; the user decided against it for now.

Figure 4's two arms already cover the *missing*-structure half of the EM error taxonomy:
synapse dropout is a cell losing a scattered fraction of its inputs, neuron removal is a
cell never reconstructed (or never matched to activity). Both resemble **split** errors,
where one cell is reconstructed as two fragments and the fragment matched to activity
carries only part of its synapses. Relabelling the existing figure in that language is
free and could be done at any time.

**Merges are the untested error**, and the only one that *adds* structure. Two cells
reconstructed as one object give a modelled unit with the union of their synapses **and a
blended activity trace** -- that second consequence is the point, and nothing in the
current sweep touches it.

How to model it:
1. Merge a fraction of neuron pairs drawn *within cell type* (and ideally within
   assembly): EM merges happen between adjacent processes, so random E-I pairs would be a
   strawman.
2. The merged unit's incoming weights are the union of the pair's; its outgoing weights
   are their sum onto each target.
3. Its forced activity is the **sum of the pair's spike trains**. This is what makes it a
   merge rather than just extra synapses.
4. Sweep the fraction of neurons involved, 0 to 0.5, scored on Figure 4's existing
   "fraction of recurrent input misassigned" axis.

**Cost:** the connectivity side is an hour in `common/structure.py`. The activity side is
the work -- merged units need their forced traces combined, which touches the
dataloader/collate path that assumes one modelled unit per teacher neuron. Roughly half a
day, then 15 runs (5 levels x 3 seeds, ~2 h each) and an analysis pass.

Until it exists, say plainly that merges are untested rather than implying the figure
covers them.

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

## Decisions log

### 2026-09-18

- **50% observed everywhere** (was 10%): roughly the fraction of reconstructed neurons expected to
  have activity. Figure 1 is the canonical 50% run set -- Figure 2's baseline bar and Figure 3's
  top point both read its runs. Old 10% runs in `bernstein/_superseded/obs-0.1/`.
- **Figure 2 is two panels**, held-out and perturbation ΔFluctuation R², three controls (full
  connectome, learnt recurrence, configuration model). The weight shuffle (`shuffle_inputs`, each
  neuron's input weights permuted among its own partners) is still trained and still in the CSVs
  but not plotted: it overlaps with Figure 5's weight noise. The schematic was dropped.
- **Figure 3 stops at 2% observed.** With three seeds (2026-09-19) the low end is bimodal, not a
  cliff: each run either trains or loses its excitatory population, seed 45 succeeding at 2% where
  44 and 46 failed, and the reverse at 5%. Mean unobserved R² is still monotonic (0.20 / 0.38 /
  0.76 / 0.97) but SD reaches 0.72 at 5% observed. Plot the seeds individually.
- **Fluctuation R² only** outside the rate scatters; scatters are linear 0-40 Hz.
- **Figure 1 gained** the scaling-factor recovery panel (its runs beside Figure 2's fully observed
  ones) and the mean-Δrate panel, split off the Δ scatter.
- **A graded weight control does not exist in the shuffle family.** Untrained damage scans: at
  equal per-neuron weight correlation (0.99), weight noise costs 0.22 R² and a shuffle costs 0.78;
  only a partial shuffle of 1-2% of synapses lands mid-range, and that is dynamically the same
  thing as mild weight noise. Scans in the session log, not kept.

## Decisions log (2026-09-17)

- **Figures 1–5 training recipe:** archived visible-driven recipe (lr 8e-3 → 5e-4, clip 5) plus
  unobserved-rate penalties (mean and SD, 0.5 each). The VR-only recipe collapsed under 10% synapse
  dropout; lr 4e-3 / clip 2 converged worse on held-out data. Earlier runs are in `bernstein/_superseded/`.
- **Evaluation:** 20 spike-flip draws per model (one extra spike at t = 0 in an unobserved neuron),
  traces averaged across draws before scoring; ceiling = perfectly specified student under the same
  forcing and flips. No shuffled floor and no Poisson noise ceiling on the figures.
- **Weight noise (Figure 5):** the mean/SD-preserving perturbation and its clipping are intentional.
- **Figure 3:** marks the band between the teacher's 80% and 90% variance PCs (104-236 neurons,
  2.1-4.7% of the population) from `generate-teacher-activity/dimensionality.py`, not the
  participation ratio (41 neurons, 0.8%) -- that sits inside the collapsed region and would imply
  the opposite of what the sweep shows.
- **Cluster:** SLURM arrays run from per-submission code snapshots; nodes call `.venv/bin/python`
  directly (never `uv run`); W&B key from `/tachyon/.../.netrc-wandb`.
