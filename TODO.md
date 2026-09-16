# Deferred — not for the Bernstein talk

## Tidy the READMEs once everything has run

These READMEs are written as instructions to an agent. Once the runs are done and the figures exist, strip out everything that was scaffolding: the "needs running / reusable" status sections, the pointers to the archive, the minimal-edits instructions, open questions that have been answered, and the rationale for design choices already made.

What should remain is a record of what each experiment *was* — configuration, evaluation protocol, panels, file contents — so the folders read as documentation of the work rather than as a task list.

## Perturbation experiments

Inference on already-trained networks, so far cheaper than training — but not needed for the 28 September talk, and **no perturbation claim should appear in the talk until these exist.**

The experiment: perturb a population in the teacher (silence inhibitory neurons, stimulate one assembly, ablate a cell type), run the trained student under the same perturbation, compare.

The claim it would support: the constrained model's advantage over controls is **larger on perturbations than on held-out stimuli**. That asymmetry is the strongest argument for connectome constraints, since it is the case where fitting activity cannot substitute for capturing mechanism.

Worth doing after the talk, on the trained networks from Figures 2 and 6.

## Maybe: an extra slide on the real dataset

Optional, only if the talk has room after rehearsal. The 10% endpoint of Figure 6 already carries the connection — it is our actual reconstruction budget — so a dedicated slide is a bonus, not a gap to fill.

## Real Dp connectome figure

Dropped from the talk. The material exists in `archive-old-figures/input_cumulative_volume.*`: for each proofread cell, how much of its input volume is captured as presynaptic partners are resolved, split spiny (E) / smooth (I).

Headline numbers, if ever wanted: spiny cells have a median of 700 presynaptic partners and need ~456 of them for 90% of input volume; smooth cells have ~48 and need ~24. At a 10% budget, resolving partners at random captures ~10% of input volume.

Caveat before any use: the smooth-cell partner count may be detection-limited on smooth dendrites, which would undercut the E/I asymmetry.

## Allocation / active connectomics

Deferred entirely. Would be the same experiment as Figure 6 with a non-random rule for choosing the reconstructed segment S — a connected subpopulation rather than a random draw — at matched budget.
