# Backup figures

Panels kept for questions from the room, not for the talk's main line. They read the CSVs
the `figNN-*/analysis.py` scripts already write, so nothing here evaluates anything.

```bash
uv run python backup-figures/figures.py     # -> backup-a-scaling-factors.svg
```

## backup-a-scaling-factors

The six tied scaling factors (learnt / true), fully observed beside 50% observed.

- **Fully observed** (`fig02-controls/connectome-fully-observed__seed-*`): all six recover
  to 1.0000 from a log-normal initialisation up to a factor of two off, with the van
  Rossum loss reaching 1e-6. With every neuron teacher-forced the student is not a closed
  loop, so this is six scalars in front of a fixed input with a noiseless optimum at the
  true value — an identifiability check, and the reason the ceilings elsewhere are real.
- **50% observed** (`fig01-full-reconstruction/seed-*`): the same factors land at
  0.57-1.19 while held-out Fluctuation R² is still 0.86/0.90. Good predictions do not
  imply recovered parameters.

**The honest reading of the gap.** The true parameters remain scaling factor 1.0 at 50%
observed, and the perfectly specified student (which *is* scaling factor 1.0) scores
*higher* on held-out data than the trained one -- 0.920/0.940 against 0.864/0.898. So the
drift is not a better solution: the training loss and held-out prediction disagree once
half the network is simulated, and the distance from 1.0 is roughly the size of the gap to
the ceiling. Do not present the drift as principled shrinkage without testing it (compare
the training loss at scaling factor 1.0 against the learnt factors on the same chunks).

Error bars need more than one run per condition: the 50% side gains seeds 45 and 46 from
the overnight grid, while the fully observed side is seed 45 alone unless those runs are
resubmitted (they left Figure 2's grid on 2026-09-18).
