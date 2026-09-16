# Run priority, novelty, and fallbacks

Two questions decide the order: **how novel is this figure** (relative to Beiran & Litwin-Kumar 2025, which covers much of the same ground in rate networks), and **what happens if the run doesn't land** — is there anything in `archive-old-figures/` that could stand in, and how honestly?

---

## Run budget

~70 runs, ~140 GPU-hours, ≈3 days on two GPUs. Freeze date **22 Sep**.

| Figure | Levels × seeds | Runs |
|---|---|---|
| 1 full reconstruction | reuse + 2 extra seeds | 2 |
| 2 controls | 4 variants × 3 | 12 |
| 3 observed sweep | 5 levels × 3 | 15 |
| 4 reconstruction errors | 2 models × 4 levels × 2 | 16 |
| 5 weight noise | 3 levels × 2 | 6 |
| 6 climax | 6 levels × 3, + 1 fully observed | 19 |

**Run the coarse grid across all six first**, then spend leftover time refining wherever a curve is interesting. If something breaks on the 20th you then have every figure at low resolution rather than three at high resolution and three missing.

**Ordering constraint:** Figure 3 before Figure 6 — it sets the observed-neuron count.

---

## Priority

| Rank | Figure | Novelty vs B&LK | Fallback if the run fails | Verdict |
|---|---|---|---|---|
| **1** | **6 — learnt feedforward** | **Highest.** Their Discussion explicitly flags input-driven circuits with unreconstructed inputs as out of scope. This is the one thing in the talk they did not do. | **None usable.** The archived scatter varies four factors at once. | **Must run.** No talk without it. |
| **2** | **2 — controls** | Low. Standard, and they do the equivalent. | **None usable.** Archived bars ran at different weight noise, so they are not comparable *to each other*. | **Must run.** Load-bearing even though unoriginal. |
| **3** | **3 — observed sweep** | Low–moderate. Their Figs 2–3 in rate networks; the spiking threshold and the dimensionality test are a modest extension. | **Partial.** Figure 1 is a single point on this curve and can carry the qualitative claim alone. | Run if possible; degrade to one point. |
| **4** | **4 — reconstruction errors** | **Moderate.** They did Gaussian weight noise plus spurious synapses; whole-neuron removal is different, and removal-vs-dropout at matched input volume is genuinely new. | **Partial, with a caveat.** `sweep-recur__ff-known__obs-100__wn-0__curve` is clean but at obs-100, so measured on observed neurons. | Run the new dropout arm first — that is the novel half. |
| **5** | **5 — weight noise** | **Lowest.** Essentially their Fig. 4 in a spiking network. | **Good, with a caveat.** `sweep-wn__ff-known__obs-100__recur-100__curve` is clean, single-factor, at obs-100. | **Cut first.** Fall back to the archived curve. |
| — | **1 — full reconstruction** | Low, but foundational. | **Already done and clean.** | No risk. |

---

## What "fallback" is allowed to mean

An archived figure may be shown **only with its actual configuration stated out loud.** Two of them are clean single-factor runs that happen to sit at a different observation level — those are honest to show as long as the axis and the caveat are on the slide:

- `sweep-wn__ff-known__obs-100__recur-100` — weight noise, measured on observed neurons at full observation
- `sweep-recur__ff-known__obs-100__wn-0` — neuron removal, same caveat

`ff-learnt__obs-45__recur-81__wn-30__scatter` (observed 0.953, unobserved 0.243) is **honest to show, and it is the most striking visual available** — but for the phenomenon, not for attribution. It varies four factors, so on its own it cannot establish that unreconstructed feedforward input is the cause.

**It can, however, be bracketed using the other archived sweeps.** At 30% weight noise the model sits around R² ≈ 0.93; at ~20% neurons removed, around 0.85. Both are measured on observed neurons at full observation and so are not directly comparable — but neither is remotely close to a collapse to 0.243. The two alternative explanations are therefore individually implausible as the cause, and saying so with those curves beside the scatter is a legitimate argument. It is weaker than a single-factor run, which is why Figure 6 remains priority 1, but it is presentable.

`controls__ff-learnt__obs-90__recur-81__wn-mixed__bar` is **not** presentable, for a different reason: the bars differ from *each other* in weight noise, so the comparison the figure exists to make — connectome vs learnt recurrence vs shuffles — is not like-for-like. No caveat repairs an invalid internal comparison.

---

## Novelty summary, for the talk itself

Figures 1, 2, 3 and 5 are a **replication in a different model class** — spiking rather than rate, calcium-like metric, six parameters rather than O(N). That is worth stating plainly and briskly: it tells the room the conclusions do not depend on the rate-network setting.

Figures 4 (removal vs dropout) and 6 (unreconstructed inputs) are where the new material is. Figure 6 is the talk.
