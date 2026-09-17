# Smoke test: perturbation evaluation, Figure 1 model only

Read `data/METHODS.md` and `data/fig01-full-reconstruction/README.md` first.

Do not retrain anything. Use the existing trained student from the Figure 1 configuration (ff-known, obs-10, recur-100, wn-0), one seed. Make minimal edits to the existing evaluation code. Write the resolved config into every output.

**Goal:** check that perturbation evaluation works under teacher forcing — the conditional claim *"during an optogenetics experiment you record part of the circuit; predict how the rest responds"*. A free-running evaluation is included as an optional extra only.

**The rule that keeps teacher forcing valid:** never inject the activity of a neuron that is being scored.

---

## 1. Sanity check — teacher-forced, held-out stimuli (existing mode)

Re-run the existing evaluation. It should reproduce the archived figure: observed Fluctuation R² ≈ 0.997, unobserved ≈ 0.995.

If it does not, stop and report — nothing below is interpretable until this matches.

Also compute the **shuffled-identity floor** once (pair each teacher neuron with a random student neuron, recompute Fluctuation R²). If it is ≈ 0, note that in `NOTES.md`; it will not need plotting elsewhere.

## 2. Perturbation, teacher-forced (primary)

**Targets:** 25% of **unobserved I cells**, chosen at random. Targeting only unobserved cells means the student must generate the targeted cells' response itself and propagate it through its own recurrence, rather than being handed it through injected activity.

**Intervention:** the same constant hyperpolarising current applied to the targeted cells in the teacher and in the student, for the whole held-out stimulus period. Calibrate the amplitude once, in the teacher, so targeted I cells lose roughly half their firing rate. Report the value used and the resulting reduction.

**Inputs:** use **frozen input spikes**, identical for the perturbed and unperturbed runs, and identical for teacher and student. Δrate is then deterministic.

**Injection:** inject the teacher's **perturbed** recorded (observed) activity, as a real recording made during the manipulation would provide.

**Metric:** Δrate = perturbed − unperturbed, per neuron. Score Fluctuation R² and Activity R² on **Δrate**, not on raw rates — baseline activity would otherwise dominate.

**Scored neurons:** unobserved, non-targeted neurons only, split E / I. Report the targeted cells separately as a check that the student reproduces the direct effect of the intervention.

**Also report the teacher's own response:** mean Δrate for non-targeted E cells and non-targeted I cells, to see whether the teacher shows a paradoxical (inhibition-stabilised) response.

## 3. Free-running (optional — only if section 2 is done)

Inject only the feedforward inputs plus the intervention; simulate all recurrent neurons freely.

Single-trial Fluctuation R² is too harsh here: with Poisson inputs, and possibly chaotic dynamics, the teacher cannot predict itself trial to trial. So:

- Repeat each held-out trajectory K = 20 times with fresh Poisson inputs, for teacher and student.
- Compare **trial-averaged** smoothed responses.
- Ceiling = the teacher's split-half reliability (average of K/2 repeats vs the other K/2), Spearman–Brown corrected to K.
- Also report single-trial teacher-vs-teacher R², to show how much of the fluctuation is unpredictable.
- Record mean firing rate over time, student vs teacher, per population, to check for drift.

---

## Outputs

```
data/fig01-full-reconstruction/smoketest/
  summary.csv                         mode{teacher_forced,free_running}, condition{heldout,perturbation},
                                      group{observed,unobserved,targeted}, cell_type, metric, value
  floor.csv                           metric, shuffled_floor_value
  teacher_perturbation_response.csv   population{E,I}, targeted{0,1}, mean_delta_rate_hz
  rates_over_time.csv                 (section 3 only) mode, population, time_s, teacher_rate_hz, student_rate_hz
  config.yaml                         resolved config actually used, incl. perturbation amplitude and targeted cell ids
  NOTES.md                            5–10 lines: what ran, the headline numbers, anything that broke
```

Stop after this. Do not extend to other figures.

---

## How to read the results

- **Section 1 must match**, or the evaluation code has drifted from the archived runs.
- **Section 2 is the result that matters.** A high R² on Δrate for unobserved, non-targeted neurons means the full-reconstruction model predicts how an optogenetic manipulation propagates. That makes it worth running on the Figure 2 controls, where it should separate the connectome from learnt recurrence.
- **Targeted cells** should also match. If they don't, the student isn't reproducing the direct effect of the current, and nothing downstream of them can be trusted.
- **The teacher's response** shows whether the paradoxical effect exists. If it doesn't, I-cell inhibition is still a fine perturbation, just a less striking one.
- **Section 3** only bears on the stronger "predict a novel experiment" claim. Nothing in the talk depends on it.
