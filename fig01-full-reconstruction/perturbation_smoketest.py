"""Figure 1 smoke test: perturbation evaluation under teacher forcing (SMOKETEST.md).

Stages, each a separate job (``slurm/submit_perturbation.sh`` chains them):

    cal-point     GPU  teacher at one current (--shift); once per teacher, run in parallel
    cal-pick      CPU  interpolate the current that halves targeted I rates; once
    teacher-on    GPU  perturbed teacher: the fixed current on this run's targets
    student-off   GPU  trained student, unperturbed, spike-flip ensemble
    perfect-off   GPU  perfectly specified student, unperturbed
    student-on    GPU  trained student, perturbed (needs teacher-on)
    perfect-on    GPU  perfectly specified student, perturbed (needs teacher-on)
    score         CPU  summary.csv, floor.csv, teacher_perturbation_response.csv, config.yaml

    uv run python fig01-full-reconstruction/perturbation_smoketest.py <stage> --run <run dir> --out <dir>

Sections 1 (held-out, teacher-forced) and 2 (perturbation, teacher-forced) of
SMOKETEST.md; section 3 (free running) is not implemented. Held-out scores are computed
from the student-off / perfect-off simulations with the standard evaluation protocol.
The rest-potential shift is exactly a constant current (see ``common/perturbation.py``).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import toml
import torch
import yaml
from connectome_snns.analysis import r_squared

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluation import (
    N_PERTURBATIONS,
    PERTURBATION_SEED,
    group_metrics,
    run_parameters,
    run_student,
    smooth,
)
from common.model import neuron_sets
from common.perturbation import (
    calibrated_shift,
    calibration_dir,
    calibration_point,
    choose_targets,
    delta_scores,
    held_out_teacher,
    pick_current,
    rate_reduction,
    simulate_teacher,
)
from common.structure import load_structure

STAGES = (
    "cal-point",
    "cal-pick",
    "teacher-on",
    "student-off",
    "perfect-off",
    "student-on",
    "perfect-on",
    "score",
)
SIMULATED = ("student-off", "perfect-off", "student-on", "perfect-on")
CELL_TYPES = {0: "E", 1: "I"}
N_FLOOR_PERMUTATIONS = 5
STARTED = time.time()


def log(message):
    print(f"[{time.time() - STARTED:5.0f}s] {message}", flush=True)


def flip_ids(sets):
    return np.random.default_rng(PERTURBATION_SEED).choice(
        sets["unobserved"], size=N_PERTURBATIONS, replace=False
    )


def save_spikes(path, spikes, **extra):
    np.savez_compressed(
        path, spikes=np.packbits(spikes, axis=-2), n_steps=spikes.shape[-2], **extra
    )


def load_spikes(path):
    with np.load(path) as data:
        n_steps = int(data["n_steps"])
        spikes = np.unpackbits(data["spikes"], axis=-2, count=n_steps).astype(bool)
        extra = {k: data[k] for k in data.files if k not in ("spikes", "n_steps")}
    return spikes, extra


def simulate(stage, run_dir, out_dir, device, shift=None):
    structure = load_structure(run_dir)
    sets = neuron_sets(structure)
    params = run_parameters(run_dir)
    burn_in_ms = params["evaluation"]["burn_in_ms"]
    if stage == "cal-point":
        reduction = calibration_point(run_dir, device, shift, burn_in_ms)
        log(f"shift {shift:+.1f} mV -> targeted rate -{100 * reduction:.0f}%")
        return
    if stage == "teacher-on":
        shift_mv = calibrated_shift(run_dir)
        targets = choose_targets(structure, sets["unobserved"])
        teacher_off, dt = held_out_teacher(run_dir, device)
        teacher_on = simulate_teacher(run_dir, device, targets, shift_mv)
        reduction = rate_reduction(
            teacher_off, teacher_on, targets, int(burn_in_ms / dt)
        )
        log(
            f"{targets.size} targets, shift {shift_mv:+.2f} mV -> "
            f"targeted rate -{100 * reduction:.0f}%"
        )
        save_spikes(
            out_dir / "teacher_on.npz",
            teacher_on,
            targets=targets,
            shift_mv=shift_mv,
            reduction=reduction,
        )
        return

    perfect = stage.startswith("perfect")
    kwargs = {}
    if stage.endswith("-on"):
        teacher_on, extra = load_spikes(out_dir / "teacher_on.npz")
        kwargs = {
            "teacher_spikes": teacher_on,
            "rest_shift": (extra["targets"], float(extra["shift_mv"])),
        }
    result = run_student(
        run_dir, device, perfect=perfect, flip_ids=flip_ids(sets), **kwargs
    )
    save_spikes(
        out_dir / f"{stage}.npz", result["student_all"], dt=np.array(result["dt"])
    )
    log(f"saved {stage}")


def score(run_dir, out_dir):
    params = run_parameters(run_dir)
    structure = load_structure(run_dir)
    sets = neuron_sets(structure)
    ct = structure["cell_type_indices"]
    evaluation_cfg = params["evaluation"]
    tau_ms = evaluation_cfg["fluctuation_tau_ms"]

    teacher_off, _ = held_out_teacher(run_dir)
    teacher_on, extra = load_spikes(out_dir / "teacher_on.npz")
    loaded = {stage: load_spikes(out_dir / f"{stage}.npz") for stage in SIMULATED}
    dt = float(loaded["student-off"][1]["dt"])
    runs = {stage: spikes for stage, (spikes, _) in loaded.items()}
    n_steps = runs["student-off"].shape[1]
    burn_in = int(evaluation_cfg["burn_in_ms"] / dt)
    window = slice(burn_in, n_steps)
    teacher_off, teacher_on = teacher_off[window], teacher_on[window]
    runs = {stage: spikes[:, window] for stage, spikes in runs.items()}
    targets, shift_mv = extra["targets"], float(extra["shift_mv"])
    rows, floor_rows = [], []

    # Section 1: held-out scores, standard protocol, plus a one-off floor.
    rng = np.random.default_rng(0)
    duration_s = teacher_off.shape[0] * dt / 1000.0
    for group in ("observed", "unobserved"):
        ids = sets[group]
        student = group_metrics(
            teacher_off[:, ids], runs["student-off"][:, :, ids], dt, tau_ms
        )
        ceiling = group_metrics(
            teacher_off[:, ids], runs["perfect-off"][:, :, ids], dt, tau_ms
        )
        for metric in ("fluctuation_r2", "activity_r2"):
            rows.append(
                {
                    "mode": "teacher_forced",
                    "condition": "heldout",
                    "group": group,
                    "cell_type": "all",
                    "metric": metric,
                    "value": student["values"][metric],
                    "ceiling_value": ceiling["values"][metric],
                }
            )
        teacher_smooth = smooth(teacher_off[:, ids], tau_ms, dt)
        student_smooth = np.zeros_like(teacher_smooth)
        for trial in runs["student-off"]:
            student_smooth += smooth(trial[:, ids], tau_ms, dt) / N_PERTURBATIONS
        floors = {"fluctuation_r2": [], "activity_r2": []}
        for _ in range(N_FLOOR_PERMUTATIONS):
            order = rng.permutation(ids.size)
            floors["fluctuation_r2"].append(
                r_squared(teacher_smooth.ravel(), student_smooth[:, order].ravel())
            )
            floors["activity_r2"].append(
                r_squared(student["teacher_rates"], student["student_rates"][order])
            )
        for metric, values in floors.items():
            floor_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "shuffled_floor_value": float(np.mean(values)),
                }
            )

    # Section 2: perturbation effect.
    targeted = np.zeros(ct.size, dtype=bool)
    targeted[targets] = True
    unobserved, observed = sets["unobserved"], sets["observed"]
    groups = {
        ("unobserved", "E"): unobserved[ct[unobserved] == 0],
        ("unobserved", "I"): unobserved[(ct[unobserved] == 1) & ~targeted[unobserved]],
        ("targeted", "I"): targets,
        ("observed", "E"): observed[ct[observed] == 0],
        ("observed", "I"): observed[ct[observed] == 1],
    }
    for (group, cell_type), ids in groups.items():
        student = delta_scores(
            teacher_off,
            teacher_on,
            runs["student-off"],
            runs["student-on"],
            ids,
            dt,
            tau_ms,
        )
        ceiling = delta_scores(
            teacher_off,
            teacher_on,
            runs["perfect-off"],
            runs["perfect-on"],
            ids,
            dt,
            tau_ms,
        )
        for metric in student:
            rows.append(
                {
                    "mode": "teacher_forced",
                    "condition": "perturbation",
                    "group": group,
                    "cell_type": cell_type,
                    "n_cells": int(ids.size),
                    "metric": metric,
                    "value": student[metric],
                    "ceiling_value": ceiling[metric],
                }
            )

    delta_rate = (teacher_on.sum(axis=0) - teacher_off.sum(axis=0)) / duration_s
    response = []
    for cell_type in (0, 1):
        for is_targeted in (False, True):
            mask = (ct == cell_type) & (targeted == is_targeted)
            if mask.any():
                response.append(
                    {
                        "population": CELL_TYPES[cell_type],
                        "targeted": int(is_targeted),
                        "n_cells": int(mask.sum()),
                        "mean_delta_rate_hz": float(delta_rate[mask].mean()),
                    }
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(floor_rows).to_csv(out_dir / "floor.csv", index=False)
    pd.DataFrame(response).to_csv(
        out_dir / "teacher_perturbation_response.csv", index=False
    )

    # A constant current I enters the voltage update exactly like an E_L shift of
    # I / g, with g = (1 - exp(-dt / tau_mem)) * C_m / dt the simulator's leak factor.
    inhibitory = params["recurrent"]["physiology"]["inhibitory"]
    tau = inhibitory["tau_mem"]
    leak = (1 - np.exp(-dt / tau)) * tau * inhibitory["g_L"] / dt
    config = {
        "run_dir": str(run_dir),
        "run_seed": int(params["simulation"]["seed"]),
        "run_student": params["student"],
        "run_optimiser": params["optimiser"],
        "run_loss": params["loss"],
        "evaluation": dict(evaluation_cfg),
        "spike_flip_draws": int(N_PERTURBATIONS),
        "spike_flip_seed": int(PERTURBATION_SEED),
        "scored_window_steps": [int(burn_in), int(n_steps)],
        "perturbation": {
            "targets": "unobserved inhibitory cells",
            "target_fraction": 0.25,
            "n_targets": int(targets.size),
            "target_ids": [int(i) for i in targets],
            "rest_potential_shift_mv": shift_mv,
            "equivalent_current_pa": float(leak * shift_mv),
            "targeted_teacher_rate_reduction": float(extra["reduction"]),
            "calibration": toml.load(calibration_dir(run_dir) / "current.toml"),
            "mitral_input": "frozen: the held-out trial's own spikes, identical for all runs",
            "teacher_forcing": "teacher's perturbed observed activity",
        },
        "not_implemented": "section 3 (free running)",
    }
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(summary)
        print(pd.DataFrame(floor_rows))
        print(pd.DataFrame(response))
    log(f"wrote {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shift", type=float, help="cal-point: rest shift in mV")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "score":
        score(args.run, args.out)
    elif args.stage == "cal-pick":
        log(f"calibrated shift {pick_current(args.run):+.2f} mV")
    else:
        simulate(args.stage, args.run, args.out, args.device, args.shift)
