"""Optogenetic-style perturbation evaluation (teacher-forced), shared by figure analyses.

A constant hyperpolarising current is applied to a random 25% of the unobserved
inhibitory cells, in the teacher and in the student, for the whole held-out trial. The
mitral input spikes are frozen (the held-out trial's own), so the teacher's perturbed
and unperturbed runs differ only by the intervention. The student is teacher-forced with
the teacher's *perturbed* observed activity, as a recording made during the manipulation
would provide, and must produce the targeted cells' response and its propagation itself.

The current is implemented as a shift of the targeted cells' leak reversal potential:
in these LIF neurons the leak term is g_L (v - E_L), so lowering E_L by d mV is exactly a
constant current of g_L * d. Its size is calibrated once, in the teacher (targets: 25% of
all its I cells), so targeted cells lose about half their rate: ``calibration_point`` at a
few currents in parallel, then ``pick_current`` interpolates. Every run then applies that
fixed current to its own targets (25% of its unobserved I cells).

Scores are on the *change* caused by the intervention (perturbed - unperturbed), so
baseline activity does not dominate: Fluctuation R² on the smoothed-trace difference and
Activity R² on the per-neuron rate difference. The student side uses the evaluation's
spike-flip ensemble (``common.evaluation.N_PERTURBATIONS`` draws, traces averaged before
differencing); the teacher side is its single deterministic pair of trajectories.
"""

import hashlib
from pathlib import Path

import numpy as np
import toml
import torch
import zarr
from connectome_snns.analysis import r_squared
from connectome_snns.analysis.test_inputs import _resolve_teacher_dir
from connectome_snns.configs.conductance_based import (
    FeedforwardLayerConfig,
    RecurrentLayerConfig,
)
from connectome_snns.network_simulators.conductance_based.simulator import (
    ConductanceLIFNetwork,
)
from connectome_snns.network_simulators.projections import make_frozen_projections

from common.evaluation import (
    N_PERTURBATIONS,
    PERTURBATION_SEED,
    held_out_trial,
    run_parameters,
    run_student,
    smooth_mean,
)
from common.model import neuron_sets
from common.structure import load_structure

TARGET_FRACTION = 0.25
TARGET_SEED = 0
TARGET_RATE_REDUCTION = 0.5
#: Rest-potential shifts (mV) simulated in parallel to calibrate the current.
CALIBRATION_SHIFTS_MV = (-40.0, -80.0, -120.0, -160.0)
# (Targeted I cells are conductance-driven: -8 mV removed only 4% of their rate, -40 mV
# 24%, -120 mV 56%. As a current, -100 mV is about -90 pA.)
CALIBRATION_DIR_NAME = "_evaluation/perturbation-calibration"
CURRENT_FILE = "current.toml"
#: Perturbed teachers, shared by every run with the same targets and current.
TEACHER_CACHE_DIR_NAME = "_evaluation/perturbation-teachers"
PERTURBATION_FILE = "perturbation.npz"
#: Bump when the cached contents change; older caches are recomputed.
PERTURBATION_VERSION = 2
#: Scored populations: the intervention's non-targeted neighbours, and its targets.
PERTURBATION_GROUPS = (("unobserved", "E"), ("unobserved", "I"), ("targeted", "I"))
DELTA_METRICS = ("activity_r2", "fluctuation_r2")
CELL_TYPE_INDICES = {"E": 0, "I": 1}
CELL_TYPE_NAMES = {0: "excitatory", 1: "inhibitory"}


def teacher_model(teacher_dir, device):
    """The teacher network exactly as ``connectome_snns.analysis.test_inputs`` builds it."""
    params = toml.load(teacher_dir / "parameters.toml")
    rec_cfg = RecurrentLayerConfig(**params["recurrent"])
    ff_cfg = FeedforwardLayerConfig(**params["feedforward"])
    ns = np.load(teacher_dir / "results" / "network_structure.npz")
    rec_projections, ff_projections = make_frozen_projections(
        rec_weights=ns["recurrent_weights"],
        ff_weights=ns["feedforward_weights"],
        cell_type_indices=ns["cell_type_indices"],
        ff_cell_type_indices=ns["feedforward_cell_type_indices"],
        cell_type_names=list(rec_cfg.cell_types.names),
        ff_cell_type_names=list(ff_cfg.cell_types.names),
    )
    model = ConductanceLIFNetwork(
        dt=params["simulation"]["dt"],
        rec_projections=rec_projections,
        ff_projections=ff_projections,
        cell_type_indices=ns["cell_type_indices"],
        cell_type_indices_FF=ns["feedforward_cell_type_indices"],
        cell_params=rec_cfg.get_cell_params(),
        cell_params_FF=ff_cfg.get_cell_params(),
        synapse_params=rec_cfg.get_synapse_params(),
        synapse_params_FF=ff_cfg.get_synapse_params(),
        surrgrad_scale=1.0,
        batch_size=1,
        track_variables=False,
        track_batch_idx=0,
    )
    model.to(device)
    model.eval()
    return model, int(params["simulation"]["chunk_size"])


def simulate_teacher(run_dir, device, target_ids=None, shift_mv=0.0):
    """Teacher spikes (time, neurons) on the held-out trial's frozen mitral input."""
    inputs = zarr.open_group(str(held_out_trial(run_dir, device)), mode="r")
    mitral = np.asarray(inputs["input_spikes"][0])
    model, chunk_size = teacher_model(_resolve_teacher_dir(Path(run_dir)), device)
    if target_ids is not None and shift_mv != 0.0:
        model.E_L[torch.as_tensor(target_ids, device=device)] += shift_mv
    chunks = []
    with torch.inference_mode():
        for start in range(0, mitral.shape[0], chunk_size):
            chunk = torch.from_numpy(mitral[None, start : start + chunk_size])
            chunks.append(model(chunk.to(device).float())[0].bool().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def choose_targets(structure, candidate_ids):
    """A random ``TARGET_FRACTION`` of the inhibitory cells among ``candidate_ids``."""
    inhibitory = candidate_ids[structure["cell_type_indices"][candidate_ids] == 1]
    rng = np.random.default_rng(TARGET_SEED)
    n = round(TARGET_FRACTION * inhibitory.size)
    return np.sort(rng.choice(inhibitory, size=n, replace=False))


def calibration_dir(run_dir):
    """Where the teacher-level current calibration lives (shared by every run)."""
    return _resolve_teacher_dir(Path(run_dir)).parent / CALIBRATION_DIR_NAME


def held_out_teacher(run_dir, device="cpu"):
    """The unperturbed held-out teacher spikes (time, neurons), as stored."""
    inputs = zarr.open_group(str(held_out_trial(run_dir, device)), mode="r")
    return np.asarray(inputs["output_spikes"][0]).astype(bool), float(
        inputs.attrs["dt"]
    )


def rate_reduction(teacher_off, teacher_on, target_ids, burn_in):
    """Fractional loss of the targeted cells' mean rate after the burn-in."""
    base = teacher_off[burn_in:, target_ids].mean()
    return float(1.0 - teacher_on[burn_in:, target_ids].mean() / base)


def calibration_point(run_dir, device, shift_mv, burn_in_ms):
    """Teacher at one current, targets = 25% of all its I cells; saves the reduction.

    The current is a property of the teacher, calibrated on a teacher-level target set
    and then applied unchanged to each run's own targets.
    """
    structure = {
        "cell_type_indices": np.load(
            _resolve_teacher_dir(Path(run_dir)) / "results" / "network_structure.npz"
        )["cell_type_indices"]
    }
    all_ids = np.arange(structure["cell_type_indices"].size)
    targets = choose_targets(structure, all_ids)
    teacher_off, dt = held_out_teacher(run_dir, device)
    teacher_on = simulate_teacher(run_dir, device, targets, shift_mv)
    reduction = rate_reduction(teacher_off, teacher_on, targets, int(burn_in_ms / dt))
    out = calibration_dir(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"shift{shift_mv:+.1f}mV.toml").write_text(
        toml.dumps(
            {
                "shift_mv": shift_mv,
                "rate_reduction": reduction,
                "n_targets": int(targets.size),
            }
        )
    )
    return reduction


def pick_current(run_dir):
    """Interpolate the calibration points to ``TARGET_RATE_REDUCTION``; saves current.toml."""
    out = calibration_dir(run_dir)
    points = sorted(
        (toml.load(f) for f in out.glob("shift*mV.toml")), key=lambda p: -p["shift_mv"]
    )
    shifts = np.array([p["shift_mv"] for p in points])  # weakest first
    reductions = np.array([p["rate_reduction"] for p in points])
    if not (reductions.min() <= TARGET_RATE_REDUCTION <= reductions.max()):
        raise SystemExit(
            f"calibration does not bracket {TARGET_RATE_REDUCTION:.0%}: "
            + ", ".join(f"{s:+g} mV -> {r:.0%}" for s, r in zip(shifts, reductions))
        )
    order = np.argsort(reductions)
    shift = float(np.interp(TARGET_RATE_REDUCTION, reductions[order], shifts[order]))
    (out / "current.toml").write_text(
        toml.dumps(
            {
                "shift_mv": shift,
                "target_rate_reduction": TARGET_RATE_REDUCTION,
                "calibration_shifts_mv": shifts.tolist(),
                "calibration_rate_reductions": reductions.tolist(),
            }
        )
    )
    return shift


def calibrated_shift(run_dir):
    return float(toml.load(calibration_dir(run_dir) / CURRENT_FILE)["shift_mv"])


# =====================================================================
# The perturbed teacher (shared by every run with the same intervention)
# =====================================================================


def pack_spikes(path, spikes, **extra):
    """Store a spike array bit-packed along time."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, spikes=np.packbits(spikes, axis=-2), n_steps=spikes.shape[-2], **extra
    )


def unpack_spikes(path):
    with np.load(path) as data:
        n_steps = int(data["n_steps"])
        spikes = np.unpackbits(data["spikes"], axis=-2, count=n_steps).astype(bool)
        extra = {
            key: data[key] for key in data.files if key not in ("spikes", "n_steps")
        }
    return spikes, extra


def teacher_cache_path(run_dir, targets, shift_mv):
    """Where the perturbed teacher for this intervention lives, keyed by its targets.

    Runs that perturb the same cells with the same current (same teacher, seed and
    observed fraction) share one simulation.
    """
    key = hashlib.sha1(
        np.ascontiguousarray(targets, dtype=np.int64).tobytes()
        + f"{shift_mv:+.4f}".encode()
    ).hexdigest()[:16]
    directory = _resolve_teacher_dir(Path(run_dir)).parent / TEACHER_CACHE_DIR_NAME
    return directory / f"{key}.npz"


def perturbed_teacher(run_dir, device, targets, shift_mv):
    """The perturbed teacher's spikes (time, neurons), simulated once and cached."""
    cache = teacher_cache_path(run_dir, targets, shift_mv)
    if cache.exists():
        spikes, _ = unpack_spikes(cache)
        return spikes
    spikes = simulate_teacher(run_dir, device, targets, shift_mv)
    pack_spikes(cache, spikes, targets=targets, shift_mv=shift_mv)
    return spikes


def delta_scores(teacher, student, ids):
    """R² of the intervention's effect on neurons ``ids``.

    ``teacher`` and ``student`` map "off"/"on" to ``(smoothed, rates)``: draw-averaged
    smoothed traces (time, neurons) and per-neuron rates, on the scoring window.
    """
    teacher_delta = teacher["on"][0][:, ids] - teacher["off"][0][:, ids]
    student_delta = student["on"][0][:, ids] - student["off"][0][:, ids]
    teacher_rate_delta = teacher["on"][1][ids] - teacher["off"][1][ids]
    student_rate_delta = student["on"][1][ids] - student["off"][1][ids]
    return {
        "fluctuation_r2": r_squared(teacher_delta.ravel(), student_delta.ravel()),
        "activity_r2": r_squared(teacher_rate_delta, student_rate_delta),
        "teacher_mean_delta_rate_hz": float(teacher_rate_delta.mean()),
        "student_mean_delta_rate_hz": float(student_rate_delta.mean()),
    }


# =====================================================================
# Evaluation, cached per run
# =====================================================================


def cached_perturbation(run_dir):
    """The run's cached scores, or None if absent or written by an older version."""
    cache = Path(run_dir) / PERTURBATION_FILE
    if not cache.exists():
        return None
    with np.load(cache, allow_pickle=False) as data:
        cached = {key: data[key] for key in data.files}
    if int(cached.get("version", 0)) != PERTURBATION_VERSION:
        return None
    return cached


def unavailable_reason(run_dir):
    """Why this run cannot be scored on the perturbation, or None if it can.

    Lets a figure's analysis skip the perturbation (fully observed runs, or a teacher
    whose current has never been calibrated) and still write its held-out CSVs.
    """
    run_dir = Path(run_dir)
    if cached_perturbation(run_dir) is not None:
        return None
    if neuron_sets(load_structure(run_dir))["unobserved"].size == 0:
        return "no unobserved neurons to perturb"
    if not (calibration_dir(run_dir) / CURRENT_FILE).exists():
        return f"no calibrated current at {calibration_dir(run_dir) / CURRENT_FILE}"
    return None


def group_ids(structure, sets, targets):
    """Teacher ids of each scored population, keyed as in ``PERTURBATION_GROUPS``."""
    ct = structure["cell_type_indices"]
    targeted = np.zeros(ct.size, dtype=bool)
    targeted[targets] = True
    unobserved = sets["unobserved"]
    ids = {}
    for group, cell_type in PERTURBATION_GROUPS:
        if group == "targeted":
            ids[(group, cell_type)] = targets
            continue
        members = sets[group]
        ids[(group, cell_type)] = members[
            (ct[members] == CELL_TYPE_INDICES[cell_type]) & ~targeted[members]
        ]
    assert np.array_equal(np.sort(targets), targets)
    assert set(targets) <= set(unobserved), "targets must be unobserved neurons"
    return ids


def evaluate_perturbation(run_dir, device="cuda", force=False):
    """Score one trained run on the perturbation; cached in ``perturbation.npz``.

    Four simulations: the trained student and the perfectly specified one (the ceiling),
    each with the intervention off and on, all with the same spike-flip draws as
    ``common.evaluation.evaluate_run``. Scores are on the difference between them.
    """
    run_dir = Path(run_dir)
    if not force:
        cached = cached_perturbation(run_dir)
        if cached is not None:
            return cached

    params = run_parameters(run_dir)
    structure = load_structure(run_dir)
    sets = neuron_sets(structure)
    ct = structure["cell_type_indices"]
    evaluation_cfg = params["evaluation"]
    tau_ms = evaluation_cfg["fluctuation_tau_ms"]
    targets = choose_targets(structure, sets["unobserved"])
    shift_mv = calibrated_shift(run_dir)
    flip_ids = np.random.default_rng(PERTURBATION_SEED).choice(
        sets["unobserved"], size=N_PERTURBATIONS, replace=False
    )
    perturbed = {
        "teacher_spikes": perturbed_teacher(run_dir, device, targets, shift_mv)
    }
    perturbed["rest_shift"] = (targets, shift_mv)

    # Simulate one model at a time and keep only its smoothed traces and rates: the
    # raw ensembles are (draws, time, neurons) and do not all fit comfortably at once.
    models, dt, n_free_params, window = {}, None, None, None
    for name, kwargs in (
        ("student_off", {}),
        ("student_on", perturbed),
        ("ceiling_off", {"perfect": True}),
        ("ceiling_on", {"perfect": True, **perturbed}),
    ):
        result = run_student(run_dir, device, flip_ids=flip_ids, **kwargs)
        if dt is None:
            dt = result["dt"]
            n_free_params = result["n_free_params"]
            window = slice(
                int(evaluation_cfg["burn_in_ms"] / dt), result["student_all"].shape[1]
            )
        trials = result["student_all"][:, window]
        duration_s = trials.shape[1] * dt / 1000.0
        models[name] = (
            smooth_mean(trials, tau_ms, dt, device),
            trials.sum(axis=(0, 1)) / (trials.shape[0] * duration_s),
        )
        del result, trials
        print(f"  simulated {name}", flush=True)

    duration_s = (window.stop - window.start) * dt / 1000.0
    teacher = {}
    for name, spikes in (
        ("off", held_out_teacher(run_dir, device)[0]),
        ("on", perturbed["teacher_spikes"]),
    ):
        spikes = spikes[window]
        teacher[name] = (
            smooth_mean(spikes[None], tau_ms, dt, device),
            spikes.sum(axis=0) / duration_s,
        )
    student = {"off": models["student_off"], "on": models["student_on"]}
    ceiling = {"off": models["ceiling_off"], "on": models["ceiling_on"]}

    out = {
        "version": np.array(PERTURBATION_VERSION),
        "seed": np.array(params["simulation"]["seed"]),
        # The same identity fields ``evaluate_run`` stores, so a figure's labeller
        # works with either cache (Figure 3 reads n_observed, Figure 5 the clipping).
        "n_free_params": np.array(n_free_params),
        "n_observed": np.array(sets["observed"].size),
        "n_unobserved": np.array(sets["unobserved"].size),
        "n_unreconstructed": np.array(sets["unreconstructed"].size),
        "kappa": structure["kappa"],
        "noise_clipped_fraction": structure["noise_clipped_fraction"],
        "dt": np.array(dt),
        "burn_in_ms": np.array(evaluation_cfg["burn_in_ms"]),
        "scored_steps": np.array([window.start, window.stop]),
        "shift_mv": np.array(shift_mv),
        "targets": targets,
        "n_targets": np.array(targets.size),
        "targeted_rate_reduction": np.array(
            1.0
            - teacher["on"][1][targets].mean()
            / max(teacher["off"][1][targets].mean(), 1e-12)
        ),
    }
    for (group, cell_type), ids in group_ids(structure, sets, targets).items():
        key = f"{group}_{cell_type}"
        out[f"{key}_n_cells"] = np.array(ids.size)
        if ids.size == 0:
            for metric in DELTA_METRICS:
                out[f"{key}_delta_{metric}"] = np.array(np.nan)
                out[f"{key}_delta_{metric}_ceiling"] = np.array(np.nan)
            continue
        scores = delta_scores(teacher, student, ids)
        ceilings = delta_scores(teacher, ceiling, ids)
        for metric in DELTA_METRICS:
            out[f"{key}_delta_{metric}"] = np.array(scores[metric])
            out[f"{key}_delta_{metric}_ceiling"] = np.array(ceilings[metric])
        for field in ("teacher_mean_delta_rate_hz", "student_mean_delta_rate_hz"):
            out[f"{key}_{field}"] = np.array(scores[field])
            out[f"{key}_{field}_ceiling"] = np.array(ceilings[field])

    # Per-neuron delta rates of every unobserved neuron, for the scatter panel.
    unobserved = sets["unobserved"]
    targeted = np.zeros(ct.size, dtype=bool)
    targeted[targets] = True
    out["delta_ids"] = unobserved
    out["delta_cell_types"] = ct[unobserved]
    out["delta_targeted"] = targeted[unobserved]
    out["teacher_delta_rates"] = (
        teacher["on"][1][unobserved] - teacher["off"][1][unobserved]
    )
    out["student_delta_rates"] = (
        student["on"][1][unobserved] - student["off"][1][unobserved]
    )
    out["ceiling_delta_rates"] = (
        ceiling["on"][1][unobserved] - ceiling["off"][1][unobserved]
    )

    np.savez_compressed(run_dir / PERTURBATION_FILE, **out)
    return out


# =====================================================================
# Tables
# =====================================================================


def perturbation_summary_rows(perturbation, **labels):
    """Delta R² rows for a figure's ``*_summary.csv`` (``evaluation = perturbation``)."""
    rows = []
    for group, cell_type in PERTURBATION_GROUPS:
        key = f"{group}_{cell_type}"
        for metric in DELTA_METRICS:
            rows.append(
                {
                    **labels,
                    "seed": int(perturbation["seed"]),
                    "evaluation": "perturbation",
                    "group": group,
                    "cell_type": CELL_TYPE_NAMES[CELL_TYPE_INDICES[cell_type]],
                    "n_cells": int(perturbation[f"{key}_n_cells"]),
                    "metric": f"delta_{metric}",
                    "value": float(perturbation[f"{key}_delta_{metric}"]),
                    "ceiling_value": float(
                        perturbation[f"{key}_delta_{metric}_ceiling"]
                    ),
                }
            )
    return rows


def perturbation_rate_rows(perturbation, **labels):
    """One row per unobserved neuron: the intervention's effect on its rate."""
    rows = []
    for i, neuron in enumerate(perturbation["delta_ids"]):
        rows.append(
            {
                **labels,
                "seed": int(perturbation["seed"]),
                "neuron_id": int(neuron),
                "cell_type": CELL_TYPE_NAMES[int(perturbation["delta_cell_types"][i])],
                "targeted": int(perturbation["delta_targeted"][i]),
                "teacher_delta_rate_hz": float(perturbation["teacher_delta_rates"][i]),
                "student_delta_rate_hz": float(perturbation["student_delta_rates"][i]),
                "ceiling_delta_rate_hz": float(perturbation["ceiling_delta_rates"][i]),
            }
        )
    return rows


def collect_perturbation(runs, labeller, device):
    """Score ``runs`` on the perturbation; returns (summary, rates) DataFrames.

    Runs that cannot be scored (fully observed, or no calibrated current) are skipped
    with a note, so a figure still gets its held-out CSVs.
    """
    summary, rates = [], []
    for run in runs:
        reason = unavailable_reason(run)
        if reason is not None:
            print(f"  no perturbation for {Path(run).name}: {reason}")
            continue
        print(f"Perturbation {run}")
        perturbation = evaluate_perturbation(run, device)
        labels = labeller(run_parameters(run), perturbation)
        summary += perturbation_summary_rows(perturbation, **labels)
        rates += perturbation_rate_rows(perturbation, **labels)
    return summary, rates
