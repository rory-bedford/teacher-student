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

from common.evaluation import held_out_trial, smooth

TARGET_FRACTION = 0.25
TARGET_SEED = 0
TARGET_RATE_REDUCTION = 0.5
#: Rest-potential shifts (mV) simulated in parallel to calibrate the current.
CALIBRATION_SHIFTS_MV = (-40.0, -80.0, -120.0, -160.0)
# (Targeted I cells are conductance-driven: -8 mV removed only 4% of their rate, -40 mV
# 24%, -120 mV 56%. As a current, -100 mV is about -90 pA.)
CALIBRATION_DIR_NAME = "_evaluation/perturbation-calibration"


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
    return float(toml.load(calibration_dir(run_dir) / "current.toml")["shift_mv"])


def delta_scores(teacher_off, teacher_on, student_off, student_on, ids, dt, tau_ms):
    """R² of the intervention's effect on neurons ``ids``.

    ``teacher_*``: (time, neurons) bool; ``student_*``: (draws, time, neurons) bool; all
    already cut to the scoring window.
    """
    duration_s = teacher_off.shape[0] * dt / 1000.0

    def mean_smooth(trials):
        out = np.zeros((trials.shape[1], ids.size), dtype=np.float32)
        for trial in trials:
            out += smooth(trial[:, ids], tau_ms, dt) / trials.shape[0]
        return out

    teacher_delta = smooth(teacher_on[:, ids], tau_ms, dt) - smooth(
        teacher_off[:, ids], tau_ms, dt
    )
    student_delta = mean_smooth(student_on) - mean_smooth(student_off)
    teacher_rate_delta = (
        teacher_on[:, ids].sum(axis=0) - teacher_off[:, ids].sum(axis=0)
    ) / duration_s
    student_rate_delta = (
        student_on[:, :, ids].sum(axis=(0, 1)) - student_off[:, :, ids].sum(axis=(0, 1))
    ) / (student_on.shape[0] * duration_s)
    return {
        "fluctuation_r2": r_squared(teacher_delta.ravel(), student_delta.ravel()),
        "activity_r2": r_squared(teacher_rate_delta, student_rate_delta),
        "teacher_mean_delta_rate_hz": float(teacher_rate_delta.mean()),
        "student_mean_delta_rate_hz": float(student_rate_delta.mean()),
    }
