"""Held-out evaluation shared by every figure's ``analysis.py``.

For each trained run: generate (or load) a held-out teacher trial on a new odour
trajectory, run the trained student on it with teacher forcing exactly as in
training, and score the observed and unobserved neurons.

Metrics, per group:
    fluctuation_r2   spike trains smoothed with a Gaussian (sigma 50 ms), R² over all
                     neurons x time (primary)
    activity_r2      R² of per-neuron firing rates (secondary)
    ceiling          the same metric for a perfectly specified student: teacher weights,
                     correct scaling factors, and the same teacher-forced sources

Why a ceiling. The teacher network is chaotic. A perfectly specified student reproduces
it spike for spike until a float32 rounding difference flips a single spike in the
free-running (unobserved) population; precise timing then decorrelates within ~1 s, at
an arbitrary moment. Mean rates and stimulus-locked fluctuations survive.

Protocol (equal terms for student and ceiling). Every simulated model is run
``N_PERTURBATIONS`` times, each with one extra spike injected at t = 0 into a different
unobserved neuron (the same draws for the trained student and the perfect one, with
identical teacher forcing). The burn-in discards the transient. Each model's smoothed
traces are averaged over draws and the average is scored against the teacher (not the
average of per-draw scores); activity R² uses rates averaged over draws. Rates and
per-neuron R² use the draw average; rasters show the first draw. Runs with no
unobserved neurons have nothing free-running and are simulated once.

The result is cached as ``evaluation.npz`` in the run directory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import toml
import torch
from connectome_snns.analysis import ensure_test_inputs, r_squared
from connectome_snns.dataloaders.supervised import CyclicSampler, ExactFFDataset
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader

from common.model import (
    StudentCollate,
    build_student,
    count_free_parameters,
    neuron_sets,
)
from common.structure import load_structure, load_teacher
from common.training import FINAL_STATE

EVALUATION_FILE = "evaluation.npz"
#: Bump when the cached contents change; older caches are recomputed.
EVALUATION_VERSION = 4
#: Spike-flip draws per run (see module docstring), and the seed choosing the neurons.
N_PERTURBATIONS = 20
PERTURBATION_SEED = 0
GROUPS = ("observed", "unobserved")
METRICS = ("activity_r2", "fluctuation_r2")
#: Seconds of post-burn-in spikes kept for rasters.
RASTER_SECONDS = 5.0


# =====================================================================
# Runs
# =====================================================================


def completed_runs(grid_dir):
    """Run directories under a grid-search parent that finished training."""
    grid_dir = Path(grid_dir)
    if (grid_dir / FINAL_STATE).exists():
        return [grid_dir]
    return sorted(p.parent for p in grid_dir.glob(f"*/{FINAL_STATE}"))


def run_parameters(run_dir):
    return toml.load(Path(run_dir) / "parameters.toml")


# =====================================================================
# Metrics
# =====================================================================


def smooth(spikes, tau_ms, dt):
    """Gaussian smoothing along time, as in ``connectome_snns.analysis.fluctuation_r_squared``."""
    return gaussian_filter1d(spikes.astype(np.float32), sigma=tau_ms / dt, axis=0)


def smooth_mean(trials, tau_ms, dt, device):
    """Mean over trials of :func:`smooth`, on ``device``; (trials, time, neurons) -> (time, neurons).

    Same kernel and boundary handling as ``scipy.ndimage.gaussian_filter1d`` (truncate 4,
    mode "reflect", which repeats the edge sample), so results match :func:`smooth` to
    float32 precision.
    """
    sigma = tau_ms / dt
    radius = int(4.0 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = (kernel / kernel.sum()).float().to(device)[None, None, :]
    total = None
    for trial in trials:
        signal = torch.as_tensor(np.ascontiguousarray(trial), device=device).float()
        signal = signal.T[:, None, :]  # (neurons, 1, time)
        padded = torch.cat(
            [
                signal[..., :radius].flip(-1),
                signal,
                signal[..., -radius:].flip(-1),
            ],
            dim=-1,
        )
        smoothed = torch.nn.functional.conv1d(padded, kernel)[:, 0, :].T
        total = smoothed if total is None else total + smoothed
    return (total / len(trials)).cpu().numpy()


def per_neuron_r2(teacher_smooth, student_smooth):
    """Fluctuation R² of each neuron separately (NaN where the teacher is silent)."""
    ss_res = ((student_smooth - teacher_smooth) ** 2).sum(axis=0)
    ss_tot = ((teacher_smooth - teacher_smooth.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)


def group_metrics(teacher, trials, dt, tau_ms):
    """Activity and fluctuation R² of one group, scored on the draw-averaged student.

    Args:
        teacher: (time, neurons) bool spikes.
        trials: (draws, time, neurons) bool student spikes.
    """
    duration_s = teacher.shape[0] * dt / 1000.0
    teacher_rates = teacher.sum(axis=0) / duration_s
    student_rates = trials.sum(axis=(0, 1)) / (trials.shape[0] * duration_s)
    teacher_smooth = smooth(teacher, tau_ms, dt)
    student_smooth = np.zeros_like(teacher_smooth)
    for trial in trials:
        student_smooth += smooth(trial, tau_ms, dt) / trials.shape[0]

    values = {
        "activity_r2": r_squared(teacher_rates, student_rates),
        "fluctuation_r2": r_squared(teacher_smooth.ravel(), student_smooth.ravel()),
    }
    return {
        "values": values,
        "teacher_rates": teacher_rates,
        "student_rates": student_rates,
        "per_neuron_fluctuation_r2": per_neuron_r2(teacher_smooth, student_smooth),
    }


# =====================================================================
# Inference
# =====================================================================


def held_out_trial(run_dir, device):
    """Path to the run's held-out teacher trial (new odour trajectory, seed + 10)."""
    return ensure_test_inputs(run_dir, device=device)


def perfect_structure(structure, teacher):
    """The perfectly specified student for ``structure``.

    Teacher weights, no perturbation, and teacher forcing on exactly the sources the
    run injects (its observed neurons and any unreconstructed units); every other
    neuron, including any the run removed, is simulated.
    """
    sets = neuron_sets(structure)
    forced = np.zeros_like(structure["modelled"])
    forced[sets["observed"]] = True
    forced[sets["unreconstructed"]] = True
    return {
        **structure,
        "ff_weights": teacher["ff_weights"],
        "rec_weights": teacher["rec_weights"],
        "perturbation": np.ones_like(structure["perturbation"]),
        "modelled": np.ones_like(structure["modelled"]),
        "observed": forced,
        "known_ff": np.ones_like(structure["known_ff"]),
        "inject_unreconstructed": np.array(False),
        "recurrent_model": "connectome",
    }


def run_student(
    run_dir, device, perfect=False, flip_ids=None, teacher_spikes=None, rest_shift=None
):
    """Teacher-forced student spikes on the held-out trial, one batch of trials.

    Args:
        perfect: Run the perfectly specified student (see :func:`perfect_structure`)
            instead of the trained one; groups keep the trained run's neuron ids.
        flip_ids: Optional teacher ids, one per trial, of an unobserved neuron made to
            spike at t = 0 in that trial.
        teacher_spikes: Optional (time, neurons) bool teacher activity replacing the
            held-out trial's for teacher forcing and scoring (e.g. a perturbed teacher);
            the mitral input is unchanged.
        rest_shift: Optional ``(teacher ids, delta mV)``: shift those unobserved neurons'
            leak reversal potential, i.e. inject a constant current g_L * delta.

    Returns:
        dict with teacher spikes (time x neurons) and student spikes
        (trials x time x neurons) for each group, the neuron sets, the structure and
        the parameters.
    """
    run_dir = Path(run_dir)
    params = run_parameters(run_dir)
    structure = load_structure(run_dir)
    sets = neuron_sets(structure)
    model_structure = (
        perfect_structure(structure, load_teacher(run_dir / "inputs"))
        if perfect
        else structure
    )
    model_sets = neuron_sets(model_structure)

    dataset = ExactFFDataset(
        spike_data_path=held_out_trial(run_dir, device),
        chunk_size=params["simulation"]["chunk_size"],
        device=device,
    )
    n_trials = 1 if flip_ids is None else len(flip_ids)
    model, parameters = build_student(
        model_structure,
        params,
        batch_size=n_trials,
        dt=dataset.dt,
        surrgrad_scale=params["optimiser"]["surrgrad_scale"],
        low_rank=params["optimiser"].get("low_rank", 1),
    )
    if not perfect:
        parameters.load_state_dict(
            torch.load(run_dir / FINAL_STATE, map_location="cpu")
        )
    model.to(device)
    model.eval()
    if flip_ids is not None:
        positions = np.searchsorted(model_sets["unobserved"], flip_ids)
        assert np.array_equal(model_sets["unobserved"][positions], flip_ids)
        layer1 = model.layer1
        rows = torch.arange(n_trials, device=layer1.v.device)
        columns = torch.as_tensor(positions, device=layer1.v.device)
        layer1.v[rows, columns] = layer1.theta[columns] + 1.0
    if rest_shift is not None:
        ids, delta_mv = rest_shift
        positions = np.searchsorted(model_sets["unobserved"], ids)
        assert np.array_equal(model_sets["unobserved"][positions], ids)
        model.layer1.E_L[torch.as_tensor(positions, device=device)] += delta_mv

    collate = StudentCollate(model_sets["observed"], model_sets["unreconstructed"])
    dataloader = DataLoader(
        dataset, batch_size=None, sampler=CyclicSampler(dataset), num_workers=0
    )
    observed_chunks, unobserved_chunks = [], []
    batches = iter(dataloader)
    chunk_size = dataset.chunk_size
    with torch.inference_mode():
        for chunk in range(dataset.num_chunks):
            batch = next(batches)
            if teacher_spikes is not None:
                start = chunk * chunk_size
                forced = torch.from_numpy(
                    teacher_spikes[None, start : start + chunk_size]
                )
                batch = batch._replace(
                    target_spikes=forced.to(
                        device=batch.target_spikes.device,
                        dtype=batch.target_spikes.dtype,
                    )
                )
            inputs = collate(batch).input_spikes.to(device)
            inputs = inputs.expand(n_trials, -1, -1).contiguous()
            out = model(inputs)
            observed_chunks.append(out["spikes"].bool().cpu().numpy())
            unobserved_chunks.append(out["hidden_spikes"].bool().cpu().numpy())

    n_steps = dataset.num_chunks * dataset.chunk_size
    if teacher_spikes is None:
        teacher = np.array(dataset.target_spike_data[0, :n_steps, :]).astype(bool)
    else:
        teacher = teacher_spikes[:n_steps].astype(bool)

    # Student spikes by teacher id, then select the trained run's groups.
    student_all = np.zeros((n_trials, *teacher.shape), dtype=bool)
    student_all[:, :, model_sets["observed"]] = np.concatenate(observed_chunks, axis=1)
    student_all[:, :, model_sets["unobserved"]] = np.concatenate(
        unobserved_chunks, axis=1
    )
    return {
        "dt": dataset.dt,
        "n_free_params": count_free_parameters(parameters),
        "sets": sets,
        "structure": structure,
        "params": params,
        "teacher": {
            "observed": teacher[:, sets["observed"]],
            "unobserved": teacher[:, sets["unobserved"]],
        },
        "student": {
            "observed": student_all[..., sets["observed"]],
            "unobserved": student_all[..., sets["unobserved"]],
        },
        "teacher_all": teacher,
        "student_all": student_all,
    }


def evaluate_run(run_dir, device="cuda", force=False, progress=None):
    """Score one trained run on held-out stimuli; cached in ``evaluation.npz``.

    Args:
        progress: Optional callable ``(stage, out)`` called after each simulation
            (``"student"``, ``"ceiling"``) with the scores so far.
    """
    run_dir = Path(run_dir)
    cache = run_dir / EVALUATION_FILE
    if cache.exists() and not force:
        with np.load(cache, allow_pickle=False) as data:
            cached = {key: data[key] for key in data.files}
        if int(cached.get("version", 0)) == EVALUATION_VERSION:
            return cached
        print(f"  outdated {EVALUATION_FILE} in {run_dir}; re-evaluating")

    params = run_parameters(run_dir)
    sets = neuron_sets(load_structure(run_dir))
    flip_ids = None
    if sets["unobserved"].size > 0:
        flip_ids = np.random.default_rng(PERTURBATION_SEED).choice(
            sets["unobserved"], size=N_PERTURBATIONS, replace=False
        )
    evaluation_cfg = params["evaluation"]
    tau_ms = evaluation_cfg["fluctuation_tau_ms"]

    result = run_student(run_dir, device, flip_ids=flip_ids)
    dt = result["dt"]
    burn_in = int(evaluation_cfg["burn_in_ms"] / dt)
    ct = result["structure"]["cell_type_indices"]
    out = {
        "version": np.array(EVALUATION_VERSION),
        "n_free_params": np.array(result["n_free_params"]),
        "n_observed": np.array(sets["observed"].size),
        "n_unobserved": np.array(sets["unobserved"].size),
        "n_unreconstructed": np.array(sets["unreconstructed"].size),
        "n_perturbations": np.array(0 if flip_ids is None else flip_ids.size),
        "dt": np.array(dt),
        "seed": np.array(params["simulation"]["seed"]),
        "kappa": result["structure"]["kappa"],
        "noise_clipped_fraction": result["structure"]["noise_clipped_fraction"],
        "raster_steps": np.array(int(RASTER_SECONDS * 1000.0 / dt)),
        "burn_in_ms": np.array(evaluation_cfg["burn_in_ms"]),
    }
    raster_steps = int(out["raster_steps"])

    for group in GROUPS:
        ids = sets[group]
        out[f"{group}_ids"] = ids
        out[f"{group}_cell_types"] = ct[ids]
        if ids.size == 0:
            for metric in METRICS:
                out[f"{group}_{metric}"] = np.array(np.nan)
            continue
        teacher = result["teacher"][group][burn_in:]
        trials = result["student"][group][:, burn_in:]
        scores = group_metrics(teacher, trials, dt, tau_ms)
        for metric in METRICS:
            out[f"{group}_{metric}"] = np.array(scores["values"][metric])
        out[f"{group}_teacher_rates"] = scores["teacher_rates"]
        out[f"{group}_student_rates"] = scores["student_rates"]
        out[f"{group}_per_neuron_fluctuation_r2"] = scores["per_neuron_fluctuation_r2"]
        out[f"{group}_teacher_raster"] = np.packbits(teacher[:raster_steps], axis=0)
        out[f"{group}_student_raster"] = np.packbits(trials[0, :raster_steps], axis=0)
    del result
    if progress:
        progress("student", out)

    perfect = run_student(run_dir, device, perfect=True, flip_ids=flip_ids)
    for group in GROUPS:
        for metric in METRICS:
            out[f"{group}_{metric}_ceiling"] = np.array(np.nan)
        if sets[group].size == 0:
            continue
        scores = group_metrics(
            perfect["teacher"][group][burn_in:],
            perfect["student"][group][:, burn_in:],
            dt,
            tau_ms,
        )
        for metric in METRICS:
            out[f"{group}_{metric}_ceiling"] = np.array(scores["values"][metric])
    if progress:
        progress("ceiling", out)

    np.savez_compressed(cache, **out)
    return out


def scaling_factor_rows(run_dir, **labels):
    """Final learnt value and true target of each tied scaling factor, one row each.

    Read from the run's ``training_metrics.csv`` rather than the model state: the run
    already logs both the value and the target it should recover.
    """
    metrics = pd.read_csv(Path(run_dir) / "training_metrics.csv")
    last = metrics.iloc[-1]
    rows = []
    for column in metrics.columns:
        if not column.endswith("_value") or not column.startswith("scaling_factors/"):
            continue
        name = column[len("scaling_factors/") : -len("_value")]
        target = f"scaling_factors/{name}_target"
        if target not in metrics:
            continue
        rows.append(
            {
                **labels,
                "scaling_factor": name,
                "value": float(last[column]),
                "target": float(last[target]),
            }
        )
    return rows


def raster(evaluation, group, source, n_steps=None):
    """Unpack a stored raster: (time, neurons) bool, first ``RASTER_SECONDS`` post burn-in."""
    packed = evaluation[f"{group}_{source}_raster"]
    steps = int(evaluation["raster_steps"]) if n_steps is None else n_steps
    return np.unpackbits(packed, axis=0, count=int(evaluation["raster_steps"]))[
        :steps
    ].astype(bool)


# =====================================================================
# Tables
# =====================================================================


def collect(runs, labeller, device):
    """Evaluate ``runs`` and return (summary, rates) DataFrames.

    ``labeller(params, evaluation)`` returns the condition columns for a run.
    """
    summary, rates = [], []
    for run in runs:
        print(f"Evaluating {run}")
        evaluation = evaluate_run(run, device)
        labels = labeller(run_parameters(run), evaluation)
        summary += summary_rows(evaluation, **labels)
        rates += rate_rows(evaluation, **labels)
    return pd.DataFrame(summary), pd.DataFrame(rates)


def summary_rows(evaluation, **labels):
    """One row per (group, metric) — the ``*_summary.csv`` layout.

    ``evaluation = "held_out"`` distinguishes these rows from the perturbation rows
    that ``common.perturbation`` appends to the same CSV.
    """
    rows = []
    for group in GROUPS:
        for metric in METRICS:
            rows.append(
                {
                    **labels,
                    "seed": int(evaluation["seed"]),
                    "evaluation": "held_out",
                    "group": group,
                    "cell_type": "all",
                    "n_cells": int(evaluation[f"n_{group}"]),
                    "metric": metric,
                    "value": float(evaluation[f"{group}_{metric}"]),
                    "ceiling_value": float(evaluation[f"{group}_{metric}_ceiling"]),
                }
            )
    return rows


def rate_rows(evaluation, **labels):
    """One row per modelled neuron — the ``*_rates.csv`` layout."""
    rows = []
    names = np.array(["excitatory", "inhibitory"])
    for group in GROUPS:
        ids = evaluation[f"{group}_ids"]
        if ids.size == 0:
            continue
        for i, neuron in enumerate(ids):
            rows.append(
                {
                    **labels,
                    "neuron_id": int(neuron),
                    "cell_type": names[evaluation[f"{group}_cell_types"][i]],
                    "observed": int(group == "observed"),
                    "seed": int(evaluation["seed"]),
                    "teacher_rate_hz": float(evaluation[f"{group}_teacher_rates"][i]),
                    "student_rate_hz": float(evaluation[f"{group}_student_rates"][i]),
                    "fluctuation_r2": float(
                        evaluation[f"{group}_per_neuron_fluctuation_r2"][i]
                    ),
                }
            )
    return rows


def spike_rows(evaluation, neurons, **labels):
    """Spike times (s, from the end of burn-in) for chosen ``(group, neuron_id)`` pairs."""
    dt = float(evaluation["dt"])
    rows = []
    for group, neuron in neurons:
        column = int(np.flatnonzero(evaluation[f"{group}_ids"] == neuron)[0])
        for source in ("teacher", "student"):
            steps = np.flatnonzero(raster(evaluation, group, source)[:, column])
            for step in steps:
                rows.append(
                    {
                        **labels,
                        "neuron_id": int(neuron),
                        "observed": int(group == "observed"),
                        "seed": int(evaluation["seed"]),
                        "source": source,
                        "time_s": step * dt / 1000.0,
                    }
                )
    return rows


def pick_raster_neurons(
    evaluation, n_observed, n_unobserved, rng_seed=0, min_hz=2.0, max_hz=20.0
):
    """Random neurons with teacher rates in a readable range, per group."""
    rng = np.random.default_rng(rng_seed)
    chosen = []
    for group, n in (("observed", n_observed), ("unobserved", n_unobserved)):
        ids = evaluation[f"{group}_ids"]
        if ids.size == 0 or n == 0:
            continue
        rates = evaluation[f"{group}_teacher_rates"]
        eligible = ids[(rates >= min_hz) & (rates <= max_hz)]
        pool = eligible if eligible.size >= n else ids
        chosen += [
            (group, int(i)) for i in np.sort(rng.choice(pool, size=n, replace=False))
        ]
    return chosen
