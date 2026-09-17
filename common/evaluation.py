"""Held-out evaluation shared by every figure's ``analysis.py``.

For each trained run: generate (or load) a held-out teacher trial on a new odour
trajectory, run the trained student on it with teacher forcing exactly as in
training, and score the observed and unobserved neurons.

Metrics, per group:
    fluctuation_r2   spike trains smoothed with a Gaussian (sigma 50 ms), R² over all
                     neurons x time (primary)
    activity_r2      R² of per-neuron firing rates (secondary)
    floor            the same metric after permuting the teacher<->student neuron
                     mapping within the group, averaged over permutations
    ceiling          the same metric for a perfectly specified student: teacher weights,
                     correct scaling factors, and the same teacher-forced sources

Why a ceiling. The teacher network is chaotic. A perfectly specified student
reproduces it spike for spike for several seconds (verified), but float32 rounding
eventually flips a single spike in the simulated population and the trajectories then
decorrelate completely within ~1 s — on the 13 s held-out window this happened after
9-10 s. Mean rates and stimulus-locked fluctuations survive, precise timing does not,
so the ceiling is below 1 and every value should be read against it.

Equal starting conditions. Left alone, the perfectly specified student would track the
teacher exactly until the first float32 rounding flip, whose timing is arbitrary. So
every simulated model here (trained student and ceiling) starts with one
extra spike injected at t = 0 into a free-running (unobserved) neuron, which decorrelates
precise timing within about a second; the burn-in discards that transient. Each value
is the mean over ``N_PERTURBATIONS`` draws of the flipped neuron, the same draws for
every model of a run. Floors, rates, per-neuron R² and rasters use the first draw.

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
EVALUATION_VERSION = 3
#: Spike-flip draws per run (see module docstring), and the seed choosing the neurons.
N_PERTURBATIONS = 5
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


def per_neuron_r2(teacher_smooth, student_smooth):
    """Fluctuation R² of each neuron separately (NaN where the teacher is silent)."""
    ss_res = ((student_smooth - teacher_smooth) ** 2).sum(axis=0)
    ss_tot = ((teacher_smooth - teacher_smooth.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)


def group_metrics(teacher, student, dt, tau_ms, n_permutations, rng):
    """Activity and fluctuation R² of one group, with shuffled-identity floors."""
    duration_s = teacher.shape[0] * dt / 1000.0
    teacher_rates = teacher.sum(axis=0) / duration_s
    student_rates = student.sum(axis=0) / duration_s
    teacher_smooth = smooth(teacher, tau_ms, dt)
    student_smooth = smooth(student, tau_ms, dt)

    values = {
        "activity_r2": r_squared(teacher_rates, student_rates),
        "fluctuation_r2": r_squared(teacher_smooth.ravel(), student_smooth.ravel()),
    }
    floors = {"activity_r2": [], "fluctuation_r2": []}
    for _ in range(n_permutations):
        order = rng.permutation(teacher.shape[1])
        floors["activity_r2"].append(r_squared(teacher_rates, student_rates[order]))
        floors["fluctuation_r2"].append(
            r_squared(teacher_smooth.ravel(), student_smooth[:, order].ravel())
        )
    return {
        "values": values,
        "floors": {k: float(np.mean(v)) if v else np.nan for k, v in floors.items()},
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


def run_student(run_dir, device, perfect=False, flip_ids=None):
    """Teacher-forced student spikes on the held-out trial, one batch of trials.

    Args:
        perfect: Run the perfectly specified student (see :func:`perfect_structure`)
            instead of the trained one; groups keep the trained run's neuron ids.
        flip_ids: Optional teacher ids, one per trial, of an unobserved neuron made to
            spike at t = 0 in that trial.

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

    collate = StudentCollate(model_sets["observed"], model_sets["unreconstructed"])
    dataloader = DataLoader(
        dataset, batch_size=None, sampler=CyclicSampler(dataset), num_workers=0
    )
    observed_chunks, unobserved_chunks = [], []
    batches = iter(dataloader)
    with torch.inference_mode():
        for _ in range(dataset.num_chunks):
            inputs = collate(next(batches)).input_spikes.to(device)
            inputs = inputs.expand(n_trials, -1, -1).contiguous()
            out = model(inputs)
            observed_chunks.append(out["spikes"].bool().cpu().numpy())
            unobserved_chunks.append(out["hidden_spikes"].bool().cpu().numpy())

    n_steps = dataset.num_chunks * dataset.chunk_size
    teacher = np.array(dataset.target_spike_data[0, :n_steps, :]).astype(bool)

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
    rng = np.random.default_rng(0)

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

    def mean_values(result, suffix):
        for group in GROUPS:
            if sets[group].size == 0:
                for metric in METRICS:
                    out[f"{group}_{metric}{suffix}"] = np.array(np.nan)
                continue
            teacher = result["teacher"][group][burn_in:]
            values = [
                group_metrics(teacher, trial[burn_in:], dt, tau_ms, 0, None)["values"]
                for trial in result["student"][group]
            ]
            for metric in METRICS:
                out[f"{group}_{metric}{suffix}"] = np.array(
                    np.mean([v[metric] for v in values])
                )

    mean_values(result, "")
    for group in GROUPS:
        ids = sets[group]
        out[f"{group}_ids"] = ids
        out[f"{group}_cell_types"] = ct[ids]
        if ids.size == 0:
            for metric in METRICS:
                out[f"{group}_{metric}_floor"] = np.array(np.nan)
            continue
        teacher = result["teacher"][group][burn_in:]
        first = result["student"][group][0][burn_in:]
        scores = group_metrics(
            teacher, first, dt, tau_ms, evaluation_cfg["n_floor_permutations"], rng
        )
        for metric in METRICS:
            out[f"{group}_{metric}_floor"] = np.array(scores["floors"][metric])
        out[f"{group}_teacher_rates"] = scores["teacher_rates"]
        out[f"{group}_student_rates"] = scores["student_rates"]
        out[f"{group}_per_neuron_fluctuation_r2"] = scores["per_neuron_fluctuation_r2"]
        out[f"{group}_teacher_raster"] = np.packbits(teacher[:raster_steps], axis=0)
        out[f"{group}_student_raster"] = np.packbits(first[:raster_steps], axis=0)
    if progress:
        progress("student", out)

    mean_values(
        run_student(run_dir, device, perfect=True, flip_ids=flip_ids), "_ceiling"
    )
    if progress:
        progress("ceiling", out)

    np.savez_compressed(cache, **out)
    return out


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
    """One row per (group, metric) — the ``*_summary.csv`` layout."""
    rows = []
    for group in GROUPS:
        for metric in METRICS:
            rows.append(
                {
                    **labels,
                    "seed": int(evaluation["seed"]),
                    "group": group,
                    "metric": metric,
                    "value": float(evaluation[f"{group}_{metric}"]),
                    "floor_value": float(evaluation[f"{group}_{metric}_floor"]),
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
