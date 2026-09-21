"""Figure 0 — everything ``figures.py`` plots about the teacher network.

Reads the teacher's read-only output directory (its zarr, its network structure and its
parameter snapshot), re-simulates the short windows the dynamics panels need, and writes
small CSV/NPZ files next to this script. Nothing is written to the teacher's directory,
and ``figures.py`` needs neither the zarr nor a GPU.

    uv run python fig00-teacher-activity/analysis.py
    uv run python fig00-teacher-activity/analysis.py --steps dynamics --force

Steps, each skipped when its outputs exist unless ``--force`` is given:

    coding          fig00_coding_schematic.csv  odourant, baseline_hz, odour_hz
                    The construction of an odourant: one input group elevated by
                    modulation_rate, the rest depressed to hold the baseline mean.
    conditions      fig00_condition_rates.csv   neuron_id, cell_type, assembly, and the
                    per-neuron rate under odourant 1, the same odourant with a different
                    Poisson seed, and the homogeneous baseline. Three re-simulations.
    dynamics        fig00_raster.npz            spike (row, time) pairs for ten rows --
                                                feedforward and recurrent, as the library's
                                                activity dashboard samples them -- and
                                                each row's id, population and rate
                    fig00_traces.npz            one neuron's voltage, currents and
                                                conductances over TRACE_SECONDS
                    The raster comes from the teacher's own zarr; the traces need a
                    re-simulation of the stored input with ``track_variables=True``.
    assemblies      fig00_assemblies.npz        the OU mixing coefficient of each odourant
                                                and each assembly's excitatory population
                                                rate, over the raster's trial
    drive           fig00_drive.csv             feedforward vs recurrent excitatory
                                                synaptic drive over the whole zarr
    dimensionality  fig00_pca_spectrum.csv      component, eigenvalue, variance_fraction,
                                                cumulative_fraction
                    fig00_dimensionality.csv    participation_ratio, n_pcs_50/80/90/95pct_var,
                                                smoothing_sigma_ms, burn_in_ms, n_trials,
                                                seconds_per_trial, n_neurons

Dimensionality: spikes of all recurrent neurons are smoothed with the Gaussian used for
Fluctuation R² (sigma and burn-in from ``fig01-full-reconstruction/parameters.toml``
[evaluation]); the covariance is pooled over every training trial after its burn-in, with
one mean per neuron over all of them. PCA is the eigendecomposition of that covariance.
It is the expensive step (tens of minutes over 50 trials), so it is the one to leave out.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import toml
import torch
import zarr
from connectome_snns.configs import SimulationConfig
from connectome_snns.configs.conductance_based import (
    FeedforwardLayerConfig,
    RecurrentLayerConfig,
)
from connectome_snns.configs.odours import OdourInputConfig
from connectome_snns.dataloaders.odourants import (
    generate_baseline_firing_rates,
    generate_odour_firing_rates,
)
from connectome_snns.dataloaders.unsupervised import HomogeneousPoissonSpikeDataLoader
from connectome_snns.network_simulators.conductance_based.simulator import (
    ConductanceLIFNetwork,
)
from connectome_snns.network_simulators.projections import make_frozen_projections
from connectome_snns.utils.reproducibility import load_experiment_config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.evaluation import smooth

EVALUATION_PARAMETERS = HERE.parent / "fig01-full-reconstruction" / "parameters.toml"
VARIANCE_THRESHOLDS = (0.5, 0.8, 0.9, 0.95)

#: The trial every dynamics panel is drawn from.
RASTER_TRIAL = 0
#: Plot windows. The raster covers the teacher's saved plotting window; the traces are a
#: shorter stretch, where single spikes are still resolvable.
RASTER_SECONDS = 10.0
TRACE_SECONDS = 5.0
#: The assembly panels get the WHOLE trial: an odourant's response is a sustained offset,
#: and a 10 s window cut off the second leader's rise (its mean deviation while leading
#: went from 1.40 Hz over 10 s to 1.81 Hz over the full trial, and its share of the trial
#: from 0.32 to 0.55).
ASSEMBLY_SECONDS = 15.0
#: One condition simulation, of which the first second is discarded before rates are
#: taken, so the network is not still relaxing from its reset.
CONDITION_SECONDS = 10.0
BURN_IN_SECONDS = 1.0
#: Steps per forward call. Tracking allocates (steps x neurons x synapses) tensors, so
#: the window is simulated in slices rather than in one call.
TRACK_CHUNK_STEPS = 1000
#: Rows in the raster, split between feedforward and recurrent as the library's activity
#: dashboard splits them (see :func:`raster_sample`).
RASTER_NEURONS = 10
#: The traced neuron is CHOSEN, not pinned. The notebook pinned neuron 13, which turns
#: out to carry a mitral synapse of weight 2.32 -- the 96th percentile of the network --
#: so a single presynaptic spike injects ~14 nS and its currents are dominated by one
#: outlier synapse. Feedforward weights are log-normal with median 0.0014 and a maximum
#: of 65.5 (see the README), so an unlucky pick is unrepresentative rather than rare.
#: The rule keeps neurons whose strongest mitral synapse is no larger than the network
#: median and, among those, takes the one firing closest to its cell type's mean rate --
#: the same "typical cell" rule the library's activity dashboard uses to pick a neuron.
TRACE_CELL_TYPE = "excitatory"
CONDUCTANCE_NEURONS = 10
SAMPLE_SEED = 42
#: The assembly panels follow the library's assembly dashboard: excitatory neurons only,
#: over the raster's window and trial. The dashboard smooths with a 200 ms Gaussian; 50 ms
#: is used here, matching Fluctuation R²'s kernel, so the rates can be read against the
#: mixing trajectories side by side without the response being smeared past its cause.
#: Odourant k drives assembly k (the library generates one input pattern per assembly), so
#: column k means the same assembly in both arrays and the two panels share colours. Held
#: static, odourant 1 raises assembly 0 by 2.7 Hz, the largest of the twenty; within one
#: trial the coefficient moves too slowly (OU tau = 700 s) for the pairing to show up as a
#: correlation, which is a property of the stimulus, not of the panels.
#: 500 ms rather than the dashboard's 200 ms: the response to an odourant is a sustained
#: offset over the whole trial (the OU process barely moves within one), so what has to be
#: averaged away is the recurrent fluctuation. At 500 ms the dominant assembly sits 1.3
#: times the other assemblies' SD above its own mean; at 50 ms, 0.7 times.
ASSEMBLY_SMOOTHING_MS = 500.0
#: The assembly panels use the trial with the cleanest SWITCH between two odourants,
#: rather than the raster's trial: a trial whose stimulus is an even blend has nothing to
#: show, and one held on a single odourant shows a level rather than a change. The trial
#: is scored by how long each of its two leading odourants leads and how strongly it is
#: mixed while leading. The general strength of the effect is the across-trial number
#: quoted above, not this trial. ``--assembly-trial`` overrides the choice.
ASSEMBLY_TRIAL = None
#: Both panels are line plots of a 10 s window, so 1 ms resolution is far finer than the
#: ink: every fifth sample is stored, which is ten per smoothing kernel.
ASSEMBLY_STRIDE = 5


def default_device():
    """A CUDA device only if one is idle; otherwise the CPU.

    The GPUs on this machine are usually running training, and every simulation here is
    a few seconds of one trial, so falling back to the CPU costs minutes, not hours.
    """
    if not torch.cuda.is_available():
        return "cpu"
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        if free > 0.8 * total:
            return f"cuda:{index}"
    return "cpu"


class Teacher:
    """The teacher's read-only outputs: configs, structure, zarr, and its model."""

    def __init__(self, teacher_dir):
        self.dir = Path(teacher_dir)
        self.results = self.dir / "results"
        parameters = toml.load(self.dir / "parameters.toml")
        self.simulation = SimulationConfig(**parameters["simulation"])
        self.recurrent = RecurrentLayerConfig(**parameters["recurrent"])
        self.feedforward = FeedforwardLayerConfig(**parameters["feedforward"])
        odours = parameters["odours"].copy()
        for key in ("tau", "temperature", "sigma"):
            odours.pop(key)
        self.odours = {
            name: OdourInputConfig(**config) for name, config in odours.items()
        }
        structure = np.load(self.results / "network_structure.npz")
        self.weights = structure["recurrent_weights"]
        self.feedforward_weights = structure["feedforward_weights"]
        self.cell_type_indices = structure["cell_type_indices"]
        self.input_source_indices = structure["feedforward_cell_type_indices"]
        self.assembly_ids = structure["assembly_ids"]
        self.dt = float(self.simulation.dt)
        self.cell_type_names = list(self.recurrent.cell_types.names)

    @property
    def zarr(self):
        return zarr.open_group(str(self.results / "spike_data.zarr"), mode="r")

    @property
    def odour_configs(self):
        return {name: config.to_dict() for name, config in self.odours.items()}

    def steps(self, seconds):
        return round(1000.0 * seconds / self.dt)

    def odour_firing_rates(self):
        """The 20 odourant input patterns, as the generating script built them."""
        torch.manual_seed(self.simulation.seed)
        np.random.seed(self.simulation.seed)
        return generate_odour_firing_rates(
            feedforward_weights=self.feedforward_weights,
            input_source_indices=self.input_source_indices,
            cell_type_indices=self.cell_type_indices,
            assembly_ids=self.assembly_ids,
            target_cell_type_idx=0,
            cell_type_names=self.feedforward.cell_types.names,
            odour_configs=self.odour_configs,
        )

    def baseline_firing_rates(self):
        return generate_baseline_firing_rates(
            n_input_neurons=self.feedforward_weights.shape[0],
            input_source_indices=self.input_source_indices,
            cell_type_names=self.feedforward.cell_types.names,
            odour_configs=self.odour_configs,
        )

    def model(self, device, track_variables=False):
        rec_projections, ff_projections = make_frozen_projections(
            rec_weights=self.weights.astype(np.float32),
            ff_weights=self.feedforward_weights.astype(np.float32),
            cell_type_indices=self.cell_type_indices,
            ff_cell_type_indices=self.input_source_indices,
            cell_type_names=self.cell_type_names,
            ff_cell_type_names=list(self.feedforward.cell_types.names),
        )
        return ConductanceLIFNetwork(
            dt=self.dt,
            rec_projections=rec_projections,
            ff_projections=ff_projections,
            cell_type_indices=self.cell_type_indices,
            cell_type_indices_FF=self.input_source_indices,
            cell_params=self.recurrent.get_cell_params(),
            cell_params_FF=self.feedforward.get_cell_params(),
            synapse_params=self.recurrent.get_synapse_params(),
            synapse_params_FF=self.feedforward.get_synapse_params(),
            surrgrad_scale=1.0,
            batch_size=1,
            track_variables=track_variables,
        ).to(device)

    def synapse_labels(self, model):
        """One label per unified synapse type, e.g. "Feedforward AMPA"."""
        labels = [None] * model.n_unified_synapse_types
        for params in model.unified_synapse_params:
            source, synapse = params["signature"]
            pathway = "Recurrent" if source in self.cell_type_names else "Feedforward"
            labels[params["unified_synapse_id"]] = f"{pathway} {synapse}"
        return labels


# ==========
# Input coding
# ==========


def step_coding(teacher, out_dir, device):
    """How one odourant is built: one group up, the rest down, same mean.

    Pure arithmetic from the odour parameters, so the schematic panel states the same
    numbers the teacher was generated with rather than its own.
    """
    config = teacher.odours["mitral"]
    n_odourants = int(teacher.recurrent.topology.num_assemblies)
    baseline = float(config.baseline_rate)
    active = baseline + float(config.modulation_rate)
    depressed = (n_odourants * baseline - active) / (n_odourants - 1)
    table = pd.DataFrame(
        {
            "odourant": np.arange(1, n_odourants + 1),
            "baseline_hz": baseline,
            "odour_hz": [active] + [depressed] * (n_odourants - 1),
        }
    )
    table.to_csv(out_dir / "fig00_coding_schematic.csv", index=False)
    print(
        f"  coding: {n_odourants} odourants, {active:.1f} Hz active, "
        f"{depressed:.3f} Hz depressed, mean {table['odour_hz'].mean():.1f} Hz"
    )


# ==========
# Network response to three input conditions
# ==========


def simulate_condition(teacher, model, firing_rates, seed, device):
    """Poisson input at ``firing_rates`` through the teacher; per-neuron rates in Hz.

    ``seed`` seeds the Poisson draw only: the odourant's rate pattern is passed in, so
    the repeat condition is the same odourant seen through different input noise.
    """
    chunk_steps = int(teacher.simulation.chunk_size)
    n_steps = teacher.steps(CONDITION_SECONDS)
    torch.manual_seed(seed)
    dataloader = HomogeneousPoissonSpikeDataLoader(
        firing_rates=firing_rates.astype(np.float32),
        chunk_size=chunk_steps * teacher.dt,
        dt=teacher.dt,
        batch_size=1,
        device=device,
    )
    model.reset_state(batch_size=1)
    counts = np.zeros(teacher.weights.shape[0], dtype=np.int64)
    burn_in = teacher.steps(BURN_IN_SECONDS)
    scored_steps = 0
    step = 0
    with torch.inference_mode():
        for input_spikes, _ in dataloader:
            spikes = model.forward(input_spikes=input_spikes.float().to(device))
            spikes = spikes.detach().cpu().numpy()[0]
            keep = spikes[max(0, burn_in - step) :]
            counts += keep.sum(axis=0).astype(np.int64)
            scored_steps += keep.shape[0]
            step += spikes.shape[0]
            if step >= n_steps:
                break
    return counts / (scored_steps * teacher.dt * 1e-3)


def step_conditions(teacher, out_dir, device):
    """Three re-simulations: odourant 1, the same odourant again, and baseline."""
    odour = teacher.odour_firing_rates()[0:1]
    baseline = teacher.baseline_firing_rates()
    model = teacher.model(device)
    seed = int(teacher.simulation.seed)
    started = time.time()
    rates = {}
    for name, firing_rates, condition_seed in (
        ("odour_hz", odour, seed),
        ("odour_repeat_hz", odour, seed + 1),
        ("baseline_hz", baseline, seed),
    ):
        rates[name] = simulate_condition(
            teacher, model, firing_rates, condition_seed, device
        )
        print(
            f"  conditions: {name} mean {rates[name].mean():.2f} Hz "
            f"({time.time() - started:.0f} s)",
            flush=True,
        )
    pd.DataFrame(
        {
            "neuron_id": np.arange(teacher.cell_type_indices.size),
            "cell_type": [
                teacher.cell_type_names[index] for index in teacher.cell_type_indices
            ],
            "assembly": teacher.assembly_ids,
            **{name: value.astype(np.float32) for name, value in rates.items()},
        }
    ).to_csv(out_dir / "fig00_condition_rates.csv", index=False)


# ==========
# Dynamics: raster and one neuron's traces
# ==========


def trace_neuron(teacher, rates):
    """A representative neuron of ``TRACE_CELL_TYPE``: no outlier feedforward synapse,
    and a firing rate close to its cell type's mean. See ``TRACE_CELL_TYPE``."""
    cell_type = teacher.cell_type_names.index(TRACE_CELL_TYPE)
    strongest = teacher.feedforward_weights.max(axis=0)
    typical_input = strongest <= np.median(strongest)
    candidates = np.flatnonzero(
        (teacher.cell_type_indices == cell_type) & typical_input
    )
    if candidates.size == 0:  # no cell of this type escapes the tail; take them all
        candidates = np.flatnonzero(teacher.cell_type_indices == cell_type)
    target = rates[teacher.cell_type_indices == cell_type].mean()
    chosen = int(candidates[np.argmin(np.abs(rates[candidates] - target))])
    print(
        f"  dynamics: tracing {TRACE_CELL_TYPE} neuron {chosen} "
        f"({rates[chosen]:.1f} Hz against the population's {target:.1f} Hz, "
        f"strongest mitral weight {strongest[chosen]:.3f} against a median of "
        f"{np.median(strongest):.3f})"
    )
    return chosen


def raster_sample(teacher, rates):
    """The raster's rows, chosen as the library's activity dashboard chooses them.

    ``create_activity_dashboard`` (``visualization/dashboards.py``) plots feedforward and
    recurrent neurons together, in proportion to the population sizes, and picks the
    recurrent ones whose rate is closest to their own cell type's mean -- so no row is a
    silent cell or one of the handful firing at 270 Hz. Feedforward rows are drawn at
    random, since they all share one Poisson process. Ten rows in total (2026-09-21: the
    first build drew 300 and was an unreadable point cloud).

    Returns the feedforward and recurrent ids separately, feedforward first, which is the
    dashboard's row order.
    """
    n_feedforward = teacher.feedforward_weights.shape[0]
    n_recurrent = teacher.cell_type_indices.size
    n_input_rows = max(
        1, int(RASTER_NEURONS * n_feedforward / (n_feedforward + n_recurrent))
    )
    n_output_rows = RASTER_NEURONS - n_input_rows
    rng = np.random.default_rng(SAMPLE_SEED)
    inputs = np.sort(rng.choice(n_feedforward, size=n_input_rows, replace=False))
    recurrent = []
    for cell_type in np.unique(teacher.cell_type_indices):
        indices = np.flatnonzero(teacher.cell_type_indices == cell_type)
        count = max(1, round(n_output_rows * indices.size / n_recurrent))
        closest = indices[np.argsort(np.abs(rates[indices] - rates[indices].mean()))]
        recurrent.append(np.sort(closest[:count]))
    return inputs, np.concatenate(recurrent)[:n_output_rows]


def step_dynamics(teacher, out_dir, device):
    """The raster from the teacher's zarr, the traces from a tracked re-simulation."""
    data = teacher.zarr
    raster_steps = teacher.steps(RASTER_SECONDS)
    seconds = raster_steps * teacher.dt * 1e-3
    output_spikes = data["output_spikes"][RASTER_TRIAL, :raster_steps, :]
    population_rates = output_spikes.sum(axis=0) / seconds
    input_rows, output_rows = raster_sample(teacher, population_rates)
    traced = trace_neuron(teacher, population_rates)
    spikes = np.concatenate(
        [
            data["input_spikes"][RASTER_TRIAL, :raster_steps, :][:, input_rows],
            output_spikes[:, output_rows],
        ],
        axis=1,
    )
    rows, times = np.nonzero(spikes.T)
    rates = spikes.sum(axis=0) / seconds
    populations = ["feedforward"] * input_rows.size + [
        teacher.cell_type_names[i] for i in teacher.cell_type_indices[output_rows]
    ]
    np.savez_compressed(
        out_dir / "fig00_raster.npz",
        row=rows.astype(np.int16),
        time_s=(times * teacher.dt * 1e-3).astype(np.float32),
        neuron_id=np.concatenate([input_rows, output_rows]).astype(np.int32),
        population=np.array(populations),
        rate_hz=rates.astype(np.float32),
        duration_s=np.float32(RASTER_SECONDS),
        trial=np.int32(RASTER_TRIAL),
    )
    print(
        f"  dynamics: raster of {len(populations)} rows "
        f"({input_rows.size} feedforward), {rows.size} spikes over "
        f"{RASTER_SECONDS:.0f} s, {rates.min():.1f}-{rates.max():.1f} Hz",
        flush=True,
    )

    # Traces need the variables the teacher's run did not save, so the stored input of
    # the same trial is pushed through the model again with tracking on.
    trace_steps = teacher.steps(TRACE_SECONDS)
    input_spikes = data["input_spikes"][RASTER_TRIAL, :trace_steps, :]
    rng = np.random.default_rng(SAMPLE_SEED)
    sampled = np.sort(
        rng.choice(
            teacher.cell_type_indices.size, size=CONDUCTANCE_NEURONS, replace=False
        )
    )
    kept = np.unique(np.concatenate([sampled, [traced]]))
    trace_row = int(np.flatnonzero(kept == traced)[0])

    model = teacher.model(device, track_variables=True)
    model.reset_state(batch_size=1)
    g_clip = model.g_clip.detach().cpu().numpy()
    voltages, currents, leak, conductances = [], [], [], []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, trace_steps, TRACK_CHUNK_STEPS):
            chunk = torch.from_numpy(
                input_spikes[start : start + TRACK_CHUNK_STEPS].astype(np.float32)
            ).to(device)[None]
            out = model.forward(input_spikes=chunk)
            voltages.append(out["voltages"][0][:, kept].cpu().numpy())
            currents.append(out["currents"][0][:, kept].cpu().numpy())
            leak.append(out["currents_leak"][0][:, kept].cpu().numpy())
            # (time, neurons, 2, synapses) -> the conductance that drives the current.
            total = out["conductances"][0][:, kept].sum(dim=2).clamp(min=0.0)
            conductances.append(
                torch.minimum(total, torch.as_tensor(g_clip, device=total.device))
                .cpu()
                .numpy()
            )
            print(
                f"  dynamics: tracked {start + chunk.shape[1]}/{trace_steps} steps "
                f"({time.time() - started:.0f} s)",
                flush=True,
            )
    physiology_name = teacher.cell_type_names[teacher.cell_type_indices[traced]]
    voltages = np.concatenate(voltages).astype(np.float32)
    currents = np.concatenate(currents).astype(np.float32)
    leak = np.concatenate(leak).astype(np.float32)
    conductances = np.concatenate(conductances).astype(np.float32)
    labels = teacher.synapse_labels(model)

    np.savez_compressed(
        out_dir / "fig00_traces.npz",
        time_s=(np.arange(voltages.shape[0]) * teacher.dt * 1e-3).astype(np.float32),
        voltage_mv=voltages[:, trace_row],
        current_pa=currents[:, trace_row],
        leak_current_pa=leak[:, trace_row],
        conductance_ns=conductances[:, trace_row],
        synapse=np.array(labels),
        spike_time_s=(
            np.flatnonzero(output_spikes[:trace_steps, traced]) * teacher.dt * 1e-3
        ).astype(np.float32),
        neuron_id=np.int32(traced),
        cell_type=np.array(physiology_name),
        threshold_mv=np.float32(teacher.recurrent.physiology[physiology_name].theta),
        rest_mv=np.float32(teacher.recurrent.physiology[physiology_name].E_L),
        trial=np.int32(RASTER_TRIAL),
    )
    print(f"  dynamics: neuron {traced} traced over {TRACE_SECONDS:.0f} s")


def choose_assembly_trial(data, steps):
    """The trial with the cleanest switch, and its two leading odourants in time order.

    Scored by how long each of the two leading odourants leads the mixture and how
    strongly it is mixed while leading, so the panels show one odourant giving way to
    another rather than a single sustained level. See ``ASSEMBLY_TRIAL``.
    """
    weights = np.asarray(data["weights"][:, :steps, :])
    best, chosen, leaders = -1.0, 0, (0, 0)
    for trial in range(weights.shape[0]):
        leading = weights[trial].argmax(axis=1)
        share = np.bincount(leading, minlength=weights.shape[2]) / leading.size
        first, second = np.argsort(share)[::-1][:2]
        strength = [weights[trial][leading == k, k].mean() for k in (first, second)]
        # Strength counts twice: a trial where both odourants are strongly on either
        # side of the switch reads better than one that splits its time evenly between
        # two weak mixtures.
        score = min(share[first], share[second]) * min(strength) ** 2
        if score > best:
            order = sorted(
                (first, second), key=lambda k: np.flatnonzero(leading == k)[0]
            )
            best, chosen, leaders = score, trial, tuple(int(k) for k in order)
    if ASSEMBLY_TRIAL is not None:
        chosen = ASSEMBLY_TRIAL
        leading = weights[chosen].argmax(axis=1)
        share = np.bincount(leading, minlength=weights.shape[2])
        top = np.argsort(share)[::-1][:2]
        leaders = tuple(
            int(k) for k in sorted(top, key=lambda k: np.flatnonzero(leading == k)[0])
        )
    return chosen, leaders


def step_assemblies(teacher, out_dir, device):
    """The OU mixing trajectories and the assembly rates they produce.

    Follows the library's assembly dashboard: one trial, excitatory neurons only, each
    assembly's population rate smoothed with a Gaussian of ``ASSEMBLY_SMOOTHING_MS``.
    """
    data = teacher.zarr
    steps = teacher.steps(ASSEMBLY_SECONDS)
    burn_in = teacher.steps(BURN_IN_SECONDS)
    trial, leaders = choose_assembly_trial(data, steps)
    dominant = np.array(leaders, dtype=np.int32)
    weights = np.asarray(data["weights"][trial, :steps, :], dtype=np.float32)
    mixing = weights.mean(axis=0)
    mixing = weights.mean(axis=0)
    spikes = np.asarray(data["output_spikes"][trial, :steps, :])
    excitatory = teacher.cell_type_indices == teacher.cell_type_names.index(
        "excitatory"
    )
    assemblies = np.unique(teacher.assembly_ids[teacher.assembly_ids >= 0])
    rates = np.zeros((steps, assemblies.size), dtype=np.float32)
    for column, assembly in enumerate(assemblies):
        members = excitatory & (teacher.assembly_ids == assembly)
        counts = spikes[:, members].mean(axis=1).astype(np.float32)
        rates[:, column] = smooth(counts[:, None], ASSEMBLY_SMOOTHING_MS, teacher.dt)[
            :, 0
        ] * (1000.0 / teacher.dt)
    # Each assembly's mean rate over EVERY trial. Assemblies differ in intrinsic rate by
    # far more than a stimulus moves them (spread SD 1.13 Hz against a deviation SD of
    # 0.41 Hz), so the raw rates show the ordering of the assemblies rather than the
    # stimulus. Subtracting this baseline exposes the response: across all 50 trials the
    # mixing coefficient correlates +0.51 with its own assembly's deviation, and the
    # dominant odourant's assembly is the largest deviator in 62% of trials, top-3 in 82%.
    members = [excitatory & (teacher.assembly_ids == a) for a in assemblies]
    all_trials = data["output_spikes"]
    burn_in = teacher.steps(BURN_IN_SECONDS)
    n_trials = all_trials.shape[0]
    baseline = np.zeros(assemblies.size, dtype=np.float64)
    for other in range(n_trials):  # one read per trial, all assemblies from it
        trial_spikes = all_trials[other, burn_in:, :]
        for column, mask in enumerate(members):
            baseline[column] += trial_spikes[:, mask].mean()
        print(f"  assemblies: baseline {other + 1}/{n_trials}", end="\r", flush=True)
    baseline = (baseline / n_trials * (1000.0 / teacher.dt)).astype(np.float32)
    np.savez_compressed(
        out_dir / "fig00_assemblies.npz",
        time_s=(np.arange(0, steps, ASSEMBLY_STRIDE) * teacher.dt * 1e-3).astype(
            np.float32
        ),
        mixing_weight=weights[::ASSEMBLY_STRIDE],
        rate_hz=rates[::ASSEMBLY_STRIDE],
        baseline_hz=baseline,
        assembly=assemblies.astype(np.int16),
        smoothing_ms=np.float32(ASSEMBLY_SMOOTHING_MS),
        dominant=dominant,
        trial=np.int32(trial),
    )
    print(
        f"  assemblies: {assemblies.size} assemblies over {ASSEMBLY_SECONDS:.0f} s, "
        f"rates {rates.min():.1f}-{rates.max():.1f} Hz, mixing weight up to "
        f"{weights.max():.2f}; trial {trial}, led by odourants "
        + " then ".join(f"{k} ({mixing[k]:.2f} mean weight)" for k in leaders)
    )


def step_drive(teacher, out_dir, device):
    """Feedforward against recurrent excitatory synaptic drive, over the whole dataset.

    Drive is the spikes a source emitted times the total weight it sends on, summed over
    sources: the charge each pathway delivers to the network, up to the synaptic kernels.
    """
    data = teacher.zarr
    output_spikes = data["output_spikes"]
    input_spikes = data["input_spikes"]
    n_trials, n_steps, n_neurons = output_spikes.shape
    chunk = output_spikes.chunks[1]
    output_counts = np.zeros(n_neurons, dtype=np.int64)
    input_counts = np.zeros(input_spikes.shape[2], dtype=np.int64)
    started = time.time()
    for start in range(0, n_steps, chunk):
        stop = start + chunk
        output_counts += output_spikes[:, start:stop, :].sum(axis=(0, 1))
        input_counts += input_spikes[:, start:stop, :].sum(axis=(0, 1))
        print(
            f"  drive: {min(stop, n_steps)}/{n_steps} steps ({time.time() - started:.0f} s)",
            flush=True,
        )
    excitatory = teacher.cell_type_indices == 0
    recurrent_drive = float(
        (output_counts[excitatory] * teacher.weights[excitatory, :].sum(axis=1)).sum()
    )
    feedforward_drive = float(
        (input_counts * teacher.feedforward_weights.sum(axis=1)).sum()
    )
    total = recurrent_drive + feedforward_drive
    pd.DataFrame(
        {
            "pathway": ["Feedforward", "Recurrent Excitatory"],
            "drive": [feedforward_drive, recurrent_drive],
            "drive_fraction": [feedforward_drive / total, recurrent_drive / total],
            "n_trials": n_trials,
            "seconds_per_trial": n_steps * teacher.dt / 1000.0,
        }
    ).to_csv(out_dir / "fig00_drive.csv", index=False)
    print(
        f"  drive: recurrent excitatory / feedforward = {recurrent_drive / feedforward_drive:.3f}"
    )


# ==========
# Dimensionality
# ==========


def step_dimensionality(teacher, out_dir, device):
    """Participation ratio and PCA spectrum of the teacher's smoothed activity."""
    evaluation = toml.load(EVALUATION_PARAMETERS)["evaluation"]
    sigma_ms = float(evaluation["fluctuation_tau_ms"])
    burn_in_ms = float(evaluation["burn_in_ms"])
    data = teacher.zarr
    spikes = data["output_spikes"]
    dt = float(data.attrs["dt"])
    burn_in = int(burn_in_ms / dt)
    n_trials, n_steps, n_neurons = spikes.shape

    # Pooled second moments, one trial at a time.
    total = torch.zeros(n_neurons, dtype=torch.float64)
    outer = torch.zeros(n_neurons, n_neurons, dtype=torch.float64)
    n_samples = 0
    started = time.time()
    for trial in range(n_trials):
        activity = torch.from_numpy(
            smooth(np.asarray(spikes[trial]), sigma_ms, dt)[burn_in:]
        ).double()
        total += activity.sum(dim=0)
        outer += activity.T @ activity
        n_samples += activity.shape[0]
        print(
            f"  trial {trial + 1}/{n_trials} ({time.time() - started:.0f} s)",
            flush=True,
        )
    mean = total / n_samples
    covariance = (outer - n_samples * torch.outer(mean, mean)) / (n_samples - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp(min=0).flip(0).numpy()

    fraction = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(fraction)
    spectrum = pd.DataFrame(
        {
            "component": np.arange(1, eigenvalues.size + 1),
            "eigenvalue": eigenvalues,
            "variance_fraction": fraction,
            "cumulative_fraction": cumulative,
        }
    )
    summary = {
        "participation_ratio": float(eigenvalues.sum() ** 2 / (eigenvalues**2).sum()),
        **{
            f"n_pcs_{round(100 * t)}pct_var": int(np.searchsorted(cumulative, t) + 1)
            for t in VARIANCE_THRESHOLDS
        },
        "smoothing_sigma_ms": sigma_ms,
        "burn_in_ms": burn_in_ms,
        "n_trials": int(n_trials),
        "seconds_per_trial": (n_steps - burn_in) * dt / 1000.0,
        "n_neurons": int(n_neurons),
    }
    spectrum.to_csv(out_dir / "fig00_pca_spectrum.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "fig00_dimensionality.csv", index=False)
    print(pd.Series(summary))


#: step name -> (function, the files it writes).
STEPS = {
    "coding": (step_coding, ["fig00_coding_schematic.csv"]),
    "conditions": (step_conditions, ["fig00_condition_rates.csv"]),
    "dynamics": (step_dynamics, ["fig00_raster.npz", "fig00_traces.npz"]),
    "assemblies": (step_assemblies, ["fig00_assemblies.npz"]),
    "drive": (step_drive, ["fig00_drive.csv"]),
    "dimensionality": (
        step_dimensionality,
        ["fig00_pca_spectrum.csv", "fig00_dimensionality.csv"],
    ),
}


def main(teacher_dir, out_dir, steps, device, force):
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher = Teacher(teacher_dir)
    print(f"Teacher {teacher.dir} on {device}")
    for name in steps:
        function, outputs = STEPS[name]
        existing = [path for path in outputs if (out_dir / path).exists()]
        if not force and len(existing) == len(outputs):
            print(f"{name}: reusing {', '.join(existing)}")
            continue
        print(f"{name}: computing {', '.join(outputs)}", flush=True)
        function(teacher, out_dir, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        type=Path,
        default=Path(load_experiment_config(HERE / "experiment.toml")["output_dir"]),
    )
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument(
        "--assembly-trial",
        type=int,
        help="trial for the assembly panels (default: the most concentrated odourant)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=list(STEPS),
        default=list(STEPS),
        help="steps to run (default: all)",
    )
    parser.add_argument("--device", default=default_device())
    parser.add_argument(
        "--force", action="store_true", help="recompute even if the outputs exist"
    )
    args = parser.parse_args()
    if args.assembly_trial is not None:
        ASSEMBLY_TRIAL = args.assembly_trial
    main(args.teacher, args.out, args.steps, args.device, args.force)
