"""Export the minimal data behind each figure to figures/data/*.csv.

Reads cached inference outputs from the (read-only) run directories and writes
small CSVs. make_figures.py regenerates the SVGs from these CSVs alone.

Run from the repo root:  uv run python figures/export_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from connectome_snns.analysis import (
    firing_rates_from_spikes,
    fluctuation_r_squared,
    r_squared,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs"
DATA_DIR = Path(__file__).resolve().parent / "data"

TAU_MS = 50.0
CELL_TYPE_NAMES = np.array(["excitatory", "inhibitory"])
FRACTIONS = np.arange(0.05, 0.55, 0.05).round(2)

# Sources (all verified to reproduce the R² values printed on the original SVGs).
# Run directory names use the old hidden/no-hidden terminology; see README.md.
FF_LEARNT_OBS45_RUN = RUNS / "full-inference/hidden/surrgrad-10"
# Labels follow each run's parameters.toml. The two shuffle-control directory
# names are swapped relative to what they did.
CONTROL_RUNS = {
    "Full Connectome": RUNS / "full-inference/no-hidden/baseline",
    "Learnt-Recurrence": RUNS / "full-inference/no-hidden/no-connectome-control",
    # shuffle_connectome = true
    "Shuffle-Connections": RUNS / "full-inference/no-hidden/shuffle-weights-control",
    # shuffle_weights = true
    "Shuffle-Weights": RUNS / "full-inference/no-hidden/shuffle-control",
}
FF_KNOWN_OBS10_RUN = RUNS / "hidden-activity/increasing-hidden-fraction/hidden-frac-0.9"
REMOVED_NEURON_GRID = RUNS / "hidden-units/increasing-hidden-fraction"
WEIGHT_NOISE_GRID = RUNS / "noisy-weights/varying-noise"

RASTER_NEURON_IDS = [2433, 2246, 1316]  # top to bottom
RASTER_T_MS = 2000.0


def export_ff_learnt_obs45_scatter():
    cache = np.load(FF_LEARNT_OBS45_RUN / "inference_cache_test.npz")
    dt = float(cache["dt"])
    frames = []
    for group, observed in [("visible", True), ("hidden", False)]:
        frames.append(
            pd.DataFrame(
                {
                    "observed": observed,
                    "cell_type": CELL_TYPE_NAMES[cache[f"{group}_cell_types"]],
                    "teacher_rate_hz": firing_rates_from_spikes(
                        cache[f"teacher_{group}_spikes"], dt
                    ),
                    "student_rate_hz": firing_rates_from_spikes(
                        cache[f"student_{group}_spikes"], dt
                    ),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def export_controls_bar():
    rows = []
    for condition, run_dir in CONTROL_RUNS.items():
        cache = np.load(run_dir / "final_state" / "inference_cache_test.npz")
        dt = float(cache["dt"])
        teacher = cache["teacher_spikes"]
        student = cache["student_spikes"]
        rows.append(
            {
                "condition": condition,
                "activity_r2": r_squared(
                    firing_rates_from_spikes(teacher, dt),
                    firing_rates_from_spikes(student, dt),
                ),
                "fluctuation_r2": fluctuation_r_squared(
                    teacher.astype(np.float32), student.astype(np.float32), TAU_MS, dt
                ),
            }
        )
    return pd.DataFrame(rows)


def export_ff_known_obs10_scatter():
    rates = np.load(FF_KNOWN_OBS10_RUN / "per_neuron_rates.npz")
    frames = []
    for group, observed in [("visible", True), ("hidden", False)]:
        idx = rates[f"{group}_indices"]
        frames.append(
            pd.DataFrame(
                {
                    "neuron_id": idx,
                    "observed": observed,
                    "cell_type": CELL_TYPE_NAMES[rates["cell_type_indices"][idx]],
                    "teacher_rate_hz": rates["teacher_rates"][idx],
                    "student_rate_hz": rates[f"{group}_rates"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def export_ff_known_obs10_raster():
    """Spike times of three unobserved neurons (dt = 1 ms)."""
    rasters = np.load(FF_KNOWN_OBS10_RUN / "spike_rasters_full.npz")
    col_of = {int(nid): col for col, nid in enumerate(rasters["hidden_indices"])}
    n_steps = int(RASTER_T_MS)
    rows = []
    for source in ["teacher", "student"]:
        spikes = rasters[f"{source}_hidden"][:n_steps]
        for nid in RASTER_NEURON_IDS:
            for step in np.flatnonzero(spikes[:, col_of[nid]]):
                rows.append(
                    {"neuron_id": nid, "source": source, "spike_time_s": step / 1000.0}
                )
    return pd.DataFrame(rows)


def export_sweep_recur_curve():
    rows = []
    for frac in FRACTIONS:
        run_dir = REMOVED_NEURON_GRID / f"hidden-{frac:.2f}"
        rates = np.load(run_dir / "per_neuron_rates.npz")
        rows.append(
            {
                "removed_neuron_fraction": frac,
                "activity_r2": r_squared(
                    rates["teacher_rates_visible"], rates["student_rates"]
                ),
                "fluctuation_r2": float(
                    np.load(run_dir / "fluct_r2.npz")["r2_visible"]
                ),
            }
        )
    return pd.DataFrame(rows)


def export_sweep_wn_curve():
    rows = []
    for frac in FRACTIONS:
        run_dir = WEIGHT_NOISE_GRID / f"noise-{frac:.2f}"
        plot_data = np.load(run_dir / "final_state" / "plot_data.npz")
        dt = float(plot_data["dt"])
        rows.append(
            {
                "weight_noise_fraction": frac,
                "activity_r2": r_squared(
                    firing_rates_from_spikes(plot_data["teacher_spikes"], dt),
                    firing_rates_from_spikes(plot_data["student_spikes"], dt),
                ),
                "fluctuation_r2": float(np.load(run_dir / "fluct_r2.npz")["r2_fluct"]),
            }
        )
    return pd.DataFrame(rows)


EXPORTS = {
    "ff-learnt__obs-45__recur-81__wn-30__scatter": export_ff_learnt_obs45_scatter,
    "controls__ff-learnt__obs-90__recur-81__wn-mixed__bar": export_controls_bar,
    "ff-known__obs-10__recur-100__wn-0__scatter": export_ff_known_obs10_scatter,
    "ff-known__obs-10__recur-100__wn-0__raster": export_ff_known_obs10_raster,
    "sweep-recur__ff-known__obs-100__wn-0__curve": export_sweep_recur_curve,
    "sweep-wn__ff-known__obs-100__recur-100__curve": export_sweep_wn_curve,
}


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for filename, export in EXPORTS.items():
        df = export()
        df.to_csv(DATA_DIR / f"{filename}.csv", index=False, float_format="%.6g")
        print(f"{filename}.csv: {len(df)} rows")


if __name__ == "__main__":
    main()
