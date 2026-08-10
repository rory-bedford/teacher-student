"""
Bias check: inference-only loss at the learnt recurrent scaling factors vs the correct ones.

Targets ``full-inference/no-hidden`` — a fully-observed, teacher-forced student
whose scaling factors update continuously, so an inference-mode evaluation at a
given set of scaling factors is directly comparable to the training loss.

Motivation: learnt scaling factors come out systematically *below* their targets,
and more so as the problem gets harder. This asks whether that shrinkage is where
the loss actually wants to be:

    - "learnt"  — the run's final state exactly as trained.
    - "correct" — the same state with only the recurrent scaling factors replaced
                  by their targets, i.e. the values that exactly undo the
                  per-cell-type-pair weight perturbation.

Both settings share the identical learnt low-rank feedforward block, so the
comparison isolates the recurrent scaling factors — a partial derivative of the
loss in exactly the direction the shrinkage happens.

Caveat worth remembering when reading the result: the target scaling factors are
*not* the true model here. ``noise_frac`` (per-synapse weight noise) and
``missing_unit_fraction`` (structurally removed neurons) are perturbations no
scaling factor can undo, so the loss-optimal scaling factors are legitimately
displaced from the targets. A gap therefore measures how far mis-specification
moves the optimum, not necessarily a pathology.

Weights come from the run's own ``initial_state``/``final_state`` snapshots rather
than being re-derived from the teacher. That is both exact and robust: input
symlinks resolve by path, so a regenerated teacher silently changes what a
historical run appears to have trained on.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import toml
import torch
from connectome_snns.configs import StudentSimulationConfig
from connectome_snns.configs.conductance_based import (
    FeedforwardLayerConfig,
    RecurrentLayerConfig,
)
from connectome_snns.dataloaders.supervised import (
    CyclicSampler,
    ExactFFDataset,
    SpikeData,
    VisibleDrivenCollate,
)
from connectome_snns.network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
from connectome_snns.network_simulators.projections import (
    make_frozen_chunked_ff_projections,
)
from connectome_snns.training_utils.losses import VanRossumLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SF_SOURCES = ("learnt", "correct")


class SubBatchDataset(Dataset):
    """Wraps an ExactFFDataset to return only the first ``n_batch`` batch elements."""

    def __init__(self, dataset, n_batch):
        self.dataset = dataset
        self.batch_size = n_batch
        self.dt = dataset.dt
        self.num_chunks = dataset.num_chunks

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return SpikeData(
            input_spikes=data.input_spikes[: self.batch_size],
            target_spikes=data.target_spikes[: self.batch_size],
        )


def load_run_config(run_dir):
    """Read the trained model's weights and configuration from a run's snapshots.

    Returns a dict with the effective concatenated weight matrix as trained
    (``learnt_weights``), the unscaled recurrent block (``initial_rec_weights``)
    that the scaling factors multiply, and the target scaling factors.
    """
    data = toml.load(run_dir / "parameters.toml")
    simulation = StudentSimulationConfig(**data["simulation"])
    recurrent = RecurrentLayerConfig(**data["recurrent"])
    feedforward = FeedforwardLayerConfig(**data["feedforward"])

    initial_state = np.load(run_dir / "initial_state" / "network_structure.npz")
    final_state = np.load(run_dir / "final_state" / "network_structure.npz")
    classification = np.load(run_dir / "targets" / "neuron_classification.npz")
    target_sf = np.load(run_dir / "targets" / "target_scaling_factors.npz")[
        "recurrent_scaling_factors"
    ]

    mask = initial_state["feedforward_connectivity"]
    n_feedforward = int(initial_state["is_feedforward_input"].sum())

    recurrent_cell_params = recurrent.get_cell_params()
    feedforward_cell_params = feedforward.get_cell_params()
    n_ff_cell_types = len(feedforward_cell_params)
    n_ff_synapse_types = len(feedforward.get_synapse_params())

    combined_cell_params_FF = feedforward_cell_params.copy()
    for cp in recurrent_cell_params:
        offset = cp.copy()
        offset["cell_id"] = cp["cell_id"] + n_ff_cell_types
        combined_cell_params_FF.append(offset)

    combined_synapse_params_FF = feedforward.get_synapse_params().copy()
    for sp in recurrent.get_synapse_params():
        offset = sp.copy()
        offset["cell_id"] = sp["cell_id"] + n_ff_cell_types
        offset["synapse_id"] = sp["synapse_id"] + n_ff_synapse_types
        combined_synapse_params_FF.append(offset)

    return {
        "learnt_weights": final_state["feedforward_weights"] * mask.astype(np.float32),
        "initial_rec_weights": initial_state["feedforward_weights"][n_feedforward:]
        * mask[n_feedforward:].astype(np.float32),
        "cell_type_indices": initial_state["cell_type_indices"],
        "feedforward_cell_type_indices": initial_state["feedforward_cell_type_indices"],
        "n_feedforward": n_feedforward,
        "target_sf": target_sf,
        "visible_indices": classification["structural_visible_indices"],
        "missing_unit_fraction": float(classification["missing_unit_fraction"]),
        "noise_frac": float(classification["noise_frac"]),
        "recurrent_cell_params": recurrent_cell_params,
        "feedforward_cell_params": feedforward_cell_params,
        "combined_cell_params_FF": combined_cell_params_FF,
        "combined_synapse_params_FF": combined_synapse_params_FF,
        "n_ff_cell_types": n_ff_cell_types,
        "output_cell_type_names": recurrent.cell_types.names,
        "chunk_size": simulation.chunk_size,
        "surrgrad_scale": data.get("phase2", {}).get("surrgrad_scale", 5.0),
        "burn_in_chunks": data["training"].get("burn_in_chunks", 0),
        "van_rossum_tau_rise": data["hyperparameters"]["van_rossum_tau_rise"],
        "van_rossum_tau_decay": data["hyperparameters"]["van_rossum_tau_decay"],
    }


def weights_for(config, sf_source):
    """Effective concatenated weights for one scaling-factor setting.

    ``learnt`` is the trained state verbatim. ``correct`` keeps the learnt
    feedforward block and rescales the recurrent block so that each cell-type
    pair sits at its target scaling factor instead of the learnt one.
    """
    weights = config["learnt_weights"].copy()
    if sf_source == "learnt":
        return weights

    cell_type_indices = config["cell_type_indices"]
    scaled = config["initial_rec_weights"] * config["target_sf"][
        cell_type_indices[:, None], cell_type_indices[None, :]
    ].astype(np.float32)
    weights[config["n_feedforward"] :] = scaled
    return weights


def learnt_scaling_factors(run_dir, target_sf, output_cell_type_names):
    """Absolute learnt recurrent scaling factors at the last logged step."""
    metrics = pd.read_csv(run_dir / "training_metrics.csv")
    last = metrics[metrics["epoch"] >= 0].iloc[-1]
    sf = np.empty_like(target_sf)
    for i, src in enumerate(output_cell_type_names):
        for j, tgt in enumerate(output_cell_type_names):
            sf[i, j] = last[f"scaling_factors/{src}_to_{tgt}_value"] * target_sf[i, j]
    return sf, float(last["van_rossum_loss"])


def evaluate(weights, *, config, spike_dataset, loss_fn, n_eval_chunks, device):
    """Run the teacher-forced student forward at fixed weights and score it.

    Mirrors the training loop's state handling: model and loss filter reset once,
    burn-in chunks passed through both without being scored, loss averaged over
    the remaining chunks.
    """
    n_feedforward = config["n_feedforward"]
    cell_type_indices = config["cell_type_indices"]
    burn_in_chunks = config["burn_in_chunks"]

    model = FeedforwardConductanceLIFNetwork(
        dt=spike_dataset.dt,
        projections=make_frozen_chunked_ff_projections(
            rec_weights=weights[n_feedforward:],
            ff_weights=weights[:n_feedforward],
            cell_type_indices=cell_type_indices,
            ff_cell_type_indices=config["feedforward_cell_type_indices"][
                :n_feedforward
            ],
            cell_type_names=config["output_cell_type_names"],
            ff_cell_type_names=[cp["name"] for cp in config["feedforward_cell_params"]],
            scaling_factors=None,  # already baked into `weights`
        ),
        cell_type_indices=cell_type_indices,
        cell_type_indices_FF=config["feedforward_cell_type_indices"],
        cell_params=config["recurrent_cell_params"],
        cell_params_FF=config["combined_cell_params_FF"],
        synapse_params_FF=config["combined_synapse_params_FF"],
        surrgrad_scale=config["surrgrad_scale"],
        batch_size=spike_dataset.batch_size,
        track_variables=False,
    )
    model.to(device)
    model.eval()
    model.reset_state(batch_size=spike_dataset.batch_size)
    loss_fn.reset_state()

    dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        num_workers=0,
        collate_fn=VisibleDrivenCollate(
            torch.from_numpy(config["visible_indices"]).long()
        ),
    )
    data_iter = iter(dataloader)

    losses = []
    student_counts = np.zeros(len(cell_type_indices))
    teacher_counts = np.zeros(len(cell_type_indices))

    with torch.inference_mode():
        for chunk_idx in tqdm(
            range(burn_in_chunks + n_eval_chunks), desc="  inference", unit="chunk"
        ):
            batch = next(data_iter)
            outputs = model.forward(input_spikes=batch.input_spikes.to(device).float())
            spikes = outputs["spikes"] if isinstance(outputs, dict) else outputs
            targets = batch.target_spikes.to(device).float()

            # Burn-in chunks still pass through the loss so its filter state is
            # warmed up, as SNNTrainer does — the value is discarded.
            loss = loss_fn(output_spikes=spikes, target_spikes=targets)
            if chunk_idx >= burn_in_chunks:
                losses.append(float(loss))
                student_counts += spikes[0].sum(dim=0).cpu().numpy()
                teacher_counts += targets[0].sum(dim=0).cpu().numpy()

    duration_s = n_eval_chunks * config["chunk_size"] * spike_dataset.dt / 1000.0
    student_rates = student_counts / duration_s
    teacher_rates = teacher_counts / duration_s

    metrics = {
        "van_rossum_loss": float(np.mean(losses)),
        "van_rossum_loss_std": float(np.std(losses)),
        "firing_rate/student_mean": float(student_rates.mean()),
        "firing_rate/teacher_mean": float(teacher_rates.mean()),
    }
    for type_idx, type_name in enumerate(config["output_cell_type_names"]):
        cells = cell_type_indices == type_idx
        metrics[f"firing_rate/student_{type_name}_mean"] = float(
            student_rates[cells].mean()
        )
        metrics[f"firing_rate/teacher_{type_name}_mean"] = float(
            teacher_rates[cells].mean()
        )

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return metrics, np.array(losses)


def main(input_dir, output_dir, params_file, wandb_config=None, resume_from=None):
    """Compute inference losses at the learnt and the correct scaling factors.

    Args:
        input_dir (Path): Contains spike_data.zarr and training_runs/ (the
            symlinked reference run, or a directory of runs).
        output_dir (Path): Directory where outputs are saved.
        params_file (Path): Path to this experiment's parameters TOML.
        wandb_config (dict, optional): Not used.
        resume_from (Path, optional): Not used.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    bias_check = toml.load(params_file).get("bias_check", {})

    training_runs_dir = input_dir / "training_runs"
    if (training_runs_dir / "final_state").exists():
        run_dirs = [training_runs_dir]
    else:
        run_dirs = sorted(
            d for d in training_runs_dir.iterdir() if (d / "final_state").exists()
        )
    if not run_dirs:
        raise FileNotFoundError(f"No completed runs found under {training_runs_dir}")
    print(f"Found {len(run_dirs)} reference run(s): {[d.name for d in run_dirs]}")

    rows, scaling_factor_record, loss_traces = [], {}, {}

    for run_dir in run_dirs:
        config = load_run_config(run_dir)
        target_sf = config["target_sf"]
        learnt_sf, training_final_loss = learnt_scaling_factors(
            run_dir, target_sf, config["output_cell_type_names"]
        )

        spike_dataset = ExactFFDataset(
            spike_data_path=input_dir / "spike_data.zarr",
            chunk_size=config["chunk_size"],
            device=device,
        )
        num_chunks = spike_dataset.num_chunks
        eval_batch_size = (
            bias_check.get("eval_batch_size", 0) or spike_dataset.batch_size
        )
        if eval_batch_size < spike_dataset.batch_size:
            spike_dataset = SubBatchDataset(spike_dataset, eval_batch_size)

        n_eval_chunks = bias_check.get("n_eval_chunks", 0) or (
            num_chunks - config["burn_in_chunks"]
        )
        if config["burn_in_chunks"] + n_eval_chunks > num_chunks:
            raise ValueError(
                f"burn_in_chunks ({config['burn_in_chunks']}) + n_eval_chunks "
                f"({n_eval_chunks}) exceeds the {num_chunks} chunks available."
            )

        loss_fn = VanRossumLoss(
            tau_rise=config["van_rossum_tau_rise"],
            tau_decay=config["van_rossum_tau_decay"],
            dt=spike_dataset.dt,
            window_size=config["chunk_size"],
            device=device,
        )

        header = (
            f"{run_dir.name}  ({len(config['cell_type_indices'])} neurons, "
            f"noise_frac {config['noise_frac']}, "
            f"missing_unit_fraction {config['missing_unit_fraction']}, "
            f"{num_chunks} chunks x {spike_dataset.batch_size})"
        )
        print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}")
        print(f"  Training final loss: {training_final_loss:.4f}")
        print(
            f"  learnt / target SF:\n{np.array2string(learnt_sf / target_sf, precision=4)}"
        )

        for sf_source in SF_SOURCES:
            print(f"\n--- {sf_source} recurrent scaling factors ---")
            metrics, losses = evaluate(
                weights_for(config, sf_source),
                config=config,
                spike_dataset=spike_dataset,
                loss_fn=loss_fn,
                n_eval_chunks=n_eval_chunks,
                device=device,
            )
            print(f"  van Rossum loss: {metrics['van_rossum_loss']:.4f}")

            key = f"{run_dir.name}__{sf_source}"
            scaling_factor_record[key] = (
                learnt_sf if sf_source == "learnt" else target_sf
            )
            loss_traces[key] = losses
            rows.append(
                {
                    "run": run_dir.name,
                    "sf_source": sf_source,
                    "n_neurons": len(config["cell_type_indices"]),
                    "noise_frac": config["noise_frac"],
                    "missing_unit_fraction": config["missing_unit_fraction"],
                    "training_final_loss": training_final_loss,
                    **metrics,
                }
            )

        del spike_dataset, loss_fn

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "inference_losses.csv", index=False)
    np.savez(
        output_dir / "scaling_factors.npz",
        output_cell_type_names=np.array(config["output_cell_type_names"]),
        **scaling_factor_record,
    )
    np.savez(output_dir / "loss_traces.npz", **loss_traces)

    print(f"\n{'=' * 60}\nBias check complete\n{'=' * 60}")
    print(results.to_string(index=False))
    print(f"\nSaved to {output_dir}")

    return float(results["van_rossum_loss"].min())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--params_file", type=Path, required=True)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.params_file)
