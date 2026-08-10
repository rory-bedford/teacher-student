"""
Bias check: inference-only loss at the learnt minimum vs the correct scaling factors.

For every hidden cell fraction of the `increasing-hidden-fraction` grid search we
evaluate two settings of the scaling factors, with no training at all:

    - "learnt"  — the scaling factors at the lowest van Rossum loss reached during
                  that training run (its minimum), reconstructed from
                  `training_metrics.csv` (logged normalised) times the run's target
                  scaling factors.
    - "correct" — the target scaling factors themselves, i.e. the values that
                  exactly undo the weight perturbation (all 1.0 after the
                  normalisation used in the plots).

Each setting is evaluated end to end exactly as during training: a clamped E-step
infers the hidden spikes with those scaling factors, then the feedforward M-step
model is run with [FF, visible teacher, inferred hidden] and the van Rossum loss is
computed on the visible neurons.

If the correct scaling factors give a *higher* loss than the learnt ones, the loss
minimum is biased away from the truth and training is doing its job — the objective
is not.

The perturbed weights, the hidden/visible split and the target scaling factors are
read from each training run's `targets/` directory rather than regenerated, so the
network here is identical to the one that was trained.

This is the pilot for a full sweep over the scaling factors; it evaluates the two
end points only.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import toml
import torch
import zarr
from connectome_snns.configs import (
    EMTrainingConfig,
    StudentHyperparameters,
    StudentSimulationConfig,
)
from connectome_snns.configs.conductance_based import (
    FeedforwardLayerConfig,
    RecurrentLayerConfig,
)
from connectome_snns.dataloaders.supervised import (
    CyclicSampler,
    ExactFFDataset,
    SpikeData,
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

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from _em_core import run_estep_clamped

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


def learnt_sf_at_minimum(
    run_dir, target_sf, input_cell_type_names, output_cell_type_names
):
    """Scaling factors at the lowest van Rossum loss reached during training.

    The metrics CSV logs scaling factors normalised by the target, so the absolute
    values are recovered by multiplying back by ``target_sf``. Rows with negative
    epochs belong to the CMA-ES phase and are skipped.

    Args:
        run_dir: Training run directory.
        target_sf: (n_input_ct, n_output_ct) target scaling factors for that run.
        input_cell_type_names: Feedforward followed by recurrent cell type names.
        output_cell_type_names: Recurrent cell type names.

    Returns:
        (sf_matrix, min_loss, epoch) — absolute scaling factors, the training loss
        at that point, and the epoch it occurred at.
    """
    metrics = pd.read_csv(run_dir / "training_metrics.csv")
    metrics = metrics[metrics["epoch"] >= 0]
    best_row = metrics.loc[metrics["van_rossum_loss"].idxmin()]

    sf = np.empty_like(target_sf)
    for src_idx, src_name in enumerate(input_cell_type_names):
        for tgt_idx, tgt_name in enumerate(output_cell_type_names):
            column = f"scaling_factors/{src_name}_to_{tgt_name}_value"
            sf[src_idx, tgt_idx] = best_row[column] * target_sf[src_idx, tgt_idx]

    return sf, float(best_row["van_rossum_loss"]), int(best_row["epoch"])


def evaluate_scaling_factors(
    sf,
    *,
    dt,
    batch_size,
    device,
    perturbed_rec_weights,
    perturbed_ff_weights,
    cell_type_indices,
    feedforward_cell_type_indices,
    recurrent_cell_params,
    feedforward_cell_params,
    recurrent_synapse_params,
    feedforward_synapse_params,
    combined_cell_params_FF,
    combined_synapse_params_FF,
    concatenated_cell_type_indices,
    recurrent_mask,
    feedforward_mask,
    visible_indices,
    hidden_indices,
    spike_dataset,
    num_chunks,
    chunk_size,
    n_ff_cell_types,
    surrgrad_scale,
    van_rossum_loss_fn,
    burn_in_chunks,
    n_eval_chunks,
    estep_path,
):
    """Run E-step + M-step inference with fixed scaling factors and return metrics.

    The scaling factors are used for both steps: the hidden spikes are inferred with
    the same model the loss is then evaluated under.
    """
    sf_feedforward = sf[:n_ff_cell_types]
    sf_recurrent = sf[n_ff_cell_types:]

    if estep_path.exists():
        shutil.rmtree(estep_path)

    run_estep_clamped(
        dt=dt,
        batch_size=batch_size,
        device=device,
        perturbed_rec_weights=perturbed_rec_weights,
        perturbed_ff_weights=perturbed_ff_weights,
        cell_type_indices=cell_type_indices,
        feedforward_cell_type_indices=feedforward_cell_type_indices,
        recurrent_cell_params=recurrent_cell_params,
        feedforward_cell_params=feedforward_cell_params,
        recurrent_synapse_params=recurrent_synapse_params,
        feedforward_synapse_params=feedforward_synapse_params,
        surrgrad_scale=surrgrad_scale,
        sf_recurrent=sf_recurrent,
        sf_feedforward=sf_feedforward,
        recurrent_mask=recurrent_mask,
        feedforward_mask=feedforward_mask,
        visible_indices=visible_indices,
        hidden_indices=hidden_indices,
        spike_dataset=spike_dataset,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        output_path=estep_path,
        n_ff_cell_types=n_ff_cell_types,
    )

    inferred_spikes = zarr.open_group(estep_path, mode="r")["output_spikes"]

    n_neurons_full = len(visible_indices) + len(hidden_indices)
    mstep_model = FeedforwardConductanceLIFNetwork(
        dt=dt,
        projections=make_frozen_chunked_ff_projections(
            rec_weights=perturbed_rec_weights[:, visible_indices]
            * recurrent_mask[:, visible_indices].astype(np.float32),
            ff_weights=perturbed_ff_weights[:, visible_indices]
            * feedforward_mask[:, visible_indices].astype(np.float32),
            cell_type_indices=cell_type_indices[visible_indices],
            ff_cell_type_indices=feedforward_cell_type_indices,
            cell_type_names=[cp["name"] for cp in recurrent_cell_params],
            ff_cell_type_names=[cp["name"] for cp in feedforward_cell_params],
            scaling_factors=sf.astype(np.float32),
        ),
        cell_type_indices=cell_type_indices[visible_indices],
        cell_type_indices_FF=concatenated_cell_type_indices,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        track_variables=False,
    )
    mstep_model.to(device)
    mstep_model.eval()
    mstep_model.reset_state(batch_size=batch_size)
    van_rossum_loss_fn.reset_state()

    visible_tensor = torch.from_numpy(visible_indices).long()
    hidden_tensor = torch.from_numpy(hidden_indices).long()

    dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        num_workers=0,
    )
    data_iter = iter(dataloader)

    total_chunks = burn_in_chunks + n_eval_chunks
    losses = []
    student_spike_counts = np.zeros(len(visible_indices))
    teacher_spike_counts = np.zeros(len(visible_indices))

    with torch.inference_mode():
        for chunk_idx in tqdm(range(total_chunks), desc="  M-step", unit="chunk"):
            batch = next(data_iter)
            time_steps = batch.input_spikes.shape[1]

            recurrent_inputs = torch.zeros(
                batch_size,
                time_steps,
                n_neurons_full,
                device=device,
                dtype=torch.float32,
            )
            recurrent_inputs[:, :, visible_tensor] = (
                batch.target_spikes[:, :, visible_tensor].float().to(device)
            )
            start_t = chunk_idx * chunk_size
            recurrent_inputs[:, :, hidden_tensor] = (
                torch.from_numpy(
                    np.array(
                        inferred_spikes[:batch_size, start_t : start_t + time_steps, :]
                    )
                )
                .float()
                .to(device)
            )

            model_input = torch.cat(
                [batch.input_spikes.float().to(device), recurrent_inputs], dim=2
            )
            spikes = mstep_model.forward(input_spikes=model_input)
            targets = batch.target_spikes[:, :, visible_tensor].float().to(device)

            # Burn-in chunks still pass through the loss so its filter state is
            # warmed up, exactly as SNNTrainer does — the value is discarded.
            loss = van_rossum_loss_fn(output_spikes=spikes, target_spikes=targets)
            if chunk_idx >= burn_in_chunks:
                losses.append(float(loss))
                student_spike_counts += spikes[0].sum(dim=0).cpu().numpy()
                teacher_spike_counts += targets[0].sum(dim=0).cpu().numpy()

    duration_s = n_eval_chunks * chunk_size * dt / 1000.0
    student_rates = student_spike_counts / duration_s
    teacher_rates = teacher_spike_counts / duration_s
    visible_cell_types = cell_type_indices[visible_indices]

    metrics = {
        "van_rossum_loss": float(np.mean(losses)),
        "van_rossum_loss_std": float(np.std(losses)),
        "firing_rate/student_visible_mean": float(student_rates.mean()),
        "firing_rate/teacher_visible_mean": float(teacher_rates.mean()),
    }
    for type_idx, type_name in enumerate(cp["name"] for cp in recurrent_cell_params):
        mask = visible_cell_types == type_idx
        metrics[f"firing_rate/student_visible_{type_name}_mean"] = float(
            student_rates[mask].mean()
        )
        metrics[f"firing_rate/teacher_visible_{type_name}_mean"] = float(
            teacher_rates[mask].mean()
        )

    shutil.rmtree(estep_path)
    del mstep_model
    if device == "cuda":
        torch.cuda.empty_cache()

    return metrics, np.array(losses)


def main(input_dir, output_dir, params_file, wandb_config=None, resume_from=None):
    """Compute inference losses for learnt and correct scaling factors.

    Args:
        input_dir (Path): Contains network_structure.npz, spike_data.zarr and
            training_runs/ (symlink to the increasing-hidden-fraction grid).
        output_dir (Path): Directory where outputs are saved.
        params_file (Path): Path to the parameters TOML file.
        wandb_config (dict, optional): Not used.
        resume_from (Path, optional): Not used.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(params_file, "r") as f:
        data = toml.load(f)

    simulation = StudentSimulationConfig(**data["simulation"])
    training = EMTrainingConfig(**data["training"])
    hyperparameters = StudentHyperparameters(**data["hyperparameters"])
    recurrent = RecurrentLayerConfig(**data["recurrent"])
    feedforward = FeedforwardLayerConfig(**data["feedforward"])
    bias_check = data.get("bias_check", {})

    chunk_size = simulation.chunk_size
    surrgrad_scale = hyperparameters.surrgrad_scale
    burn_in_chunks = training.burn_in_chunks

    # ================================
    # Load Network Structure
    # ================================

    network_structure = np.load(input_dir / "network_structure.npz")
    weights = network_structure["recurrent_weights"]
    feedforward_weights = network_structure["feedforward_weights"]
    cell_type_indices = network_structure["cell_type_indices"]
    feedforward_cell_type_indices = network_structure["feedforward_cell_type_indices"]
    recurrent_mask = network_structure["recurrent_connectivity"]
    feedforward_mask = network_structure["feedforward_connectivity"]

    n_neurons_full = weights.shape[0]
    n_feedforward = feedforward_weights.shape[0]

    # ===============================================
    # Cell and Synapse Parameters
    # ===============================================

    recurrent_cell_params = recurrent.get_cell_params()
    feedforward_cell_params = feedforward.get_cell_params()
    n_ff_cell_types = len(feedforward_cell_params)
    recurrent_synapse_params = recurrent.get_synapse_params()
    feedforward_synapse_params = feedforward.get_synapse_params()
    n_ff_synapse_types = len(feedforward_synapse_params)

    input_cell_type_names = feedforward.cell_types.names + recurrent.cell_types.names
    output_cell_type_names = recurrent.cell_types.names

    concatenated_cell_type_indices = np.concatenate(
        [feedforward_cell_type_indices, cell_type_indices + n_ff_cell_types]
    )

    combined_cell_params_FF = feedforward_cell_params.copy()
    for cp in recurrent_cell_params:
        offset_cp = cp.copy()
        offset_cp["cell_id"] = cp["cell_id"] + n_ff_cell_types
        combined_cell_params_FF.append(offset_cp)

    combined_synapse_params_FF = feedforward_synapse_params.copy()
    for sp in recurrent_synapse_params:
        offset_sp = sp.copy()
        offset_sp["cell_id"] = sp["cell_id"] + n_ff_cell_types
        offset_sp["synapse_id"] = sp["synapse_id"] + n_ff_synapse_types
        combined_synapse_params_FF.append(offset_sp)

    concatenated_weights = np.concatenate([feedforward_weights, weights], axis=0)

    # ======================
    # Load Dataset
    # ======================

    spike_dataset = ExactFFDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=chunk_size,
        device=device,
    )
    dt = spike_dataset.dt
    num_chunks = spike_dataset.num_chunks

    eval_batch_size = bias_check.get("eval_batch_size", 0) or spike_dataset.batch_size
    if eval_batch_size < spike_dataset.batch_size:
        spike_dataset = SubBatchDataset(spike_dataset, eval_batch_size)

    n_eval_chunks = bias_check.get("n_eval_chunks", 0) or (num_chunks - burn_in_chunks)
    if burn_in_chunks + n_eval_chunks > num_chunks:
        raise ValueError(
            f"burn_in_chunks ({burn_in_chunks}) + n_eval_chunks ({n_eval_chunks}) "
            f"exceeds the {num_chunks} chunks in the dataset."
        )

    print(
        f"Loaded {num_chunks} chunks x {spike_dataset.batch_size} batch size "
        f"({burn_in_chunks} burn-in + {n_eval_chunks} evaluated)"
    )

    van_rossum_loss_fn = VanRossumLoss(
        tau_rise=hyperparameters.van_rossum_tau_rise,
        tau_decay=hyperparameters.van_rossum_tau_decay,
        dt=dt,
        window_size=chunk_size,
        device=device,
    )

    # ======================================
    # Sweep Training Runs
    # ======================================

    training_runs_dir = input_dir / "training_runs"
    run_dirs = sorted(
        d
        for d in training_runs_dir.iterdir()
        if (d / "targets" / "hidden_neurons.npz").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"No completed training runs found in {training_runs_dir}"
        )
    print(f"Found {len(run_dirs)} training runs: {[d.name for d in run_dirs]}")

    rows = []
    scaling_factor_record = {}
    loss_traces = {}

    for run_dir in run_dirs:
        hidden_neurons = np.load(run_dir / "targets" / "hidden_neurons.npz")
        hidden_indices = hidden_neurons["hidden_indices"]
        visible_indices = hidden_neurons["visible_indices"]
        hidden_cell_fraction = float(hidden_neurons["hidden_cell_fraction"])

        if int(hidden_neurons["n_neurons_full"]) != n_neurons_full:
            raise ValueError(
                f"{run_dir.name} was trained on {int(hidden_neurons['n_neurons_full'])} "
                f"neurons but the teacher data has {n_neurons_full}."
            )

        target_sf = np.load(run_dir / "targets" / "target_scaling_factors.npz")[
            "feedforward_scaling_factors"
        ]
        learnt_sf, training_min_loss, training_min_epoch = learnt_sf_at_minimum(
            run_dir, target_sf, input_cell_type_names, output_cell_type_names
        )

        # The perturbation applied to the weights is the reciprocal of the target
        # scaling factors, so the correct scaling factors exactly undo it.
        perturbation = 1.0 / target_sf
        perturbed_weights = concatenated_weights * perturbation[
            concatenated_cell_type_indices[:, None], cell_type_indices[None, :]
        ].astype(np.float32)
        perturbed_ff_weights = perturbed_weights[:n_feedforward, :]
        perturbed_rec_weights = perturbed_weights[n_feedforward:, :]

        header = (
            f"{run_dir.name}  (hidden fraction {hidden_cell_fraction:.2f}, "
            f"{len(hidden_indices)} hidden / {len(visible_indices)} visible)"
        )
        print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}")
        print(
            f"  Training minimum: {training_min_loss:.4f} at epoch {training_min_epoch}"
        )

        for sf_source in SF_SOURCES:
            sf = learnt_sf if sf_source == "learnt" else target_sf
            print(f"\n--- {sf_source} scaling factors ---")
            print(f"  normalised: {np.array2string(sf / target_sf, precision=4)}")

            metrics, losses = evaluate_scaling_factors(
                sf,
                dt=dt,
                batch_size=spike_dataset.batch_size,
                device=device,
                perturbed_rec_weights=perturbed_rec_weights,
                perturbed_ff_weights=perturbed_ff_weights,
                cell_type_indices=cell_type_indices,
                feedforward_cell_type_indices=feedforward_cell_type_indices,
                recurrent_cell_params=recurrent_cell_params,
                feedforward_cell_params=feedforward_cell_params,
                recurrent_synapse_params=recurrent_synapse_params,
                feedforward_synapse_params=feedforward_synapse_params,
                combined_cell_params_FF=combined_cell_params_FF,
                combined_synapse_params_FF=combined_synapse_params_FF,
                concatenated_cell_type_indices=concatenated_cell_type_indices,
                recurrent_mask=recurrent_mask,
                feedforward_mask=feedforward_mask,
                visible_indices=visible_indices,
                hidden_indices=hidden_indices,
                spike_dataset=spike_dataset,
                num_chunks=num_chunks,
                chunk_size=chunk_size,
                n_ff_cell_types=n_ff_cell_types,
                surrgrad_scale=surrgrad_scale,
                van_rossum_loss_fn=van_rossum_loss_fn,
                burn_in_chunks=burn_in_chunks,
                n_eval_chunks=n_eval_chunks,
                estep_path=output_dir / "_temp_estep.zarr",
            )
            print(f"  van Rossum loss: {metrics['van_rossum_loss']:.4f}")

            key = f"{run_dir.name}__{sf_source}"
            scaling_factor_record[key] = sf
            loss_traces[key] = losses
            rows.append(
                {
                    "run": run_dir.name,
                    "hidden_cell_fraction": hidden_cell_fraction,
                    "sf_source": sf_source,
                    "n_hidden": len(hidden_indices),
                    "n_visible": len(visible_indices),
                    "training_min_loss": training_min_loss,
                    "training_min_epoch": training_min_epoch,
                    **metrics,
                }
            )

    # ======================
    # Save Results
    # ======================

    results = pd.DataFrame(rows).sort_values(["hidden_cell_fraction", "sf_source"])
    results.to_csv(output_dir / "inference_losses.csv", index=False)

    np.savez(
        output_dir / "scaling_factors.npz",
        input_cell_type_names=np.array(input_cell_type_names),
        output_cell_type_names=np.array(output_cell_type_names),
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
