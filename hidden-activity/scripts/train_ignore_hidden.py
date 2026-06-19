"""
Ignore-hidden training strategy.

No EM. The student is a FeedforwardConductanceLIFNetwork operating on visible
neurons only. Inputs are [FF_spikes, teacher_visible_rec_spikes]; hidden neurons
are absent from the model entirely.

This is the feedforward equivalent of the EM M-step: the teacher provides the
visible recurrent inputs rather than the model generating its own recurrent
activity (as in fully-recurrent). The weight matrix is sliced to visible×visible
recurrence — no hidden rows or columns are allocated or simulated.
"""

from pathlib import Path
import sys

import numpy as np
import toml
import torch
import wandb
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))  # scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))  # hidden-activity/

from configs import DATALOADER_KWARGS, StudentHyperparameters, StudentSimulationConfig
from configs.conductance_based import FeedforwardLayerConfig, RecurrentLayerConfig
from collate import VisibleDrivenCollate
from dataloaders.supervised import CyclicSampler, ExactFFDataset
from network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
from network_simulators.projections import (
    make_chunked_ff_projections,
)
from snn_runners import SNNTrainer, EvolutionarySearch
from training_utils import AsyncLogger
from training_utils.losses import VanRossumLoss
from visualization import plot_spike_trains


def main(input_dir, output_dir, params_file, wandb_config=None, resume_from=None):
    print("\n" + "=" * 60)
    print("Ignore-Hidden Training (Feedforward, Visible Neurons Only)")
    print("=" * 60 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(params_file, "r") as f:
        data = toml.load(f)

    simulation = StudentSimulationConfig(**data["simulation"])
    training_cfg = data["training"]
    hyperparameters = StudentHyperparameters(**data["hyperparameters"])
    recurrent = RecurrentLayerConfig(**data["recurrent"])
    feedforward = FeedforwardLayerConfig(**data["feedforward"])
    scaling_factors_cfg = data.get("scaling_factors", {})
    cma_es_config = data.get("cma_es", {})

    total_epochs = training_cfg["total_epochs"]
    chunk_size = simulation.chunk_size
    seed = simulation.seed
    chunks_per_update = training_cfg["chunks_per_update"]
    log_interval = training_cfg["log_interval"]
    checkpoint_interval = training_cfg["checkpoint_interval"]
    plot_size = training_cfg["plot_size"]
    mixed_precision = training_cfg["mixed_precision"]
    burn_in_chunks = training_cfg.get("burn_in_chunks", 0)
    weight_perturbation_variance = training_cfg["weight_perturbation_variance"]
    beta1 = hyperparameters.beta1
    beta2 = hyperparameters.beta2
    grad_norm_clip = hyperparameters.grad_norm_clip
    learning_rate = hyperparameters.learning_rate
    surrgrad_scale = hyperparameters.surrgrad_scale
    van_rossum_tau_rise = hyperparameters.van_rossum_tau_rise
    van_rossum_tau_decay = hyperparameters.van_rossum_tau_decay
    loss_weight_van_rossum = hyperparameters.loss_weight.van_rossum
    hidden_cell_fraction = data["simulation"].get("hidden_cell_fraction", 0.5)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ================================
    # Load Network Structure
    # ================================

    network_structure = np.load(input_dir / "network_structure.npz")
    weights = network_structure["recurrent_weights"]
    feedforward_weights = network_structure["feedforward_weights"]
    cell_type_indices = network_structure["cell_type_indices"]
    feedforward_cell_type_indices = network_structure["feedforward_cell_type_indices"]

    n_neurons_full = weights.shape[0]
    n_feedforward = feedforward_weights.shape[0]

    # ================================
    # Select Hidden and Visible Neurons
    # ================================

    n_hidden = int(n_neurons_full * hidden_cell_fraction)
    n_visible = n_neurons_full - n_hidden

    if n_hidden > 0:
        all_indices = np.arange(n_neurons_full)
        np.random.shuffle(all_indices)
        hidden_indices = np.sort(all_indices[:n_hidden])
        visible_indices = np.sort(all_indices[n_hidden:])
    else:
        hidden_indices = np.array([], dtype=int)
        visible_indices = np.arange(n_neurons_full)

    print("Hidden cell configuration:")
    print(f"  Total: {n_neurons_full}, Hidden: {n_hidden}, Visible: {n_visible}")

    # ===================================================
    # Cell and Synapse Parameters
    # ===================================================

    recurrent_cell_params = recurrent.get_cell_params()
    feedforward_cell_params = feedforward.get_cell_params()
    n_ff_cell_types = len(feedforward_cell_params)
    n_ff_synapse_types = len(feedforward.get_synapse_params())
    recurrent_synapse_params = recurrent.get_synapse_params()
    feedforward_synapse_params = feedforward.get_synapse_params()
    vis_cell_type_indices = cell_type_indices[visible_indices]

    # Combined FF + visible-rec cell/synapse params (with offset IDs)
    combined_cell_params_FF = feedforward_cell_params.copy()
    for cp in recurrent_cell_params:
        o = cp.copy()
        o["cell_id"] = cp["cell_id"] + n_ff_cell_types
        combined_cell_params_FF.append(o)

    combined_synapse_params_FF = feedforward_synapse_params.copy()
    for sp in recurrent_synapse_params:
        o = sp.copy()
        o["cell_id"] = sp["cell_id"] + n_ff_cell_types
        o["synapse_id"] = sp["synapse_id"] + n_ff_synapse_types
        combined_synapse_params_FF.append(o)

    # Input cell type indices: FF + visible recurrent
    model_input_ct = np.concatenate(
        [feedforward_cell_type_indices, vis_cell_type_indices + n_ff_cell_types]
    )

    # ============================================
    # Scaling Factors and Perturbation
    # ============================================

    sf_feedforward = np.array(scaling_factors_cfg["feedforward"])
    sf_recurrent = np.array(scaling_factors_cfg["recurrent"])
    concatenated_scaling_factors = np.concatenate(
        [sf_feedforward, sf_recurrent], axis=0
    )

    sigma = np.sqrt(weight_perturbation_variance)
    mu = -(sigma**2) / 2.0
    perturbation_factors = np.random.lognormal(
        mean=mu, sigma=sigma, size=concatenated_scaling_factors.shape
    )
    target_scaling_factors = 1.0 / perturbation_factors

    # ===================================================
    # Build Visible-Only Weight Matrix
    # ===================================================

    # Perturb the full concatenated weight matrix
    all_ct = np.concatenate(
        [feedforward_cell_type_indices, cell_type_indices + n_ff_cell_types]
    )
    n_total = n_feedforward + n_neurons_full
    full_weights = np.concatenate([feedforward_weights, weights], axis=0)

    perturbed = full_weights.copy()
    for i in range(n_total):
        it = all_ct[i]
        for j in range(n_neurons_full):
            ot = cell_type_indices[j]
            perturbed[i, j] *= perturbation_factors[it, ot]

    perturbed_ff = perturbed[:n_feedforward, :]
    perturbed_rec = perturbed[n_feedforward:, :]

    # Slice to visible outputs; visible rec inputs only
    vis_ff_w = perturbed_ff[:, visible_indices]
    vis_rec_w = perturbed_rec[visible_indices, :][:, visible_indices]
    model_weights = np.concatenate([vis_ff_w, vis_rec_w], axis=0)

    print(
        f"  Model weight matrix: {model_weights.shape} "
        f"({n_feedforward} FF + {n_visible} vis-rec inputs → {n_visible} visible outputs)"
    )

    # ===================================================
    # Save Targets
    # ===================================================

    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        targets_dir / "target_scaling_factors.npz",
        feedforward_scaling_factors=target_scaling_factors,
    )
    np.savez(
        targets_dir / "hidden_neurons.npz",
        hidden_indices=hidden_indices,
        visible_indices=visible_indices,
        hidden_cell_fraction=hidden_cell_fraction,
        n_neurons_full=n_neurons_full,
    )

    # ======================
    # Load Dataset
    # ======================

    spike_dataset = ExactFFDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=chunk_size,
        device=device,
    )
    batch_size = spike_dataset.batch_size
    dt = spike_dataset.dt
    num_chunks = spike_dataset.num_chunks

    print(f"Loaded {num_chunks} chunks x {batch_size} batch size")

    # Collate: input = [FF, teacher_visible_rec], target = teacher_visible_rec
    visible_tensor = torch.from_numpy(visible_indices).long()
    collate_fn = VisibleDrivenCollate(visible_tensor)

    # ==============================================
    # Common Model Keyword Arguments
    # ==============================================

    model_kwargs = dict(
        dt=dt,
        cell_type_indices=vis_cell_type_indices,
        cell_type_indices_FF=model_input_ct,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        track_variables=False,
    )

    # Combined input cell-type names (for projection keys): FF + recurrent.
    combined_input_cell_type_names = (
        feedforward.cell_types.names + recurrent.cell_types.names
    )
    rec_output_cell_type_names = recurrent.cell_types.names

    # Split model_weights into FF rows and visible-recurrent rows. The
    # chunked-FF projection builder expects them separately.
    model_ff_block = model_weights[:n_feedforward, :]
    model_rec_block = model_weights[n_feedforward:, :]

    # ==============================
    # Loss Functions
    # ==============================

    # Model outputs visible neurons directly — no VisibleOnlyLoss wrapper needed
    van_rossum_loss_fn = VanRossumLoss(
        tau_rise=van_rossum_tau_rise,
        tau_decay=van_rossum_tau_decay,
        dt=dt,
        window_size=chunk_size,
        device=device,
    )
    gradient_loss_functions = {"van_rossum": van_rossum_loss_fn}
    gradient_loss_weights = {"van_rossum": loss_weight_van_rossum}
    cma_loss_weights = {"van_rossum": loss_weight_van_rossum}

    # ================================================
    # Logging Setup
    # ================================================

    input_cell_type_names = feedforward.cell_types.names + recurrent.cell_types.names
    output_cell_type_names = recurrent.cell_types.names
    metrics_logger = AsyncLogger(log_dir=output_dir, max_queue_size=10)

    wandb_logger = None
    if wandb_config:
        run_name = wandb_config.pop("name", None) or output_dir.name
        wandb_logger = wandb.init(
            name=run_name,
            dir=str(output_dir),
            config={
                "strategy": "ignore_hidden",
                "hidden_cell_fraction": hidden_cell_fraction,
                "n_hidden": n_hidden,
                "n_visible": n_visible,
                "total_epochs": total_epochs,
            },
            **wandb_config,
        )
        wandb.define_metric("epoch")

    # ================================================
    # Phase 1: CMA-ES Evolutionary Search
    # ================================================

    best_sf = concatenated_scaling_factors.copy()

    if cma_es_config:
        header = "PHASE 1: CMA-ES Evolutionary Search"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))

        # ScalingFactorProjections for CMA-ES: connectome blocks come from
        # the perturbed visible weight matrix; the configured teacher SFs are
        # baked into each pair's initial ``log_sf``. CMA's update_fn writes
        # candidate SFs back into the same ``log_sf`` buffers.
        cma_projections = make_chunked_ff_projections(
            rec_weights=model_rec_block,
            ff_weights=model_ff_block,
            cell_type_indices=vis_cell_type_indices,
            ff_cell_type_indices=feedforward_cell_type_indices,
            cell_type_names=rec_output_cell_type_names,
            ff_cell_type_names=feedforward.cell_types.names,
            init_sf=1.0,
        )
        with torch.no_grad():
            for (src_name, tgt_name), proj in cma_projections.items():
                src_id = combined_input_cell_type_names.index(src_name)
                tgt_id = rec_output_cell_type_names.index(tgt_name)
                proj.log_sf.copy_(
                    torch.tensor(
                        float(np.log(concatenated_scaling_factors[src_id, tgt_id])),
                        dtype=proj.log_sf.dtype,
                    )
                )

        cma_model = FeedforwardConductanceLIFNetwork(
            **model_kwargs,
            projections=cma_projections,
        )
        cma_model.to(device)

        sf_shape = concatenated_scaling_factors.shape

        def cma_update_fn(flat_log_params: np.ndarray) -> None:
            sf_matrix = np.exp(flat_log_params).reshape(sf_shape).astype(np.float32)
            with torch.no_grad():
                for (src_name, tgt_name), proj in cma_projections.items():
                    src_id = combined_input_cell_type_names.index(src_name)
                    tgt_id = rec_output_cell_type_names.index(tgt_name)
                    proj.log_sf.copy_(
                        torch.tensor(
                            float(np.log(sf_matrix[src_id, tgt_id])),
                            dtype=proj.log_sf.dtype,
                            device=proj.log_sf.device,
                        )
                    )

        cma_van_rossum_fn = VanRossumLoss(
            tau_rise=van_rossum_tau_rise,
            tau_decay=van_rossum_tau_decay,
            dt=dt,
            window_size=chunk_size,
            device=device,
        )
        cma_loss_functions = {"van_rossum": cma_van_rossum_fn}
        cma_dataloader = DataLoader(
            spike_dataset,
            batch_size=None,
            sampler=CyclicSampler(spike_dataset),
            **DATALOADER_KWARGS,
            collate_fn=collate_fn,
        )

        def cma_es_callback(metrics):
            log_dict = {
                "cma_es/best_loss": metrics["best_loss"],
                "cma_es/sigma": metrics["sigma"],
                "cma_es/n_evals": metrics["n_evals"],
            }
            for ln, lv in metrics["mean_losses"].items():
                log_dict[f"cma_es_loss/{ln}"] = lv

            current_sf = cma_model.scaling_factors_FF.detach().cpu().numpy()
            for si in range(current_sf.shape[0]):
                for ti in range(current_sf.shape[1]):
                    name = (
                        f"{input_cell_type_names[si]}_to_{output_cell_type_names[ti]}"
                    )
                    tv = target_scaling_factors[si, ti]
                    log_dict[f"cma_es_scaling_factors/{name}_value"] = float(
                        current_sf[si, ti] / tv if tv != 0 else current_sf[si, ti]
                    )

            out = metrics["output_spikes"].numpy()[0]
            rates = out.sum(axis=0) / (out.shape[0] * dt / 1000.0)
            for ti, tn in enumerate(output_cell_type_names):
                mask = vis_cell_type_indices == ti
                if mask.sum() > 0:
                    log_dict[f"cma_es_firing_rate/student_{tn}_mean"] = float(
                        rates[mask].mean()
                    )

            metrics_logger.log(epoch=-metrics["generation"], **log_dict)
            if wandb_logger:
                wandb.log(log_dict)

        searcher = EvolutionarySearch(
            model=cma_model,
            initial_params=np.log(concatenated_scaling_factors).ravel(),
            update_fn=cma_update_fn,
            dataloader=cma_dataloader,
            loss_functions=cma_loss_functions,
            loss_weights=cma_loss_weights,
            device=device,
            n_chunks=cma_es_config.get("n_chunks", 10),
            initial_sigma=cma_es_config.get("initial_sigma", 0.5),
            max_evaluations=cma_es_config.get("max_evaluations", 200),
            popsize=cma_es_config.get("popsize", None),
            batch_size=cma_es_config.get("batch_size", None),
            seed=cma_es_config.get("seed", None),
            burn_in_chunks=burn_in_chunks,
            callback=cma_es_callback,
        )

        best_cma_loss = searcher.search()
        best_sf = cma_model.scaling_factors_FF.detach().cpu().numpy()

        cma_state_dir = output_dir / "cma_es_state"
        cma_state_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            cma_state_dir / "scaling_factors.npz",
            scaling_factors_FF=best_sf,
            best_loss=best_cma_loss,
        )
        del cma_model
        print(f"\nCMA-ES best loss: {best_cma_loss:.6f}")

    # ======================================
    # Phase 2: Gradient Training
    # ======================================

    # Gradient phase: ScalingFactorProjection per pair; only trainable params
    # are the per-pair log_sf scalars. ``optimisable`` was hardcoded to
    # "scaling_factors" in the old API.
    optimisable = "scaling_factors"
    if optimisable != "scaling_factors":
        raise NotImplementedError(
            f"ignore-hidden supports optimisable='scaling_factors' only; "
            f"got {optimisable!r}"
        )

    gradient_projections = make_chunked_ff_projections(
        rec_weights=model_rec_block,
        ff_weights=model_ff_block,
        cell_type_indices=vis_cell_type_indices,
        ff_cell_type_indices=feedforward_cell_type_indices,
        cell_type_names=rec_output_cell_type_names,
        ff_cell_type_names=feedforward.cell_types.names,
        init_sf=1.0,
    )
    with torch.no_grad():
        for (src_name, tgt_name), proj in gradient_projections.items():
            src_id = combined_input_cell_type_names.index(src_name)
            tgt_id = rec_output_cell_type_names.index(tgt_name)
            proj.log_sf.copy_(
                torch.tensor(
                    float(np.log(best_sf[src_id, tgt_id])),
                    dtype=proj.log_sf.dtype,
                )
            )

    model = FeedforwardConductanceLIFNetwork(
        **model_kwargs,
        projections=gradient_projections,
    )
    model.to(device)
    print(f"\nModel: FeedforwardConductanceLIFNetwork ({n_visible} visible neurons)")

    def stats_computer(snapshot):
        student_spikes = snapshot["spikes"][0, :, :]  # (time, n_visible)
        n_timesteps = student_spikes.shape[0]
        duration_s = n_timesteps * dt / 1000.0

        # target_spikes from collate_fn: already visible-only
        teacher_vis = snapshot["target_spikes"][0, :n_timesteps, :]

        student_rates = student_spikes.sum(axis=0) / duration_s
        teacher_rates = teacher_vis.sum(axis=0) / duration_s

        stats = {}
        for ti, tn in enumerate(output_cell_type_names):
            mask = vis_cell_type_indices == ti
            if mask.sum() > 0:
                stats[f"firing_rate/student_visible_{tn}_mean"] = float(
                    student_rates[mask].mean()
                )
                stats[f"firing_rate/student_visible_{tn}_std"] = float(
                    student_rates[mask].std()
                )
                stats[f"firing_rate/teacher_visible_{tn}_mean"] = float(
                    teacher_rates[mask].mean()
                )
                stats[f"firing_rate/teacher_visible_{tn}_std"] = float(
                    teacher_rates[mask].std()
                )

        current_sf = snapshot["scaling_factors_FF"]
        for si in range(current_sf.shape[0]):
            for ti in range(current_sf.shape[1]):
                name = f"{input_cell_type_names[si]}_to_{output_cell_type_names[ti]}"
                tv = target_scaling_factors[si, ti]
                stats[f"scaling_factors/{name}_value"] = float(
                    current_sf[si, ti] / tv if tv != 0 else current_sf[si, ti]
                )
                stats[f"scaling_factors/{name}_target"] = 1.0
        return stats

    def plot_generator(spikes, target_spikes, input_spikes, **kwargs):
        n_plot = min(10, spikes.shape[2])
        interleaved = np.zeros((1, spikes.shape[1], 2 * n_plot))
        for i in range(n_plot):
            interleaved[0, :, 2 * i] = target_spikes[0, :, i]
            interleaved[0, :, 2 * i + 1] = spikes[0, :, i]
        fig = plot_spike_trains(
            spikes=interleaved,
            dt=dt,
            cell_type_indices=np.array([0, 1] * n_plot),
            cell_type_names=["Target", "Student"],
            n_neurons_plot=2 * n_plot,
            n_compared=2,
            fraction=1.0,
            random_seed=None,
            title=f"Ignore-Hidden: Target vs Student ({n_plot} visible neurons)",
            ylabel="Neuron",
            figsize=(14, 8),
        )
        return {"spike_comparison_visible": fig}

    total_chunks = total_epochs * num_chunks
    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(beta1, beta2)
    )
    lr_min = getattr(hyperparameters, "lr_min", None)
    if lr_min is not None:
        t_max = total_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=t_max, eta_min=lr_min
        )
        print(
            f"  LR schedule: cosine {learning_rate} → {lr_min} over {total_epochs} epochs"
        )
    else:
        scheduler = None
    scaler = GradScaler("cuda", enabled=mixed_precision and device == "cuda")

    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        **DATALOADER_KWARGS,
        collate_fn=collate_fn,
    )

    pbar = tqdm(range(total_chunks), desc="Ignore-hidden", unit="chunk")

    trainer = SNNTrainer(
        model=model,
        optimizer=optimiser,
        scaler=scaler,
        dataloader=spike_dataloader,
        loss_functions=gradient_loss_functions,
        loss_weights=gradient_loss_weights,
        device=device,
        num_epochs=total_chunks,
        chunks_per_update=chunks_per_update,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
        plot_size=plot_size,
        mixed_precision=mixed_precision,
        grad_norm_clip=grad_norm_clip,
        wandb_config=None,
        progress_bar=pbar,
        plot_generator=plot_generator,
        stats_computer=stats_computer,
        chunks_per_data_epoch=num_chunks,
        burn_in_chunks=burn_in_chunks,
        scheduler=scheduler,
    )

    if wandb_logger:
        trainer.wandb_logger = wandb_logger
    trainer.metrics_logger = metrics_logger

    best_loss = trainer.train(output_dir=output_dir)

    print(f"\n{'=' * 60}")
    print("Ignore-Hidden Training Complete")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"{'=' * 60}")

    metrics_logger.close()
    if wandb_logger:
        wandb.finish()

    return best_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--params_file", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.params_file)
