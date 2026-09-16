"""
Fully-recurrent training with surrogate gradient backpropagation.

Instead of alternating E-step (hidden inference) and M-step (parameter update),
this approach trains the full recurrent network directly. Scaling factors are
optimized by backpropagating through the entire recurrent dynamics using
surrogate gradients. Loss is computed on visible neurons only.

This bypasses the EM instability caused by chaotic hidden->hidden recurrence:
there is no iterative E-step/M-step alternation, so there is no divergent
fixed-point iteration.
"""

from pathlib import Path
import sys

import numpy as np
import toml
import torch
import wandb
import zarr
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))  # scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))  # hidden-activity/

from connectome_snns.configs import (
    DATALOADER_KWARGS,
    StudentHyperparameters,
    StudentSimulationConfig,
)
from connectome_snns.configs.conductance_based import FeedforwardLayerConfig, RecurrentLayerConfig
from collate import VisibleTargetCollate
from connectome_snns.dataloaders.supervised import CyclicSampler, ExactFFDataset
from connectome_snns.network_simulators.conductance_based.simulator import (
    ConductanceLIFNetwork,
)
from connectome_snns.network_simulators.projections import (
    make_scaling_factor_projections,
)
from connectome_snns.snn_runners import SNNTrainer, EvolutionarySearch
from connectome_snns.training_utils import AsyncLogger
from connectome_snns.training_utils.losses import VanRossumLoss
from connectome_snns.visualization import plot_spike_trains


class VisibleOnlyLoss:
    """Wraps a loss function to only compute on visible neurons."""

    required_inputs = ["output_spikes"]
    requires_target = True

    def __init__(self, loss_fn, visible_indices):
        self.loss_fn = loss_fn
        self.visible_indices = visible_indices
        self._visible_idx = None

    def __call__(self, output_spikes, target_spikes):
        if (
            self._visible_idx is None
            or self._visible_idx.device != output_spikes.device
        ):
            self._visible_idx = (
                torch.from_numpy(self.visible_indices).long().to(output_spikes.device)
            )
        visible_output = output_spikes[:, :, self._visible_idx]
        return self.loss_fn(output_spikes=visible_output, target_spikes=target_spikes)

    def reset_state(self):
        self.loss_fn.reset_state()


def _bake_sf_into_projections(rec_projs, ff_projs, sf_rec, sf_ff, rec_names, ff_names):
    """Copy per-pair scaling factors into each ScalingFactorProjection's log_sf."""
    with torch.no_grad():
        for (src_name, tgt_name), proj in rec_projs.items():
            src_id = rec_names.index(src_name)
            tgt_id = rec_names.index(tgt_name)
            proj.log_sf.copy_(
                torch.tensor(
                    float(np.log(sf_rec[src_id, tgt_id])),
                    dtype=proj.log_sf.dtype,
                    device=proj.log_sf.device,
                )
            )
        for (src_name, tgt_name), proj in ff_projs.items():
            src_id = ff_names.index(src_name)
            tgt_id = rec_names.index(tgt_name)
            proj.log_sf.copy_(
                torch.tensor(
                    float(np.log(sf_ff[src_id, tgt_id])),
                    dtype=proj.log_sf.dtype,
                    device=proj.log_sf.device,
                )
            )


def main(input_dir, output_dir, params_file, wandb_config=None, resume_from=None):
    print("\n" + "=" * 60)
    print("Fully-Recurrent Training (Surrogate Gradient Backpropagation)")
    print("=" * 60 + "\n")

    # ======================================
    # Device Selection and Parameter Loading
    # ======================================

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

    print("\nHidden cell configuration:")
    print(f"  Total: {n_neurons_full}, Hidden: {n_hidden}, Visible: {n_visible}")

    # ===============================================
    # Cell and Synapse Parameters
    # ===============================================

    recurrent_cell_params = recurrent.get_cell_params()
    feedforward_cell_params = feedforward.get_cell_params()
    n_ff_cell_types = len(feedforward_cell_params)

    recurrent_synapse_params = recurrent.get_synapse_params()
    feedforward_synapse_params = feedforward.get_synapse_params()

    # ============================================
    # Scaling Factors and Perturbation
    # ============================================

    sf_feedforward = np.array(scaling_factors_cfg["feedforward"])
    sf_recurrent = np.array(scaling_factors_cfg["recurrent"])

    concatenated_scaling_factors = np.concatenate(
        [sf_feedforward, sf_recurrent], axis=0
    )

    # Apply perturbation to weights
    sigma = np.sqrt(weight_perturbation_variance)
    mu = -(sigma**2) / 2.0
    perturbation_factors = np.random.lognormal(
        mean=mu, sigma=sigma, size=concatenated_scaling_factors.shape
    )
    target_scaling_factors = 1.0 / perturbation_factors

    # ===============================================
    # Combined Input Cell Types
    # ===============================================

    concatenated_cell_type_indices = np.concatenate(
        [feedforward_cell_type_indices, cell_type_indices + n_ff_cell_types]
    )

    # ===============================================
    # Perturb Weights
    # ===============================================

    n_total_inputs = n_feedforward + n_neurons_full
    concatenated_weights_full = np.concatenate([feedforward_weights, weights], axis=0)

    perturbed_weights_full = concatenated_weights_full.copy()
    for input_idx in range(n_total_inputs):
        input_type = concatenated_cell_type_indices[input_idx]
        for output_idx in range(n_neurons_full):
            output_type = cell_type_indices[output_idx]
            perturbed_weights_full[input_idx, output_idx] *= perturbation_factors[
                input_type, output_type
            ]

    perturbed_ff_weights = perturbed_weights_full[:n_feedforward, :]
    perturbed_rec_weights = perturbed_weights_full[n_feedforward:, :]

    # ===============================================
    # Save Targets
    # ===============================================

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

    # ==============================================
    # Common Model Keyword Arguments
    # ==============================================

    model_kwargs = dict(
        dt=dt,
        cell_type_indices=cell_type_indices,
        cell_type_indices_FF=feedforward_cell_type_indices,
        cell_params=recurrent_cell_params,
        cell_params_FF=feedforward_cell_params,
        synapse_params=recurrent_synapse_params,
        synapse_params_FF=feedforward_synapse_params,
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        track_variables=False,
    )

    rec_cell_type_names = recurrent.cell_types.names
    ff_cell_type_names = feedforward.cell_types.names

    # ==============================
    # Setup Loss Functions
    # ==============================

    van_rossum_loss_fn = VanRossumLoss(
        tau_rise=van_rossum_tau_rise,
        tau_decay=van_rossum_tau_decay,
        dt=dt,
        window_size=chunk_size,
        device=device,
    )

    visible_loss = VisibleOnlyLoss(van_rossum_loss_fn, visible_indices)

    cma_loss_weights = {"van_rossum": loss_weight_van_rossum}

    gradient_loss_functions = {"van_rossum": visible_loss}
    gradient_loss_weights = {"van_rossum": loss_weight_van_rossum}

    # ================================================
    # Initialise wandb
    # ================================================

    wandb_logger = None
    if wandb_config:
        run_name = wandb_config.pop("name", None) or output_dir.name
        wandb_config_dict = {
            **wandb_config,
            "config": {
                "strategy": "fully_recurrent",
                "hidden_cell_fraction": hidden_cell_fraction,
                "n_hidden": n_hidden,
                "n_visible": n_visible,
                "total_epochs": total_epochs,
            },
        }
        wandb_logger = wandb.init(
            name=run_name,
            dir=str(output_dir),
            **wandb_config_dict,
        )
        wandb.define_metric("epoch")

    # ===============================================
    # Cell Type Names (for logging)
    # ===============================================

    input_cell_type_names = ff_cell_type_names + rec_cell_type_names
    output_cell_type_names = rec_cell_type_names

    # Metrics logger
    metrics_logger = AsyncLogger(log_dir=output_dir, max_queue_size=10)

    # ================================================
    # Phase 1: CMA-ES Evolutionary Search
    # ================================================

    best_sf_recurrent = sf_recurrent.copy()
    best_sf_feedforward = sf_feedforward.copy()

    if cma_es_config:
        header = "PHASE 1: CMA-ES Evolutionary Search"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))

        cma_rec_projs, cma_ff_projs = make_scaling_factor_projections(
            rec_weights=perturbed_rec_weights,
            ff_weights=perturbed_ff_weights,
            cell_type_indices=cell_type_indices,
            ff_cell_type_indices=feedforward_cell_type_indices,
            cell_type_names=rec_cell_type_names,
            ff_cell_type_names=ff_cell_type_names,
            init_sf=1.0,
        )
        _bake_sf_into_projections(
            cma_rec_projs,
            cma_ff_projs,
            sf_recurrent,
            sf_feedforward,
            rec_cell_type_names,
            ff_cell_type_names,
        )

        cma_model = ConductanceLIFNetwork(
            **model_kwargs,
            rec_projections=cma_rec_projs,
            ff_projections=cma_ff_projs,
        )
        cma_model.to(device)

        sf_ff_shape = sf_feedforward.shape
        sf_rec_shape = sf_recurrent.shape
        n_ff_params = sf_feedforward.size

        def cma_update_fn(flat_log_params: np.ndarray) -> None:
            params = np.exp(flat_log_params).astype(np.float32)
            sf_ff = params[:n_ff_params].reshape(sf_ff_shape)
            sf_rec = params[n_ff_params:].reshape(sf_rec_shape)
            _bake_sf_into_projections(
                cma_rec_projs,
                cma_ff_projs,
                sf_rec,
                sf_ff,
                rec_cell_type_names,
                ff_cell_type_names,
            )

        cma_van_rossum_fn = VanRossumLoss(
            tau_rise=van_rossum_tau_rise,
            tau_decay=van_rossum_tau_decay,
            dt=dt,
            window_size=chunk_size,
            device=device,
        )
        cma_visible_loss = VisibleOnlyLoss(cma_van_rossum_fn, visible_indices)
        cma_loss_functions = {"van_rossum": cma_visible_loss}

        # CMA-ES model is a full ConductanceLIFNetwork with its own recurrence,
        # so it only needs FF inputs. Targets are visible-only.
        cma_collate = VisibleTargetCollate(torch.from_numpy(visible_indices).long())

        cma_dataloader = DataLoader(
            spike_dataset,
            batch_size=None,
            sampler=CyclicSampler(spike_dataset),
            **DATALOADER_KWARGS,
            collate_fn=cma_collate,
        )

        initial_params = np.log(
            np.concatenate([sf_feedforward.ravel(), sf_recurrent.ravel()])
        )

        # Wandb + disk callback for CMA-ES generation logging
        def cma_es_callback(metrics):
            log_dict = {
                "cma_es/best_loss": metrics["best_loss"],
                "cma_es/sigma": metrics["sigma"],
                "cma_es/n_evals": metrics["n_evals"],
            }
            for loss_name, loss_val in metrics["mean_losses"].items():
                log_dict[f"cma_es_loss/{loss_name}"] = loss_val

            # Scaling factors vs targets
            sf_ff = cma_model.scaling_factors_FF.detach().cpu().numpy()
            sf_rec = cma_model.scaling_factors.detach().cpu().numpy()
            current_sf = np.concatenate([sf_ff, sf_rec], axis=0)
            for source_idx in range(current_sf.shape[0]):
                source_name = input_cell_type_names[source_idx]
                for target_idx in range(current_sf.shape[1]):
                    target_name = output_cell_type_names[target_idx]
                    synapse_name = f"{source_name}_to_{target_name}"
                    target_val = target_scaling_factors[source_idx, target_idx]
                    if target_val != 0:
                        normalized = current_sf[source_idx, target_idx] / target_val
                    else:
                        normalized = current_sf[source_idx, target_idx]
                    log_dict[f"cma_es_scaling_factors/{synapse_name}_value"] = float(
                        normalized
                    )
                    log_dict[f"cma_es_scaling_factors/{synapse_name}_target"] = 1.0

            # Firing rates from output spikes (all neurons)
            output_spikes = metrics["output_spikes"].numpy()
            student_spikes = output_spikes[0]  # (time, n_neurons_full)
            duration_s = student_spikes.shape[0] * dt / 1000.0
            student_rates = student_spikes.sum(axis=0) / duration_s
            visible_cell_types = cell_type_indices[visible_indices]
            hidden_cell_types = cell_type_indices[hidden_indices]
            for type_idx, type_name in enumerate(output_cell_type_names):
                vis_mask = visible_cell_types == type_idx
                hid_mask = hidden_cell_types == type_idx
                vis_rates = student_rates[visible_indices]
                hid_rates = student_rates[hidden_indices]
                if vis_mask.sum() > 0:
                    log_dict[f"cma_es_firing_rate/student_visible_{type_name}_mean"] = (
                        float(vis_rates[vis_mask].mean())
                    )
                if hid_mask.sum() > 0:
                    log_dict[f"cma_es_firing_rate/student_hidden_{type_name}_mean"] = (
                        float(hid_rates[hid_mask].mean())
                    )

            # Log to disk
            metrics_logger.log(epoch=-metrics["generation"], **log_dict)

            # Log to wandb
            if wandb_logger:
                wandb.log(log_dict)

        searcher = EvolutionarySearch(
            model=cma_model,
            initial_params=initial_params,
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

        best_sf_feedforward = cma_model.scaling_factors_FF.detach().cpu().numpy()
        best_sf_recurrent = cma_model.scaling_factors.detach().cpu().numpy()

        print(f"\nCMA-ES best loss: {best_cma_loss:.6f}")

        cma_state_dir = output_dir / "cma_es_state"
        cma_state_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            cma_state_dir / "scaling_factors.npz",
            scaling_factors_FF=best_sf_feedforward,
            scaling_factors=best_sf_recurrent,
            best_loss=best_cma_loss,
        )

        del cma_model

    # ======================================
    # Create Gradient Training Model (Phase 2)
    # ======================================

    gradient_rec_projs, gradient_ff_projs = make_scaling_factor_projections(
        rec_weights=perturbed_rec_weights,
        ff_weights=perturbed_ff_weights,
        cell_type_indices=cell_type_indices,
        ff_cell_type_indices=feedforward_cell_type_indices,
        cell_type_names=rec_cell_type_names,
        ff_cell_type_names=ff_cell_type_names,
        init_sf=1.0,
    )
    _bake_sf_into_projections(
        gradient_rec_projs,
        gradient_ff_projs,
        best_sf_recurrent,
        best_sf_feedforward,
        rec_cell_type_names,
        ff_cell_type_names,
    )

    model = ConductanceLIFNetwork(
        **model_kwargs,
        rec_projections=gradient_rec_projs,
        ff_projections=gradient_ff_projs,
    )
    model.to(device)

    print(
        f"Model: ConductanceLIFNetwork ({n_neurons_full} neurons, optimising scaling_factors)"
    )

    # ==============================
    # Collate Function
    # ==============================

    visible_tensor = torch.from_numpy(visible_indices).long()
    collate_fn = VisibleTargetCollate(visible_tensor)

    # ==============================
    # Teacher Data for Stats (hidden neuron rates need zarr)
    # ==============================

    # Hidden teacher firing rates require the full teacher zarr because
    # target_spikes from the collate only contains visible neurons.
    teacher_zarr_root = zarr.open_group(input_dir / "spike_data.zarr", mode="r")
    teacher_spikes_zarr = teacher_zarr_root["output_spikes"]

    # ==============================
    # Stats Computer
    # ==============================

    def stats_computer(snapshot):
        """Compute summary statistics for logging."""
        # spikes shape: (1, time, n_neurons_full) — model outputs all neurons
        student_spikes = snapshot["spikes"][0, :, :]
        n_timesteps = student_spikes.shape[0]
        duration_s = n_timesteps * dt / 1000.0

        # Visible teacher spikes from collate (already filtered to visible neurons)
        teacher_visible_spikes = snapshot["target_spikes"][0, :n_timesteps, :]
        teacher_visible_rates = teacher_visible_spikes.sum(axis=0) / duration_s

        # Hidden teacher spikes: use epoch from trainer to compute correct zarr window
        epoch = snapshot.get("epoch", 0)
        n_chunks_accumulated = n_timesteps // chunk_size
        start_chunk = (epoch - n_chunks_accumulated + 1) % num_chunks
        start_t = start_chunk * chunk_size
        end_t = start_t + n_timesteps

        total_timesteps = num_chunks * chunk_size
        if end_t <= total_timesteps:
            teacher_hidden_spikes = np.array(
                teacher_spikes_zarr[0, start_t:end_t, hidden_indices]
            )
        else:
            part1 = np.array(
                teacher_spikes_zarr[0, start_t:total_timesteps, hidden_indices]
            )
            part2 = np.array(
                teacher_spikes_zarr[0, : end_t - total_timesteps, hidden_indices]
            )
            teacher_hidden_spikes = np.concatenate([part1, part2], axis=0)
        teacher_hidden_rates = teacher_hidden_spikes.sum(axis=0) / duration_s

        # Student firing rates — directly from model output (all neurons)
        student_full_rates = student_spikes.sum(axis=0) / duration_s

        stats = {}

        # Visible neuron firing rates (teacher from collate target_spikes)
        visible_ct = cell_type_indices[visible_indices]
        for type_idx, type_name in enumerate(output_cell_type_names):
            mask = visible_ct == type_idx
            if mask.sum() > 0:
                stats[f"firing_rate/teacher_visible_{type_name}_mean"] = float(
                    teacher_visible_rates[mask].mean()
                )
                stats[f"firing_rate/teacher_visible_{type_name}_std"] = float(
                    teacher_visible_rates[mask].std()
                )
                student_vis_rates = student_full_rates[visible_indices]
                stats[f"firing_rate/student_visible_{type_name}_mean"] = float(
                    student_vis_rates[mask].mean()
                )
                stats[f"firing_rate/student_visible_{type_name}_std"] = float(
                    student_vis_rates[mask].std()
                )

        # Hidden neuron firing rates (teacher from zarr)
        hidden_ct = cell_type_indices[hidden_indices]
        for type_idx, type_name in enumerate(output_cell_type_names):
            mask = hidden_ct == type_idx
            if mask.sum() > 0:
                stats[f"firing_rate/teacher_hidden_{type_name}_mean"] = float(
                    teacher_hidden_rates[mask].mean()
                )
                stats[f"firing_rate/teacher_hidden_{type_name}_std"] = float(
                    teacher_hidden_rates[mask].std()
                )
                student_hid_rates = student_full_rates[hidden_indices]
                stats[f"firing_rate/student_hidden_{type_name}_mean"] = float(
                    student_hid_rates[mask].mean()
                )
                stats[f"firing_rate/student_hidden_{type_name}_std"] = float(
                    student_hid_rates[mask].std()
                )

        # Scaling factors (normalized so target=1)
        # Concatenate FF + recurrent SFs to match EM format
        sf_ff = snapshot["scaling_factors_FF"]
        sf_rec = snapshot["scaling_factors"]
        current_sf = np.concatenate([sf_ff, sf_rec], axis=0)

        for source_idx in range(current_sf.shape[0]):
            source_type_name = input_cell_type_names[source_idx]
            for target_idx in range(current_sf.shape[1]):
                target_type_name = output_cell_type_names[target_idx]
                synapse_name = f"{source_type_name}_to_{target_type_name}"
                target_val = target_scaling_factors[source_idx, target_idx]
                if target_val != 0:
                    normalized_value = current_sf[source_idx, target_idx] / target_val
                else:
                    normalized_value = current_sf[source_idx, target_idx]
                stats[f"scaling_factors/{synapse_name}_value"] = float(normalized_value)
                stats[f"scaling_factors/{synapse_name}_target"] = 1.0

        return stats

    # ==============================
    # Training Setup
    # ==============================

    total_chunks = total_epochs * num_chunks
    print(
        f"\nTraining: {total_epochs} epochs x {num_chunks} chunks = {total_chunks} total chunks"
    )

    # Optimizer
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

    # Dataloader
    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        **DATALOADER_KWARGS,
        collate_fn=collate_fn,
    )

    # Progress bar
    pbar = tqdm(range(total_chunks), desc="Fully-recurrent", unit="chunk")

    # Plot generator for checkpoint spike comparisons
    def plot_generator(spikes, target_spikes, input_spikes, epoch=0, **kwargs):
        # spikes: (1, time, n_neurons_full), target_spikes: (1, time, n_visible)
        figures = {}

        # --- Visible neuron comparison ---
        visible_spikes = spikes[:, :, visible_indices]
        n_plot_vis = min(10, visible_spikes.shape[2])
        interleaved_vis = np.zeros((1, spikes.shape[1], 2 * n_plot_vis))
        for i in range(n_plot_vis):
            interleaved_vis[0, :, 2 * i] = target_spikes[0, :, i]
            interleaved_vis[0, :, 2 * i + 1] = visible_spikes[0, :, i]

        figures["spike_comparison_visible"] = plot_spike_trains(
            spikes=interleaved_vis,
            dt=dt,
            cell_type_indices=np.array([0, 1] * n_plot_vis),
            cell_type_names=["Target", "Trained"],
            n_neurons_plot=2 * n_plot_vis,
            n_compared=2,
            fraction=1.0,
            random_seed=None,
            title=f"Fully-recurrent: Target vs Trained ({n_plot_vis} visible)",
            ylabel="Neuron",
            figsize=(14, 8),
        )

        # --- Hidden neuron comparison ---
        if n_hidden > 0:
            hidden_spikes = spikes[:, :, hidden_indices]
            n_plot_hid = min(10, hidden_spikes.shape[2])

            # Load teacher hidden for the same time window as the accumulated
            # plot data.  The trainer accumulates the last plot_size chunks;
            # at trainer epoch `epoch` these correspond to dataloader indices
            # (epoch - n_chunks_in_plot + 1) .. epoch, wrapping via num_chunks.
            n_time = spikes.shape[1]
            n_chunks_in_plot = n_time // chunk_size
            start_chunk = (epoch - n_chunks_in_plot + 1) % num_chunks
            start_t = start_chunk * chunk_size
            end_t = start_t + n_time

            total_timesteps = num_chunks * chunk_size
            if end_t <= total_timesteps:
                teacher_hid = np.array(
                    teacher_spikes_zarr[0, start_t:end_t, hidden_indices]
                )
            else:
                part1 = np.array(
                    teacher_spikes_zarr[0, start_t:total_timesteps, hidden_indices]
                )
                part2 = np.array(
                    teacher_spikes_zarr[0, : end_t - total_timesteps, hidden_indices]
                )
                teacher_hid = np.concatenate([part1, part2], axis=0)

            interleaved_hid = np.zeros((1, n_time, 2 * n_plot_hid))
            for i in range(n_plot_hid):
                interleaved_hid[0, :, 2 * i] = teacher_hid[:, i]
                interleaved_hid[0, :, 2 * i + 1] = hidden_spikes[0, :, i]

            figures["spike_comparison_hidden"] = plot_spike_trains(
                spikes=interleaved_hid,
                dt=dt,
                cell_type_indices=np.array([0, 1] * n_plot_hid),
                cell_type_names=["Teacher", "Student"],
                n_neurons_plot=2 * n_plot_hid,
                n_compared=2,
                fraction=1.0,
                random_seed=None,
                title=f"Fully-recurrent: Teacher vs Student ({n_plot_hid} hidden)",
                ylabel="Neuron",
                figsize=(14, 8),
            )

        return figures

    # Trainer
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

    # ==============================
    # Train
    # ==============================

    best_loss = trainer.train(output_dir=output_dir)

    # ==============================
    # Cleanup
    # ==============================

    print(f"\n{'=' * 60}")
    print("Fully-Recurrent Training Complete")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"{'=' * 60}")

    if metrics_logger:
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
