"""
Visible-driven training with surrogate gradient backpropagation.

Uses TwoLayerSNN: a recurrent hidden layer (ConductanceLIFNetwork) followed
by a feedforward visible layer (FeedforwardConductanceLIFNetwork). Teacher
visible spikes are injected as feedforward input to both layers. Gradients
flow from the visible loss through the hidden→visible feedforward path to
FF→hidden weights.

Architecture:
  Layer 1 (hidden): [FF spikes, teacher visible spikes] → hidden spikes
    - Hidden→hidden recurrence kept (detached gradients)
    - FF→hidden and visible→hidden as feedforward inputs
  Layer 2 (visible): [FF spikes, hidden spikes, teacher visible spikes] → visible spikes
    - FF→visible, hidden→visible, visible→visible as feedforward inputs
    - Gradients flow from loss through hidden spikes to Layer 1 FF weights
  Loss: visible neurons only (Layer 2 output)
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

from configs import (
    DATALOADER_KWARGS,
    StudentHyperparameters,
    StudentSimulationConfig,
)
from configs.conductance_based import FeedforwardLayerConfig, RecurrentLayerConfig
from collate import VisibleDrivenCollate
from dataloaders.supervised import CyclicSampler, ExactFFDataset
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
from network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
from network_simulators.projections import (
    ScalingFactorProjection,
    make_chunked_ff_projections,
    make_scaling_factor_projections,
)
from network_simulators.two_layer import TwoLayerSNN
from snn_runners import SNNTrainer
from training_utils import AsyncLogger
from training_utils.losses import VanRossumLoss
from visualization import plot_spike_trains


def main(input_dir, output_dir, params_file, wandb_config=None, resume_from=None):
    print("\n" + "=" * 60)
    print("Visible-Driven Training (Hidden Recurrent + Visible Feedforward)")
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
    share_ff_scaling = training_cfg.get("share_ff_scaling", True)

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
    grad_norm_clip_ff = getattr(hyperparameters, "grad_norm_clip_ff", grad_norm_clip)
    adam_eps = getattr(hyperparameters, "eps", 1e-8)
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
    recurrent_mask = network_structure["recurrent_connectivity"]
    feedforward_mask_np = network_structure["feedforward_connectivity"]

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
    n_ff_synapse_types = len(feedforward.get_synapse_params())

    recurrent_synapse_params = recurrent.get_synapse_params()
    feedforward_synapse_params = feedforward.get_synapse_params()

    # Combined FF cell/synapse params: original FF + recurrent types (offset IDs)
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

    # Cell type name lists for projection keys.
    rec_cell_type_names = [cp["name"] for cp in recurrent_cell_params]
    ff_cell_type_names = [cp["name"] for cp in feedforward_cell_params]
    combined_input_cell_type_names = [cp["name"] for cp in combined_cell_params_FF]

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

    # ===============================================
    # Perturb Weights
    # ===============================================

    all_ct = np.concatenate(
        [feedforward_cell_type_indices, cell_type_indices + n_ff_cell_types]
    )
    n_total_inputs = n_feedforward + n_neurons_full
    concatenated_weights_full = np.concatenate([feedforward_weights, weights], axis=0)

    perturbed_weights_full = concatenated_weights_full.copy()
    for input_idx in range(n_total_inputs):
        input_type = all_ct[input_idx]
        for output_idx in range(n_neurons_full):
            output_type = cell_type_indices[output_idx]
            perturbed_weights_full[input_idx, output_idx] *= perturbation_factors[
                input_type, output_type
            ]

    # Apply connectivity masks up-front so projection blocks contain zeros on
    # absent connections (the projection API has no separate mask argument).
    perturbed_ff_weights = perturbed_weights_full[
        :n_feedforward, :
    ] * feedforward_mask_np.astype(np.float32)
    perturbed_rec_weights = perturbed_weights_full[
        n_feedforward:, :
    ] * recurrent_mask.astype(np.float32)

    # ===============================================
    # Split Into Two-Layer Weight Matrices
    # ===============================================

    # Layer 1 (hidden neurons):
    #   FF input: [FF→hidden, visible→hidden] as feedforward
    #   Recurrent: hidden→hidden

    layer1_ff_weights = np.concatenate(
        [
            perturbed_ff_weights[:, hidden_indices],  # FF→hidden
            perturbed_rec_weights[visible_indices][:, hidden_indices],  # vis→hidden
        ],
        axis=0,
    )
    layer1_rec_weights = perturbed_rec_weights[hidden_indices][:, hidden_indices]

    layer1_cell_type_indices = cell_type_indices[hidden_indices]
    layer1_ff_cell_type_indices = np.concatenate(
        [
            feedforward_cell_type_indices,
            cell_type_indices[visible_indices] + n_ff_cell_types,
        ]
    )

    # Layer 2 (visible neurons):
    #   FF input: [FF→visible, hidden→visible, visible→visible]
    #   No recurrence

    layer2_ff_weights = np.concatenate(
        [
            perturbed_ff_weights[:, visible_indices],  # FF→visible
            perturbed_rec_weights[hidden_indices][:, visible_indices],  # hidden→visible
            perturbed_rec_weights[visible_indices][
                :, visible_indices
            ],  # visible→visible
        ],
        axis=0,
    )
    layer2_cell_type_indices = cell_type_indices[visible_indices]
    layer2_ff_cell_type_indices = np.concatenate(
        [
            feedforward_cell_type_indices,
            cell_type_indices[hidden_indices] + n_ff_cell_types,
            cell_type_indices[visible_indices] + n_ff_cell_types,
        ]
    )

    print(
        f"\n  Layer 1 (hidden): FF {layer1_ff_weights.shape}, "
        f"Rec {layer1_rec_weights.shape}"
    )
    print(f"  Layer 2 (visible): FF {layer2_ff_weights.shape}")

    # ===============================================
    # Scaling Factors for Each Layer
    # ===============================================

    # Both layers: FF SF = [ff_sf, rec_sf], Rec SF = rec_sf
    model_sf_ff = np.concatenate([sf_feedforward, sf_recurrent], axis=0)
    model_sf_rec = sf_recurrent.copy()

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
    # Build Projections for Each Layer
    # ==============================================

    # Layer 1: hidden→hidden recurrence + (FF + visible→hidden) feedforward.
    # ``make_scaling_factor_projections`` returns one ScalingFactorProjection
    # per (src_ct, tgt_ct) pair, each with init_sf=1.0 and a connectome buffer
    # holding the (already mask-applied) per-pair weight block.
    layer1_rec_projections, layer1_ff_projections = make_scaling_factor_projections(
        rec_weights=layer1_rec_weights,
        ff_weights=layer1_ff_weights,
        cell_type_indices=layer1_cell_type_indices,
        ff_cell_type_indices=layer1_ff_cell_type_indices,
        cell_type_names=rec_cell_type_names,
        ff_cell_type_names=combined_input_cell_type_names,
        init_sf=1.0,
    )

    # Initialise log_sf from teacher scaling factors.
    with torch.no_grad():
        for (src_name, tgt_name), proj in layer1_rec_projections.items():
            src_id = rec_cell_type_names.index(src_name)
            tgt_id = rec_cell_type_names.index(tgt_name)
            proj.log_sf.copy_(
                torch.tensor(
                    float(np.log(model_sf_rec[src_id, tgt_id])),
                    dtype=proj.log_sf.dtype,
                )
            )
        for (src_name, tgt_name), proj in layer1_ff_projections.items():
            src_id = combined_input_cell_type_names.index(src_name)
            tgt_id = rec_cell_type_names.index(tgt_name)
            proj.log_sf.copy_(
                torch.tensor(
                    float(np.log(model_sf_ff[src_id, tgt_id])),
                    dtype=proj.log_sf.dtype,
                )
            )

    # Layer 2: chunked-FF model. All inputs (FF + hidden + visible recurrent)
    # are treated as feedforward.  ``make_chunked_ff_projections`` builds a
    # single combined dict over (ff names + rec names) → rec names.
    layer2_ff_block = layer2_ff_weights[:n_feedforward, :]
    layer2_rec_block = layer2_ff_weights[n_feedforward:, :]
    layer2_rec_ct_indices = (
        layer2_ff_cell_type_indices[n_feedforward:] - n_ff_cell_types
    )

    layer2_projections = make_chunked_ff_projections(
        rec_weights=layer2_rec_block,
        ff_weights=layer2_ff_block,
        cell_type_indices=layer2_cell_type_indices,
        ff_cell_type_indices=feedforward_cell_type_indices,
        cell_type_names=rec_cell_type_names,
        ff_cell_type_names=ff_cell_type_names,
        init_sf=1.0,
    )
    # The recurrent rows of layer 2 input span hidden+visible neurons.  The
    # default builder slices using a single rec source index per type, so
    # rows from hidden vs visible with the same cell type are collapsed.
    # Rebuild rec-source pairs explicitly so the per-pair block aligns with
    # the (hidden+visible) source ordering implied by layer2_ff_cell_type_indices.
    rec_idx_src_l2 = {
        name: np.flatnonzero(layer2_rec_ct_indices == ct_id)
        for ct_id, name in enumerate(rec_cell_type_names)
    }
    rec_idx_tgt_l2 = {
        name: np.flatnonzero(layer2_cell_type_indices == ct_id)
        for ct_id, name in enumerate(rec_cell_type_names)
    }
    for src_name in rec_cell_type_names:
        src_rows = rec_idx_src_l2[src_name]
        for tgt_name in rec_cell_type_names:
            tgt_cols = rec_idx_tgt_l2[tgt_name]
            block = layer2_rec_block[np.ix_(src_rows, tgt_cols)].astype(np.float32)
            layer2_projections[(src_name, tgt_name)] = ScalingFactorProjection(
                connectome=block, init_sf=1.0
            )

    with torch.no_grad():
        for (src_name, tgt_name), proj in layer2_projections.items():
            src_id = combined_input_cell_type_names.index(src_name)
            tgt_id = rec_cell_type_names.index(tgt_name)
            proj.log_sf.copy_(
                torch.tensor(
                    float(np.log(model_sf_ff[src_id, tgt_id])),
                    dtype=proj.log_sf.dtype,
                )
            )

    # Share the FF scaling factor parameters across layers: layer 2 reuses
    # the same ScalingFactorProjection objects as layer 1 ff_projections for
    # matching (src_name, tgt_name) pairs. Because Projections own their
    # nn.Parameter, sharing the Python object shares the param automatically.
    if share_ff_scaling:
        for key, proj in layer1_ff_projections.items():
            if key in layer2_projections:
                layer2_projections[key] = proj

    # ==============================================
    # Build Two-Layer Model
    # ==============================================

    layer1 = ConductanceLIFNetwork(
        dt=dt,
        rec_projections=layer1_rec_projections,
        ff_projections=layer1_ff_projections,
        cell_type_indices=layer1_cell_type_indices,
        cell_type_indices_FF=layer1_ff_cell_type_indices,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params=recurrent_synapse_params,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        track_variables=False,
    )

    layer2 = FeedforwardConductanceLIFNetwork(
        dt=dt,
        projections=layer2_projections,
        cell_type_indices=layer2_cell_type_indices,
        cell_type_indices_FF=layer2_ff_cell_type_indices,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        track_variables=False,
    )

    model = TwoLayerSNN(layer1, layer2, n_ff=n_feedforward)
    model.to(device)

    print(
        f"\nModel: TwoLayerSNN ({n_hidden} hidden + {n_visible} visible, "
        f"optimising scaling_factors)"
    )

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

    # No VisibleOnlyLoss needed — Layer 2 output IS visible-only
    gradient_loss_functions = {"van_rossum": van_rossum_loss_fn}
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
                "strategy": "visible_driven",
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

    input_cell_type_names = feedforward.cell_types.names + recurrent.cell_types.names
    output_cell_type_names = recurrent.cell_types.names

    metrics_logger = AsyncLogger(log_dir=output_dir, max_queue_size=10)

    # ================================================
    # Collate Function
    # ================================================

    # [FF, teacher_visible] input, teacher_visible target
    visible_tensor = torch.from_numpy(visible_indices).long()
    collate_fn = VisibleDrivenCollate(visible_tensor)

    # ==============================
    # Teacher Data for Hidden Stats
    # ==============================

    teacher_zarr_root = zarr.open_group(input_dir / "spike_data.zarr", mode="r")
    teacher_spikes_zarr = teacher_zarr_root["output_spikes"]

    # Cache teacher hidden spikes in memory to avoid network filesystem reads
    if n_hidden > 0:
        teacher_hidden_cache = np.array(teacher_spikes_zarr[0, :, hidden_indices])
    else:
        teacher_hidden_cache = None

    # ==============================
    # Stats Computer
    # ==============================

    def _current_sf_matrices():
        """Read current scaling factors from layer1 projections into matrices.

        Returns ``(sf_ff, sf_rec)`` arrays matching the old API shapes used by
        the stats / plot code (``sf_ff`` is the combined-input × rec_target
        matrix; ``sf_rec`` is the rec × rec matrix).
        """
        sf_ff = np.ones(
            (len(combined_input_cell_type_names), len(rec_cell_type_names)),
            dtype=np.float32,
        )
        sf_rec = np.ones(
            (len(rec_cell_type_names), len(rec_cell_type_names)), dtype=np.float32
        )
        for (src_name, tgt_name), proj in layer1_ff_projections.items():
            src_id = combined_input_cell_type_names.index(src_name)
            tgt_id = rec_cell_type_names.index(tgt_name)
            sf_ff[src_id, tgt_id] = float(np.exp(proj.log_sf.detach().cpu().numpy()))
        for (src_name, tgt_name), proj in layer1_rec_projections.items():
            src_id = rec_cell_type_names.index(src_name)
            tgt_id = rec_cell_type_names.index(tgt_name)
            sf_rec[src_id, tgt_id] = float(np.exp(proj.log_sf.detach().cpu().numpy()))
        return sf_ff, sf_rec

    def stats_computer(snapshot):
        """Compute summary statistics for logging."""
        # spikes shape: (1, time, n_visible) — Layer 2 outputs visible only
        student_visible_spikes = snapshot["spikes"][0, :, :]
        n_timesteps = student_visible_spikes.shape[0]
        duration_s = n_timesteps * dt / 1000.0

        teacher_visible_spikes = snapshot["target_spikes"][0, :n_timesteps, :]
        teacher_visible_rates = teacher_visible_spikes.sum(axis=0) / duration_s

        stats = {}

        # Visible neuron firing rates
        student_visible_rates = student_visible_spikes.sum(axis=0) / duration_s
        visible_ct = cell_type_indices[visible_indices]
        for type_idx, type_name in enumerate(output_cell_type_names):
            mask = visible_ct == type_idx
            if mask.sum() > 0:
                stats[f"firing_rate/teacher_visible_{type_name}_mean"] = float(
                    teacher_visible_rates[mask].mean()
                )
                stats[f"firing_rate/student_visible_{type_name}_mean"] = float(
                    student_visible_rates[mask].mean()
                )

        # Hidden neuron firing rates (student from hidden_spikes, teacher from zarr)
        if n_hidden > 0 and "hidden_spikes" in snapshot:
            student_hidden_spikes = snapshot["hidden_spikes"][0, :n_timesteps, :]
            student_hidden_rates = student_hidden_spikes.sum(axis=0) / duration_s

            epoch = snapshot.get("epoch", 0)
            n_chunks_accumulated = n_timesteps // chunk_size
            start_chunk = (epoch - n_chunks_accumulated + 1) % num_chunks
            total_timesteps = num_chunks * chunk_size
            start_t = start_chunk * chunk_size
            end_t = start_t + n_timesteps

            if end_t <= total_timesteps:
                teacher_hidden_spikes = teacher_hidden_cache[start_t:end_t]
            else:
                teacher_hidden_spikes = np.concatenate(
                    [
                        teacher_hidden_cache[start_t:total_timesteps],
                        teacher_hidden_cache[: end_t - total_timesteps],
                    ],
                    axis=0,
                )
            teacher_hidden_rates = teacher_hidden_spikes.sum(axis=0) / duration_s

            hidden_ct = cell_type_indices[hidden_indices]
            for type_idx, type_name in enumerate(output_cell_type_names):
                mask = hidden_ct == type_idx
                if mask.sum() > 0:
                    stats[f"firing_rate/teacher_hidden_{type_name}_mean"] = float(
                        teacher_hidden_rates[mask].mean()
                    )
                    stats[f"firing_rate/student_hidden_{type_name}_mean"] = float(
                        student_hidden_rates[mask].mean()
                    )

        # Scaling factors (from Layer 1 projections)
        sf_ff, sf_rec = _current_sf_matrices()

        # Match the old reporting shape: first FF row + recurrent rows.
        report_sf = np.concatenate([sf_ff[:1], sf_rec], axis=0)

        for source_idx in range(report_sf.shape[0]):
            source_name = input_cell_type_names[source_idx]
            for target_idx in range(report_sf.shape[1]):
                target_name = output_cell_type_names[target_idx]
                synapse_name = f"{source_name}_to_{target_name}"
                target_val = target_scaling_factors[source_idx, target_idx]
                if target_val != 0:
                    normalized = report_sf[source_idx, target_idx] / target_val
                else:
                    normalized = report_sf[source_idx, target_idx]
                stats[f"scaling_factors/{synapse_name}_value"] = float(normalized)
                stats[f"scaling_factors/{synapse_name}_target"] = 1.0

        return stats

    # ==============================
    # Plot Generator
    # ==============================

    def plot_generator(spikes, target_spikes, input_spikes, epoch=0, **kwargs):
        figures = {}

        # Visible neuron comparison (spikes are already visible-only from Layer 2)
        n_plot_vis = min(10, spikes.shape[2])
        interleaved_vis = np.zeros((1, spikes.shape[1], 2 * n_plot_vis))
        for i in range(n_plot_vis):
            interleaved_vis[0, :, 2 * i] = target_spikes[0, :, i]
            interleaved_vis[0, :, 2 * i + 1] = spikes[0, :, i]

        figures["spike_comparison_visible"] = plot_spike_trains(
            spikes=interleaved_vis,
            dt=dt,
            cell_type_indices=np.array([0, 1] * n_plot_vis),
            cell_type_names=["Target", "Trained"],
            n_neurons_plot=2 * n_plot_vis,
            n_compared=2,
            fraction=1.0,
            random_seed=None,
            title=f"Visible-driven: Target vs Trained ({n_plot_vis} visible)",
            ylabel="Neuron",
            figsize=(14, 8),
        )

        # Hidden neuron comparison
        hidden_spikes = kwargs.get("hidden_spikes")
        if n_hidden > 0 and hidden_spikes is not None:
            n_plot_hid = min(10, hidden_spikes.shape[2])
            n_time = spikes.shape[1]
            n_chunks_in_plot = n_time // chunk_size
            start_chunk = (epoch - n_chunks_in_plot + 1) % num_chunks
            start_t = start_chunk * chunk_size
            end_t = start_t + n_time

            total_timesteps = num_chunks * chunk_size
            if end_t <= total_timesteps:
                teacher_hid = teacher_hidden_cache[start_t:end_t]
            else:
                teacher_hid = np.concatenate(
                    [
                        teacher_hidden_cache[start_t:total_timesteps],
                        teacher_hidden_cache[: end_t - total_timesteps],
                    ],
                    axis=0,
                )

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
                title=f"Visible-driven: Teacher vs Student ({n_plot_hid} hidden)",
                ylabel="Neuron",
                figsize=(14, 8),
            )

        return figures

    # ==============================
    # Training Setup
    # ==============================

    total_chunks = total_epochs * num_chunks
    print(
        f"\nTraining: {total_epochs} epochs x {num_chunks} chunks "
        f"= {total_chunks} total chunks"
    )

    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=adam_eps
    )
    lr_min = getattr(hyperparameters, "lr_min", None)
    if lr_min is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=total_chunks, eta_min=lr_min
        )
        print(f"  LR schedule: cosine {learning_rate} -> {lr_min}")
    else:
        scheduler = None
    scaler = GradScaler("cuda", enabled=mixed_precision and device == "cuda")

    # Collect log_sf parameters per group (rec vs ff) for per-group grad clipping.
    rec_log_sf_params = [proj.log_sf for proj in layer1_rec_projections.values()]
    # ff group: union of layer1 ff log_sf and any layer2-only log_sf parameters
    # (when share_ff_scaling is False or when (src_name, tgt_name) pairs differ).
    ff_log_sf_params: list = []
    seen_ids: set[int] = set()
    for proj in layer1_ff_projections.values():
        if id(proj.log_sf) not in seen_ids:
            ff_log_sf_params.append(proj.log_sf)
            seen_ids.add(id(proj.log_sf))
    for proj in layer2_projections.values():
        if (
            isinstance(proj, ScalingFactorProjection)
            and id(proj.log_sf) not in seen_ids
        ):
            ff_log_sf_params.append(proj.log_sf)
            seen_ids.add(id(proj.log_sf))

    def per_group_grad_clip(m):
        rec_with_grad = [p for p in rec_log_sf_params if p.grad is not None]
        if rec_with_grad:
            torch.nn.utils.clip_grad_norm_(rec_with_grad, max_norm=grad_norm_clip)
        ff_with_grad = [p for p in ff_log_sf_params if p.grad is not None]
        if ff_with_grad:
            torch.nn.utils.clip_grad_norm_(ff_with_grad, max_norm=grad_norm_clip_ff)

    # Dataloader
    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        **DATALOADER_KWARGS,
        collate_fn=collate_fn,
    )

    pbar = tqdm(range(total_chunks), desc="Visible-driven", unit="chunk")

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
        grad_clip_fn=per_group_grad_clip,
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
    print("Visible-Driven Training Complete")
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
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("params_file", type=Path)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.params_file)
