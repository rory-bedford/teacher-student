"""
Core EM training module for hidden activity experiments.

This module implements the EM algorithm for training a student network when some
neurons are hidden (unobserved). It supports different E-step and loss configurations.

E-step Types:
    - "free": Run full recurrent network; visible neurons are simulated by the model
    - "clamped": Clamp visible neurons to teacher spikes; only hidden neurons are inferred
    - "clamped_ff": First iteration uses the full recurrent clamped model (same as
      "clamped") to get a good initial estimate of hidden spikes. Subsequent iterations
      replace hidden→hidden recurrence with feedforward input from the previous
      iteration's inferred hidden spikes. Avoids chaotic divergence from within-simulation
      hidden recurrence while starting from a well-initialised hidden state.

Loss Types:
    - "visible": Compute loss on visible neurons only (compare to teacher)
    - "full": Compute loss on all neurons (visible=teacher target, hidden=E-step inference)

Usage:
    from _em_core import run_em_training
    run_em_training(input_dir, output_dir, params_file, e_step_type="clamped", loss_type="visible")
"""

import gc
import json

import numpy as np
from pathlib import Path
from connectome_snns.dataloaders.supervised import (
    ExactFFDataset,
    CyclicSampler,
    SpikeData,
)
from connectome_snns.network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
from connectome_snns.network_simulators.conductance_based.simulator import (
    ConductanceLIFNetwork,
)
from connectome_snns.network_simulators.projections import (
    FrozenProjection,
    make_frozen_projections,
    make_frozen_chunked_ff_projections,
    make_chunked_ff_projections,
)
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm
from connectome_snns.training_utils.losses import VanRossumLoss
from connectome_snns.training_utils import AsyncLogger, load_checkpoint
from connectome_snns.configs import (
    StudentSimulationConfig,
    EMTrainingConfig,
    StudentHyperparameters,
)
from connectome_snns.configs.conductance_based import RecurrentLayerConfig, FeedforwardLayerConfig
from connectome_snns.snn_runners import SNNTrainer, EvolutionarySearch
from connectome_snns.snn_runners.inference_runner import SNNInference
import toml
import wandb
import zarr
import shutil
from connectome_snns.visualization import plot_spike_trains


def _rebake_frozen_projections(
    projections_module,
    pair_blocks: dict,
    sf_matrix: np.ndarray,
    src_name_to_row: dict,
    tgt_name_to_col: dict,
) -> None:
    """Rewrite each FrozenProjection's weight buffer in-place with new SFs.

    Args:
        projections_module: nn.ModuleDict of FrozenProjections keyed by
            ``f"{src}__{tgt}"``.
        pair_blocks: ``{(src_name, tgt_name): np.ndarray}`` of unscaled
            connectome blocks (float32) — multiplied by the per-pair SF.
        sf_matrix: scaling factor matrix indexed by ``(src_row, tgt_col)``.
        src_name_to_row: maps source cell type name -> row index in sf_matrix.
        tgt_name_to_col: maps target cell type name -> column index in sf_matrix.
    """
    for (src_name, tgt_name), block in pair_blocks.items():
        sf_val = float(sf_matrix[src_name_to_row[src_name], tgt_name_to_col[tgt_name]])
        scaled = block * sf_val
        proj = projections_module[f"{src_name}__{tgt_name}"]
        proj._weights.copy_(torch.from_numpy(scaled).to(proj._weights.device))


class CMAEMWrapper(torch.nn.Module):
    """Wrapper combining E-step and M-step models for CMA-ES evaluation.

    For each forward pass: infers hidden activity via the E-step model
    (clamped visible), then runs the M-step feedforward model with
    inferred hidden + visible teacher spikes. This avoids giving CMA-ES
    access to hidden teacher spikes.
    """

    def __init__(
        self,
        estep_model: ConductanceLIFNetwork,
        mstep_model: FeedforwardConductanceLIFNetwork,
        n_feedforward: int,
        visible_indices: np.ndarray,
        hidden_indices: np.ndarray,
        n_ff_cell_types: int,
        estep_rec_pair_blocks: dict,
        estep_rec_src_name_to_row: dict,
        estep_rec_tgt_name_to_col: dict,
        estep_ff_pair_blocks: dict,
        estep_ff_src_name_to_row: dict,
        estep_ff_tgt_name_to_col: dict,
        mstep_pair_blocks: dict,
        mstep_src_name_to_row: dict,
        mstep_tgt_name_to_col: dict,
    ):
        super().__init__()
        self.estep_model = estep_model
        self.mstep_model = mstep_model
        self.n_feedforward = n_feedforward
        self.n_ff_cell_types = n_ff_cell_types
        self._visible_tensor = torch.from_numpy(visible_indices).long()
        self._hidden_tensor = torch.from_numpy(hidden_indices).long()
        self._n_neurons_full = len(visible_indices) + len(hidden_indices)
        # Unscaled connectome blocks + index lookups for in-place SF rebaking.
        self._estep_rec_pair_blocks = estep_rec_pair_blocks
        self._estep_rec_src_name_to_row = estep_rec_src_name_to_row
        self._estep_rec_tgt_name_to_col = estep_rec_tgt_name_to_col
        self._estep_ff_pair_blocks = estep_ff_pair_blocks
        self._estep_ff_src_name_to_row = estep_ff_src_name_to_row
        self._estep_ff_tgt_name_to_col = estep_ff_tgt_name_to_col
        self._mstep_pair_blocks = mstep_pair_blocks
        self._mstep_src_name_to_row = mstep_src_name_to_row
        self._mstep_tgt_name_to_col = mstep_tgt_name_to_col

    @property
    def track_variables(self):
        return self.mstep_model.track_variables

    @track_variables.setter
    def track_variables(self, value):
        self.mstep_model.track_variables = value
        self.estep_model.track_variables = value

    @property
    def track_batch_idx(self):
        return self.mstep_model.track_batch_idx

    @track_batch_idx.setter
    def track_batch_idx(self, value):
        self.mstep_model.track_batch_idx = value
        self.estep_model.track_batch_idx = value

    @property
    def dt(self):
        return self.mstep_model.dt

    @property
    def batch_size(self):
        return self.mstep_model.batch_size

    def reset_state(self, batch_size):
        self.estep_model.reset_state(batch_size=batch_size)
        self.mstep_model.reset_state(batch_size=batch_size)

    def update_scaling_factors(self, scaling_factors_FF):
        """Update scaling factors on both E-step and M-step models.

        Walks the projections on each submodel and rewrites the underlying
        FrozenProjection weight buffers in-place (since CMA-ES E-step and
        M-step models are inference-only with baked SFs).

        Args:
            scaling_factors_FF: Concatenated [sf_feedforward; sf_recurrent] tensor
                with shape (n_ff_cell_types + n_rec_cell_types, n_rec_cell_types).
        """
        sf_np = scaling_factors_FF.detach().cpu().numpy().astype(np.float32)
        sf_feedforward = sf_np[: self.n_ff_cell_types]
        sf_recurrent = sf_np[self.n_ff_cell_types :]

        # E-step model: rec uses sf_recurrent (indexed by rec cell type names);
        # ff uses [sf_feedforward; sf_recurrent] (FF inputs followed by visible
        # rec inputs offset by n_ff_cell_types).
        _rebake_frozen_projections(
            self.estep_model.rec_projections,
            self._estep_rec_pair_blocks,
            sf_recurrent,
            self._estep_rec_src_name_to_row,
            self._estep_rec_tgt_name_to_col,
        )
        _rebake_frozen_projections(
            self.estep_model.ff_projections,
            self._estep_ff_pair_blocks,
            np.concatenate([sf_feedforward, sf_recurrent], axis=0),
            self._estep_ff_src_name_to_row,
            self._estep_ff_tgt_name_to_col,
        )
        self.estep_model._precompute_weight_products()

        # M-step model: single combined projections dict.
        _rebake_frozen_projections(
            self.mstep_model.projections,
            self._mstep_pair_blocks,
            sf_np,
            self._mstep_src_name_to_row,
            self._mstep_tgt_name_to_col,
        )
        self.mstep_model._precompute_weight_products()

    def forward(self, input_spikes):
        """Run E-step inference then M-step evaluation.

        Args:
            input_spikes: (batch, time, n_ff + n_neurons_full) — FF + all teacher rec.

        Returns:
            Visible neuron output spikes from M-step model.
        """
        ff_spikes = input_spikes[:, :, : self.n_feedforward]
        teacher_rec = input_spikes[:, :, self.n_feedforward :]

        vis = self._visible_tensor.to(input_spikes.device)
        hid = self._hidden_tensor.to(input_spikes.device)

        visible_teacher = teacher_rec[:, :, vis]

        # E-step: infer hidden spikes from [FF, visible_teacher]
        estep_input = torch.cat([ff_spikes, visible_teacher], dim=2).float()
        hidden_spikes = self.estep_model.forward(input_spikes=estep_input)

        # Construct full recurrent input (float, since model outputs are float)
        full_rec = torch.zeros(
            input_spikes.shape[0],
            input_spikes.shape[1],
            self._n_neurons_full,
            device=input_spikes.device,
            dtype=torch.float32,
        )
        full_rec[:, :, vis] = visible_teacher.float()
        full_rec[:, :, hid] = hidden_spikes

        # M-step: run feedforward model with [FF, full_rec]
        mstep_input = torch.cat([ff_spikes.float(), full_rec], dim=2)
        return self.mstep_model.forward(input_spikes=mstep_input)


class MStepCollate:
    """Collate for M-step training: reads inferred hidden spikes from zarr.

    Builds model input as ``[FF spikes, visible teacher spikes, inferred hidden spikes]``
    and targets as visible-only teacher spikes.

    This class is picklable (required for ``num_workers > 0`` dataloaders).
    Stateful: tracks chunk position for sequential zarr reads and resets
    the loss filter when cycling through the dataset.

    Args:
        visible_indices: Array of visible neuron indices.
        hidden_indices: Array of hidden neuron indices.
        inferred_spikes_zarr_path: Path to zarr with inferred hidden spikes.
        n_neurons_full: Total neuron count.
        num_chunks: Chunks in the dataset (for cycling detection).
        loss_type: ``"visible"`` or ``"full"``.
    """

    def __init__(
        self,
        visible_indices: np.ndarray,
        hidden_indices: np.ndarray,
        inferred_spikes_zarr_path: Path,
        n_neurons_full: int,
        num_chunks: int,
        loss_type: str,
    ):
        self.visible_tensor = torch.from_numpy(visible_indices).long()
        self.hidden_tensor = torch.from_numpy(hidden_indices).long()
        self.inferred_spikes_zarr_path = inferred_spikes_zarr_path
        self.n_neurons_full = n_neurons_full
        self.num_chunks = num_chunks
        self.loss_type = loss_type
        self.chunk_counter = [0]
        self.loss_ref = [None]
        self._zarr = None

    @property
    def inferred_spikes_zarr(self):
        if self._zarr is None:
            root = zarr.open_group(self.inferred_spikes_zarr_path, mode="r")
            self._zarr = root["output_spikes"]
        return self._zarr

    def reset_counter(self):
        self.chunk_counter[0] = 0

    def set_loss(self, loss):
        self.loss_ref[0] = loss

    def __call__(self, batch: SpikeData) -> SpikeData:
        # Reset loss filter when cycling
        if self.chunk_counter[0] % self.num_chunks == 0 and self.chunk_counter[0] > 0:
            if self.loss_ref[0] is not None:
                self.loss_ref[0].reset_state()

        batch_size, time_steps, _ = batch.target_spikes.shape
        device = batch.target_spikes.device

        # Build full recurrent input: [visible teacher spikes, inferred hidden spikes]
        recurrent_inputs = torch.zeros(
            batch_size,
            time_steps,
            self.n_neurons_full,
            device=device,
            dtype=torch.float32,
        )

        # Visible: exact teacher spikes
        recurrent_inputs[:, :, self.visible_tensor] = batch.target_spikes[
            :, :, self.visible_tensor
        ].float()

        # Hidden: inferred spikes from zarr
        chunk_idx = self.chunk_counter[0] % self.num_chunks
        start_t = chunk_idx * time_steps
        end_t = start_t + time_steps
        inferred_chunk = self.inferred_spikes_zarr[:, start_t:end_t, :]
        hidden_spikes_chunk = (
            torch.from_numpy(np.array(inferred_chunk)).float().to(device)
        )
        recurrent_inputs[:, :, self.hidden_tensor] = hidden_spikes_chunk

        self.chunk_counter[0] += 1

        # Concatenate FF + recurrent as inputs
        concatenated_inputs = torch.cat([batch.input_spikes, recurrent_inputs], dim=2)

        # Target depends on loss_type
        if self.loss_type == "visible":
            targets = batch.target_spikes[:, :, self.visible_tensor]
        else:  # loss_type == "full"
            targets = torch.zeros(
                batch_size,
                time_steps,
                self.n_neurons_full,
                device=device,
                dtype=torch.float32,
            )
            targets[:, :, self.visible_tensor] = batch.target_spikes[
                :, :, self.visible_tensor
            ].float()
            targets[:, :, self.hidden_tensor] = hidden_spikes_chunk

        return SpikeData(input_spikes=concatenated_inputs, target_spikes=targets)


class CMACollateVisibleTarget:
    """Collate for CMA-ES: concatenate [FF, all teacher rec] as input, visible-only target.

    Args:
        visible_indices: Tensor of visible neuron indices.
    """

    def __init__(self, visible_indices: torch.Tensor):
        self.visible_indices = visible_indices

    def __call__(self, batch: SpikeData) -> SpikeData:
        concatenated_inputs = torch.cat(
            [batch.input_spikes, batch.target_spikes], dim=2
        )
        return SpikeData(
            input_spikes=concatenated_inputs,
            target_spikes=batch.target_spikes[:, :, self.visible_indices],
        )


def run_estep_free(
    dt: float,
    batch_size: int,
    device: str,
    perturbed_rec_weights: np.ndarray,
    perturbed_ff_weights: np.ndarray,
    cell_type_indices: np.ndarray,
    feedforward_cell_type_indices: np.ndarray,
    recurrent_cell_params: list,
    feedforward_cell_params: list,
    recurrent_synapse_params: list,
    feedforward_synapse_params: list,
    surrgrad_scale: float,
    sf_recurrent: np.ndarray,
    sf_feedforward: np.ndarray,
    recurrent_mask: np.ndarray,
    feedforward_mask: np.ndarray,
    hidden_indices: np.ndarray,
    spike_dataset: ExactFFDataset,
    num_chunks: int,
    chunk_size: int,
    output_path: Path,
):
    """
    E-step with FREE visible neurons: run full recurrent network.

    Both visible and hidden neurons are simulated by the model.
    Returns path to zarr file containing inferred hidden spikes.
    """
    n_hidden = len(hidden_indices)

    # Cell type name lists (full recurrent + FF spaces).
    rec_cell_type_names = [cp["name"] for cp in recurrent_cell_params]
    ff_cell_type_names = [cp["name"] for cp in feedforward_cell_params]

    # Apply masks so projections see zeros on inactive connections; SFs are
    # baked directly into the FrozenProjection weight buffers.
    masked_rec = perturbed_rec_weights * recurrent_mask.astype(np.float32)
    masked_ff = perturbed_ff_weights * feedforward_mask.astype(np.float32)
    rec_projs, ff_projs = make_frozen_projections(
        rec_weights=masked_rec,
        ff_weights=masked_ff,
        cell_type_indices=cell_type_indices,
        ff_cell_type_indices=feedforward_cell_type_indices,
        cell_type_names=rec_cell_type_names,
        ff_cell_type_names=ff_cell_type_names,
        scaling_factors_rec=sf_recurrent,
        scaling_factors_ff=sf_feedforward,
    )

    estep_network = ConductanceLIFNetwork(
        dt=dt,
        rec_projections=rec_projs,
        ff_projections=ff_projs,
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
    estep_network.to(device)
    estep_network.reset_state(batch_size=batch_size)

    # Run inference
    inference_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        num_workers=0,
    )

    temp_zarr_path = output_path.parent / "temp_estep.zarr"
    inference_runner = SNNInference(
        model=estep_network,
        dataloader=inference_dataloader,
        device=device,
        output_mode="zarr",
        zarr_path=temp_zarr_path,
        save_tracked_variables=False,
        max_chunks=num_chunks,
        progress_bar=True,
    )
    inference_runner.run()

    # Extract hidden neurons only
    temp_zarr_root = zarr.open_group(temp_zarr_path, mode="r")
    temp_output_spikes = temp_zarr_root["output_spikes"]

    final_zarr_root = zarr.open_group(output_path, mode="w")
    hidden_spikes_only = final_zarr_root.create_dataset(
        "output_spikes",
        shape=(batch_size, temp_output_spikes.shape[1], n_hidden),
        dtype=temp_output_spikes.dtype,
        chunks=(1, chunk_size, n_hidden),
    )

    for batch_idx in range(batch_size):
        hidden_spikes_only[batch_idx, :, :] = temp_output_spikes[
            batch_idx, :, hidden_indices
        ]

    # Cleanup
    shutil.rmtree(temp_zarr_path)
    del estep_network
    if device == "cuda":
        torch.cuda.empty_cache()

    return output_path


def run_estep_clamped(
    dt: float,
    batch_size: int,
    device: str,
    perturbed_rec_weights: np.ndarray,
    perturbed_ff_weights: np.ndarray,
    cell_type_indices: np.ndarray,
    feedforward_cell_type_indices: np.ndarray,
    recurrent_cell_params: list,
    feedforward_cell_params: list,
    recurrent_synapse_params: list,
    feedforward_synapse_params: list,
    surrgrad_scale: float,
    sf_recurrent: np.ndarray,
    sf_feedforward: np.ndarray,
    recurrent_mask: np.ndarray,
    feedforward_mask: np.ndarray,
    visible_indices: np.ndarray,
    hidden_indices: np.ndarray,
    spike_dataset: ExactFFDataset,
    num_chunks: int,
    chunk_size: int,
    output_path: Path,
    n_ff_cell_types: int,
    spike_mode: str = "deterministic",
):
    """
    E-step with CLAMPED visible neurons: infer hidden given observed visible.

    Visible neurons are fixed to teacher spikes. Hidden neurons receive:
    - Feedforward (mitral) inputs
    - Visible neuron inputs (clamped to teacher)
    - Hidden-to-hidden recurrence

    Returns path to zarr file containing inferred hidden spikes.
    """
    # Build inference network: inputs=[FF, visible_teacher], recurrence=hidden->hidden
    # FF -> hidden weights
    ff_to_hidden_weights = perturbed_ff_weights[:, hidden_indices]
    ff_to_hidden_mask = feedforward_mask[:, hidden_indices]

    # Visible -> hidden weights
    visible_to_hidden_weights = perturbed_rec_weights[visible_indices, :][
        :, hidden_indices
    ]
    visible_to_hidden_mask = recurrent_mask[visible_indices, :][:, hidden_indices]

    # Concatenate: [FF, visible] -> hidden
    inference_weights_FF = np.concatenate(
        [ff_to_hidden_weights, visible_to_hidden_weights], axis=0
    )
    inference_mask_FF = np.concatenate(
        [ff_to_hidden_mask, visible_to_hidden_mask], axis=0
    )

    # Hidden -> hidden recurrence
    inference_weights_rec = perturbed_rec_weights[hidden_indices, :][:, hidden_indices]
    inference_mask_rec = recurrent_mask[hidden_indices, :][:, hidden_indices]

    # Cell type indices
    cell_type_indices_hidden = cell_type_indices[hidden_indices]
    inference_cell_type_indices_FF = np.concatenate(
        [
            feedforward_cell_type_indices,
            cell_type_indices[visible_indices] + n_ff_cell_types,
        ]
    )

    # Scaling factors for inference
    inference_sf_FF = np.concatenate([sf_feedforward, sf_recurrent], axis=0)
    inference_sf_rec = sf_recurrent.copy()

    # Combined cell/synapse params
    combined_cell_params_FF = feedforward_cell_params.copy()
    for cp in recurrent_cell_params:
        offset_cp = cp.copy()
        offset_cp["cell_id"] = cp["cell_id"] + n_ff_cell_types
        combined_cell_params_FF.append(offset_cp)

    n_ff_synapse_types = len(feedforward_synapse_params)
    combined_synapse_params_FF = feedforward_synapse_params.copy()
    for sp in recurrent_synapse_params:
        offset_sp = sp.copy()
        offset_sp["cell_id"] = sp["cell_id"] + n_ff_cell_types
        offset_sp["synapse_id"] = sp["synapse_id"] + n_ff_synapse_types
        combined_synapse_params_FF.append(offset_sp)

    # Cell type names for the hidden-only subnetwork.
    rec_cell_type_names = [cp["name"] for cp in recurrent_cell_params]
    combined_input_cell_type_names = [cp["name"] for cp in combined_cell_params_FF]

    masked_inference_rec = inference_weights_rec * inference_mask_rec.astype(np.float32)
    masked_inference_ff = inference_weights_FF * inference_mask_FF.astype(np.float32)
    rec_projs, ff_projs = make_frozen_projections(
        rec_weights=masked_inference_rec,
        ff_weights=masked_inference_ff,
        cell_type_indices=cell_type_indices_hidden,
        ff_cell_type_indices=inference_cell_type_indices_FF,
        cell_type_names=rec_cell_type_names,
        ff_cell_type_names=combined_input_cell_type_names,
        scaling_factors_rec=inference_sf_rec,
        scaling_factors_ff=inference_sf_FF,
    )

    estep_network = ConductanceLIFNetwork(
        dt=dt,
        rec_projections=rec_projs,
        ff_projections=ff_projs,
        cell_type_indices=cell_type_indices_hidden,
        cell_type_indices_FF=inference_cell_type_indices_FF,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params=recurrent_synapse_params,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        track_variables=False,
        spike_mode=spike_mode,
    )
    estep_network.to(device)
    estep_network.reset_state(batch_size=batch_size)

    # Collate function to provide [FF, visible_teacher] as input
    visible_tensor = torch.from_numpy(visible_indices).long()

    def clamped_collate_fn(batch: SpikeData) -> SpikeData:
        visible_teacher_spikes = batch.target_spikes[:, :, visible_tensor].float()
        concatenated_inputs = torch.cat(
            [batch.input_spikes, visible_teacher_spikes], dim=2
        )
        return SpikeData(
            input_spikes=concatenated_inputs,
            target_spikes=batch.target_spikes,  # Not used during inference
        )

    # Run inference
    inference_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        num_workers=0,
        collate_fn=clamped_collate_fn,
    )

    inference_runner = SNNInference(
        model=estep_network,
        dataloader=inference_dataloader,
        device=device,
        output_mode="zarr",
        zarr_path=output_path,
        save_tracked_variables=False,
        max_chunks=num_chunks,
        progress_bar=True,
    )
    inference_runner.run()

    # Cleanup
    del estep_network
    if device == "cuda":
        torch.cuda.empty_cache()

    return output_path


def run_estep_clamped_feedforward(
    dt: float,
    batch_size: int,
    device: str,
    perturbed_rec_weights: np.ndarray,
    perturbed_ff_weights: np.ndarray,
    cell_type_indices: np.ndarray,
    feedforward_cell_type_indices: np.ndarray,
    recurrent_cell_params: list,
    feedforward_cell_params: list,
    recurrent_synapse_params: list,
    feedforward_synapse_params: list,
    surrgrad_scale: float,
    sf_recurrent: np.ndarray,
    sf_feedforward: np.ndarray,
    recurrent_mask: np.ndarray,
    feedforward_mask: np.ndarray,
    visible_indices: np.ndarray,
    hidden_indices: np.ndarray,
    spike_dataset: ExactFFDataset,
    num_chunks: int,
    chunk_size: int,
    output_path: Path,
    n_ff_cell_types: int,
    previous_hidden_spikes_path: Path = None,
):
    """
    E-step with CLAMPED visible, FEEDFORWARD hidden: no within-simulation hidden recurrence.

    Like the clamped E-step, visible neurons are fixed to teacher spikes. However,
    hidden→hidden connections are treated as feedforward input from the previous
    EM iteration's inferred hidden spikes, rather than from the current simulation's
    own recurrent output. This breaks the chaotic sensitivity of hidden recurrence.

    Note: The first EM iteration should use run_estep_clamped (full recurrent model)
    to get a good initial estimate. This function is used from iteration 2 onwards.
    """
    n_hidden = len(hidden_indices)

    # Weights: [FF→hidden, visible→hidden, hidden→hidden] all as feedforward
    ff_to_hidden = perturbed_ff_weights[:, hidden_indices]
    ff_to_hidden_mask = feedforward_mask[:, hidden_indices]

    vis_to_hidden = perturbed_rec_weights[visible_indices][:, hidden_indices]
    vis_to_hidden_mask = recurrent_mask[visible_indices][:, hidden_indices]

    hid_to_hidden = perturbed_rec_weights[hidden_indices][:, hidden_indices]
    hid_to_hidden_mask = recurrent_mask[hidden_indices][:, hidden_indices]

    inference_weights_FF = np.concatenate(
        [ff_to_hidden, vis_to_hidden, hid_to_hidden], axis=0
    )
    inference_mask_FF = np.concatenate(
        [ff_to_hidden_mask, vis_to_hidden_mask, hid_to_hidden_mask], axis=0
    )

    # Cell type indices for input: [FF types, visible rec types, hidden rec types]
    cell_type_indices_hidden = cell_type_indices[hidden_indices]
    inference_cell_type_indices_FF = np.concatenate(
        [
            feedforward_cell_type_indices,
            cell_type_indices[visible_indices] + n_ff_cell_types,
            cell_type_indices[hidden_indices] + n_ff_cell_types,
        ]
    )

    # Scaling factors: same as clamped E-step (visible and hidden share rec SF)
    inference_sf_FF = np.concatenate([sf_feedforward, sf_recurrent], axis=0)

    # Combined cell/synapse params
    combined_cell_params_FF = feedforward_cell_params.copy()
    for cp in recurrent_cell_params:
        offset_cp = cp.copy()
        offset_cp["cell_id"] = cp["cell_id"] + n_ff_cell_types
        combined_cell_params_FF.append(offset_cp)

    n_ff_synapse_types = len(feedforward_synapse_params)
    combined_synapse_params_FF = feedforward_synapse_params.copy()
    for sp in recurrent_synapse_params:
        offset_sp = sp.copy()
        offset_sp["cell_id"] = sp["cell_id"] + n_ff_cell_types
        offset_sp["synapse_id"] = sp["synapse_id"] + n_ff_synapse_types
        combined_synapse_params_FF.append(offset_sp)

    # Cell type names. The chunked-FF model treats all input rows as FF;
    # the combined input space is [FF, visible-rec, hidden-rec] but with
    # the rec cell-type names shared, so positions in combined_cell_params_FF
    # map id -> name.
    rec_cell_type_names = [cp["name"] for cp in recurrent_cell_params]
    combined_input_cell_type_names = [cp["name"] for cp in combined_cell_params_FF]

    masked_inference_ff = inference_weights_FF * inference_mask_FF.astype(np.float32)
    # No recurrence in the chunked-FF model: empty rec block (still a 2D
    # (0, n_hidden) array — provided to make_frozen_chunked_ff_projections,
    # but here we explicitly construct a single FF dict via make_frozen_projections
    # on a degenerate path. Simpler: directly build per-(src,tgt) FF blocks.
    ff_idx_per_ct = {
        name: np.flatnonzero(inference_cell_type_indices_FF == ct_id)
        for ct_id, name in enumerate(combined_input_cell_type_names)
    }
    rec_idx_per_ct = {
        name: np.flatnonzero(cell_type_indices_hidden == ct_id)
        for ct_id, name in enumerate(rec_cell_type_names)
    }
    estep_projections: dict = {}
    for src_id, src_name in enumerate(combined_input_cell_type_names):
        src_idx = ff_idx_per_ct[src_name]
        for tgt_id, tgt_name in enumerate(rec_cell_type_names):
            tgt_idx = rec_idx_per_ct[tgt_name]
            block = masked_inference_ff[np.ix_(src_idx, tgt_idx)].astype(np.float32)
            block = block * float(inference_sf_FF[src_id, tgt_id])
            estep_projections[(src_name, tgt_name)] = FrozenProjection(block)

    estep_network = FeedforwardConductanceLIFNetwork(
        dt=dt,
        projections=estep_projections,
        cell_type_indices=cell_type_indices_hidden,
        cell_type_indices_FF=inference_cell_type_indices_FF,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        track_variables=False,
    )
    estep_network.to(device)
    estep_network.reset_state(batch_size=batch_size)

    # Load previous hidden spikes (if available)
    prev_hidden_zarr = None
    if previous_hidden_spikes_path is not None:
        prev_hidden_zarr = zarr.open_group(previous_hidden_spikes_path, mode="r")

    visible_tensor = torch.from_numpy(visible_indices).long()
    chunk_counter = [0]

    def clamped_ff_collate_fn(batch: SpikeData) -> SpikeData:
        visible_teacher_spikes = batch.target_spikes[:, :, visible_tensor].float()
        bs, ts, _ = batch.input_spikes.shape

        if prev_hidden_zarr is not None:
            ci = chunk_counter[0] % num_chunks
            start_t = ci * chunk_size
            end_t = start_t + ts
            prev_hidden = np.array(
                prev_hidden_zarr["output_spikes"][:bs, start_t:end_t, :]
            )
            prev_hidden_tensor = (
                torch.from_numpy(prev_hidden).float().to(batch.input_spikes.device)
            )
        else:
            prev_hidden_tensor = torch.zeros(
                bs,
                ts,
                n_hidden,
                device=batch.input_spikes.device,
                dtype=torch.float32,
            )

        chunk_counter[0] += 1

        concatenated = torch.cat(
            [batch.input_spikes, visible_teacher_spikes, prev_hidden_tensor], dim=2
        )
        return SpikeData(
            input_spikes=concatenated,
            target_spikes=batch.target_spikes,
        )

    # Run inference
    inference_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        num_workers=0,
        collate_fn=clamped_ff_collate_fn,
    )

    inference_runner = SNNInference(
        model=estep_network,
        dataloader=inference_dataloader,
        device=device,
        output_mode="zarr",
        zarr_path=output_path,
        save_tracked_variables=False,
        max_chunks=num_chunks,
        progress_bar=True,
    )
    inference_runner.run()

    # Cleanup
    del estep_network
    if device == "cuda":
        torch.cuda.empty_cache()

    return output_path


def rescale_hidden_firing_rates(
    inferred_spikes_path: Path,
    teacher_spikes_zarr,
    visible_indices: np.ndarray,
    hidden_indices: np.ndarray,
    cell_type_indices: np.ndarray,
    cell_type_names: list[str],
    dt: float,
):
    """Rescale inferred hidden firing rates to match visible firing rates by cell type.

    Applies uniform Poisson thinning per cell type so that the mean hidden firing rate
    matches the mean visible (teacher) firing rate. This corrects for the systematic
    upward bias in inferred hidden activity when scaling factors are imperfect.

    Modifies the zarr in-place.
    """
    zarr_root = zarr.open_group(inferred_spikes_path, mode="r+")
    inferred_spikes = zarr_root["output_spikes"]  # (batch, time, n_hidden)

    batch_size = inferred_spikes.shape[0]
    total_time = inferred_spikes.shape[1]
    duration_s = total_time * dt / 1000.0

    hidden_cell_types = cell_type_indices[hidden_indices]
    visible_cell_types = cell_type_indices[visible_indices]
    n_cell_types = len(cell_type_names)

    # Compute mean rates from batch 0
    inferred_arr = np.array(inferred_spikes[0])  # (time, n_hidden)
    hidden_rates = inferred_arr.sum(axis=0) / duration_s

    teacher_visible = np.array(teacher_spikes_zarr[0, :total_time, visible_indices])
    visible_rates = teacher_visible.sum(axis=0) / duration_s

    # Compute ratios by cell type
    ratios = {}
    print("  Firing rate rescaling:")
    for type_idx in range(n_cell_types):
        hid_mask = hidden_cell_types == type_idx
        vis_mask = visible_cell_types == type_idx

        mean_hidden = hidden_rates[hid_mask].mean() if hid_mask.sum() > 0 else 0.0
        mean_visible = visible_rates[vis_mask].mean() if vis_mask.sum() > 0 else 0.0

        ratio = mean_visible / mean_hidden if mean_hidden > 0 else 1.0
        ratios[type_idx] = ratio
        print(
            f"    {cell_type_names[type_idx]}: "
            f"hidden={mean_hidden:.2f} Hz, visible={mean_visible:.2f} Hz, "
            f"ratio={ratio:.4f}"
        )

    # Apply thinning in-place
    for b in range(batch_size):
        arr = np.array(inferred_spikes[b])  # (time, n_hidden)
        for type_idx in range(n_cell_types):
            ratio = ratios[type_idx]
            if ratio >= 1.0:
                continue
            neuron_indices = np.where(hidden_cell_types == type_idx)[0]
            spikes = arr[:, neuron_indices]
            rand = np.random.random(spikes.shape)
            arr[:, neuron_indices] = spikes * (rand < ratio)
        inferred_spikes[b] = arr

    # Verify
    new_arr = np.array(inferred_spikes[0])
    new_rates = new_arr.sum(axis=0) / duration_s
    print("  After rescaling:")
    for type_idx in range(n_cell_types):
        hid_mask = hidden_cell_types == type_idx
        vis_mask = visible_cell_types == type_idx
        new_mean = new_rates[hid_mask].mean() if hid_mask.sum() > 0 else 0.0
        vis_mean = visible_rates[vis_mask].mean() if vis_mask.sum() > 0 else 0.0
        print(
            f"    {cell_type_names[type_idx]}: hidden={new_mean:.2f} Hz, visible={vis_mean:.2f} Hz"
        )

    return ratios


def run_em_training(
    input_dir: Path,
    output_dir: Path,
    params_file: Path,
    e_step_type: str = "free",
    loss_type: str = "visible",
    poisson_trimming: bool = False,
    wandb_config: dict = None,
    resume_from: Path = None,
):
    """
    Run EM training for hidden activity inference.

    Args:
        input_dir: Directory containing teacher data (network_structure.npz, spike_data.zarr)
        output_dir: Directory for training outputs
        params_file: Path to parameter TOML file
        e_step_type: "free" (visible simulated) or "clamped" (visible fixed to teacher)
        loss_type: "visible" (loss on visible only) or "full" (loss on all neurons)
        wandb_config: Optional W&B configuration
        resume_from: Optional checkpoint path to resume from
    """
    assert e_step_type in ["free", "clamped", "clamped_ff"], (
        f"Invalid e_step_type: {e_step_type}"
    )
    assert loss_type in ["visible", "full"], f"Invalid loss_type: {loss_type}"

    print(f"\n{'=' * 60}")
    print("EM Training Configuration:")
    print(f"  E-step: {e_step_type}")
    print(f"  Loss: {loss_type}")
    print(f"  Poisson trimming: {poisson_trimming}")
    print(f"{'=' * 60}\n")

    # ======================================
    # Device Selection and Parameter Loading
    # ======================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(params_file, "r") as f:
        data = toml.load(f)

    simulation = StudentSimulationConfig(**data["simulation"])
    training = EMTrainingConfig(**data["training"])
    hyperparameters = StudentHyperparameters(**data["hyperparameters"])
    recurrent = RecurrentLayerConfig(**data["recurrent"])
    feedforward = FeedforwardLayerConfig(**data["feedforward"])
    scaling_factors = data.get("scaling_factors", {})
    em_config = data.get("em", {})
    cma_es_config = data.get("cma_es", {})

    total_iterations = em_config.get("total_iterations", 10)
    epochs_per_update = em_config.get("epochs_per_update", 5)
    chunk_size = simulation.chunk_size
    seed = simulation.seed
    chunks_per_update = training.chunks_per_update
    log_interval = training.log_interval
    checkpoint_interval = training.checkpoint_interval
    plot_size = training.plot_size
    mixed_precision = training.mixed_precision
    grad_norm_clip = hyperparameters.grad_norm_clip
    burn_in_chunks = getattr(training, "burn_in_chunks", 0)
    weight_perturbation_variance = training.weight_perturbation_variance
    optimisable = training.optimisable
    beta1 = hyperparameters.beta1
    beta2 = hyperparameters.beta2
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

    recurrent_synapse_params = recurrent.get_synapse_params()
    feedforward_synapse_params = feedforward.get_synapse_params()
    n_ff_synapse_types = len(feedforward_synapse_params)

    # ============================================
    # Scaling Factors and Perturbation
    # ============================================

    sf_feedforward = np.array(scaling_factors["feedforward"])
    sf_recurrent = np.array(scaling_factors["recurrent"])

    concatenated_scaling_factors = np.concatenate(
        [sf_feedforward, sf_recurrent], axis=0
    )

    # Apply perturbation
    sigma = np.sqrt(weight_perturbation_variance)
    mu = -(sigma**2) / 2.0
    perturbation_factors = np.random.lognormal(
        mean=mu, sigma=sigma, size=concatenated_scaling_factors.shape
    )
    target_scaling_factors_FF = 1.0 / perturbation_factors

    # ===============================================
    # Combined Input Cell Types and Parameters
    # ===============================================

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
    # Setup Models Based on Loss Type
    # ===============================================

    if loss_type == "visible":
        # M-step model outputs visible neurons only
        model_weights = np.concatenate(
            [
                perturbed_ff_weights[:, visible_indices],
                perturbed_rec_weights[:, visible_indices],
            ],
            axis=0,
        )
        model_mask = np.concatenate(
            [
                feedforward_mask_np[:, visible_indices],
                recurrent_mask[:, visible_indices],
            ],
            axis=0,
        )
        model_cell_type_indices = cell_type_indices[visible_indices]
        n_model_outputs = n_visible
    else:  # loss_type == "full"
        # M-step model outputs all neurons
        model_weights = np.concatenate(
            [
                perturbed_ff_weights,
                perturbed_rec_weights,
            ],
            axis=0,
        )
        model_mask = np.concatenate(
            [
                feedforward_mask_np,
                recurrent_mask,
            ],
            axis=0,
        )
        model_cell_type_indices = cell_type_indices
        n_model_outputs = n_neurons_full

    # Mask the perturbed weights up-front so projection builders see zeros
    # on inactive connections.
    masked_model_weights = model_weights * model_mask.astype(np.float32)

    # Cell type name lists. The chunked-FF M-step model treats both true-FF
    # and recurrent inputs as feedforward; the combined input-cell-type space
    # is FF names followed by recurrent names (matching combined_cell_params_FF).
    rec_output_cell_type_names = [cp["name"] for cp in recurrent_cell_params]
    ff_input_cell_type_names = [cp["name"] for cp in feedforward_cell_params]
    combined_input_cell_type_names = [cp["name"] for cp in combined_cell_params_FF]

    def _build_mstep_projections(sf_matrix: np.ndarray) -> dict:
        """Build trainable ScalingFactorProjections for the M-step model.

        Each pair holds the (masked, unscaled) connectome block as a buffer
        and a trainable ``log_sf`` initialised to ``log(sf_matrix[src, tgt])``.
        """
        projs = make_chunked_ff_projections(
            rec_weights=masked_model_weights[n_feedforward:, :],
            ff_weights=masked_model_weights[:n_feedforward, :],
            cell_type_indices=model_cell_type_indices,
            ff_cell_type_indices=concatenated_cell_type_indices[:n_feedforward],
            cell_type_names=rec_output_cell_type_names,
            ff_cell_type_names=ff_input_cell_type_names,
            init_sf=1.0,
        )
        with torch.no_grad():
            for (src_name, tgt_name), proj in projs.items():
                src_id = combined_input_cell_type_names.index(src_name)
                tgt_id = rec_output_cell_type_names.index(tgt_name)
                proj.log_sf.copy_(
                    torch.tensor(
                        float(np.log(sf_matrix[src_id, tgt_id])),
                        dtype=proj.log_sf.dtype,
                    )
                )
        return projs

    if optimisable != "scaling_factors":
        raise NotImplementedError(
            f"hidden-activity EM only handles optimisable='scaling_factors', "
            f"got {optimisable!r}."
        )

    # Save targets
    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        targets_dir / "target_scaling_factors.npz",
        feedforward_scaling_factors=target_scaling_factors_FF,
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

    # ======================================
    # Initialize M-step Model
    # ======================================

    feedforward_model = FeedforwardConductanceLIFNetwork(
        dt=dt,
        projections=_build_mstep_projections(concatenated_scaling_factors),
        cell_type_indices=model_cell_type_indices,
        cell_type_indices_FF=concatenated_cell_type_indices,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        track_variables=False,
    )
    feedforward_model.to(device)

    print(f"M-step model: {n_model_outputs} output neurons")

    # ==============================
    # Loss Function
    # ==============================

    van_rossum_loss_fn = VanRossumLoss(
        tau_rise=van_rossum_tau_rise,
        tau_decay=van_rossum_tau_decay,
        dt=dt,
        window_size=chunk_size,
        device=device,
    )

    cma_loss_weights = {"van_rossum": loss_weight_van_rossum}

    gradient_loss_functions = {"van_rossum": van_rossum_loss_fn}
    gradient_loss_weights = {"van_rossum": loss_weight_van_rossum}

    # ================================================
    # Initialise wandb
    # ================================================

    wandb_logger = None
    if wandb_config:
        # Use name from config if provided and non-empty, otherwise use output_dir name
        run_name = wandb_config.pop("name", None) or output_dir.name
        wandb_config_dict = {
            **wandb_config,
            "config": {
                "e_step_type": e_step_type,
                "loss_type": loss_type,
                "hidden_cell_fraction": hidden_cell_fraction,
                "n_hidden": n_hidden,
                "n_visible": n_visible,
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

    # Create master metrics logger at root output directory
    master_metrics_logger = AsyncLogger(
        log_dir=output_dir,
        max_queue_size=10,
    )

    # ================================================
    # Phase 0: CMA-ES Evolutionary Search (before EM)
    # ================================================

    # Check if CMA-ES already completed (resume loads saved results)
    cma_state_path = output_dir / "cma_es_state" / "scaling_factors.npz"
    if resume_from is not None and cma_state_path.exists():
        print("\nLoading CMA-ES results from previous run...")
        cma_saved = np.load(cma_state_path)
        best_sf = cma_saved["scaling_factors_FF"]
        best_cma_loss = float(cma_saved["best_loss"])
        print(f"  Scaling factors: {best_sf}")
        print(f"  Best CMA-ES loss: {best_cma_loss:.6f}")

        concatenated_scaling_factors = best_sf
        sf_feedforward = best_sf[:n_ff_cell_types]
        sf_recurrent = best_sf[n_ff_cell_types:]

        feedforward_model = FeedforwardConductanceLIFNetwork(
            dt=dt,
            projections=_build_mstep_projections(concatenated_scaling_factors),
            cell_type_indices=model_cell_type_indices,
            cell_type_indices_FF=concatenated_cell_type_indices,
            cell_params=recurrent_cell_params,
            cell_params_FF=combined_cell_params_FF,
            synapse_params_FF=combined_synapse_params_FF,
            surrgrad_scale=surrgrad_scale,
            batch_size=batch_size,
            track_variables=False,
        )
        feedforward_model.to(device)

    elif cma_es_config:
        header = "PHASE 0: CMA-ES Evolutionary Search"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))

        # --- Build E-step model (clamped visible, infer hidden) ---
        # Weight slicing: same logic as run_estep_clamped
        cma_estep_weights_FF = np.concatenate(
            [
                perturbed_ff_weights[:, hidden_indices],
                perturbed_rec_weights[visible_indices][:, hidden_indices],
            ],
            axis=0,
        )
        cma_estep_mask_FF = np.concatenate(
            [
                feedforward_mask_np[:, hidden_indices],
                recurrent_mask[visible_indices][:, hidden_indices],
            ],
            axis=0,
        )
        cma_estep_weights_rec = perturbed_rec_weights[hidden_indices][:, hidden_indices]
        cma_estep_mask_rec = recurrent_mask[hidden_indices][:, hidden_indices]
        cma_estep_cell_type_indices_FF = np.concatenate(
            [
                feedforward_cell_type_indices,
                cell_type_indices[visible_indices] + n_ff_cell_types,
            ]
        )

        # Build CMA E-step projections (frozen, SFs baked in). The rec part
        # uses sf_recurrent (n_rec_ct x n_rec_ct); the FF part uses
        # [sf_feedforward; sf_recurrent] indexed by combined input names.
        cma_estep_rec_masked = cma_estep_weights_rec * cma_estep_mask_rec.astype(
            np.float32
        )
        cma_estep_ff_masked = cma_estep_weights_FF * cma_estep_mask_FF.astype(
            np.float32
        )
        cma_estep_initial_sf_ff = np.concatenate([sf_feedforward, sf_recurrent], axis=0)
        cma_estep_rec_projs, cma_estep_ff_projs = make_frozen_projections(
            rec_weights=cma_estep_rec_masked,
            ff_weights=cma_estep_ff_masked,
            cell_type_indices=cell_type_indices[hidden_indices],
            ff_cell_type_indices=cma_estep_cell_type_indices_FF,
            cell_type_names=rec_output_cell_type_names,
            ff_cell_type_names=combined_input_cell_type_names,
            scaling_factors_rec=sf_recurrent,
            scaling_factors_ff=cma_estep_initial_sf_ff,
        )

        cma_estep_model = ConductanceLIFNetwork(
            dt=dt,
            rec_projections=cma_estep_rec_projs,
            ff_projections=cma_estep_ff_projs,
            cell_type_indices=cell_type_indices[hidden_indices],
            cell_type_indices_FF=cma_estep_cell_type_indices_FF,
            cell_params=recurrent_cell_params,
            cell_params_FF=combined_cell_params_FF,
            synapse_params=recurrent_synapse_params,
            synapse_params_FF=combined_synapse_params_FF,
            surrgrad_scale=surrgrad_scale,
            batch_size=batch_size,
            track_variables=False,
        )
        cma_estep_model.to(device)

        # Per-pair unscaled connectome blocks for rebake during update_scaling_factors.
        def _collect_pair_blocks(weights, src_indices_per_ct, tgt_indices_per_ct):
            return {
                (src, tgt): weights[
                    np.ix_(src_indices_per_ct[src], tgt_indices_per_ct[tgt])
                ].astype(np.float32)
                for src in src_indices_per_ct
                for tgt in tgt_indices_per_ct
            }

        _rec_idx_hidden = {
            name: np.flatnonzero(cell_type_indices[hidden_indices] == ct_id)
            for ct_id, name in enumerate(rec_output_cell_type_names)
        }
        _ff_idx_estep = {
            name: np.flatnonzero(cma_estep_cell_type_indices_FF == ct_id)
            for ct_id, name in enumerate(combined_input_cell_type_names)
        }
        cma_estep_rec_blocks = _collect_pair_blocks(
            cma_estep_rec_masked, _rec_idx_hidden, _rec_idx_hidden
        )
        cma_estep_ff_blocks = _collect_pair_blocks(
            cma_estep_ff_masked, _ff_idx_estep, _rec_idx_hidden
        )
        cma_estep_rec_src_name_to_row = {
            name: i for i, name in enumerate(rec_output_cell_type_names)
        }
        cma_estep_rec_tgt_name_to_col = {
            name: i for i, name in enumerate(rec_output_cell_type_names)
        }
        cma_estep_ff_src_name_to_row = {
            name: i for i, name in enumerate(combined_input_cell_type_names)
        }
        cma_estep_ff_tgt_name_to_col = dict(cma_estep_rec_tgt_name_to_col)

        # --- Build M-step model (always visible-only output for CMA-ES) ---
        cma_mstep_weights = np.concatenate(
            [
                perturbed_ff_weights[:, visible_indices],
                perturbed_rec_weights[:, visible_indices],
            ],
            axis=0,
        )
        cma_mstep_mask = np.concatenate(
            [
                feedforward_mask_np[:, visible_indices],
                recurrent_mask[:, visible_indices],
            ],
            axis=0,
        )

        cma_mstep_masked_weights = cma_mstep_weights * cma_mstep_mask.astype(np.float32)
        cma_mstep_projections = make_frozen_chunked_ff_projections(
            rec_weights=cma_mstep_masked_weights[n_feedforward:, :],
            ff_weights=cma_mstep_masked_weights[:n_feedforward, :],
            cell_type_indices=cell_type_indices[visible_indices],
            ff_cell_type_indices=concatenated_cell_type_indices[:n_feedforward],
            cell_type_names=rec_output_cell_type_names,
            ff_cell_type_names=ff_input_cell_type_names,
            scaling_factors=concatenated_scaling_factors.astype(np.float32),
        )
        cma_mstep_model = FeedforwardConductanceLIFNetwork(
            dt=dt,
            projections=cma_mstep_projections,
            cell_type_indices=cell_type_indices[visible_indices],
            cell_type_indices_FF=concatenated_cell_type_indices,
            cell_params=recurrent_cell_params,
            cell_params_FF=combined_cell_params_FF,
            synapse_params_FF=combined_synapse_params_FF,
            surrgrad_scale=surrgrad_scale,
            batch_size=batch_size,
            track_variables=False,
        )
        cma_mstep_model.to(device)

        # Per-pair unscaled blocks + index maps for in-place SF rewrites.
        _rec_idx_visible = {
            name: np.flatnonzero(cell_type_indices[visible_indices] == ct_id)
            for ct_id, name in enumerate(rec_output_cell_type_names)
        }
        _ff_idx_mstep_full = {
            name: np.flatnonzero(concatenated_cell_type_indices == ct_id)
            for ct_id, name in enumerate(combined_input_cell_type_names)
        }
        # Source-row offsets: for FF source names rows live at [0, n_feedforward);
        # for rec source names rows live at [n_feedforward, total).
        cma_mstep_blocks: dict = {}
        for src_name in combined_input_cell_type_names:
            src_rows = _ff_idx_mstep_full[src_name]
            for tgt_name in rec_output_cell_type_names:
                tgt_cols = _rec_idx_visible[tgt_name]
                cma_mstep_blocks[(src_name, tgt_name)] = cma_mstep_masked_weights[
                    np.ix_(src_rows, tgt_cols)
                ].astype(np.float32)
        cma_mstep_src_name_to_row = {
            name: i for i, name in enumerate(combined_input_cell_type_names)
        }
        cma_mstep_tgt_name_to_col = {
            name: i for i, name in enumerate(rec_output_cell_type_names)
        }

        # --- Create wrapper ---
        cma_wrapper = CMAEMWrapper(
            estep_model=cma_estep_model,
            mstep_model=cma_mstep_model,
            n_feedforward=n_feedforward,
            visible_indices=visible_indices,
            hidden_indices=hidden_indices,
            n_ff_cell_types=n_ff_cell_types,
            estep_rec_pair_blocks=cma_estep_rec_blocks,
            estep_rec_src_name_to_row=cma_estep_rec_src_name_to_row,
            estep_rec_tgt_name_to_col=cma_estep_rec_tgt_name_to_col,
            estep_ff_pair_blocks=cma_estep_ff_blocks,
            estep_ff_src_name_to_row=cma_estep_ff_src_name_to_row,
            estep_ff_tgt_name_to_col=cma_estep_ff_tgt_name_to_col,
            mstep_pair_blocks=cma_mstep_blocks,
            mstep_src_name_to_row=cma_mstep_src_name_to_row,
            mstep_tgt_name_to_col=cma_mstep_tgt_name_to_col,
        )

        sf_shape = concatenated_scaling_factors.shape

        # Track the most-recent SFs written into the wrapper, so the callback
        # (and post-search readout) can recover them without relying on
        # `mstep_model.scaling_factors_FF` which is all-ones for a
        # FrozenProjection-only model.
        current_cma_sf = concatenated_scaling_factors.astype(np.float32).copy()

        def cma_update_fn(flat_log_params: np.ndarray) -> None:
            sf_np = np.exp(flat_log_params).reshape(sf_shape).astype(np.float32)
            current_cma_sf[:] = sf_np
            sf_tensor = torch.from_numpy(sf_np).to(device)
            cma_wrapper.update_scaling_factors(sf_tensor)

        cma_van_rossum_fn = VanRossumLoss(
            tau_rise=van_rossum_tau_rise,
            tau_decay=van_rossum_tau_decay,
            dt=dt,
            window_size=chunk_size,
            device=device,
        )
        cma_loss_functions = {"van_rossum": cma_van_rossum_fn}

        # Collate: [FF, all_teacher_rec] as input, visible-only targets
        cma_collate = CMACollateVisibleTarget(torch.from_numpy(visible_indices).long())

        cma_dataloader = DataLoader(
            spike_dataset,
            batch_size=None,
            sampler=CyclicSampler(spike_dataset),
            num_workers=0,
            collate_fn=cma_collate,
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

            # Scaling factors vs targets (read from externally-tracked array;
            # the wrapper's submodels are FrozenProjection-only).
            current_sf = current_cma_sf
            for source_idx in range(current_sf.shape[0]):
                source_name = input_cell_type_names[source_idx]
                for target_idx in range(current_sf.shape[1]):
                    target_name = output_cell_type_names[target_idx]
                    synapse_name = f"{source_name}_to_{target_name}"
                    target_val = target_scaling_factors_FF[source_idx, target_idx]
                    if target_val != 0:
                        normalized = current_sf[source_idx, target_idx] / target_val
                    else:
                        normalized = current_sf[source_idx, target_idx]
                    log_dict[f"cma_es_scaling_factors/{synapse_name}_value"] = float(
                        normalized
                    )
                    log_dict[f"cma_es_scaling_factors/{synapse_name}_target"] = 1.0

            # Firing rates from output spikes (visible neurons)
            output_spikes = metrics["output_spikes"].numpy()
            student_spikes = output_spikes[0]  # (time, n_visible)
            duration_s = student_spikes.shape[0] * dt / 1000.0
            student_rates = student_spikes.sum(axis=0) / duration_s
            visible_cell_types = cell_type_indices[visible_indices]
            for type_idx, type_name in enumerate(output_cell_type_names):
                mask = visible_cell_types == type_idx
                if mask.sum() > 0:
                    log_dict[f"cma_es_firing_rate/student_visible_{type_name}_mean"] = (
                        float(student_rates[mask].mean())
                    )

            # Log to disk
            master_metrics_logger.log(epoch=-metrics["generation"], **log_dict)

            # Log to wandb
            if wandb_logger:
                wandb.log(log_dict)

        searcher = EvolutionarySearch(
            model=cma_wrapper,
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
            callback=cma_es_callback,
            burn_in_chunks=burn_in_chunks,
        )

        best_cma_loss = searcher.search()

        # Extract best scaling factors from the externally-tracked array;
        # EvolutionarySearch writes the best candidate via update_fn at end.
        best_sf = current_cma_sf.copy()
        print(f"\nCMA-ES best loss: {best_cma_loss:.6f}")

        # Save post-CMA-ES state
        cma_state_dir = output_dir / "cma_es_state"
        cma_state_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            cma_state_dir / "scaling_factors.npz",
            scaling_factors_FF=best_sf,
            best_loss=best_cma_loss,
        )

        # Update concatenated_scaling_factors so EM loop uses CMA-ES result
        concatenated_scaling_factors = best_sf
        sf_feedforward = best_sf[:n_ff_cell_types]
        sf_recurrent = best_sf[n_ff_cell_types:]

        # Rebuild M-step model with CMA-ES scaling factors
        feedforward_model = FeedforwardConductanceLIFNetwork(
            dt=dt,
            projections=_build_mstep_projections(concatenated_scaling_factors),
            cell_type_indices=model_cell_type_indices,
            cell_type_indices_FF=concatenated_cell_type_indices,
            cell_params=recurrent_cell_params,
            cell_params_FF=combined_cell_params_FF,
            synapse_params_FF=combined_synapse_params_FF,
            surrgrad_scale=surrgrad_scale,
            batch_size=batch_size,
            track_variables=False,
        )
        feedforward_model.to(device)

        # Free all CMA-ES GPU memory before EM loop
        searcher.update_fn = None
        searcher.callback = None
        searcher._chunks = []
        searcher.model = None
        searcher.loss_functions = {}
        del cma_wrapper, cma_estep_model, cma_mstep_model, searcher
        del cma_van_rossum_fn, cma_loss_functions
        del cma_dataloader, cma_update_fn, cma_es_callback
        gc.collect()
        torch.cuda.empty_cache()

    # ================================================
    # Open Teacher Data for Stats Computation
    # ================================================

    teacher_zarr_root = zarr.open_group(input_dir / "spike_data.zarr", mode="r")
    teacher_spikes_zarr = teacher_zarr_root["output_spikes"]

    # ================================================
    # Stats Computer Function
    # ================================================

    def make_stats_computer(num_chunks_local, inferred_spikes_path):
        """Create stats_computer function."""

        # Load inferred hidden spikes for student hidden firing rates
        inferred_zarr = zarr.open_group(inferred_spikes_path, mode="r")
        inferred_hidden_spikes = np.array(inferred_zarr["output_spikes"][0, :, :])
        inferred_duration_s = inferred_hidden_spikes.shape[0] * dt / 1000.0
        inferred_hidden_rates = inferred_hidden_spikes.sum(axis=0) / inferred_duration_s
        hidden_cell_types = cell_type_indices[hidden_indices]

        def stats_computer(snapshot):
            """Compute summary statistics for logging."""
            # spikes shape: (batch, time, n_neurons)
            student_spikes = snapshot["spikes"][0, :, :]  # (time, n_neurons)
            n_timesteps = student_spikes.shape[0]
            duration_s = n_timesteps * dt / 1000.0

            # Visible teacher rates from collate target_spikes.
            # For loss_type=="visible": target_spikes has visible neurons only.
            # For loss_type=="full": target_spikes has all neurons (visible from
            #   teacher, hidden from E-step) — index visible portion.
            target_spikes = snapshot["target_spikes"][0, :n_timesteps, :]
            if loss_type == "visible":
                teacher_visible_rates = target_spikes.sum(axis=0) / duration_s
            else:
                teacher_visible_rates = (
                    target_spikes[:, visible_indices].sum(axis=0) / duration_s
                )

            # Hidden teacher rates: use epoch from trainer to compute correct zarr window
            epoch = snapshot.get("epoch", 0)
            n_chunks_accumulated = n_timesteps // chunk_size
            start_chunk = (epoch - n_chunks_accumulated + 1) % num_chunks_local
            start_t = start_chunk * chunk_size
            end_t = start_t + n_timesteps

            total_timesteps = num_chunks_local * chunk_size
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

            visible_cell_types = cell_type_indices[visible_indices]

            stats = {}
            for type_idx, type_name in enumerate(output_cell_type_names):
                vis_mask = visible_cell_types == type_idx
                hid_mask = hidden_cell_types == type_idx
                if vis_mask.sum() > 0:
                    stats[f"firing_rate/teacher_visible_{type_name}_mean"] = float(
                        teacher_visible_rates[vis_mask].mean()
                    )
                    stats[f"firing_rate/teacher_visible_{type_name}_std"] = float(
                        teacher_visible_rates[vis_mask].std()
                    )
                if hid_mask.sum() > 0:
                    stats[f"firing_rate/teacher_hidden_{type_name}_mean"] = float(
                        teacher_hidden_rates[hid_mask].mean()
                    )
                    stats[f"firing_rate/teacher_hidden_{type_name}_std"] = float(
                        teacher_hidden_rates[hid_mask].std()
                    )

            # Student visible firing rates by cell type (from M-step model)
            student_firing_rates = student_spikes.sum(axis=0) / duration_s
            if loss_type == "visible":
                # Model only outputs visible neurons
                student_visible_rates = student_firing_rates
                student_visible_cell_types = model_cell_type_indices
            else:
                # Model outputs all neurons, index by visible
                student_visible_rates = student_firing_rates[visible_indices]
                student_visible_cell_types = cell_type_indices[visible_indices]

            for type_idx, type_name in enumerate(output_cell_type_names):
                vis_mask = student_visible_cell_types == type_idx
                if vis_mask.sum() > 0:
                    stats[f"firing_rate/student_visible_{type_name}_mean"] = float(
                        student_visible_rates[vis_mask].mean()
                    )
                    stats[f"firing_rate/student_visible_{type_name}_std"] = float(
                        student_visible_rates[vis_mask].std()
                    )

            # Student hidden firing rates by cell type (from E-step)
            if len(hidden_indices) > 0:
                for type_idx, type_name in enumerate(output_cell_type_names):
                    hid_mask = hidden_cell_types == type_idx
                    if hid_mask.sum() > 0:
                        stats[f"firing_rate/student_hidden_{type_name}_mean"] = float(
                            inferred_hidden_rates[hid_mask].mean()
                        )
                        stats[f"firing_rate/student_hidden_{type_name}_std"] = float(
                            inferred_hidden_rates[hid_mask].std()
                        )

            # Scaling factors (normalized so target=1)
            current_sf = snapshot["scaling_factors_FF"]
            target_sf = target_scaling_factors_FF

            for source_idx in range(current_sf.shape[0]):
                source_type_name = input_cell_type_names[source_idx]
                for target_idx in range(current_sf.shape[1]):
                    target_type_name = output_cell_type_names[target_idx]
                    synapse_name = f"{source_type_name}_to_{target_type_name}"
                    target_val = target_sf[source_idx, target_idx]
                    if target_val != 0:
                        normalized_value = (
                            current_sf[source_idx, target_idx] / target_val
                        )
                    else:
                        normalized_value = current_sf[source_idx, target_idx]
                    stats[f"scaling_factors/{synapse_name}_value"] = float(
                        normalized_value
                    )
                    stats[f"scaling_factors/{synapse_name}_target"] = 1.0

            return stats

        return stats_computer

    # ===================
    # EM Training Loop
    # ===================

    print("\nStarting EM training:")
    print(f"  Total iterations: {total_iterations}")
    print(f"  Epochs per M-step: {epochs_per_update}")

    best_loss_overall = float("inf")
    global_chunk_counter = 0
    previous_hidden_spikes_path = None  # For clamped_ff E-step
    start_em_iter = 0

    # Resume: detect last completed EM iteration
    em_state_path = output_dir / "em_state.json"
    if resume_from is not None and em_state_path.exists():
        with open(em_state_path, "r") as f:
            em_state = json.load(f)
        start_em_iter = em_state["completed_iterations"]
        global_chunk_counter = em_state["global_chunk_counter"]
        best_loss_overall = em_state["best_loss_overall"]
        sf_feedforward = np.array(em_state["sf_feedforward"])
        sf_recurrent = np.array(em_state["sf_recurrent"])
        if em_state.get("previous_hidden_spikes_path"):
            previous_hidden_spikes_path = (
                output_dir / em_state["previous_hidden_spikes_path"]
            )

        # Restore scaling factors into the feedforward model by writing
        # log(sf) directly into each ScalingFactorProjection's log_sf.
        concatenated_scaling_factors = np.concatenate(
            [sf_feedforward, sf_recurrent], axis=0
        )
        with torch.no_grad():
            for pair, proj in zip(
                feedforward_model._pairs, feedforward_model.projections.values()
            ):
                src_name, tgt_name = pair
                src_id = combined_input_cell_type_names.index(src_name)
                tgt_id = rec_output_cell_type_names.index(tgt_name)
                proj.log_sf.copy_(
                    torch.tensor(
                        float(np.log(concatenated_scaling_factors[src_id, tgt_id])),
                        dtype=proj.log_sf.dtype,
                        device=proj.log_sf.device,
                    )
                )

        print(f"\n  Resuming from EM iteration {start_em_iter + 1}")
        print(f"  Global chunk counter: {global_chunk_counter}")
        print(f"  Best loss so far: {best_loss_overall:.6f}")

    for em_iter in range(start_em_iter, total_iterations):
        print(f"\n{'=' * 60}")
        print(f"EM Iteration {em_iter + 1}/{total_iterations}")
        print(f"{'=' * 60}")

        em_iter_output_dir = output_dir / f"em_iter_{em_iter + 1:03d}"
        em_iter_output_dir.mkdir(parents=True, exist_ok=True)

        # ===== E-STEP =====
        inferred_spikes_path = em_iter_output_dir / "inferred_spikes.zarr"

        # Skip E-step if inferred spikes already exist (resume case)
        if inferred_spikes_path.exists():
            print(f"\n--- E-Step ({e_step_type}) --- SKIPPED (already exists)")
            # For clamped_ff mode, update previous_hidden_spikes_path
            if e_step_type == "clamped_ff":
                previous_hidden_spikes_path = inferred_spikes_path
        else:
            print(f"\n--- E-Step ({e_step_type}) ---")

            if e_step_type == "free":
                run_estep_free(
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
                    feedforward_mask=feedforward_mask_np,
                    hidden_indices=hidden_indices,
                    spike_dataset=spike_dataset,
                    num_chunks=num_chunks,
                    chunk_size=chunk_size,
                    output_path=inferred_spikes_path,
                )
            elif e_step_type == "clamped":
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
                    feedforward_mask=feedforward_mask_np,
                    visible_indices=visible_indices,
                    hidden_indices=hidden_indices,
                    spike_dataset=spike_dataset,
                    num_chunks=num_chunks,
                    chunk_size=chunk_size,
                    output_path=inferred_spikes_path,
                    n_ff_cell_types=n_ff_cell_types,
                )
            elif e_step_type == "clamped_ff":
                if previous_hidden_spikes_path is None:
                    # First iteration: use full recurrent clamped model for good init
                    print("  (Using recurrent clamped model for first E-step)")
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
                        feedforward_mask=feedforward_mask_np,
                        visible_indices=visible_indices,
                        hidden_indices=hidden_indices,
                        spike_dataset=spike_dataset,
                        num_chunks=num_chunks,
                        chunk_size=chunk_size,
                        output_path=inferred_spikes_path,
                        n_ff_cell_types=n_ff_cell_types,
                    )
                else:
                    # Subsequent iterations: feedforward with previous hidden spikes
                    run_estep_clamped_feedforward(
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
                        feedforward_mask=feedforward_mask_np,
                        visible_indices=visible_indices,
                        hidden_indices=hidden_indices,
                        spike_dataset=spike_dataset,
                        num_chunks=num_chunks,
                        chunk_size=chunk_size,
                        output_path=inferred_spikes_path,
                        n_ff_cell_types=n_ff_cell_types,
                        previous_hidden_spikes_path=previous_hidden_spikes_path,
                    )
                previous_hidden_spikes_path = inferred_spikes_path

        print(f"  Inferred spikes saved to: {inferred_spikes_path}")

        # Rescale hidden firing rates to match visible rates
        if poisson_trimming and n_hidden > 0:
            rescale_hidden_firing_rates(
                inferred_spikes_path=inferred_spikes_path,
                teacher_spikes_zarr=teacher_spikes_zarr,
                visible_indices=visible_indices,
                hidden_indices=hidden_indices,
                cell_type_indices=cell_type_indices,
                cell_type_names=output_cell_type_names,
                dt=dt,
            )

        # ===== INITIAL SPIKE COMPARISON (first EM iter only) =====
        if em_iter == 0:
            print("\n--- Initial inference spike comparison ---")

            # Run M-step model forward (no training) to get student output
            visible_tensor_plot = torch.from_numpy(visible_indices).long()
            hidden_tensor_plot = torch.from_numpy(hidden_indices).long()
            inferred_zarr_plot = zarr.open_group(inferred_spikes_path, mode="r")
            inferred_spikes_plot = inferred_zarr_plot["output_spikes"]

            feedforward_model.reset_state(batch_size=batch_size)
            feedforward_model.track_variables = False

            n_chunks_plot = min(plot_size, num_chunks - burn_in_chunks)
            n_total_chunks = burn_in_chunks + n_chunks_plot
            data_iter = iter(
                DataLoader(
                    spike_dataset,
                    batch_size=None,
                    sampler=CyclicSampler(spike_dataset),
                    num_workers=0,
                )
            )

            student_chunks = []
            with torch.inference_mode():
                for i in range(n_total_chunks):
                    batch = next(data_iter)
                    ff = batch.input_spikes.to(device).float()
                    rec = torch.zeros(
                        batch_size,
                        chunk_size,
                        n_neurons_full,
                        device=device,
                        dtype=torch.float32,
                    )
                    rec[:, :, visible_tensor_plot] = (
                        batch.target_spikes[:, :, visible_tensor_plot]
                        .float()
                        .to(device)
                    )
                    start_t = i * chunk_size
                    end_t = start_t + chunk_size
                    hid_chunk = (
                        torch.from_numpy(
                            np.array(
                                inferred_spikes_plot[:batch_size, start_t:end_t, :]
                            )
                        )
                        .float()
                        .to(device)
                    )
                    rec[:, :, hidden_tensor_plot] = hid_chunk
                    output = feedforward_model.forward(
                        input_spikes=torch.cat([ff, rec], dim=2)
                    )
                    # Only collect chunks after burn-in
                    if i >= burn_in_chunks:
                        student_chunks.append(output.detach().cpu().numpy())

            # shape: (batch, total_time, n_model_out)
            student_spikes = np.concatenate(student_chunks, axis=1)
            total_time_plot = n_chunks_plot * chunk_size
            burn_in_time = burn_in_chunks * chunk_size

            import matplotlib.pyplot as plt

            figures_dir = output_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

            # --- Visible neuron comparison ---
            teacher_vis = np.array(
                teacher_spikes_zarr[
                    0, burn_in_time : burn_in_time + total_time_plot, visible_indices
                ]
            )
            if loss_type == "visible":
                student_vis = student_spikes[0]
            else:
                student_vis = student_spikes[0][:, visible_indices]

            n_plot_vis = min(10, student_vis.shape[1])
            interleaved_vis = np.zeros((1, total_time_plot, 2 * n_plot_vis))
            for i in range(n_plot_vis):
                interleaved_vis[0, :, 2 * i] = teacher_vis[:, i]
                interleaved_vis[0, :, 2 * i + 1] = student_vis[:, i]

            fig_vis = plot_spike_trains(
                spikes=interleaved_vis,
                dt=dt,
                cell_type_indices=np.array([0, 1] * n_plot_vis),
                cell_type_names=["Target", "Student"],
                n_neurons_plot=2 * n_plot_vis,
                n_compared=2,
                fraction=1.0,
                random_seed=None,
                title=f"Initial: Target vs Student (first {n_plot_vis} visible neurons)",
                ylabel="Neuron",
                figsize=(14, 8),
            )
            fig_vis.savefig(
                figures_dir / "initial_spike_comparison_visible.png", dpi=150
            )
            plt.close(fig_vis)

            # --- Hidden neuron comparison ---
            if n_hidden > 0:
                teacher_hid = np.array(
                    teacher_spikes_zarr[
                        0, burn_in_time : burn_in_time + total_time_plot, hidden_indices
                    ]
                )
                inferred_hid = np.array(
                    inferred_spikes_plot[
                        0, burn_in_time : burn_in_time + total_time_plot, :
                    ]
                )

                n_plot_hid = min(10, inferred_hid.shape[1])
                interleaved_hid = np.zeros((1, total_time_plot, 2 * n_plot_hid))
                for i in range(n_plot_hid):
                    interleaved_hid[0, :, 2 * i] = teacher_hid[:, i]
                    interleaved_hid[0, :, 2 * i + 1] = inferred_hid[:, i]

                fig_hid = plot_spike_trains(
                    spikes=interleaved_hid,
                    dt=dt,
                    cell_type_indices=np.array([0, 1] * n_plot_hid),
                    cell_type_names=["Teacher", "Inferred"],
                    n_neurons_plot=2 * n_plot_hid,
                    n_compared=2,
                    fraction=1.0,
                    random_seed=None,
                    title=f"Initial: Teacher vs Inferred (first {n_plot_hid} hidden neurons)",
                    ylabel="Neuron",
                    figsize=(14, 8),
                )
                fig_hid.savefig(
                    figures_dir / "initial_spike_comparison_hidden.png", dpi=150
                )
                plt.close(fig_hid)

            print(f"  Saved to: {figures_dir}")

        # ===== M-STEP =====
        print(f"\n--- M-Step (loss={loss_type}) ---")

        collate_fn = MStepCollate(
            visible_indices=visible_indices,
            hidden_indices=hidden_indices,
            inferred_spikes_zarr_path=inferred_spikes_path,
            n_neurons_full=n_neurons_full,
            num_chunks=num_chunks,
            loss_type=loss_type,
        )
        collate_fn.set_loss(van_rossum_loss_fn)

        # Create optimizer
        optimiser = torch.optim.Adam(
            feedforward_model.parameters(), lr=learning_rate, betas=(beta1, beta2)
        )
        scaler = GradScaler("cuda", enabled=mixed_precision and device == "cuda")

        van_rossum_loss_fn.reset_state()

        num_chunks_m_step = epochs_per_update * num_chunks

        lr_min = getattr(hyperparameters, "lr_min", None)
        if lr_min is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=epochs_per_update, eta_min=lr_min
            )
            print(
                f"  LR schedule: cosine {learning_rate} → {lr_min} over {epochs_per_update} epochs"
            )
        else:
            scheduler = None

        spike_dataloader = DataLoader(
            spike_dataset,
            batch_size=None,
            sampler=CyclicSampler(spike_dataset),
            num_workers=0,
            collate_fn=collate_fn,
        )

        pbar = tqdm(
            range(num_chunks_m_step),
            desc=f"M-step (EM {em_iter + 1})",
            unit="chunk",
        )

        # Create stats computer for this M-step iteration
        stats_computer = make_stats_computer(num_chunks, inferred_spikes_path)

        # Plot generator for checkpoint spike comparisons
        def plot_generator(spikes, target_spikes, input_spikes, **kwargs):
            figures = {}

            # --- Visible neuron comparison ---
            if loss_type == "visible":
                student_vis = spikes[0]
                target_vis = target_spikes[0]
            else:
                student_vis = spikes[0][:, visible_indices]
                target_vis = target_spikes[0][:, visible_indices]

            n_plot_vis = min(10, student_vis.shape[1])
            interleaved_vis = np.zeros((1, spikes.shape[1], 2 * n_plot_vis))
            for i in range(n_plot_vis):
                interleaved_vis[0, :, 2 * i] = target_vis[:, i]
                interleaved_vis[0, :, 2 * i + 1] = student_vis[:, i]

            figures["spike_comparison_visible"] = plot_spike_trains(
                spikes=interleaved_vis,
                dt=dt,
                cell_type_indices=np.array([0, 1] * n_plot_vis),
                cell_type_names=["Target", "Trained"],
                n_neurons_plot=2 * n_plot_vis,
                n_compared=2,
                fraction=1.0,
                random_seed=None,
                title=f"EM {em_iter + 1}: Target vs Trained ({n_plot_vis} visible)",
                ylabel="Neuron",
                figsize=(14, 8),
            )

            # --- Hidden neuron comparison ---
            if loss_type == "full" and n_hidden > 0:
                student_hid = spikes[0][:, hidden_indices]
                target_hid = target_spikes[0][:, hidden_indices]

                n_plot_hid = min(10, student_hid.shape[1])
                interleaved_hid = np.zeros((1, spikes.shape[1], 2 * n_plot_hid))
                for i in range(n_plot_hid):
                    interleaved_hid[0, :, 2 * i] = target_hid[:, i]
                    interleaved_hid[0, :, 2 * i + 1] = student_hid[:, i]

                figures["spike_comparison_hidden"] = plot_spike_trains(
                    spikes=interleaved_hid,
                    dt=dt,
                    cell_type_indices=np.array([0, 1] * n_plot_hid),
                    cell_type_names=["Inferred", "Trained"],
                    n_neurons_plot=2 * n_plot_hid,
                    n_compared=2,
                    fraction=1.0,
                    random_seed=None,
                    title=f"EM {em_iter + 1}: Inferred vs Trained ({n_plot_hid} hidden)",
                    ylabel="Neuron",
                    figsize=(14, 8),
                )

            return figures

        # Create trainer
        trainer = SNNTrainer(
            model=feedforward_model,
            optimizer=optimiser,
            scaler=scaler,
            dataloader=spike_dataloader,
            loss_functions=gradient_loss_functions,
            loss_weights=gradient_loss_weights,
            device=device,
            num_epochs=global_chunk_counter + num_chunks_m_step,
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

        # Use master metrics logger (logs to root output_dir)
        trainer.metrics_logger = master_metrics_logger

        trainer.current_epoch = global_chunk_counter
        collate_fn.reset_counter()
        feedforward_model.reset_state(batch_size=batch_size)

        # Resume M-step from checkpoint if one exists (interrupted iteration)
        ckpt_dir = em_iter_output_dir / "checkpoints"
        if resume_from is not None and ckpt_dir.exists():
            pts = sorted(ckpt_dir.glob("*.pt"))
            if pts:
                print(f"  Resuming M-step from {pts[-1].name}")
                start_epoch, ckpt_best_loss = load_checkpoint(
                    checkpoint_path=pts[-1],
                    model=feedforward_model,
                    optimiser=optimiser,
                    scaler=scaler,
                    device=device,
                )
                trainer.set_checkpoint_state(start_epoch, ckpt_best_loss)
                pbar.initial = start_epoch - global_chunk_counter
                pbar.refresh()

        best_loss = trainer.train(output_dir=em_iter_output_dir)

        if best_loss < best_loss_overall:
            best_loss_overall = best_loss

        print(f"  M-step best loss: {best_loss:.6f}")

        global_chunk_counter += num_chunks_m_step

        # Update scaling factors for next E-step
        updated_sf = feedforward_model.scaling_factors_FF.detach().cpu().numpy()
        sf_feedforward = updated_sf[:n_ff_cell_types]
        sf_recurrent = updated_sf[n_ff_cell_types:]

        # Save EM state for resume
        prev_path_rel = None
        if previous_hidden_spikes_path is not None:
            try:
                prev_path_rel = str(previous_hidden_spikes_path.relative_to(output_dir))
            except ValueError:
                prev_path_rel = str(previous_hidden_spikes_path)
        em_state = {
            "completed_iterations": em_iter + 1,
            "global_chunk_counter": global_chunk_counter,
            "best_loss_overall": best_loss_overall,
            "sf_feedforward": sf_feedforward.tolist(),
            "sf_recurrent": sf_recurrent.tolist(),
            "previous_hidden_spikes_path": prev_path_rel,
        }
        with open(output_dir / "em_state.json", "w") as f:
            json.dump(em_state, f, indent=2)

        # Clear resume flag after first resumed iteration completes
        resume_from = None

    # Final summary
    print(f"\n{'=' * 60}")
    print("EM Training Complete")
    print(f"  Best loss overall: {best_loss_overall:.6f}")
    print(f"{'=' * 60}")

    # Close master metrics logger
    if master_metrics_logger:
        master_metrics_logger.close()

    if wandb_logger:
        wandb.finish()

    return best_loss_overall
