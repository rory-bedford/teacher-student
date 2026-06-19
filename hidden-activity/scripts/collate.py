"""Collate functions for hidden-activity experiments."""

import torch

from connectome_snns.dataloaders.supervised import SpikeData


class VisibleTargetCollate:
    """Slice target spikes to visible neurons only.

    For recurrent models where the model outputs all neurons but loss is
    computed on visible neurons only. Input spikes are passed through unchanged.

    Args:
        visible_indices: Tensor of visible neuron indices into target_spikes.
    """

    def __init__(self, visible_indices: torch.Tensor):
        self.visible_indices = visible_indices

    def __call__(self, batch: SpikeData) -> SpikeData:
        return SpikeData(
            input_spikes=batch.input_spikes,
            target_spikes=batch.target_spikes[:, :, self.visible_indices],
        )


class VisibleDrivenCollate:
    """Concatenate [FF, teacher_visible] as input; teacher_visible as target.

    For models that receive teacher visible spikes as feedforward input
    (ignore-hidden, visible-driven, full-inference). The visible spikes are
    extracted from target_spikes and appended to input_spikes.

    Args:
        visible_indices: Tensor of visible neuron indices into target_spikes.
    """

    def __init__(self, visible_indices: torch.Tensor):
        self.visible_indices = visible_indices

    def __call__(self, batch: SpikeData) -> SpikeData:
        visible_rec = batch.target_spikes[:, :, self.visible_indices]
        return SpikeData(
            input_spikes=torch.cat([batch.input_spikes, visible_rec], dim=2),
            target_spikes=visible_rec,
        )
