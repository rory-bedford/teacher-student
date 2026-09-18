"""Training entry point shared by every figure's ``train.py``.

Settings are read explicitly from ``parameters.toml``; a missing key is an error rather
than a silent default. (The archived scripts read ``eps`` and ``grad_norm_clip_ff``
through a config class that dropped them, so neither ever took effect — the values
here are the ones that actually ran.)
"""

import math
from pathlib import Path

import numpy as np
import toml
import torch
import wandb
from connectome_snns.configs import DATALOADER_KWARGS
from connectome_snns.dataloaders.supervised import CyclicSampler, ExactFFDataset
from connectome_snns.snn_runners import SNNTrainer
from connectome_snns.training_utils import AsyncLogger
from connectome_snns.training_utils.losses import VanRossumLoss
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.model import (
    StudentCollate,
    build_student,
    count_free_parameters,
    neuron_sets,
    physiology,
    scaling_factors_relative_to_target,
)
from common.structure import build_student_structure, load_teacher, save_structure

FINAL_STATE = "final_model_state.pt"


# =====================================================================
# Auxiliary losses (archived ff-learnt run, unchanged)
# =====================================================================


class HiddenRateMeanPenalty(nn.Module):
    """MSE between the unobserved population's mean rate and a per-cell-type target.

    Targets come from the observed teacher neurons, which are drawn from the same
    cell-type distribution, so no unobserved activity leaks into training.
    """

    required_inputs: list = ["hidden_spikes"]

    def __init__(self, cell_type_indices, target_rates_hz, dt_ms):
        super().__init__()
        self.register_buffer("cell_type_indices", cell_type_indices)
        self.register_buffer("target_rates_hz", target_rates_hz)
        self.dt_ms = dt_ms

    def forward(self, hidden_spikes):
        batch, n_t, _ = hidden_spikes.shape
        duration_s = n_t * self.dt_ms / 1000.0
        loss = hidden_spikes.new_zeros(())
        for ct in torch.unique(self.cell_type_indices):
            mask = self.cell_type_indices == ct
            n_ct = mask.sum().clamp(min=1).float()
            mean_rate = hidden_spikes[:, :, mask].sum() / (n_ct * duration_s * batch)
            loss = loss + (mean_rate - self.target_rates_hz[int(ct)]).pow(2)
        return loss


class HiddenRateStdPenalty(nn.Module):
    """MSE between the unobserved population's rate spread and a per-cell-type target."""

    required_inputs: list = ["hidden_spikes"]

    def __init__(self, cell_type_indices, target_rate_stds_hz, dt_ms):
        super().__init__()
        self.register_buffer("cell_type_indices", cell_type_indices)
        self.register_buffer("target_rate_stds_hz", target_rate_stds_hz)
        self.dt_ms = dt_ms

    def forward(self, hidden_spikes):
        batch, n_t, _ = hidden_spikes.shape
        duration_s = n_t * self.dt_ms / 1000.0
        per_neuron_rate = hidden_spikes.sum(dim=(0, 1)) / (duration_s * batch)
        loss = hidden_spikes.new_zeros(())
        for ct in torch.unique(self.cell_type_indices):
            mask = self.cell_type_indices == ct
            if mask.sum() < 2:
                continue
            std_rate = per_neuron_rate[mask].std(unbiased=False)
            loss = loss + (std_rate - self.target_rate_stds_hz[int(ct)]).pow(2)
        return loss


class LearntWeightL1Penalty(nn.Module):
    """L1 penalty ``mean(exp(U @ V))`` on one block of learnt weights."""

    required_inputs: list = []

    def __init__(self, U, V):
        super().__init__()
        object.__setattr__(self, "_U", U)
        object.__setattr__(self, "_V", V)

    def forward(self):
        return torch.exp(self._U @ self._V).mean()


def cosine_lambda(lr_init, lr_min, n_epochs):
    """Cosine decay from ``lr_init`` to ``lr_min`` over ``n_epochs`` data epochs.

    The trainer steps the scheduler once per data epoch. (The archived scripts passed
    the number of *chunks* here, so their learning rate barely decayed at all.)
    """
    frac = lr_min / lr_init

    def schedule(epoch):
        return (
            frac
            + (1 - frac) * (1 + math.cos(math.pi * min(epoch, n_epochs) / n_epochs)) / 2
        )

    return schedule


# =====================================================================
# Train
# =====================================================================


def train_student(
    input_dir, output_dir, params_file, wandb_config=None, resume_from=None
):
    """Build the student described by ``parameters.toml`` and train it.

    Writes to ``output_dir``: ``student_structure.npz`` (everything needed to rebuild
    the network), the trainer's checkpoints and ``training_metrics.csv``, and
    ``final_model_state.pt`` (the trained parameters evaluation loads).
    """
    if resume_from is not None:
        raise NotImplementedError("resuming is not supported; rerun from scratch")

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    params = toml.load(params_file)
    simulation = params["simulation"]
    training = params["training"]
    optimiser_cfg = params["optimiser"]
    loss_cfg = params["loss"]

    seed = simulation["seed"]
    chunk_size = simulation["chunk_size"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Student
    # ------------------------------------------------------------------
    teacher = load_teacher(input_dir)
    structure = build_student_structure(
        teacher, params["student"], seed, training["weight_perturbation_variance"]
    )
    save_structure(structure, output_dir)
    sets = neuron_sets(structure)
    print(
        f"\nStudent: {sets['observed'].size} observed, {sets['unobserved'].size} "
        f"unobserved, {sets['unreconstructed'].size} unreconstructed (injected), "
        f"{int((~structure['modelled']).sum()) - sets['unreconstructed'].size} removed; "
        f"recurrent model: {structure['recurrent_model']}"
    )

    dataset = ExactFFDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=chunk_size,
        device=device,
    )
    batch_size, dt, num_chunks = dataset.batch_size, dataset.dt, dataset.num_chunks
    print(f"Teacher data: {num_chunks} chunks x {batch_size} trials, dt {dt} ms")

    model, parameters = build_student(
        structure,
        params,
        batch_size=batch_size,
        dt=dt,
        surrgrad_scale=optimiser_cfg["surrgrad_scale"],
        low_rank=optimiser_cfg.get("low_rank", 1),
    )
    model.to(device)
    n_free = count_free_parameters(parameters)
    print(f"Free parameters: {n_free:,}")

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    losses = {
        "van_rossum": VanRossumLoss(
            tau_rise=loss_cfg["van_rossum_tau_rise"],
            tau_decay=loss_cfg["van_rossum_tau_decay"],
            dt=dt,
            window_size=chunk_size,
            device=device,
        )
    }
    loss_weights = {"van_rossum": loss_cfg["van_rossum"]}

    ct = structure["cell_type_indices"]
    rec_names = physiology(params)["rec_names"]
    rate_penalties = loss_cfg.get("hidden_rate_mean", 0.0) > 0 or (
        loss_cfg.get("hidden_rate_std", 0.0) > 0
    )
    if rate_penalties and sets["unobserved"].size > 0:
        # Which neurons define the target rates. "observed" (default) uses only the
        # recorded cells, so no unobserved activity enters training; with few observed
        # cells and this teacher's heavy-tailed rates a typical sample lands well below
        # the population mean (25 cells: E 1.4 vs 3.9 Hz), which drags the unobserved
        # population towards silence. "population" uses every neuron's rate instead: the
        # same target for every condition, at the cost of assuming the region's mean rate
        # is known.
        target_from = loss_cfg.get("rate_target", "observed")
        if target_from == "population":
            target_ids = np.arange(ct.size)
        elif target_from == "observed":
            target_ids = sets["observed"]
        else:
            raise ValueError('rate_target must be "observed" or "population"')
        teacher_trial = np.array(dataset.target_spike_data[0, :, target_ids])
        rates = teacher_trial.sum(axis=0) / (teacher_trial.shape[0] * dt / 1000.0)
        target_types = ct[target_ids]
        target_mean = np.array(
            [rates[target_types == b].mean() for b in range(len(rec_names))]
        )
        target_std = np.array(
            [rates[target_types == b].std() for b in range(len(rec_names))]
        )
        print(
            f"  Rate-penalty targets from the {target_from} neurons: "
            + ", ".join(
                f"{name} {m:.2f} +- {sd:.2f} Hz"
                for name, m, sd in zip(rec_names, target_mean, target_std)
            )
        )
        hidden_types = torch.from_numpy(ct[sets["unobserved"]]).long()
        if loss_cfg["hidden_rate_mean"] > 0:
            losses["hidden_rate_mean"] = HiddenRateMeanPenalty(
                hidden_types, torch.from_numpy(target_mean).float(), dt
            ).to(device)
            loss_weights["hidden_rate_mean"] = loss_cfg["hidden_rate_mean"]
        if loss_cfg["hidden_rate_std"] > 0:
            losses["hidden_rate_std"] = HiddenRateStdPenalty(
                hidden_types, torch.from_numpy(target_std).float(), dt
            ).to(device)
            loss_weights["hidden_rate_std"] = loss_cfg["hidden_rate_std"]

    if loss_cfg.get("learnt_weight_l1", 0.0) > 0:
        for key in parameters.U:
            # The trainer abbreviates loss names at underscores, so no "__".
            name = f"l1_{key.replace('__', '_to_')}"
            losses[name] = LearntWeightL1Penalty(parameters.U[key], parameters.V[key])
            loss_weights[name] = loss_cfg["learnt_weight_l1"]

    # ------------------------------------------------------------------
    # Optimiser: scaling factors and learnt weights as separate groups.
    # ------------------------------------------------------------------
    total_epochs = training["total_epochs"]
    groups, schedules, clips = [], [], []
    if parameters.scaling_factor_parameters():
        groups.append(
            {
                "params": parameters.scaling_factor_parameters(),
                "lr": optimiser_cfg["lr_scaling"],
            }
        )
        schedules.append(
            cosine_lambda(
                optimiser_cfg["lr_scaling"],
                optimiser_cfg["lr_min_scaling"],
                total_epochs,
            )
        )
        clips.append(optimiser_cfg["grad_clip_scaling"])
    if parameters.weight_parameters():
        groups.append(
            {
                "params": parameters.weight_parameters(),
                "lr": optimiser_cfg["lr_weights"],
            }
        )
        schedules.append(
            cosine_lambda(
                optimiser_cfg["lr_weights"],
                optimiser_cfg["lr_min_weights"],
                total_epochs,
            )
        )
        clips.append(optimiser_cfg["grad_clip_weights"])

    optimiser = torch.optim.Adam(
        groups,
        betas=(optimiser_cfg["beta1"], optimiser_cfg["beta2"]),
        eps=optimiser_cfg["eps"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedules)

    def grad_clip_fn(_model):
        for group, clip in zip(optimiser.param_groups, clips):
            with_grad = [p for p in group["params"] if p.grad is not None]
            if with_grad:
                torch.nn.utils.clip_grad_norm_(with_grad, max_norm=clip)

    scaler = GradScaler(
        "cuda", enabled=training["mixed_precision"] and device == "cuda"
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    observed_types = ct[sets["observed"]]
    unobserved_types = ct[sets["unobserved"]]

    def stats_computer(snapshot):
        student = snapshot["spikes"][0]
        teacher_obs = snapshot["target_spikes"][0, : student.shape[0]]
        duration_s = student.shape[0] * dt / 1000.0
        stats = {}
        for b, name in enumerate(rec_names):
            members = observed_types == b
            if members.any():
                stats[f"firing_rate/student_observed_{name}"] = float(
                    student[:, members].sum() / duration_s / members.sum()
                )
                stats[f"firing_rate/teacher_observed_{name}"] = float(
                    teacher_obs[:, members].sum() / duration_s / members.sum()
                )
            hidden = snapshot.get("hidden_spikes")
            members = unobserved_types == b
            if hidden is not None and members.any():
                stats[f"firing_rate/student_unobserved_{name}"] = float(
                    hidden[0, : student.shape[0]][:, members].sum()
                    / duration_s
                    / members.sum()
                )
        for pathway, value in scaling_factors_relative_to_target(
            parameters, structure, params
        ).items():
            stats[f"scaling_factors/{pathway}_value"] = value
            stats[f"scaling_factors/{pathway}_target"] = 1.0
        return stats

    wandb_logger = None
    if wandb_config:
        # Grid runs are named <figure>/<run> so they are unambiguous in one project.
        run_name = (
            wandb_config.pop("name", None)
            or f"{output_dir.parent.name}/{output_dir.name}"
        )
        wandb_logger = wandb.init(
            name=run_name,
            dir=str(output_dir),
            config={
                "student": params["student"],
                "seed": seed,
                "n_observed": int(sets["observed"].size),
                "n_unobserved": int(sets["unobserved"].size),
                "n_unreconstructed": int(sets["unreconstructed"].size),
                "n_free_parameters": n_free,
            },
            **wandb_config,
        )
        wandb.define_metric("epoch")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        sampler=CyclicSampler(dataset),
        **DATALOADER_KWARGS,
        collate_fn=StudentCollate(sets["observed"], sets["unreconstructed"]),
    )
    total_chunks = total_epochs * num_chunks
    print(
        f"\nTraining {total_epochs} epochs x {num_chunks} chunks = {total_chunks} chunks"
    )

    trainer = SNNTrainer(
        model=model,
        optimizer=optimiser,
        scaler=scaler,
        dataloader=dataloader,
        loss_functions=losses,
        loss_weights=loss_weights,
        device=device,
        num_epochs=total_chunks,
        chunks_per_update=training["chunks_per_update"],
        log_interval=training["log_interval"],
        checkpoint_interval=training["checkpoint_interval"],
        plot_size=training["plot_size"],
        mixed_precision=training["mixed_precision"],
        grad_clip_fn=grad_clip_fn,
        progress_bar=tqdm(range(total_chunks), desc="Training", unit="chunk"),
        stats_computer=stats_computer,
        chunks_per_data_epoch=num_chunks,
        burn_in_chunks=training["burn_in_chunks"],
        scheduler=scheduler,
    )
    trainer.metrics_logger = AsyncLogger(log_dir=output_dir, max_queue_size=10)
    if wandb_logger:
        trainer.wandb_logger = wandb_logger

    best_loss = trainer.train(output_dir=output_dir)

    # Only the trainable parameters: evaluation rebuilds the network itself from
    # student_structure.npz, so the connectome buffers need not be stored again.
    torch.save(parameters.state_dict(), output_dir / FINAL_STATE)
    relative = scaling_factors_relative_to_target(parameters, structure, params)
    np.savez(output_dir / "final_scaling_factors.npz", **relative)
    print("\nScaling factors relative to their correct values (1.0 = recovered):")
    for pathway, value in sorted(relative.items()):
        print(f"  {pathway:32s} {value:.4f}")
    print(f"Best training loss: {best_loss:.6f}")

    trainer.metrics_logger.close()
    if wandb_logger:
        wandb.finish()
    return best_loss
