"""
Poisson NMF of feedforward spike trains (torchnmf, GPU).

Uses beta=1 (KL divergence), the principled noise model for spike count data.
No Gaussian pre-smoothing — raw binary spike trains are fed directly.

All batches are concatenated along the time axis, factorised jointly on GPU,
then split back into (n_batches, n_timesteps, rank) for saving.

Reconstruction: activations @ components  (same convention as sklearn NMF)
  where activations = H  (n_batches, n_timesteps, rank)
        components  = W.T (rank, n_feedforward)

Outputs
-------
output_dir/results/nmf_inputs.zarr
    activations   (n_batches, n_timesteps, rank)  float32
    components    (rank, n_feedforward)            float32
    Attributes: rank, beta, dt, reconstruction_r2, n_iter
"""

from pathlib import Path

import numpy as np
import torch
import zarr
import tomllib
from torchnmf.nmf import NMF


def main(input_dir, output_dir, params_file, wandb_config=None, resume_from=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    params_file = Path(params_file)

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load parameters
    # ------------------------------------------------------------------
    print("=" * 60)
    print("COMPUTE NMFs — Poisson NMF on raw spike trains (torchnmf)")
    print("=" * 60)

    with open(params_file, "rb") as f:
        params = tomllib.load(f)

    rank = int(params["nmf"]["rank"])
    beta = float(params["nmf"].get("beta", 1))
    max_iter = int(params["nmf"].get("max_iter", 500))
    tol = float(params["nmf"].get("tol", 1e-4))
    random_state = int(params["nmf"].get("random_state", 0))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice               : {device}")
    print("\nParameters")
    print(f"  Rank               : {rank}")
    print(f"  Beta               : {beta}  (1 = KL divergence / Poisson noise model)")
    print(f"  Max iterations     : {max_iter}")
    print(f"  Tolerance          : {tol}")
    print(f"  Random seed        : {random_state}")

    # ------------------------------------------------------------------
    # Load spike data
    # ------------------------------------------------------------------
    print(f"\nLoading spike data from {input_dir / 'spike_data.zarr'} ...")
    spike_data = zarr.open_group(input_dir / "spike_data.zarr", mode="r")
    dt = float(spike_data.attrs["dt"])

    input_spikes = spike_data["input_spikes"][:]  # (n_batches, n_timesteps, n_ff)

    n_batches, n_timesteps, n_ff = input_spikes.shape
    duration_s = n_timesteps * dt / 1000.0

    print(
        f"  Input spikes shape : {input_spikes.shape}  (batches × timesteps × neurons)"
    )
    print(f"  dt                 : {dt} ms")
    print(f"  Trial duration     : {duration_s:.1f} s  ({n_timesteps} timesteps)")
    print(f"  Feedforward neurons: {n_ff}")
    print(f"  Batches            : {n_batches}")

    mean_ff_rate = input_spikes.mean() / (dt / 1000.0)
    print(f"  Mean FF firing rate: {mean_ff_rate:.2f} Hz")

    # ------------------------------------------------------------------
    # Concatenate batches → (N, n_ff) and move to GPU
    # ------------------------------------------------------------------
    N = n_batches * n_timesteps
    flat = input_spikes.reshape(N, n_ff).astype(np.float32)

    print(f"\nMoving data to {device} ...")
    print(f"  Data matrix shape  : {flat.shape}  ({N} timesteps × {n_ff} neurons)")
    print(f"  Memory             : {flat.nbytes / 1e9:.2f} GB")

    V = torch.from_numpy(flat).to(device)
    del flat

    # ------------------------------------------------------------------
    # Fit Poisson NMF
    # ------------------------------------------------------------------
    print(f"\nFitting Poisson NMF  (rank={rank}, beta={beta}, max_iter={max_iter}) ...")
    torch.manual_seed(random_state)

    model = NMF(V.shape, rank=rank).to(device)
    model.fit(V, beta=beta, max_iter=max_iter, tol=tol)

    H = model.H.detach().cpu().numpy()  # (N, rank)       — activations
    W = model.W.detach().cpu().numpy()  # (n_ff, rank)    — basis vectors

    del V
    if device == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Reconstruction quality (Frobenius R² on raw spikes)
    # ------------------------------------------------------------------
    flat_np = input_spikes.reshape(N, n_ff).astype(np.float64)
    recon = H.astype(np.float64) @ W.T.astype(np.float64)  # (N, n_ff)
    ss_res = np.sum((flat_np - recon) ** 2)
    ss_tot = np.sum((flat_np - flat_np.mean(axis=0)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)
    del flat_np, recon

    print(f"\n  Reconstruction R²  : {r2:.4f}  (Frobenius, on raw spikes)")

    # ------------------------------------------------------------------
    # Save results  (split H back into batches)
    # ------------------------------------------------------------------
    activations = H.reshape(n_batches, n_timesteps, rank).astype(np.float32)
    components = W.T.astype(np.float32)  # (rank, n_ff) — matches sklearn convention

    out_path = results_dir / "nmf_inputs.zarr"
    print(f"\nSaving results to {out_path} ...")

    nmf_out = zarr.open_group(out_path, mode="w")

    nmf_out.create_dataset(
        "activations",
        shape=activations.shape,
        dtype=np.float32,
        data=activations,
        chunks=(1, n_timesteps, rank),
    )
    print(f"  activations        : {activations.shape}")

    nmf_out.create_dataset(
        "components",
        shape=components.shape,
        dtype=np.float32,
        data=components,
    )
    print(
        f"  components         : {components.shape}  "
        f"(reconstruction: activations @ components)"
    )

    nmf_out.attrs["rank"] = rank
    nmf_out.attrs["beta"] = beta
    nmf_out.attrs["dt"] = dt
    nmf_out.attrs["reconstruction_r2"] = r2

    print(f"\n✓ Done. Results written to {results_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--params_file", type=Path, required=True)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.params_file)
