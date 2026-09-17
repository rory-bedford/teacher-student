#!/bin/bash
# Two-minute check that the uv environment runs on a cluster GPU node: driver, torch
# build, and a torch.compile'd CUDA kernel (Triton), which is what training relies on.
#
#   sbatch --output=check_env_%j.log slurm/check_env.sbatch
#   (add --gres=gpu:a40:1 or --gres=gpu:v100:1 to check the other GPU types)
#SBATCH --account=iris00000
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00

set -euo pipefail
cd /tachyon/groups/scratch/gzenke/bedfrory/teacher-student
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run python - <<'PY'
import torch

print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))


@torch.compile
def step(v, w, s):
    return 0.9 * v + s @ w


v = torch.zeros(50, 1000, device="cuda")
w = torch.randn(1000, 1000, device="cuda", requires_grad=True)
s = (torch.rand(50, 1000, device="cuda") < 0.01).float()
step(v, w, s).sum().backward()
print("torch.compile forward/backward OK; grad norm", w.grad.norm().item())
PY
