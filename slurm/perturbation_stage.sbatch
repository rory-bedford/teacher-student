#!/bin/bash
# One stage of fig01-full-reconstruction/perturbation_smoketest.py. No GPU by default:
# GPU stages get --gres=gpu:<type>:1 from slurm/submit_perturbation.sh.
#   sbatch [--gres=gpu:v100:1] slurm/perturbation_stage.sbatch <stage> <run dir> <out dir> [extra args]
#SBATCH --account=connectome
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
REPO="/tachyon/groups/scratch/gzenke/bedfrory/teacher-student"
cd "$REPO"
stage="$1"; run="$2"; out="$3"; shift 3
# Never `uv run` on a node: it rebuilds the shared .venv when it cannot see the interpreter.
echo "Stage $stage $* on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}, commit $(git rev-parse --short HEAD)"
"$REPO/.venv/bin/python" -u fig01-full-reconstruction/perturbation_smoketest.py "$stage" --run "$run" --out "$out" "$@"
