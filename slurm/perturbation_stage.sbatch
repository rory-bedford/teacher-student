#!/bin/bash
# One stage of fig01-full-reconstruction/perturbation_smoketest.py on one GPU.
#   sbatch --gres=gpu:v100:1 slurm/perturbation_stage.sbatch <stage> <run dir> <out dir>
# Use slurm/submit_perturbation.sh to chain all stages.
#SBATCH --account=connectome
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=02:00:00

set -euo pipefail
REPO="/tachyon/groups/scratch/gzenke/bedfrory/teacher-student"
cd "$REPO"
# Never `uv run` on a node: it rebuilds the shared .venv when it cannot see the interpreter.
echo "Stage $1 on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}, commit $(git rev-parse --short HEAD)"
"$REPO/.venv/bin/python" -u fig01-full-reconstruction/perturbation_smoketest.py "$1" --run "$2" --out "$3"
