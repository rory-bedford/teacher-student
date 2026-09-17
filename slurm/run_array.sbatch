#!/bin/bash
# One grid run per array task. Generate the array with slurm/prepare_array.py, which
# prints the matching sbatch command (array range, time limit, log path).
#
#   sbatch --array=0-N --time=HH:MM:SS --output=<array dir>/logs/%a.log \
#          slurm/run_array.sbatch <array dir>
#
# Each run needs one GPU with >= 12 GB (measured peak 9.3 GB), ~32 GB RAM, 4 CPUs.
# GPU type: a100 by default; override on the command line with --gres=gpu:a40:1 or
# --gres=gpu:v100:1. Keep the array throttle (%8) in the sbatch line.
#SBATCH --account=iris00000
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

ARRAY_DIR="$1"
REPO="/tachyon/groups/scratch/gzenke/bedfrory/teacher-student"

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ARRAY_DIR/tasks.txt")
config=$(cut -f1 <<<"$line")
description=$(cut -f2 <<<"$line")

cd "$REPO"

# The configs were generated at a specific commit and the run scripts are loaded from
# this checkout, so refuse to run if the code has moved on.
expected=$(cat "$ARRAY_DIR/commit.txt")
actual=$(git rev-parse HEAD)
if [[ "$actual" != "$expected" ]]; then
    echo "Checkout is at $actual but the array was prepared at $expected; aborting." >&2
    exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Repository has uncommitted changes; aborting." >&2
    exit 1
fi

output_dir=$(uv run python -c "import toml,sys; print(toml.load(sys.argv[1])['output_dir'])" "$config")
if [[ -f "$output_dir/final_model_state.pt" ]]; then
    echo "$description already finished; nothing to do."
    exit 0
fi
if [[ -e "$output_dir" ]]; then
    echo "$output_dir exists but never finished; move or delete it and resubmit this task." >&2
    exit 1
fi

echo "Task $SLURM_ARRAY_TASK_ID: $description on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}"
./run --no-commit "$config" < /dev/null
