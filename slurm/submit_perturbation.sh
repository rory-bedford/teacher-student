#!/bin/bash
# Submit the Figure 1 perturbation smoke test as chained SLURM jobs (run on the login node):
#   slurm/submit_perturbation.sh <run dir> <out dir> [gres type, default v100]
# Stage 1 (3 GPUs): calibrate, student-off, perfect-off. Stage 2 (2 GPUs, after
# calibrate): student-on, perfect-on. Stage 3 (after all): score. Logs go to <out dir>/logs.
set -euo pipefail
RUN="$(realpath "$1")"; OUT="$(realpath -m "$2")"; GPU="${3:-v100}"
REPO="/tachyon/groups/scratch/gzenke/bedfrory/teacher-student"
mkdir -p "$OUT/logs"
submit() {  # submit <stage> [dependency]
    sbatch --parsable --gres=gpu:"$GPU":1 --job-name="pert-$1" \
        --output="$OUT/logs/$1.log" ${2:+--dependency=afterok:$2} \
        "$REPO/slurm/perturbation_stage.sbatch" "$1" "$RUN" "$OUT"
}
calibrate=$(submit calibrate)
student_off=$(submit student-off)
perfect_off=$(submit perfect-off)
student_on=$(submit student-on "$calibrate")
perfect_on=$(submit perfect-on "$calibrate")
score=$(submit score "$calibrate:$student_off:$perfect_off:$student_on:$perfect_on")
echo "submitted: calibrate $calibrate, student-off $student_off, perfect-off $perfect_off,"
echo "           student-on $student_on, perfect-on $perfect_on, score $score"
echo "logs: $OUT/logs"
