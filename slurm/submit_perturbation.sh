#!/bin/bash
# Submit the Figure 1 perturbation smoke test for one trained run (on the login node):
#   slurm/submit_perturbation.sh <run dir> <out dir> [GPU type, default v100]
#
# If the teacher's current is not calibrated yet (no current.toml), first submits one GPU
# job per calibration current in parallel and a CPU job that picks the current. Then:
# teacher-on (GPU, after calibration), student-off and perfect-off (GPU, immediately),
# student-on and perfect-on (GPU, after teacher-on), score (CPU, after all).
# Jobs waiting on dependencies hold no GPU. Logs: <out dir>/logs.
set -euo pipefail
RUN="$(realpath "$1")"; OUT="$(realpath -m "$2")"; GPU="${3:-v100}"
REPO="/tachyon/groups/scratch/gzenke/bedfrory/teacher-student"
PY="$REPO/.venv/bin/python"
mkdir -p "$OUT/logs"
cd "$REPO"

submit() {  # submit <gpu|cpu> <stage> <dependency ids or ""> [extra args]
    local kind="$1" stage="$2" after="$3"; shift 3
    local name="pert-$stage${1:+-${2:-}}"
    sbatch --parsable --job-name="$name" --output="$OUT/logs/$stage${2:+_$2}.log" \
        ${after:+--dependency=afterok:$after} \
        $([[ "$kind" == gpu ]] && echo "--gres=gpu:$GPU:1") \
        "$REPO/slurm/perturbation_stage.sbatch" "$stage" "$RUN" "$OUT" "$@"
}

CAL_DIR=$("$PY" -c "import sys; sys.path.insert(0, '$REPO'); from common.perturbation import calibration_dir; print(calibration_dir('$RUN'))")
calibrated=""
if [[ ! -f "$CAL_DIR/current.toml" ]]; then
    shifts=$("$PY" -c "import sys; sys.path.insert(0, '$REPO'); from common.perturbation import CALIBRATION_SHIFTS_MV as s; print(' '.join(map(str, s)))")
    points=()
    for s in $shifts; do
        if [[ -f "$CAL_DIR/$(printf 'shift%+.1fmV.toml' "$s")" ]]; then
            echo "calibration point $s mV already done"
        else
            points+=("$(submit gpu cal-point "" --shift "$s")")
        fi
    done
    calibrated=$(submit cpu cal-pick "$(IFS=:; echo "${points[*]:-}")")
    echo "calibration: points ${points[*]}, pick $calibrated -> $CAL_DIR"
else
    echo "using calibrated current in $CAL_DIR/current.toml"
fi
teacher_on=$(submit gpu teacher-on "$calibrated")
student_off=$(submit gpu student-off "")
perfect_off=$(submit gpu perfect-off "")
student_on=$(submit gpu student-on "$teacher_on")
perfect_on=$(submit gpu perfect-on "$teacher_on")
score=$(submit cpu score "$teacher_on:$student_off:$perfect_off:$student_on:$perfect_on")
echo "teacher-on $teacher_on, student-off $student_off, perfect-off $perfect_off,"
echo "student-on $student_on, perfect-on $perfect_on, score $score"
echo "logs: $OUT/logs"
