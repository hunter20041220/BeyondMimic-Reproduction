#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 MOTION_NAME [GPU_INDEX]" >&2
  exit 2
fi

MOTION="$1"
GPU="${2:-7}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
RUNTIME="${RUNTIME:-/dev/shm/BeyondMimic_Official_Stage1_runtime/whole_body_tracking}"
PYTHON_SH="${PYTHON_SH:-/dev/shm/BeyondMimic_Official_Stage1_runtime/envs/isaacsim-4.5.0/python.sh}"
EXPERIENCE="${EXPERIENCE:-/dev/shm/BeyondMimic_Official_Stage1_runtime/IsaacLab/apps/isaaclab.python.headless.loop_isaac.single_gpu.kit}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_flat_branchC_25hz}"
SHARED_ROOT="${SHARED_ROOT:-/shared_disk/zzy/BeyondMimic/stage1_branchC_25hz}"
RUNTIME_EXP_ROOT="${RUNTIME}/logs/rsl_rl/${EXPERIMENT_NAME}"
SHARED_EXP_ROOT="${SHARED_ROOT}/rsl_rl/${EXPERIMENT_NAME}"
LAUNCH_LOG_ROOT="${SHARED_ROOT}/launch_logs/$(date +%Y-%m-%d_%H-%M-%S)"

case "${MOTION}" in
  walk1_subject1|walk2_subject3|walk2_subject4|run2_subject1|sprint1_subject2) ;;
  *)
    echo "unsupported Branch-C locomotion motion: ${MOTION}" >&2
    exit 3
    ;;
esac

MOTION_FILE="${RUNTIME}/artifacts/${MOTION}:v0/motion.npz"
RUN_NAME="${MOTION}_branchC_25hz_stage1"
TRAIN_LOG="${LAUNCH_LOG_ROOT}/${MOTION}.log"

if [[ ! -f "${MOTION_FILE}" ]]; then
  echo "missing motion file: ${MOTION_FILE}" >&2
  exit 4
fi

mkdir -p "${SHARED_EXP_ROOT}" "${LAUNCH_LOG_ROOT}" "$(dirname "${RUNTIME_EXP_ROOT}")"
if [[ -e "${RUNTIME_EXP_ROOT}" && ! -L "${RUNTIME_EXP_ROOT}" ]]; then
  echo "runtime experiment path exists and is not a symlink: ${RUNTIME_EXP_ROOT}" >&2
  exit 5
fi
if [[ ! -e "${RUNTIME_EXP_ROOT}" ]]; then
  ln -s "${SHARED_EXP_ROOT}" "${RUNTIME_EXP_ROOT}"
fi

setsid -f bash -lc "
  cd '${RUNTIME}' &&
  export WANDB_MODE=disabled &&
  export CUDA_VISIBLE_DEVICES='${GPU}' &&
  exec '${PYTHON_SH}' -u scripts/rsl_rl/train_motion_file.py \
    --headless \
    --device cuda:0 \
    --experience '${EXPERIENCE}' \
    --task Tracking-Flat-G1-Low-Freq-v0 \
    --motion_file '${MOTION_FILE}' \
    --max_iterations '${MAX_ITERATIONS}' \
    --logger tensorboard \
    --experiment_name '${EXPERIMENT_NAME}' \
    --run_name '${RUN_NAME}' \
    > '${TRAIN_LOG}' 2>&1 < /dev/null
"

{
  printf "motion\tgpu\trun_name\tmotion_file\tshared_experiment_root\ttrain_log\n"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${MOTION}" "${GPU}" "${RUN_NAME}" "${MOTION_FILE}" "${SHARED_EXP_ROOT}" "${TRAIN_LOG}"
} > "${LAUNCH_LOG_ROOT}/launch_manifest.tsv"

echo "launched ${MOTION} 25Hz teacher on GPU ${GPU}"
echo "shared_experiment_root=${SHARED_EXP_ROOT}"
echo "train_log=${TRAIN_LOG}"
