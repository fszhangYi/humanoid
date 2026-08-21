#!/usr/bin/env bash
# 启动自有 G1 强化学习训练；支持 RESUME / EARLY_STOP
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_PREFIX="${G1_TRAIN_ENV:-/root/autodl-tmp/conda-envs/g1_train}"
CFG="${CFG:-$ROOT/my_rl_train/configs/train_g1.yaml}"
export CFG

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"
# shellcheck source=/dev/null
source "$ROOT/my_scripts/env.sh"

yaml_get() {
  local key="$1"
  python - <<PY
import yaml
c = yaml.safe_load(open("$CFG")) or {}
v = c.get("$key")
print("" if v is None else v)
PY
}

ENV_NAME="${ENV_NAME:-$(yaml_get env_name)}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-$(yaml_get num_timesteps)}"
NUM_VIDEOS="${NUM_VIDEOS:-$(yaml_get num_videos)}"
NUM_VIDEOS="${NUM_VIDEOS:-0}"
NUM_ENVS="${NUM_ENVS:-$(yaml_get num_envs)}"
BATCH_SIZE="${BATCH_SIZE:-$(yaml_get batch_size)}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-$(yaml_get num_minibatches)}"
NUM_EVALS="${NUM_EVALS:-$(yaml_get num_evals)}"
RESUME="${RESUME:-$(yaml_get resume)}"
RESUME="${RESUME:-off}"
RESUME_MODE="${RESUME_MODE:-$(yaml_get resume_mode)}"
RESUME_MODE="${RESUME_MODE:-additional}"
TARGET_TIMESTEPS="${TARGET_TIMESTEPS:-$(yaml_get target_timesteps)}"

# early_stop.enabled from yaml unless EARLY_STOP already set
if [[ -z "${EARLY_STOP:-}" ]]; then
  EARLY_STOP="$(python - <<PY
import yaml
c = yaml.safe_load(open("$CFG")) or {}
es = c.get("early_stop") or {}
print("1" if es.get("enabled") else "0")
PY
)"
fi
export RESUME RESUME_MODE TARGET_TIMESTEPS EARLY_STOP ENV_NAME

USE_TB="${USE_TB:-$(python - <<PY
import yaml
c = yaml.safe_load(open("$CFG")) or {}
print("1" if c.get("use_tb") else "0")
PY
)}"

EXTRA_FLAGS=()
[[ -n "${NUM_ENVS}" ]] && EXTRA_FLAGS+=(--num_envs "$NUM_ENVS")
[[ -n "${BATCH_SIZE}" ]] && EXTRA_FLAGS+=(--batch_size "$BATCH_SIZE")
[[ -n "${NUM_MINIBATCHES}" ]] && EXTRA_FLAGS+=(--num_minibatches "$NUM_MINIBATCHES")
[[ -n "${NUM_EVALS}" ]] && EXTRA_FLAGS+=(--num_evals "$NUM_EVALS")
[[ "${USE_TB}" == "1" || "${USE_TB}" == "true" ]] && EXTRA_FLAGS+=(--use_tb)
echo "use_tb=$USE_TB"

mkdir -p "$MY_PARTY_ROOT/logs" "$MY_PARTY_ROOT/checkpoints"
LOG="$MY_PARTY_ROOT/logs/train_${ENV_NAME}_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="$MY_PARTY_ROOT/logs/train.pid"

echo "env=$ENV_NAME  timesteps=$NUM_TIMESTEPS  videos=$NUM_VIDEOS"
echo "num_envs=${NUM_ENVS:-default}  batch=${BATCH_SIZE:-default}  minibatches=${NUM_MINIBATCHES:-default}  evals=${NUM_EVALS:-default}"
echo "resume=$RESUME  resume_mode=$RESUME_MODE  target_timesteps=${TARGET_TIMESTEPS:-}  early_stop=$EARLY_STOP"
echo "log=$LOG"

cd "$MY_PARTY_ROOT"
export PYTHONUNBUFFERED=1
nohup python -u "$ROOT/my_scripts/train_my_g1.py" \
  --env_name "$ENV_NAME" \
  --num_timesteps "$NUM_TIMESTEPS" \
  --num_videos "$NUM_VIDEOS" \
  "${EXTRA_FLAGS[@]}" \
  >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "pid=$(cat "$PIDFILE")"
sleep 5
tail -n 50 "$LOG" || true
