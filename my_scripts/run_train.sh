#!/usr/bin/env bash
# 启动自有 G1 强化学习训练；日志与 checkpoint 落在 my_party/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_PREFIX="${G1_TRAIN_ENV:-/root/autodl-tmp/conda-envs/g1_train}"
CFG="${CFG:-$ROOT/my_rl_train/configs/train_g1.yaml}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"
# shellcheck source=/dev/null
source "$ROOT/my_scripts/env.sh"

yaml_get() {
  local key="$1"
  python - <<PY
import yaml
c = yaml.safe_load(open("$CFG"))
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

EXTRA_FLAGS=()
[[ -n "${NUM_ENVS}" ]] && EXTRA_FLAGS+=(--num_envs "$NUM_ENVS")
[[ -n "${BATCH_SIZE}" ]] && EXTRA_FLAGS+=(--batch_size "$BATCH_SIZE")
[[ -n "${NUM_MINIBATCHES}" ]] && EXTRA_FLAGS+=(--num_minibatches "$NUM_MINIBATCHES")
[[ -n "${NUM_EVALS}" ]] && EXTRA_FLAGS+=(--num_evals "$NUM_EVALS")

mkdir -p "$MY_PARTY_ROOT/logs" "$MY_PARTY_ROOT/checkpoints"
LOG="$MY_PARTY_ROOT/logs/train_${ENV_NAME}_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="$MY_PARTY_ROOT/logs/train.pid"

echo "env=$ENV_NAME  timesteps=$NUM_TIMESTEPS  videos=$NUM_VIDEOS"
echo "num_envs=${NUM_ENVS:-default}  batch=${BATCH_SIZE:-default}  minibatches=${NUM_MINIBATCHES:-default}  evals=${NUM_EVALS:-default}"
echo "log=$LOG"
echo "cwd logs -> $MY_PARTY_ROOT/logs"

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
tail -n 40 "$LOG" || true
