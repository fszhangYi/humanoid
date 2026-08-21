#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/root/autodl-tmp/conda-envs/g1_train}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR" "$ROOT/checkpoints"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"
# shellcheck source=/dev/null
source "$ROOT/scripts/env.sh"
cd "$ROOT"

NUM_TIMESTEPS="${NUM_TIMESTEPS:-8000}"
ENV_NAME="${ENV_NAME:-CartpoleBalance}"

echo "smoke train: env=$ENV_NAME timesteps=$NUM_TIMESTEPS"
echo "log: $LOG_DIR/smoke_train.log"

python "$ROOT/scripts/train_g1.py" \
  --env_name "$ENV_NAME" \
  --num_timesteps "$NUM_TIMESTEPS" \
  --num_videos 0 \
  >"$LOG_DIR/smoke_train.log" 2>&1
rc=$?
tail -n 40 "$LOG_DIR/smoke_train.log" || true
exit $rc
