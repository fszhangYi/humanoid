#!/usr/bin/env bash
# Start mjviser on port 6008. ROBOT=g1|h1|humanoid (default g1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/root/autodl-tmp/conda-envs/mjviser}"
PORT="${PORT:-6008}"
ROBOT="${ROBOT:-g1}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"

case "$ROBOT" in
  g1) MODEL="${MODEL:-$ROOT/models/unitree_g1/scene.xml}" ;;
  h1) MODEL="${MODEL:-$ROOT/models/unitree_h1/scene.xml}" ;;
  humanoid) MODEL="${MODEL:-$ROOT/models/humanoid.xml}" ;;
  *)
    echo "unknown ROBOT=$ROBOT (use g1|h1|humanoid)"
    exit 1
    ;;
esac

# shellcheck source=/dev/null
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
  echo "port ${PORT} already in use; refuse to start"
  ss -tlnp | grep ":${PORT} " || true
  exit 1
fi

echo "robot: $ROBOT"
echo "model: $MODEL"
echo "url:   http://localhost:${PORT}"
nohup mjviser "$MODEL" --port "$PORT" \
  >"$LOG_DIR/${ROBOT}_${PORT}.log" 2>&1 &
echo "pid=$!  log=$LOG_DIR/${ROBOT}_${PORT}.log"
sleep 2
tail -n 30 "$LOG_DIR/${ROBOT}_${PORT}.log" || true
ss -tlnp | grep ":${PORT} " || echo "(waiting for bind; check log)"
