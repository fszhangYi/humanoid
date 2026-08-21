#!/usr/bin/env bash
# Run Unitree G1 walking (RL policy) on port 6008.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/root/autodl-tmp/conda-envs/mjviser}"
PORT="${PORT:-6008}"
VX="${VX:-0.5}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"

# shellcheck source=/dev/null
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
  echo "port ${PORT} in use — stop it first, e.g.:"
  ss -tlnp | grep ":${PORT} " || true
  exit 1
fi

echo "G1 walk  vx=${VX}  http://localhost:${PORT}"
nohup python "$ROOT/scripts/walk_g1.py" --port "$PORT" --vx "$VX" \
  >"$LOG_DIR/walk_g1_${PORT}.log" 2>&1 &
echo "pid=$!  log=$LOG_DIR/walk_g1_${PORT}.log"
sleep 3
tail -n 20 "$LOG_DIR/walk_g1_${PORT}.log" || true
ss -tlnp | grep ":${PORT} " || echo "(waiting for bind)"
