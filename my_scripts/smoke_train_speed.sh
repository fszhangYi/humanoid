#!/usr/bin/env bash
# 冒烟测速：加大 num_envs 后短训，打印 SPS / 峰值显存
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_PREFIX="${G1_TRAIN_ENV:-/root/autodl-tmp/conda-envs/g1_train}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"
# shellcheck source=/dev/null
source "$ROOT/my_scripts/env.sh"

NUM_ENVS="${NUM_ENVS:-16384}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-64}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-1000000}"
NUM_EVALS="${NUM_EVALS:-2}"
ENV_NAME="${ENV_NAME:-G1JoystickFlatTerrain}"

mkdir -p "$MY_PARTY_ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$MY_PARTY_ROOT/logs/smoke_speed_${NUM_ENVS}_${STAMP}.log"
VRAM_LOG="$MY_PARTY_ROOT/logs/smoke_vram_${NUM_ENVS}_${STAMP}.csv"

echo "smoke: envs=$NUM_ENVS batch=$BATCH_SIZE mb=$NUM_MINIBATCHES steps=$NUM_TIMESTEPS evals=$NUM_EVALS"
echo "log=$LOG"

# 采样显存
(
  echo "ts_iso,mem_used_mib"
  while true; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "$(date -Iseconds),$u"
    sleep 2
  done
) >"$VRAM_LOG" &
VRAM_PID=$!
trap 'kill $VRAM_PID 2>/dev/null || true' EXIT

cd "$MY_PARTY_ROOT"
export PYTHONUNBUFFERED=1
python -u "$ROOT/my_scripts/train_my_g1.py" \
  --env_name "$ENV_NAME" \
  --num_timesteps "$NUM_TIMESTEPS" \
  --num_videos 0 \
  --num_envs "$NUM_ENVS" \
  --batch_size "$BATCH_SIZE" \
  --num_minibatches "$NUM_MINIBATCHES" \
  --num_evals "$NUM_EVALS" \
  2>&1 | tee "$LOG"

kill "$VRAM_PID" 2>/dev/null || true
trap - EXIT

python - <<PY
from pathlib import Path
import re
log = Path("$LOG").read_text(errors="ignore")
vram = Path("$VRAM_LOG").read_text().strip().splitlines()[1:]
peak = max(int(l.split(",")[1]) for l in vram if l.strip()) if vram else -1
m_jit = re.search(r"Time to JIT compile:\s*([0-9.]+)", log)
m_tr = re.search(r"Time to train:\s*([0-9.]+)", log)
m_ne = re.search(r"num_envs:\s*([0-9]+)", log)
steps = int("$NUM_TIMESTEPS")
jit = float(m_jit.group(1)) if m_jit else float("nan")
tr = float(m_tr.group(1)) if m_tr else float("nan")
ne = m_ne.group(1) if m_ne else "?"
sps = steps / tr if tr == tr and tr > 0 else float("nan")
print("==== SMOKE SPEED SUMMARY ====")
print(f"num_envs={ne}")
print(f"num_timesteps={steps}")
print(f"JIT_s={jit:.1f}")
print(f"train_s={tr:.1f}")
print(f"sps={sps:.0f}")
print(f"peak_vram_MiB={peak}")
print(f"hours_per_100M={100e6/steps*tr/3600 if tr==tr else float('nan'):.2f}")
PY
