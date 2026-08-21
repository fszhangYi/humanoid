#!/usr/bin/env bash
# 启动 / 复用 TensorBoard：优先写到 AutoDL 默认 /root/tf-logs（端口 6007 常已有服务）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_PREFIX="${G1_TRAIN_ENV:-/root/autodl-tmp/conda-envs/g1_train}"
PORT="${TB_PORT:-6006}"
TF_LOGS="${TF_LOGS:-/root/tf-logs}"
EXP_LOGDIR="${1:-}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

mkdir -p "$TF_LOGS" "$ROOT/my_party/logs"

# 若传入实验目录，软链到 tf-logs 便于 6007 默认面板看到
if [[ -n "$EXP_LOGDIR" && -d "$EXP_LOGDIR" ]]; then
  name="$(basename "$EXP_LOGDIR")"
  ln -sfn "$EXP_LOGDIR" "$TF_LOGS/$name"
  echo "linked $EXP_LOGDIR -> $TF_LOGS/$name"
fi

# 6007 已有 AutoDL 默认 TB 时，只保证数据进 /root/tf-logs 即可
if ss -tlnp 2>/dev/null | grep -q ":6007"; then
  echo "TensorBoard already on :6007 (logdir usually /root/tf-logs)"
  echo "Open AutoDL 自定义服务 6007 查看。"
fi

# 额外在 PORT 起一个指向 my_party/logs + tf-logs 的 TB（可选）
if ss -tlnp 2>/dev/null | grep -q ":${PORT}"; then
  echo "port $PORT already in use; skip starting another tensorboard"
  exit 0
fi

LOG="$ROOT/my_party/logs/tensorboard_${PORT}.log"
PIDFILE="$ROOT/my_party/logs/tensorboard_${PORT}.pid"
nohup tensorboard --host 0.0.0.0 --port "$PORT" \
  --logdir_spec "g1_party:${ROOT}/my_party/logs,tf:${TF_LOGS}" \
  >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "tensorboard pid=$(cat "$PIDFILE") port=$PORT log=$LOG"
sleep 2
tail -n 15 "$LOG" || true
