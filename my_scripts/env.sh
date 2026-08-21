# shellcheck shell=bash
# 在 conda activate .../g1_train 之后 source 本文件
# 工程根：mujoco_humanoid
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MUJOCO_HUMANOID_ROOT="${MUJOCO_HUMANOID_ROOT:-$_ROOT}"
export MY_RL_TRAIN_ROOT="${MY_RL_TRAIN_ROOT:-$MUJOCO_HUMANOID_ROOT/my_rl_train}"
export MY_SCRIPTS_ROOT="${MY_SCRIPTS_ROOT:-$MUJOCO_HUMANOID_ROOT/my_scripts}"
export MY_PARTY_ROOT="${MY_PARTY_ROOT:-$MUJOCO_HUMANOID_ROOT/my_party}"

export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-highest}"
mkdir -p "$TMPDIR" "$MY_PARTY_ROOT/logs" "$MY_PARTY_ROOT/checkpoints" "$MY_PARTY_ROOT/policies"

_NV_ROOT="${G1_TRAIN_ENV:-/root/autodl-tmp/conda-envs/g1_train}/lib/python3.11/site-packages/nvidia"
_NV_LIBS=""
if [[ -d "$_NV_ROOT" ]]; then
  while IFS= read -r d; do
    _NV_LIBS="${_NV_LIBS:+$_NV_LIBS:}$d"
  done < <(find "$_NV_ROOT" -type d -name lib 2>/dev/null)
fi
export LD_LIBRARY_PATH="${_NV_LIBS}"
unset _NV_ROOT _NV_LIBS _ROOT
