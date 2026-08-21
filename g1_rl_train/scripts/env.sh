# shellcheck shell=bash
# Source after: conda activate /root/autodl-tmp/conda-envs/g1_train
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-highest}"
mkdir -p "$TMPDIR"

# Prefer pip-nvidia CUDA libs; drop system CUDA that breaks JAX plugin discovery.
_NV_ROOT="/root/autodl-tmp/conda-envs/g1_train/lib/python3.11/site-packages/nvidia"
_NV_LIBS=""
if [[ -d "$_NV_ROOT" ]]; then
  while IFS= read -r d; do
    _NV_LIBS="${_NV_LIBS:+$_NV_LIBS:}$d"
  done < <(find "$_NV_ROOT" -type d -name lib 2>/dev/null)
fi
export LD_LIBRARY_PATH="${_NV_LIBS}"
unset _NV_ROOT _NV_LIBS
