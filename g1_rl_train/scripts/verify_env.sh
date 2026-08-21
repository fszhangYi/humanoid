#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/root/autodl-tmp/conda-envs/g1_train}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"
# shellcheck source=/dev/null
source "$ROOT/scripts/env.sh"

python - <<'PY'
import jax
import mujoco
from mujoco import mjx
from pathlib import Path

assert jax.default_backend() == "gpu", jax.default_backend()
print("[ok] jax", jax.__version__, "devices", jax.devices())
print("[ok] mujoco", mujoco.__version__)

xml = Path("/root/autodl-tmp/mujoco_humanoid/models/unitree_g1/scene_mjx.xml")
m = mujoco.MjModel.from_xml_path(str(xml))
mx = mjx.put_model(m)
dx = mjx.make_data(m)
dx = jax.jit(lambda d: mjx.step(mx, d))(dx)
print("[ok] mjx step on local scene_mjx.xml  nq=", m.nq, "nu=", m.nu)

from mujoco_playground import locomotion
env = locomotion.load("G1JoystickFlatTerrain")
print("[ok] playground G1JoystickFlatTerrain", type(env).__name__)
print("ALL CHECKS PASSED")
PY
