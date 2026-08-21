#!/usr/bin/env bash
# 验收自有训练环境（GPU + MJX + Playground G1）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${G1_TRAIN_ENV:-/root/autodl-tmp/conda-envs/g1_train}"
# shellcheck source=/dev/null
source "$ROOT/my_scripts/env.sh"

python - <<'PY'
import jax, mujoco
from mujoco import mjx
from pathlib import Path
import os

root = Path(os.environ["MUJOCO_HUMANOID_ROOT"])
assert jax.default_backend() == "gpu", jax.default_backend()
print("[ok] jax", jax.__version__, jax.devices())
print("[ok] mujoco", mujoco.__version__)

xml = root / "models/unitree_g1/scene_mjx.xml"
m = mujoco.MjModel.from_xml_path(str(xml))
mx = mjx.put_model(m)
dx = mjx.make_data(m)
dx = jax.jit(lambda d: mjx.step(mx, d))(dx)
print("[ok] mjx step nq=", m.nq, "nu=", m.nu)

from mujoco_playground import locomotion
env = locomotion.load("G1JoystickFlatTerrain")
print("[ok] G1JoystickFlatTerrain", type(env).__name__)
print("ALL CHECKS PASSED")
PY
