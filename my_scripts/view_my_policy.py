#!/usr/bin/env python3
"""用自有训练 checkpoint 驱动 G1，经 mjviser 在指定端口可视化。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import mujoco
import numpy as np
import viser
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_ckpt
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen
from mjviser import Viewer
from mujoco_playground import locomotion
from mujoco_playground._src import wrapper
from mujoco_playground.config import locomotion_params

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "my_rl_train"))
from my_rl import jax_compat  # noqa: F401

DEFAULT_CKPT_ROOT = (
    ROOT
    / "my_party"
    / "logs"
    / "G1JoystickFlatTerrain-20260821-134220"
    / "checkpoints"
)


def latest_checkpoint(ckpt_root: Path) -> Path:
  dirs = [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit()]
  if not dirs:
    raise FileNotFoundError(f"no checkpoints under {ckpt_root}")
  return sorted(dirs, key=lambda p: int(p.name))[-1]


def load_inference_fn(ckpt: Path):
  """brax load_policy 会因 config 里 null init 失败；这里手动还原网络。"""
  ckpt = ckpt.resolve()
  cfg = json.loads((ckpt / "ppo_network_config.json").read_text())
  params = ppo_ckpt.load(ckpt)
  obs_size = {
      k: int(np.prod(v["shape"])) for k, v in cfg["observation_size"].items()
  }
  nfk = cfg["network_factory_kwargs"]
  network = ppo_networks.make_ppo_networks(
      observation_size=obs_size,
      action_size=cfg["action_size"],
      preprocess_observations_fn=running_statistics.normalize,
      policy_hidden_layer_sizes=tuple(nfk["policy_hidden_layer_sizes"]),
      value_hidden_layer_sizes=tuple(nfk["value_hidden_layer_sizes"]),
      activation=linen.silu,
      policy_obs_key=nfk["policy_obs_key"],
      value_obs_key=nfk["value_obs_key"],
      distribution_type=nfk["distribution_type"],
      noise_std_type=nfk["noise_std_type"],
      init_noise_std=nfk["init_noise_std"],
      state_dependent_std=nfk["state_dependent_std"],
  )
  return ppo_networks.make_inference_fn(network)(params, deterministic=True)


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--port", type=int, default=6008)
  p.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
  p.add_argument("--ckpt", type=Path, default=None)
  p.add_argument("--env-name", default="G1JoystickFlatTerrain")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  ckpt = (args.ckpt or latest_checkpoint(args.ckpt_root.resolve())).resolve()
  print(f"loading policy: {ckpt}", flush=True)

  env = locomotion.load(args.env_name)
  ppo_params = locomotion_params.brax_ppo_config(args.env_name)
  wrapped = wrapper.wrap_for_brax_training(
      env,
      episode_length=int(ppo_params.episode_length),
      action_repeat=int(ppo_params.get("action_repeat", 1)),
  )

  policy_fn = load_inference_fn(ckpt)
  jit_policy = jax.jit(policy_fn)
  jit_reset = jax.jit(wrapped.reset)
  jit_step = jax.jit(wrapped.step)

  rng = jax.random.PRNGKey(args.seed)
  carry = {
      "state": None,
      "rng": rng,
  }

  def soft_reset():
    carry["rng"], key = jax.random.split(carry["rng"])
    keys = jax.random.split(key, 1)
    carry["state"] = jit_reset(keys)

  print("JIT compile (first reset/step, may take ~1–2 min)...", flush=True)
  soft_reset()
  # warm up one policy+step
  carry["rng"], act_key = jax.random.split(carry["rng"])
  act, _ = jit_policy(carry["state"].obs, act_key)
  carry["state"] = jit_step(carry["state"], act)
  print("JIT ready", flush=True)

  mj_model = env.mj_model
  mj_data = mujoco.MjData(mj_model)

  def sync(d: mujoco.MjData) -> None:
    qpos = np.asarray(carry["state"].data.qpos)
    qvel = np.asarray(carry["state"].data.qvel)
    if qpos.ndim == 2:
      qpos, qvel = qpos[0], qvel[0]
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    mujoco.mj_forward(mj_model, d)

  sync(mj_data)

  def step_fn(m: mujoco.MjModel, d: mujoco.MjData) -> None:
    carry["rng"], act_key = jax.random.split(carry["rng"])
    act, _ = jit_policy(carry["state"].obs, act_key)
    carry["state"] = jit_step(carry["state"], act)
    sync(d)

  def reset_fn(m: mujoco.MjModel, d: mujoco.MjData) -> None:
    soft_reset()
    sync(d)

  print(f"mjviser: http://0.0.0.0:{args.port}  (checkpoint {ckpt.name})", flush=True)
  server = viser.ViserServer(host="0.0.0.0", port=args.port)
  Viewer(mj_model, mj_data, step_fn=step_fn, reset_fn=reset_fn, server=server).run()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
