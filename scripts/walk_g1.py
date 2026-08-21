#!/usr/bin/env python3
"""Walk Unitree G1 in MuJoCo using the official unitree_rl_gym Torch policy.

Policy: third_party/unitree_rl/motion.pt (12-DoF legs, torque via joint PD)
Model:  models/unitree_g1_12dof/scene.xml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "models" / "unitree_g1_12dof" / "scene.xml"
DEFAULT_POLICY = ROOT / "third_party" / "unitree_rl" / "motion.pt"
DEFAULT_CFG = ROOT / "third_party" / "unitree_rl" / "g1.yaml"


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Match unitree_rl_gym/deploy_mujoco.py observation feature."""
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


class G1RLWalker:
    def __init__(
        self,
        model: mujoco.MjModel,
        policy_path: Path,
        cfg: dict,
        cmd: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.simulation_dt = float(cfg["simulation_dt"])
        self.control_decimation = int(cfg["control_decimation"])
        self.kps = np.asarray(cfg["kps"], dtype=np.float32)
        self.kds = np.asarray(cfg["kds"], dtype=np.float32)
        self.default_angles = np.asarray(cfg["default_angles"], dtype=np.float32)
        self.ang_vel_scale = float(cfg["ang_vel_scale"])
        self.dof_pos_scale = float(cfg["dof_pos_scale"])
        self.dof_vel_scale = float(cfg["dof_vel_scale"])
        self.action_scale = float(cfg["action_scale"])
        self.cmd_scale = np.asarray(cfg["cmd_scale"], dtype=np.float32)
        self.num_actions = int(cfg["num_actions"])
        self.num_obs = int(cfg["num_obs"])
        self.cmd = np.asarray(cmd if cmd is not None else cfg["cmd_init"], dtype=np.float32)

        if model.nu != self.num_actions:
            raise ValueError(f"model.nu={model.nu} != num_actions={self.num_actions}")

        self.policy = torch.jit.load(str(policy_path), map_location="cpu")
        self.policy.eval()

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.counter = 0
        self.period = 0.8

    def reset(self, data: mujoco.MjData) -> None:
        mujoco.mj_resetData(self.model, data)
        # Start slightly crouched like training default.
        data.qpos[2] = 0.78
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qpos[7 : 7 + self.num_actions] = self.default_angles
        data.qvel[:] = 0
        self.action[:] = 0
        self.target_dof_pos = self.default_angles.copy()
        self.counter = 0
        mujoco.mj_forward(self.model, data)

    def _update_policy(self, data: mujoco.MjData) -> None:
        qj = data.qpos[7 : 7 + self.num_actions].astype(np.float32)
        dqj = data.qvel[6 : 6 + self.num_actions].astype(np.float32)
        quat = data.qpos[3:7].astype(np.float32)
        omega = data.qvel[3:6].astype(np.float32)

        qj = (qj - self.default_angles) * self.dof_pos_scale
        dqj = dqj * self.dof_vel_scale
        gravity_orientation = get_gravity_orientation(quat)
        omega = omega * self.ang_vel_scale

        count = self.counter * self.simulation_dt
        phase = (count % self.period) / self.period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        n = self.num_actions
        self.obs[:3] = omega
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.cmd_scale
        self.obs[9 : 9 + n] = qj
        self.obs[9 + n : 9 + 2 * n] = dqj
        self.obs[9 + 2 * n : 9 + 3 * n] = self.action
        self.obs[9 + 3 * n : 9 + 3 * n + 2] = np.array([sin_phase, cos_phase], dtype=np.float32)

        with torch.no_grad():
            obs_t = torch.from_numpy(self.obs).unsqueeze(0)
            self.action = self.policy(obs_t).detach().cpu().numpy().squeeze().astype(np.float32)
        self.target_dof_pos = self.action * self.action_scale + self.default_angles

    def apply(self, data: mujoco.MjData) -> None:
        tau = pd_control(
            self.target_dof_pos,
            data.qpos[7 : 7 + self.num_actions],
            self.kps,
            np.zeros_like(self.kds),
            data.qvel[6 : 6 + self.num_actions],
            self.kds,
        )
        data.ctrl[:] = tau
        self.counter += 1
        if self.counter % self.control_decimation == 0:
            self._update_policy(data)


def run_headless(model, data, walker: G1RLWalker, seconds: float) -> dict:
    walker.reset(data)
    model.opt.timestep = walker.simulation_dt
    n = int(seconds / model.opt.timestep)
    x0 = float(data.qpos[0])
    z_min = float(data.qpos[2])
    for _ in range(n):
        walker.apply(data)
        mujoco.mj_step(model, data)
        z_min = min(z_min, float(data.qpos[2]))
    return {
        "dx": float(data.qpos[0]) - x0,
        "dy": float(data.qpos[1]),
        "z": float(data.qpos[2]),
        "z_min": z_min,
        "fell": z_min < 0.35,
    }


def run_viewer(model, data, walker: G1RLWalker, port: int) -> None:
    import viser
    from mjviser import Viewer

    walker.reset(data)
    model.opt.timestep = walker.simulation_dt

    def step_fn(m, d):
        walker.apply(d)
        mujoco.mj_step(m, d)

    def reset_fn(m, d):
        walker.reset(d)

    print(f"G1 RL walk viewer: http://localhost:{port}")
    print(f"cmd (vx, vy, yaw_rate) = {walker.cmd.tolist()}")
    server = viser.ViserServer(host="0.0.0.0", port=port)
    Viewer(model, data, step_fn=step_fn, reset_fn=reset_fn, server=server).run()


def main() -> int:
    p = argparse.ArgumentParser(description="Walk Unitree G1 with RL policy")
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--port", type=int, default=6008)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--vx", type=float, default=0.5, help="forward command (m/s)")
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--no-view", action="store_true")
    args = p.parse_args()

    for path, label in (
        (args.xml, "xml"),
        (args.policy, "policy"),
        (args.config, "config"),
    ):
        if not path.is_file():
            print(f"missing {label}: {path}", file=sys.stderr)
            return 1

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    cmd = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)
    walker = G1RLWalker(model, args.policy, cfg, cmd=cmd)

    st = run_headless(model, data, walker, seconds=args.seconds)
    print(
        f"headless {args.seconds:.1f}s: dx={st['dx']:+.3f} m, dy={st['dy']:+.3f}, "
        f"z={st['z']:.3f}, z_min={st['z_min']:.3f}, fell={st['fell']}"
    )
    if args.no_view:
        return 0 if (not st["fell"] and st["dx"] > 0.3) else 1

    run_viewer(model, data, walker, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
