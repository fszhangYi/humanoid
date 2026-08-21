#!/usr/bin/env python3
"""Load MuJoCo humanoid (Unitree G1/H1 or DeepMind humanoid), optionally view."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
MODELS: dict[str, Path] = {
    "g1": ROOT / "models" / "unitree_g1" / "scene.xml",
    "h1": ROOT / "models" / "unitree_h1" / "scene.xml",
    "humanoid": ROOT / "models" / "humanoid.xml",
}


def resolve_xml(robot: str | None, xml: Path | None) -> Path:
    if xml is not None:
        return xml.resolve()
    key = (robot or "g1").lower()
    if key not in MODELS:
        raise SystemExit(f"unknown robot {key!r}; choose from {sorted(MODELS)}")
    path = MODELS[key]
    if not path.is_file():
        raise SystemExit(f"model missing: {path}")
    return path.resolve()


def load_model(xml_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    if not xml_path.is_file():
        raise FileNotFoundError(f"model not found: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def summarize(model: mujoco.MjModel, xml_path: Path) -> None:
    bodies = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)
    ]
    acts = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    name = bytes(model.names).split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    print(f"xml        : {xml_path}")
    print(f"model name : {name!r}")
    print(f"nq / nv / nu: {model.nq} / {model.nv} / {model.nu}")
    print(f"nbody / njnt: {model.nbody} / {model.njnt}")
    print(f"bodies ({len(bodies)}): {bodies}")
    print(f"actuators ({len(acts)}): {acts}")


def step_demo(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> None:
    n = int(seconds / model.opt.timestep)
    for _ in range(n):
        mujoco.mj_step(model, data)
    print(
        f"stepped {seconds:.2f}s -> time={data.time:.3f}, "
        f"root_xyz=({data.qpos[0]:.3f}, {data.qpos[1]:.3f}, {data.qpos[2]:.3f})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Load MuJoCo humanoid / Unitree models")
    parser.add_argument(
        "--robot",
        choices=sorted(MODELS),
        default="g1",
        help="preset: g1 (Unitree), h1 (Unitree), humanoid (DeepMind)",
    )
    parser.add_argument("--xml", type=Path, default=None, help="override MJCF path")
    parser.add_argument(
        "--step",
        type=float,
        default=1.0,
        help="simulate this many seconds under gravity (0 to skip)",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="open mjviser web viewer (needs mjviser installed)",
    )
    parser.add_argument("--port", type=int, default=6008, help="mjviser port")
    args = parser.parse_args()

    xml_path = resolve_xml(args.robot, args.xml)
    model, data = load_model(xml_path)
    summarize(model, xml_path)
    if args.step > 0:
        step_demo(model, data, args.step)

    if args.view:
        try:
            import viser
            from mjviser import Viewer
        except ImportError:
            print("mjviser/viser not installed; pip install mjviser", file=sys.stderr)
            return 1
        print(f"mjviser: http://localhost:{args.port}")
        server = viser.ViserServer(host="0.0.0.0", port=args.port)
        Viewer(model, data, server=server).run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
