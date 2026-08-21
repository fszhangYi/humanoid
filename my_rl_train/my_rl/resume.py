"""Checkpoint discovery helpers for resume / warm-start."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def is_ckpt_dir(path: Path) -> bool:
  return path.is_dir() and path.name.isdigit()


def list_checkpoints(checkpoints_dir: Path) -> list[Path]:
  if not checkpoints_dir.is_dir():
    return []
  ckpts = [p for p in checkpoints_dir.iterdir() if is_ckpt_dir(p)]
  return sorted(ckpts, key=lambda p: int(p.name))


def latest_checkpoint(checkpoints_dir: Path) -> Optional[Path]:
  ckpts = list_checkpoints(checkpoints_dir)
  return ckpts[-1] if ckpts else None


def checkpoint_step(ckpt: Path) -> int:
  return int(ckpt.name)


def find_experiment_dirs(logs_root: Path, env_name: Optional[str] = None) -> list[Path]:
  if not logs_root.is_dir():
    return []
  out = []
  for p in logs_root.iterdir():
    if not p.is_dir():
      continue
    if env_name and not p.name.startswith(env_name):
      continue
    if (p / "checkpoints").is_dir():
      out.append(p)
  return sorted(out, key=lambda p: p.stat().st_mtime)


def resolve_resume_checkpoint(
    resume: str,
    logs_root: Path,
    env_name: str,
) -> Optional[Path]:
  """Resolve resume spec to a concrete checkpoint directory.

  resume:
    - off / empty / none → None
    - auto / 1 / true → latest ckpt under newest matching experiment
    - path to experiment dir, checkpoints dir, or a numbered ckpt dir
  """
  spec = (resume or "off").strip()
  if spec.lower() in ("", "off", "none", "false", "0"):
    return None

  if spec.lower() in ("auto", "1", "true", "yes"):
    exps = find_experiment_dirs(logs_root, env_name=env_name)
    for exp in reversed(exps):
      ckpt = latest_checkpoint(exp / "checkpoints")
      if ckpt is not None:
        return ckpt
    return None

  path = Path(spec).expanduser().resolve()
  if is_ckpt_dir(path):
    return path
  if path.name == "checkpoints" and path.is_dir():
    return latest_checkpoint(path)
  if (path / "checkpoints").is_dir():
    return latest_checkpoint(path / "checkpoints")
  if path.is_dir():
    # maybe checkpoints root itself with numeric children
    ckpt = latest_checkpoint(path)
    if ckpt is not None:
      return ckpt
  raise FileNotFoundError(f"cannot resolve resume checkpoint from: {spec}")


def write_run_pointer(logs_root: Path, exp_dir: Path, ckpt: Optional[Path] = None) -> None:
  logs_root.mkdir(parents=True, exist_ok=True)
  (logs_root / "LATEST_RUN").write_text(str(exp_dir.resolve()) + "\n", encoding="utf-8")
  if ckpt is not None:
    (logs_root / "LATEST_CKPT").write_text(str(ckpt.resolve()) + "\n", encoding="utf-8")


def save_resume_meta(exp_dir: Path, meta: dict) -> None:
  exp_dir.mkdir(parents=True, exist_ok=True)
  path = exp_dir / "resume_meta.json"
  path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
