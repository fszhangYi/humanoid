#!/usr/bin/env python3
"""自有策略训练入口：支持断点续训（warm-start）与智能早停。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MY_RL = ROOT / "my_rl_train"
MY_PARTY = ROOT / "my_party"
sys.path.insert(0, str(MY_RL))

from my_rl import jax_compat  # noqa: F401
from my_rl.early_stop import EarlyStop, EarlyStopper
from my_rl.resume import (
    checkpoint_step,
    resolve_resume_checkpoint,
    write_run_pointer,
)

os.chdir(MY_PARTY)
(MY_PARTY / "logs").mkdir(parents=True, exist_ok=True)
(MY_PARTY / "checkpoints").mkdir(parents=True, exist_ok=True)


def _env_bool(name: str, default: bool = False) -> bool:
  v = os.environ.get(name)
  if v is None:
    return default
  return v.strip().lower() in ("1", "true", "yes", "on")


def _load_yaml_config() -> dict:
  cfg_path = Path(
      os.environ.get("CFG", ROOT / "my_rl_train" / "configs" / "train_g1.yaml")
  )
  if not cfg_path.is_file():
    return {}
  try:
    import yaml

    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
  except Exception:
    return {}


def _prepare_resume(argv: list[str], cfg: dict) -> list[str]:
  """Inject --load_checkpoint_path and adjust --num_timesteps if needed."""
  # Respect explicit CLI load path.
  if any(a.startswith("--load_checkpoint_path") for a in argv):
    print("[resume] using CLI --load_checkpoint_path", flush=True)
    return argv

  resume = os.environ.get("RESUME", cfg.get("resume", "off"))
  resume_mode = os.environ.get(
      "RESUME_MODE", cfg.get("resume_mode", "additional")
  )
  env_name = os.environ.get("ENV_NAME", cfg.get("env_name", "G1JoystickFlatTerrain"))
  logs_root = MY_PARTY / "logs"

  try:
    ckpt = resolve_resume_checkpoint(str(resume), logs_root, env_name)
  except FileNotFoundError as e:
    print(f"[resume] ERROR: {e}", flush=True)
    raise SystemExit(2) from e

  if ckpt is None:
    print("[resume] off (fresh run)", flush=True)
    return argv

  step = checkpoint_step(ckpt)
  print(f"[resume] warm-start from {ckpt} (step={step})", flush=True)
  argv = list(argv) + [f"--load_checkpoint_path={ckpt}"]

  # remaining: train only until target_timesteps (additional env steps)
  target = os.environ.get("TARGET_TIMESTEPS")
  if target is None and cfg.get("target_timesteps") is not None:
    target = str(cfg["target_timesteps"])
  if str(resume_mode).lower() == "remaining" and target:
    target_i = int(target)
    remaining = max(0, target_i - step)
    print(
        f"[resume] mode=remaining target={target_i} done={step} "
        f"-> num_timesteps={remaining}",
        flush=True,
    )
    # replace or append num_timesteps
    out = []
    replaced = False
    for a in argv:
      if a.startswith("--num_timesteps="):
        out.append(f"--num_timesteps={remaining}")
        replaced = True
      elif a == "--num_timesteps":
        continue
      else:
        out.append(a)
    if not replaced:
      # handle "--num_timesteps 123" form
      skip = False
      out2 = []
      i = 0
      while i < len(out):
        if out[i] == "--num_timesteps" and i + 1 < len(out):
          out2.extend(["--num_timesteps", str(remaining)])
          i += 2
          replaced = True
          continue
        out2.append(out[i])
        i += 1
      out = out2
      if not replaced:
        out.extend(["--num_timesteps", str(remaining)])
    argv = out
    if remaining == 0:
      print("[resume] already reached target; nothing to train.", flush=True)
      raise SystemExit(0)
  else:
    print(
        f"[resume] mode=additional "
        f"(will train configured num_timesteps more from warm weights)",
        flush=True,
    )

  write_run_pointer(logs_root, ckpt.parents[1], ckpt)
  return argv


def _build_early_stopper(cfg: dict) -> EarlyStopper | None:
  es = cfg.get("early_stop") or {}
  enabled = _env_bool("EARLY_STOP", bool(es.get("enabled", False)))
  if not enabled:
    print("[early_stop] disabled", flush=True)
    return None

  def _get(key, default, cast=float):
    env_key = f"EARLY_STOP_{key.upper()}"
    if os.environ.get(env_key) is not None:
      return cast(os.environ[env_key])
    return cast(es.get(key, default)) if es.get(key, default) is not None else default

  patience = int(_get("patience", es.get("patience", 5), int))
  min_delta = float(_get("min_delta", es.get("min_delta", 0.05), float))
  min_evals = int(_get("min_evals", es.get("min_evals", 3), int))
  target = es.get("target")
  if os.environ.get("EARLY_STOP_TARGET") is not None:
    target = float(os.environ["EARLY_STOP_TARGET"])
  metric_key = os.environ.get(
      "EARLY_STOP_METRIC", es.get("metric_key", "eval/episode_reward")
  )
  mode = os.environ.get("EARLY_STOP_MODE", es.get("mode", "max"))

  stopper = EarlyStopper(
      enabled=True,
      metric_key=metric_key,
      mode=mode,
      patience=patience,
      min_delta=min_delta,
      min_evals=min_evals,
      target=float(target) if target is not None else None,
  )
  print(
      f"[early_stop] enabled patience={patience} min_delta={min_delta} "
      f"min_evals={min_evals} target={target} metric={metric_key}",
      flush=True,
  )
  return stopper


def _install_ppo_hooks(stopper: EarlyStopper | None) -> None:
  from brax.training.agents.ppo import train as ppo_train_mod

  orig_train = ppo_train_mod.train

  def train_with_hooks(*args, **kwargs):
    user_progress = kwargs.get("progress_fn", lambda *_a, **_k: None)

    def progress(num_steps, metrics):
      user_progress(num_steps, metrics)
      if stopper is not None:
        stopper.on_progress(int(num_steps), metrics)

    kwargs["progress_fn"] = progress
    return orig_train(*args, **kwargs)

  ppo_train_mod.train = train_with_hooks  # type: ignore[assignment]


def main() -> int:
  cfg = _load_yaml_config()
  argv = _prepare_resume(sys.argv[1:], cfg)
  sys.argv = [sys.argv[0]] + argv

  stopper = _build_early_stopper(cfg)
  _install_ppo_hooks(stopper)

  # 把实验目录链到 AutoDL 默认 TB 路径（:6007）
  def _link_tf_logs_later() -> None:
    tf_root = Path(os.environ.get("TF_LOGS", "/root/tf-logs"))
    logs = MY_PARTY / "logs"
    if not logs.is_dir():
      return
    # 最新实验目录
    exps = sorted(
        [p for p in logs.iterdir() if p.is_dir() and (p / "checkpoints").exists()],
        key=lambda p: p.stat().st_mtime,
    )
    if not exps:
      return
    exp = exps[-1]
    tf_root.mkdir(parents=True, exist_ok=True)
    link = tf_root / exp.name
    try:
      if link.is_symlink() or not link.exists():
        link.unlink(missing_ok=True)
        link.symlink_to(exp.resolve())
        print(f"[tb] linked {exp} -> {link}", flush=True)
    except OSError as e:
      print(f"[tb] link failed: {e}", flush=True)

  from learning.train_jax_ppo import run

  try:
    _link_tf_logs_later()
    run()
  except EarlyStop as e:
    print(f"Done training (early stop): {e.reason}", flush=True)
    print(
        f"Time note: stopped at step={e.step} metric={e.metric}",
        flush=True,
    )
    return 0
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
