#!/usr/bin/env python3
"""自有策略训练入口：Playground PPO + MJX（代码仅依赖 my_rl_train / my_scripts / my_party）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# mujoco_humanoid 根目录
ROOT = Path(__file__).resolve().parents[1]
MY_RL = ROOT / "my_rl_train"
MY_PARTY = ROOT / "my_party"
sys.path.insert(0, str(MY_RL))

from my_rl import jax_compat  # noqa: F401

# 默认把 Playground/Brax 日志写到 my_party/logs（通过 cwd）
os.chdir(MY_PARTY)
(MY_PARTY / "logs").mkdir(parents=True, exist_ok=True)
(MY_PARTY / "checkpoints").mkdir(parents=True, exist_ok=True)

from learning.train_jax_ppo import run

if __name__ == "__main__":
  run()
