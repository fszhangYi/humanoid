#!/usr/bin/env python3
"""Train G1 (or other Playground envs) with JAX PPO + MJX.

Applies a small JAX 0.10 compat shim, then delegates to Playground's train-jax-ppo.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_rl import jax_compat  # noqa: F401  — patch jax.device_put_replicated

from learning.train_jax_ppo import run

if __name__ == "__main__":
  run()
