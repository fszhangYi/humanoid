"""Eval-based early stopping for Playground/Brax PPO progress callbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


class EarlyStop(Exception):
  """Raised from progress_fn to abort the Brax training loop cleanly."""

  def __init__(self, reason: str, step: int = 0, metric: float | None = None):
    super().__init__(reason)
    self.reason = reason
    self.step = step
    self.metric = metric


@dataclass
class EarlyStopper:
  """Stop when eval reward plateaus (or hits an optional target)."""

  enabled: bool = True
  metric_key: str = "eval/episode_reward"
  mode: str = "max"  # max: higher better; min: lower better
  patience: int = 5
  min_delta: float = 0.05
  min_evals: int = 3  # trained evals (excludes step==0) before stop allowed
  target: Optional[float] = None

  best: Optional[float] = field(default=None, init=False)
  best_step: int = field(default=0, init=False)
  bad_count: int = field(default=0, init=False)
  trained_evals: int = field(default=0, init=False)

  def _extract(self, metrics: Mapping[str, Any]) -> Optional[float]:
    if self.metric_key in metrics:
      return float(metrics[self.metric_key])
    # fallbacks seen in brax logs
    for k in ("episode/sum_reward", "eval/episode_reward"):
      if k in metrics:
        return float(metrics[k])
    return None

  def _is_improvement(self, value: float) -> bool:
    if self.best is None:
      return True
    if self.mode == "min":
      return value < self.best - self.min_delta
    return value > self.best + self.min_delta

  def _hit_target(self, value: float) -> bool:
    if self.target is None:
      return False
    if self.mode == "min":
      return value <= self.target
    return value >= self.target

  def on_progress(self, step: int, metrics: Mapping[str, Any]) -> None:
    if not self.enabled:
      return
    value = self._extract(metrics)
    if value is None:
      return

    # Initial eval (step 0): seed best, do not consume patience.
    if step == 0:
      self.best = value
      self.best_step = 0
      print(
          f"[early_stop] init metric={value:.4f} key={self.metric_key}",
          flush=True,
      )
      return

    self.trained_evals += 1
    if self._is_improvement(value):
      prev = self.best
      self.best = value
      self.best_step = step
      self.bad_count = 0
      print(
          f"[early_stop] improve step={step} {value:.4f}"
          f" (was {prev if prev is not None else float('nan'):.4f})"
          f" patience_reset",
          flush=True,
      )
    else:
      self.bad_count += 1
      print(
          f"[early_stop] no_improve step={step} {value:.4f}"
          f" best={self.best:.4f}@{self.best_step}"
          f" bad={self.bad_count}/{self.patience}",
          flush=True,
      )

    if self._hit_target(value):
      raise EarlyStop(
          f"hit target {self.target} at step={step} metric={value:.4f}",
          step=step,
          metric=value,
      )

    if (
        self.trained_evals >= self.min_evals
        and self.bad_count >= self.patience
    ):
      raise EarlyStop(
          f"plateau: no improve for {self.patience} evals "
          f"(best={self.best:.4f}@{self.best_step}, last={value:.4f}@{step})",
          step=step,
          metric=value,
      )
