#!/usr/bin/env python3
"""从训练 stdout 日志解析 eval reward，写入 TensorBoard（适配未开 --use_tb 的在跑任务）。"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from tensorboardX import SummaryWriter

RE_REWARD = re.compile(
    r"^(?P<step>\d+):\s+reward=(?P<rew>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
RE_EARLY = re.compile(r"\[early_stop\].*step=(?P<step>\d+).*?(?P<val>[-+]?\d+\.\d+)")


def parse_line(line: str) -> list[tuple[str, int, float]]:
  out: list[tuple[str, int, float]] = []
  m = RE_REWARD.match(line.strip())
  if m:
    out.append(("eval/episode_reward", int(m.group("step")), float(m.group("rew"))))
  return out


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--log", type=Path, required=True, help="train_*.log")
  p.add_argument(
      "--logdir",
      type=Path,
      required=True,
      help="TensorBoard event 目录（可多个，逗号分隔见 --also）",
  )
  p.add_argument(
      "--also",
      type=Path,
      nargs="*",
      default=[],
      help="额外写入的 logdir（如 /root/tf-logs/...）",
  )
  p.add_argument("--poll", type=float, default=2.0)
  p.add_argument("--once", action="store_true", help="只回填已有内容后退出")
  args = p.parse_args()

  logdirs = [args.logdir, *args.also]
  writers = []
  for d in logdirs:
    d.mkdir(parents=True, exist_ok=True)
    writers.append(SummaryWriter(logdir=str(d)))
    print(f"[tb_bridge] writing -> {d}", flush=True)

  seen: set[tuple[str, int]] = set()
  pos = 0

  def ingest(text: str) -> int:
    n = 0
    for line in text.splitlines():
      for tag, step, val in parse_line(line):
        key = (tag, step)
        if key in seen:
          continue
        seen.add(key)
        for w in writers:
          w.add_scalar(tag, val, step)
        print(f"[tb_bridge] {step}: {tag}={val:.4f}", flush=True)
        n += 1
    if n:
      for w in writers:
        w.flush()
    return n

  # initial backfill
  if args.log.is_file():
    data = args.log.read_text(encoding="utf-8", errors="ignore")
    pos = len(data)
    ingest(data)

  if args.once:
    for w in writers:
      w.close()
    return 0

  print(f"[tb_bridge] watching {args.log} poll={args.poll}s", flush=True)
  while True:
    time.sleep(args.poll)
    if not args.log.is_file():
      continue
    with args.log.open("r", encoding="utf-8", errors="ignore") as f:
      f.seek(pos)
      chunk = f.read()
      pos = f.tell()
    if chunk:
      ingest(chunk)


if __name__ == "__main__":
  raise SystemExit(main())
