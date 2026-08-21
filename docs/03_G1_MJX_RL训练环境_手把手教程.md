# 宇树 G1 × MuJoCo MJX RL 训练环境手把手教程（方案 B · 本机实测细化版）

> 目标：在 AutoDL 上搭建 **独立于可视化环境** 的强化学习训练栈，用 **MuJoCo MJX（GPU 并行仿真）+ MuJoCo Playground + Brax PPO** 训练人形行走。  
> 路线：文档《训练自有策略》中的 **方案 B**（不用 Isaac Gym / Isaac Lab）。  
> 实测机：AutoDL；GPU **RTX 5090 D 32GB**；数据盘 `/root/autodl-tmp`；系统盘 `/`。  
> **进度（2026-08-21）：** `g1_train` 环境创建 → JAX CUDA / MJX / Playground 安装 → `verify_env` 通过 → Cartpole PPO 冒烟 `Done training.` → 可启动 `G1JoystickFlatTerrain`。

---

## 这项工作是什么：你装完能做什么

先分清两个项目，避免装错环境：

| 项目 | 路径 | 做什么 | 用哪个 conda |
| --- | --- | --- | --- |
| **部署 / 看机器人走路** | `/root/autodl-tmp/mujoco_humanoid` | 加载预训练 `motion.pt`，mjviser 网页看 | `conda-envs/mjviser`（可 CPU） |
| **训练（本教程）** | `/root/autodl-tmp/g1_rl_train` | GPU 上大规模仿真 + PPO 学策略 | **`conda-envs/g1_train`（必须 GPU/JAX）** |

本教程装完后你应能：

1. `jax.default_backend()` 打印 **`gpu`**；  
2. 用 MJX 对本地 `scene_mjx.xml` 做一步仿真；  
3. `locomotion.load("G1JoystickFlatTerrain")` 成功；  
4. 跑通一次短 PPO（Cartpole 冒烟），日志出现 **`Done training.`**；  
5. 用同一入口对 G1 开训（长跑另算时间）。

> **本教程不覆盖：** 把训好的权重导出并替换 `mujoco_humanoid` 里的 `motion.pt`（观测对齐见该项目 `docs/02_*.md`）。也不覆盖真机部署。

---

## 读前须知（零基础请先读完）

1. **不要用 `mjviser` 环境训练。** 那个环境是 CPU 版 PyTorch + 网页查看器，和本教程依赖冲突。  
2. **分盘是硬要求：** `jax[cuda12]` 会下载数 GB 的 NVIDIA wheel；**pip 临时目录若落在系统盘 `/tmp`，很容易把 30G 系统盘写爆。** 必须 `export TMPDIR=/root/autodl-tmp/tmp`。  
3. **磁盘建议：** 数据盘空闲 **≥ 25GB** 再开始（环境约 8GB + menagerie 缓存 + checkpoint）。本机装完后数据盘曾只剩约十几 GB，长训前建议再清理。  
4. **网络：** 装 pip 包时一般 **关掉** `source /etc/network_turbo`，用清华源；首次加载 Playground G1 会从 GitHub **clone mujoco_menagerie**，那时再开 turbo 或代理。  
5. **本机踩过的两个兼容坑（教程已写进脚本）：**  
   - 系统 `/usr/local/cuda` 进 `LD_LIBRARY_PATH` 会导致 JAX 找不到 pip 自带的 cuSPARSE → 掉到 CPU；  
   - JAX 0.10 删除了 `jax.device_put_replicated`，Brax 0.14 仍在用 → 需兼容补丁（`g1_rl/jax_compat.py`）。  
6. **无显示器机器不要渲染视频：** 训练结束默认可能调 OpenGL；请加 **`--num_videos 0`**。

### 本机实测版本（写教程时锁定）

| 项 | 版本 / 路径 |
| --- | --- |
| conda | Miniconda `/root/miniconda3` |
| Python | 3.11 |
| 环境前缀 | `/root/autodl-tmp/conda-envs/g1_train` |
| 工程目录 | `/root/autodl-tmp/g1_rl_train` |
| JAX | 0.10.2 + `jax[cuda12]`（backend=`gpu`） |
| mujoco / mujoco-mjx | 3.11.0 |
| warp-lang | 1.14.0 |
| brax | 0.14.2 |
| playground | 0.2.0 |
| 冒烟任务 | `CartpoleBalance` |
| G1 训练任务名 | `G1JoystickFlatTerrain` |

⏱ 干净机器墙钟粗估：**30–60 分钟**（含大包下载；首次拉 menagerie 再加数分钟）。

---

## 0. 前置检查

### 0.1 GPU 与驱动

```bash
nvidia-smi
```

应能看到 GPU 名称与显存。本教程在 **RTX 5090 D** 上实测通过；其它 NVIDIA GPU 一般也可，但需 CUDA 12 可用的 JAX 轮子。

### 0.2 磁盘

```bash
df -h / /root/autodl-tmp
```

建议：

| 分区 | 建议空闲 |
| --- | --- |
| `/` 系统盘 | ≥ **5GB**（仍可能被别的程序占用；**不要把 pip 临时文件放这里**） |
| `/root/autodl-tmp` 数据盘 | ≥ **25GB**（越充足越好） |

空间不够时可先清理（按需，勿误删业务数据）：

```bash
# 示例：清空 conda/pip 缓存（可再下）
rm -rf /root/autodl-tmp/conda-pkgs/* /root/autodl-tmp/.pip-cache/* /root/autodl-tmp/tmp/*
conda clean -a -y
```

### 0.3 conda

```bash
which conda
conda --version
source /root/miniconda3/etc/profile.d/conda.sh
```

没有 conda 时先装 Miniconda（略）。

### 0.4 目录约定

后面所有命令默认：

```bash
export CONDA_ENVS_DIRS=/root/autodl-tmp/conda-envs
export CONDA_PKGS_DIRS=/root/autodl-tmp/conda-pkgs
export PIP_CACHE_DIR=/root/autodl-tmp/.pip-cache
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$CONDA_ENVS_DIRS" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
```

**每次新开终端装包前，都建议重新 export `TMPDIR`。**

---

## 1. 创建 conda 环境（Python 3.11）

本机清华 `pkgs/r` 频道曾 **HTTP 404**，创建时用 `--override-channels` 只走 main：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_DIRS=/root/autodl-tmp/conda-envs
export CONDA_PKGS_DIRS=/root/autodl-tmp/conda-pkgs

conda create -y -p /root/autodl-tmp/conda-envs/g1_train python=3.11 \
  --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
```

激活并确认路径：

```bash
conda activate /root/autodl-tmp/conda-envs/g1_train
which python
# 应类似：.../conda-envs/g1_train/bin/python
python -V   # Python 3.11.x
```

升级 pip：

```bash
export TMPDIR=/root/autodl-tmp/tmp PIP_CACHE_DIR=/root/autodl-tmp/.pip-cache
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
python -m pip install -U pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 2. 安装 JAX（CUDA 12）——最关键、也最大的一步

### 2.1 安装

```bash
conda activate /root/autodl-tmp/conda-envs/g1_train
export TMPDIR=/root/autodl-tmp/tmp
export PIP_CACHE_DIR=/root/autodl-tmp/.pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# 会拉取 jaxlib、cudnn、cublas 等大包，耐心等待
python -m pip install -U "jax[cuda12]"
```

> 若必须用国内镜像：可加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`（NVIDIA 大 wheel 也能下，只是有时较慢）。

### 2.2 验证 GPU

**不要**把错误的系统 CUDA 路径留在 `LD_LIBRARY_PATH`。验收时：

```bash
# 若之前 source 过 network_turbo 等，先清掉系统 cuda 路径干扰：
# 下面两行：先清空，再指到 pip 自带的 nvidia 库（与 scripts/env.sh 一致）
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(find /root/autodl-tmp/conda-envs/g1_train/lib/python3.11/site-packages/nvidia -type d -name lib | paste -sd:)

python - <<'PY'
import jax
print("backend:", jax.default_backend())
print("devices:", jax.devices())
assert jax.default_backend() == "gpu"
print("JAX GPU OK")
PY
```

期望：

```text
backend: gpu
devices: [CudaDevice(id=0)]
JAX GPU OK
```

若出现 `Unable to load cuSPARSE` / `Falling back to cpu`：检查是否误设了 `LD_LIBRARY_PATH=/usr/local/cuda/lib64`，按上面改回 pip-nvidia 路径。

---

## 3. 安装 MuJoCo MJX、Warp、训练栈、Playground

```bash
conda activate /root/autodl-tmp/conda-envs/g1_train
export TMPDIR=/root/autodl-tmp/tmp
export PIP_CACHE_DIR=/root/autodl-tmp/.pip-cache

python -m pip install -U mujoco mujoco-mjx \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# NVIDIA GPU 上可用的 Warp 后端（可选但推荐）
python -m pip install -U "mujoco-mjx[warp]" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# PPO 相关
python -m pip install -U brax flax optax orbax-checkpoint pyyaml \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# DeepMind 任务套件（含 G1JoystickFlatTerrain）
python -m pip install -U playground \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

快速 import 检查：

```bash
python - <<'PY'
import mujoco
from mujoco import mjx
import brax, flax
import importlib.metadata as m
print("mujoco", mujoco.__version__)
print("mujoco-mjx", m.version("mujoco-mjx"))
print("brax", m.version("brax"))
print("playground", m.version("playground"))
print("warp", m.version("warp-lang"))
PY
```

---

## 4. 准备工程目录与脚本

若本机已有 `/root/autodl-tmp/g1_rl_train`，可跳到 **§5**。  
从零创建时：

```bash
mkdir -p /root/autodl-tmp/g1_rl_train/{g1_rl,scripts,configs,checkpoints,logs}
```

### 4.1 `g1_rl/jax_compat.py`（JAX 0.10 ↔ Brax 兼容）

保存为 `/root/autodl-tmp/g1_rl_train/g1_rl/jax_compat.py`：

```python
"""Import this before brax/playground training under JAX>=0.10."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _device_put_replicated(value, devices):
  devices = list(devices)
  mesh = Mesh(np.array(devices), axis_names=("i",))
  sharding = NamedSharding(mesh, P("i"))

  def _put(x):
    x = jnp.asarray(x)
    x = jnp.broadcast_to(x, (len(devices),) + x.shape)
    return jax.device_put(x, sharding)

  return jax.tree_util.tree_map(_put, value)


def apply() -> None:
  jax.device_put_replicated = _device_put_replicated  # type: ignore[attr-defined]


apply()
```

### 4.2 `g1_rl/__init__.py`

```python
"""G1 MJX RL helpers (Scheme B)."""
from . import jax_compat
jax_compat.apply()
```

### 4.3 `scripts/env.sh`（每次训练前 source）

保存为 `/root/autodl-tmp/g1_rl_train/scripts/env.sh`：

```bash
# shellcheck shell=bash
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-highest}"
mkdir -p "$TMPDIR"

_NV_ROOT="/root/autodl-tmp/conda-envs/g1_train/lib/python3.11/site-packages/nvidia"
_NV_LIBS=""
if [[ -d "$_NV_ROOT" ]]; then
  while IFS= read -r d; do
    _NV_LIBS="${_NV_LIBS:+$_NV_LIBS:}$d"
  done < <(find "$_NV_ROOT" -type d -name lib 2>/dev/null)
fi
export LD_LIBRARY_PATH="${_NV_LIBS}"
unset _NV_ROOT _NV_LIBS
```

### 4.4 `scripts/train_g1.py`

```python
#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_rl import jax_compat  # noqa: F401
from learning.train_jax_ppo import run

if __name__ == "__main__":
  run()
```

```bash
chmod +x /root/autodl-tmp/g1_rl_train/scripts/train_g1.py
```

### 4.5（推荐）同步修补 site-packages 里的 Brax

仅靠 `train_g1.py` 导入补丁即可；为让直接调用 `train-jax-ppo` 也稳妥，可把同一 `device_put_replicated` 写进：

`.../site-packages/brax/training/pmap.py`

本机已按此修补；若你重装了 brax，需重新打补丁或始终用 `scripts/train_g1.py` 入口。

---

## 5. 验收一：环境自检（必做）

需要本地已有 MJX 友好模型（本机部署项目里已有）：

`/root/autodl-tmp/mujoco_humanoid/models/unitree_g1/scene_mjx.xml`

若还没有，可从 [MuJoCo Menagerie `unitree_g1`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) 只取 `scene_mjx.xml` / `g1_mjx.xml` + `assets`。

创建 `/root/autodl-tmp/g1_rl_train/scripts/verify_env.sh`（或直接用仓库里已有文件）：

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/g1_train
source "$ROOT/scripts/env.sh"

python - <<'PY'
import jax, mujoco
from mujoco import mjx
from pathlib import Path

assert jax.default_backend() == "gpu", jax.default_backend()
print("[ok] jax", jax.__version__, "devices", jax.devices())
print("[ok] mujoco", mujoco.__version__)

xml = Path("/root/autodl-tmp/mujoco_humanoid/models/unitree_g1/scene_mjx.xml")
m = mujoco.MjModel.from_xml_path(str(xml))
mx = mjx.put_model(m)
dx = mjx.make_data(m)
dx = jax.jit(lambda d: mjx.step(mx, d))(dx)
print("[ok] mjx step  nq=", m.nq, "nu=", m.nu)

from mujoco_playground import locomotion
env = locomotion.load("G1JoystickFlatTerrain")
print("[ok] playground", type(env).__name__)
print("ALL CHECKS PASSED")
PY
```

运行：

```bash
chmod +x /root/autodl-tmp/g1_rl_train/scripts/verify_env.sh
bash /root/autodl-tmp/g1_rl_train/scripts/verify_env.sh
```

### 首次加载 G1 时

Playground 若提示 `mujoco_menagerie not found. Downloading...`：

```bash
# 需要访问 GitHub 时再开加速（装 pip 包时请关掉）
source /etc/network_turbo   # AutoDL
bash /root/autodl-tmp/g1_rl_train/scripts/verify_env.sh
```

menagerie 会落到类似：

`.../site-packages/mujoco_playground/external_deps/mujoco_menagerie`

成功标志：

```text
[ok] jax ... devices [CudaDevice(id=0)]
[ok] mjx step ...
[ok] playground Joystick
ALL CHECKS PASSED
```

### 重要：哪个 XML 能进 MJX？

| 模型 | 能否 `mjx.put_model` | 用途 |
| --- | --- | --- |
| `unitree_g1/scene_mjx.xml` | ✅ | MJX 训练 / 验收 |
| Playground `G1JoystickFlatTerrain` | ✅ | 正式训行走 |
| `unitree_g1_12dof/scene.xml`（部署用） | ❌（cylinder–mesh 碰撞未实现） | 只给 CPU MuJoCo + `walk_g1.py` |

---

## 6. 验收二：PPO 冒烟（Cartpole）

无头服务器请禁用视频：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/g1_train
source /root/autodl-tmp/g1_rl_train/scripts/env.sh
cd /root/autodl-tmp/g1_rl_train

python scripts/train_g1.py \
  --env_name CartpoleBalance \
  --num_timesteps 8000 \
  --num_videos 0
```

或：

```bash
bash scripts/smoke_train.sh
```

本机实测日志特征：

```text
PPO Training Parameters:
...
Done training.
Time to JIT compile: ~18s
Time to train: ~2min
```

`exit code = 0` 即冒烟通过。  
日志与 checkpoint 在：`g1_rl_train/logs/CartpoleBalance-<时间戳>/`。

> 若失败在 `mjr_makeContext` / OpenGL：多半忘了 `--num_videos 0`。

---

## 7. 开始训练宇树 G1

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/g1_train
source /root/autodl-tmp/g1_rl_train/scripts/env.sh
cd /root/autodl-tmp/g1_rl_train

# 短跑试通（分钟～十分钟级，视 JIT/步数）
python scripts/train_g1.py \
  --env_name G1JoystickFlatTerrain \
  --num_timesteps 200000 \
  --num_videos 0

# 正式长训（Playground 默认配置量级可达上亿步，请按显存与时间自行调整）
python scripts/train_g1.py \
  --env_name G1JoystickFlatTerrain \
  --num_timesteps 50000000 \
  --num_videos 0
```

查看全部参数：

```bash
python scripts/train_g1.py --helpfull
```

常用相关 flag：`--env_name`、`--num_timesteps`、`--num_videos`。

建议用 `tmux` / `nohup` 挂机，并盯磁盘：

```bash
df -h /root/autodl-tmp
du -sh /root/autodl-tmp/g1_rl_train/logs
```

---

## 8. 日常激活速查卡（复制即用）

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/g1_train
source /root/autodl-tmp/g1_rl_train/scripts/env.sh
cd /root/autodl-tmp/g1_rl_train

bash scripts/verify_env.sh
# python scripts/train_g1.py --env_name G1JoystickFlatTerrain --num_timesteps 200000 --num_videos 0
```

---

## 9. 常见坑（本机实录）

### 坑 A：`pip install jax[cuda12]` 把系统盘写满

- **现象：** `/tmp` 或 `/` 暴涨，`No space left on device`。  
- **处理：** `export TMPDIR=/root/autodl-tmp/tmp`，清 `/tmp/pip-*`，重装。

### 坑 B：JAX 掉到 CPU（`Falling back to cpu`）

- **现象：** `Unable to load cuSPARSE`。  
- **处理：** `source scripts/env.sh`，不要用裸的 `/usr/local/cuda/lib64` 作为唯一 `LD_LIBRARY_PATH`。

### 坑 C：`AttributeError: jax.device_put_replicated is deprecated`

- **原因：** JAX 0.10 + Brax 0.14。  
- **处理：** 使用本教程的 `g1_rl/jax_compat.py` + `scripts/train_g1.py`，不要裸跑未打补丁的旧入口。

### 坑 D：训练已结束却报 OpenGL / `mjr_makeContext`

- **处理：** 加 `--num_videos 0`。

### 坑 E：`NotImplementedError: (CYLINDER, MESH) collisions`

- **原因：** 对部署用的 `g1_12dof` 做了 `mjx.put_model`。  
- **处理：** 训练用 `scene_mjx.xml` 或 Playground 任务。

### 坑 F：conda `pkgs/r` 404

- **处理：** `conda create ... --override-channels -c .../pkgs/main`（见 §1）。

### 坑 G：首次 G1 卡住很久

- **原因：** 正在 clone menagerie。  
- **处理：** 开 GitHub 加速，等进度条到 100%；或预先放好 `external_deps/mujoco_menagerie`。

---

## 10. 和「已经能走路」的部署项目如何衔接

| 阶段 | 位置 |
| --- | --- |
| 现在（本教程） | 在 MJX 里 **训练** 新策略 |
| 下一步 | 导出权重，对齐观测/动作/PD，再接到 `mujoco_humanoid/scripts/walk_g1.py` |
| 观看 | `mjviser` 环境，端口 **6008** |

详细部署原理：`mujoco_humanoid/docs/01_G1_MuJoCo行走实现详解.md`  
训练规划总述：`mujoco_humanoid/docs/02_训练自有策略_环境与步骤.md`  
工程速查：`g1_rl_train/README.md`

---

## 11. 验收清单（全部勾上即复现成功）

- [ ] `conda activate .../g1_train` 成功，`which python` 指向该环境  
- [ ] `jax.default_backend() == "gpu"`  
- [ ] `bash scripts/verify_env.sh` 输出 `ALL CHECKS PASSED`  
- [ ] `python scripts/train_g1.py --env_name CartpoleBalance --num_timesteps 8000 --num_videos 0` 日志含 `Done training.`  
- [ ] 能执行（至少启动）`G1JoystickFlatTerrain` 训练命令  

---

## 12. 一键复现脚本（可选）

把下列内容存为 `/root/autodl-tmp/setup_g1_train.sh`（已有环境可只跑后半验收）：

```bash
#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_DIRS=/root/autodl-tmp/conda-envs
export CONDA_PKGS_DIRS=/root/autodl-tmp/conda-pkgs
export PIP_CACHE_DIR=/root/autodl-tmp/.pip-cache
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$CONDA_ENVS_DIRS" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true

ENV=/root/autodl-tmp/conda-envs/g1_train
if [[ ! -x "$ENV/bin/python" ]]; then
  conda create -y -p "$ENV" python=3.11 \
    --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
fi
conda activate "$ENV"
python -m pip install -U pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -U "jax[cuda12]"
python -m pip install -U mujoco "mujoco-mjx[warp]" brax flax optax orbax-checkpoint pyyaml playground \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "---- 请确认已有 g1_rl_train 脚本后执行 ----"
echo "source $ENV/../g1_rl_train/scripts/env.sh  # 路径按实际修改"
echo "bash /root/autodl-tmp/g1_rl_train/scripts/verify_env.sh"
```

---

## 参考链接

- MuJoCo MJX：<https://mujoco.readthedocs.io/en/stable/mjx.html>  
- MuJoCo Playground：<https://github.com/google-deepmind/mujoco_playground>  
- Brax：<https://github.com/google/brax>  
- Menagerie G1：<https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1>  

---

*文档依据 2026-08-21 本机 AutoDL（RTX 5090 D）实测步骤整理；PyPI 小版本号会变动，以你环境中 `pip show jax mujoco mujoco-mjx brax playground` 为准。*
