# G1 RL 训练（方案 B：MuJoCo MJX）

独立于 `mujoco_humanoid` 部署项目的训练工程。使用 **JAX + MuJoCo MJX + MuJoCo Playground** 在 GPU 上并行仿真并训练行走策略。

**零基础完整复现教程（推荐先读）：**  
[`../docs/03_G1_MJX_RL训练环境_手把手教程.md`](../docs/03_G1_MJX_RL训练环境_手把手教程.md)

## 本机环境（已搭建）

| 项 | 路径 / 版本 |
| --- | --- |
| conda | `/root/autodl-tmp/conda-envs/g1_train` |
| Python | 3.11 |
| JAX | 0.10.x + `cuda12`（`jax.default_backend()==gpu`） |
| MuJoCo / MJX | 3.11.x / `mujoco-mjx` |
| Warp | 1.14（可选 MJX-Warp 后端） |
| 训练栈 | `brax` / `flax` / `optax` + **`playground`** |
| GPU | RTX 5090 D（本机已验证） |
| 入口任务 | `G1JoystickFlatTerrain` |

**不要**用 `mjviser`（CPU torch）环境训练。

## 激活

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/g1_train
source /root/autodl-tmp/g1_rl_train/scripts/env.sh
```

`env.sh` 会设置：`TMPDIR`、`JAX_DEFAULT_MATMUL_PRECISION`，以及指向 pip 自带 NVIDIA 库的 `LD_LIBRARY_PATH`（避免 `/usr/local/cuda` 干扰 JAX）。

## 验收

```bash
bash /root/autodl-tmp/g1_rl_train/scripts/verify_env.sh
```

期望输出含：`ALL CHECKS PASSED`。

冒烟（Cartpole，确认 PPO 管线）：

```bash
bash /root/autodl-tmp/g1_rl_train/scripts/smoke_train.sh
# EXIT=0，日志含 Done training.
```

## 训练 G1

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/g1_train
source /root/autodl-tmp/g1_rl_train/scripts/env.sh
cd /root/autodl-tmp/g1_rl_train

# 短跑试通
python scripts/train_g1.py --env_name G1JoystickFlatTerrain --num_timesteps 200000 --num_videos 0

# 正式训练（时长与显存占用显著增加）
python scripts/train_g1.py --env_name G1JoystickFlatTerrain --num_timesteps 50000000 --num_videos 0
```

`train_g1.py` 会先打上 JAX 0.10 与 Brax 的兼容补丁，再调用 Playground 的 PPO。  
完整 flags：`python scripts/train_g1.py --helpfull`  

日志与 checkpoint 默认写在 `g1_rl_train/logs/<Env>-<timestamp>/`。

## 与部署项目的关系

| 项目 | 作用 |
| --- | --- |
| `g1_rl_train`（本目录） | **训练**（MJX / Playground） |
| `mujoco_humanoid` | **部署推理**（`walk_g1.py` + `motion.pt` + mjviser@6008） |

训完后需导出策略并做观测/动作对齐，才能替换 `mujoco_humanoid/third_party/unitree_rl/motion.pt`。  
说明见：`mujoco_humanoid/docs/02_训练自有策略_环境与步骤.md`。

### 模型注意

- Playground `G1JoystickFlatTerrain`：自带 MJX 友好资产（首次会拉 menagerie）。  
- 本地 `mujoco_humanoid/models/unitree_g1/scene_mjx.xml`：可用于 MJX-JAX 步进验收。  
- `unitree_g1_12dof/scene.xml`（部署用）：含 MJX-JAX 尚未实现的 cylinder–mesh 碰撞，**不要**直接 `mjx.put_model`；部署推理仍用 CPU MuJoCo + PD。

## 磁盘提醒

训练环境 + menagerie 约占用数 GB～十余 GB。数据盘空闲建议长期保持 **≥20GB**；pip 临时目录请用：

```bash
export TMPDIR=/root/autodl-tmp/tmp
```

## 依赖冻结

见 `requirements.txt`（`pip freeze` 快照）。重装时可：

```bash
conda activate /root/autodl-tmp/conda-envs/g1_train
export TMPDIR=/root/autodl-tmp/tmp
pip install -r requirements.txt
```
