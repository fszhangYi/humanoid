# MuJoCo 人形模型加载示例

在 conda 环境 `/root/autodl-tmp/conda-envs/mjviser` 中加载人形 MJCF，并用 mjviser 在 **6008** 网页查看。

## 已内置模型

| 名字 | 路径 | 说明 |
| --- | --- | --- |
| **g1**（默认查看） | `models/unitree_g1/scene.xml` | 宇树 G1 29-DoF（Menagerie） |
| **g1 walk** | `models/unitree_g1_12dof/scene.xml` | 宇树 G1 12-DoF 腿部 + RL 行走 |
| **h1** | `models/unitree_h1/scene.xml` | 宇树 H1（Menagerie） |
| **humanoid** | `models/humanoid.xml` | DeepMind 简化人形 |

来源：[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)（公开仿真描述，不是整机闭源 SDK）。

Menagerie 里还有其它商业/准商业人形可再拉取，例如 Booster T1、Fourier N1、Apptronik Apollo、PAL TALOS 等。

## 环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/mjviser
cd /root/autodl-tmp/mujoco_humanoid
```

## 用法

```bash
# 加载并步进（默认 G1）
python scripts/load_humanoid.py
python scripts/load_humanoid.py --robot h1
python scripts/load_humanoid.py --robot humanoid

# 网页查看（端口必须 6008）
bash scripts/run_viewer.sh                 # G1
ROBOT=h1 bash scripts/run_viewer.sh        # H1
```

浏览器打开 AutoDL 映射的 `6008`。

## 让 G1 走起来（RL 策略）

使用宇树 [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) 预训练 `motion.pt`（12 自由度腿部 + 关节 PD 力矩）。

```bash
# 无界面自检（约 8s，期望前进数米且不倒）
python scripts/walk_g1.py --no-view --seconds 8 --vx 0.5

# 网页查看（端口 6008）
bash scripts/run_walk.sh
# 或
python scripts/walk_g1.py --port 6008 --vx 0.5
```

速度命令：`--vx` 前进 (m/s)，`--vy` 侧移，`--yaw` 转向。

## 文档

- [当前如何实现 G1 在 MuJoCo 中走路（详解）](docs/01_G1_MuJoCo行走实现详解.md)
- [训练自有策略：环境与步骤](docs/02_训练自有策略_环境与步骤.md)

若 6008 被占用：

```bash
ss -tlnp | grep 6008
# kill <pid>
bash scripts/run_walk.sh
```
