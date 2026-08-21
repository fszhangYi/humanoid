# 自有策略训练说明（my_* 三目录）

对应关系：

| 原目录 | 自有目录 | 内容 |
| --- | --- | --- |
| `g1_rl_train` | `my_rl_train` | 训练包 `my_rl/`、训练配置 |
| `scripts` | `my_scripts` | 激活 / 验收 / 开训入口 |
| `third_party` | `my_party` | 部署 yaml、policies、logs、checkpoints |

## 当前训练任务

- 环境：`G1JoystickFlatTerrain`（MuJoCo Playground + MJX）
- 步数：`my_rl_train/configs/train_g1.yaml` → 默认 **2_000_000**
- 启动：`bash my_scripts/run_train.sh`
- 进程：`my_party/logs/train.pid`
- 实验目录：`my_party/logs/G1JoystickFlatTerrain-<时间戳>/`

查看：

```bash
tail -f my_party/logs/train_*.log   # 若用 PYTHONUNBUFFERED 启动
cat my_party/logs/train.pid
nvidia-smi
```

停止：

```bash
kill $(cat my_party/logs/train.pid)
```
