# 自有策略脚本（对应原 `scripts`）

| 脚本 | 作用 |
| --- | --- |
| `env.sh` | 激活后设置 `TMPDIR` / CUDA `LD_LIBRARY_PATH` / 三个目录环境变量 |
| `verify_my_env.sh` | GPU + MJX + Playground G1 验收 |
| `train_my_g1.py` | PPO 训练入口（先加载 `my_rl_train.my_rl` 补丁） |
| `run_train.sh` | 后台启动训练，日志写入 `my_party/logs/` |

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/g1_train
source /root/autodl-tmp/mujoco_humanoid/my_scripts/env.sh
bash /root/autodl-tmp/mujoco_humanoid/my_scripts/verify_my_env.sh
bash /root/autodl-tmp/mujoco_humanoid/my_scripts/run_train.sh
```
