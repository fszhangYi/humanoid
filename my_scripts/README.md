# 自有策略脚本（对应原 `scripts`）

| 脚本 | 作用 |
| --- | --- |
| `env.sh` | 激活后设置 `TMPDIR` / CUDA `LD_LIBRARY_PATH` / 三个目录环境变量 |
| `verify_my_env.sh` | GPU + MJX + Playground G1 验收 |
| `train_my_g1.py` | PPO 训练入口（续训 / 早停钩子 + `my_rl` 补丁） |
| `run_train.sh` | 后台启动训练，日志写入 `my_party/logs/` |

```bash
# 全新长训（yaml 默认 early_stop=on）
bash my_scripts/run_train.sh

# 从最新 checkpoint 热启动，再训配置里的 num_timesteps
RESUME=auto bash my_scripts/run_train.sh

# 从最新 ckpt 续训到 target_timesteps（只补剩余步数）
RESUME=auto RESUME_MODE=remaining TARGET_TIMESTEPS=270000000 bash my_scripts/run_train.sh

# 关掉早停
EARLY_STOP=0 bash my_scripts/run_train.sh
```
