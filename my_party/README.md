# 自有策略产物目录（对应原 `third_party`）

| 子目录 | 用途 |
| --- | --- |
| `configs/g1_deploy.yaml` | 训完后部署对齐用（观测/PD/默认角） |
| `policies/` | 导出的自有策略权重（如 `my_policy.pt`） |
| `checkpoints/` | 训练过程 checkpoint（若框架写入此处） |
| `logs/` | 训练标准输出日志 + Playground 实验目录 |

训练时工作目录会切到本目录，便于产物集中存放。
