# 自有 RL 训练包（对应原 `g1_rl_train`）

本目录只放**训练用 Python 包与训练配置**。入口脚本在 `../my_scripts/`，权重/日志在 `../my_party/`。

```
my_rl_train/
  my_rl/           # JAX–Brax 兼容补丁等
  configs/         # train_g1.yaml
  README.md
```

激活与开训见 `../my_scripts/run_train.sh`。
