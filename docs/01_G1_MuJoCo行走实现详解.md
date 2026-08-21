# 文档一：当前如何让宇树 G1 在 MuJoCo 中走路（详解）

> 项目路径：`/root/autodl-tmp/mujoco_humanoid`  
> 性质：**推理 / 部署**，不是训练。使用宇树公开预训练策略在 MuJoCo 里驱动 G1 行走，并用 mjviser 在网页端口 **6008** 可视化。  
> 实测（本机）：无界面 8 秒前进约 **3.7 m**，未摔倒。

---

## 1. 一句话原理

```
速度命令 cmd
    ↓
每 0.02s 拼观测 obs(47维) → TorchScript 策略 motion.pt → 动作 action(12维)
    ↓
目标关节角 target = default_angles + action × 0.25
    ↓
每个物理步(0.002s) 用 PD 算力矩 τ → 写入 MuJoCo data.ctrl → mj_step
    ↓
（可选）mjviser 把仿真状态推到浏览器
```

也就是说：**策略不直接输出力矩**，只输出「相对默认站姿的关节位置残差」；底层用经典 PD 把位置误差变成电机力矩，再交给 MuJoCo 的 `<motor>` 执行器。

这与宇树仓库 [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) 中 `deploy/deploy_mujoco/deploy_mujoco.py` 的部署回路一致；本项目把它封装进 `scripts/walk_g1.py`，并把桌面 `mujoco.viewer` 换成网页 `mjviser`。

---

## 2. 项目里和「走路」相关的文件

```
mujoco_humanoid/
├── models/unitree_g1_12dof/     # 走路用的机器人 + 场景
│   ├── scene.xml                # 地面 + 灯光，include g1_12dof.xml
│   ├── g1_12dof.xml             # 12 自由度腿部 + <motor> 力矩执行器
│   └── meshes/                  # STL 网格
├── third_party/unitree_rl/
│   ├── motion.pt                # 预训练 TorchScript 策略（约 145KB）
│   ├── g1.yaml                  # PD 增益、默认角、观测/动作尺度等
│   └── deploy_mujoco.py         # 上游官方部署脚本（参考用）
├── scripts/
│   ├── walk_g1.py               # 本项目行走控制 + 可选网页查看
│   └── run_walk.sh              # 后台在 6008 启动行走
└── logs/walk_g1_6008.log
```

说明：

| 模型 | 用途 |
| --- | --- |
| `models/unitree_g1/`（Menagerie，29-DoF，`<position>` 执行器） | 只适合「加载查看」，**不能**直接套本策略 |
| `models/unitree_g1_12dof/`（unitree_rl_gym 的 12-DoF，`<motor>`） | **走路必须用这个**，与 `motion.pt` 训练/部署设定一致 |

---

## 3. 运行环境（推理侧）

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/mjviser
cd /root/autodl-tmp/mujoco_humanoid
```

该环境已具备：`mujoco`、`torch`（本机部署可用 CPU 版）、`yaml`、`mjviser` / `viser`。

**注意：** 推理可以跑在 CPU 上（策略很小）；**训练**需要另一套 CUDA 环境，见《文档二》。

---

## 4. 怎么启动

### 4.1 无界面验收（推荐先跑）

```bash
python scripts/walk_g1.py --no-view --seconds 8 --vx 0.5
```

期望类似：

```text
headless 8.0s: dx=+3.657 m, dy=-0.219, z=0.772, z_min=0.762, fell=False
```

### 4.2 网页可视化（端口 6008）

```bash
bash scripts/run_walk.sh
# 等价于：
# python scripts/walk_g1.py --port 6008 --vx 0.5
```

浏览器打开 AutoDL 映射的 **6008**。  
速度命令：`--vx` 前进(m/s)、`--vy` 侧移、`--yaw` 偏航角速度。

若端口占用：

```bash
ss -tlnp | grep 6008
# kill <pid> 后再启动
```

---

## 5. 机器人模型细节（为何是 12-DoF）

### 5.1 自由度与状态维度

`g1_12dof.xml` + `scene.xml` 加载后（本机实测）：

| 量 | 值 | 含义 |
| --- | --- | --- |
| `nq` | 19 | 广义坐标：自由根 7（xyz+四元数）+ 12 关节角 |
| `nv` | 18 | 广义速度：自由根 6 + 12 关节速度 |
| `nu` | 12 | 执行器数 = 左右腿各 6 个关节 |

腿关节顺序（与策略 12 维动作一致）：

1. left_hip_pitch / roll / yaw  
2. left_knee  
3. left_ankle_pitch / roll  
4. right_hip_pitch / roll / yaw  
5. right_knee  
6. right_ankle_pitch / roll  

上身在 12-DoF 模型里基本是固定/简化几何，**策略不控手臂与腰**，只控双腿。

### 5.2 执行器类型：`<motor>`（力矩）

与 Menagerie 29-DoF 的 `<position>`（ctrl=目标角）不同，12-DoF 部署模型是：

```xml
<actuator>
  <motor name="left_hip_pitch_joint" joint="left_hip_pitch_joint"/>
  ...
</actuator>
```

因此 `data.ctrl[i]` 的含义是 **力矩 τ**，不是目标角度。  
策略输出位置目标后，必须经 PD 转成力矩再写入 `ctrl`。

### 5.3 状态在 `qpos` / `qvel` 中的切片

与 `walk_g1.py` 一致：

| 数据 | 切片 | 内容 |
| --- | --- | --- |
| 根位置 | `qpos[0:3]` | x, y, z |
| 根姿态 | `qpos[3:7]` | 四元数 w,x,y,z |
| 关节角 | `qpos[7:19]` | 12 个关节 |
| 根线速度 | `qvel[0:3]` | |
| 根角速度 | `qvel[3:6]` | |
| 关节速度 | `qvel[6:18]` | 12 个 |

---

## 6. 配置文件 `g1.yaml`（控制律超参）

路径：`third_party/unitree_rl/g1.yaml`（来自 unitree_rl_gym）。

| 字段 | 本机值 | 作用 |
| --- | --- | --- |
| `simulation_dt` | 0.002 | 物理步长 → 500 Hz |
| `control_decimation` | 10 | 每 10 个物理步推理一次策略 → **50 Hz** |
| `kps` / `kds` | 各 12 维 | 关节 PD 刚度/阻尼（膝更大：kp=150, kd=4） |
| `default_angles` | 12 维 | 默认蹲站姿态（策略残差叠加在此之上） |
| `action_scale` | 0.25 | `target = default + action * 0.25` |
| `ang_vel_scale` | 0.25 | 观测里角速度缩放 |
| `dof_pos_scale` | 1.0 | 关节位置（相对默认）缩放 |
| `dof_vel_scale` | 0.05 | 关节速度缩放 |
| `cmd_scale` | [2, 2, 0.25] | 对 `[vx, vy, yaw_rate]` 的缩放 |
| `num_actions` | 12 | 策略输出维 |
| `num_obs` | 47 | 策略输入维 |
| `cmd_init` | [0.5, 0, 0] | 默认命令（可被 CLI 覆盖） |

默认角（左右腿对称蹲姿）：

```text
[-0.1, 0, 0, 0.3, -0.2, 0,   -0.1, 0, 0, 0.3, -0.2, 0]
  hipP  R  Y  knee ankP ankR    （右腿同上）
```

---

## 7. 控制程序 `walk_g1.py` 逐步拆解

### 7.1 启动流程（`main`）

1. 读 `g1.yaml`、加载 `scene.xml` → `MjModel` / `MjData`  
2. 构造 `G1RLWalker`（加载 `motion.pt`，设置 `cmd=[vx,vy,yaw]`）  
3. 先跑一段 **headless** 自检并打印 `dx / z_min / fell`  
4. 若未加 `--no-view`，再进入 mjviser 实时循环  

### 7.2 复位（`G1RLWalker.reset`）

- `mj_resetData`  
- 根高度 `z=0.78`，姿态单位四元数  
- 关节角设为 `default_angles`，速度清零  
- `action`、策略计数器清零  
- `mj_forward` 更新派生量  

### 7.3 每个物理步（`apply`）——双频率结构

```text
频率 A：500 Hz（每个 mj_step 前）
  τ = kp*(target_q - q) + kd*(0 - dq)
  data.ctrl[:] = τ

频率 B：50 Hz（counter % 10 == 0）
  拼 obs → policy(obs) → action → 更新 target_q
```

PD 公式（与上游一致）：

\[
\tau = k_p (q_{\mathrm{target}} - q) + k_d (0 - \dot q)
\]

期望关节速度恒为 0（在两次策略更新之间把腿「钉」在目标角附近）。

### 7.4 观测向量 `obs`（47 维，必须与训练时一致）

| 下标 | 长度 | 内容 |
| --- | ---: | --- |
| 0:3 | 3 | 根角速度 × `ang_vel_scale` |
| 3:6 | 3 | 重力在机体坐标下的方向（由四元数算出） |
| 6:9 | 3 | `cmd * cmd_scale`（速度指令） |
| 9:21 | 12 | `(q - default_angles) * dof_pos_scale` |
| 21:33 | 12 | `dq * dof_vel_scale` |
| 33:45 | 12 | **上一步** 的 `action`（动作历史） |
| 45:47 | 2 | 步态相位 `[sin(2πφ), cos(2πφ)]`，周期 `period=0.8s` |

相位：

\[
\phi = \frac{(counter \cdot dt) \bmod 0.8}{0.8}
\]

给策略一个「时钟」，便于学出周期性左右脚交替。

重力方向特征由 `get_gravity_orientation(quat)` 计算，公式与官方 `deploy_mujoco.py` 保持一致，**不要随意改**，否则与 `motion.pt` 分布不匹配会摔倒。

### 7.5 策略与动作映射

```text
action = policy(obs)           # TorchScript，shape (12,)
target_dof_pos = action * 0.25 + default_angles
```

- `motion.pt`：`torch.jit.load`，`eval()`，推理时 `torch.no_grad()`  
- 本机部署用 CPU 即可（策略很小，瓶颈在仿真与可视化）

### 7.6 可视化如何接上

`run_viewer`：

1. `viser.ViserServer(host="0.0.0.0", port=6008)`  
2. `mjviser.Viewer(model, data, step_fn=..., reset_fn=...)`  
3. `step_fn` 内：`walker.apply(d)` → `mujoco.mj_step(m, d)`  

即：**网页刷新的每一步仿真，都先走一遍完整控制律**，所以浏览器里看到的是「带策略的行走」，不是无控自由落体。

---

## 8. 端到端时序图

```text
t = 0, 0.002, 0.004, ...                t = 0.02, 0.04, ...
─────────────────────────────────       ─────────────────────────
读 q, dq                                 拼 obs(47)
算 τ = PD(target, q, dq)                 action = π(obs)
ctrl ← τ                                 target ← default + 0.25*action
mj_step
渲染（若开 mjviser）
```

---

## 9. 和「开环摆腿」的对比（为何必须用策略）

本项目早期曾尝试：

- 正弦开环髋/膝目标 + 简易倾角反馈  
- 准静态「移重心 → 抬脚 → 落脚」状态机  

在 **29-DoF position 执行器** 上均易摔倒：单脚支撑时开环无法稳定维持质心在支撑多边形内。

当前方案成功的原因：

1. 使用与训练匹配的 **12-DoF + motor + PD** 回路  
2. 使用在大量并行仿真里用 RL 学出的 **闭环策略**（含相位、历史动作、角速度等）  
3. 观测/尺度/默认角与 checkpoint **严格对齐**

---

## 10. 验收标准与常见问题

### 验收

- [ ] `python scripts/walk_g1.py --no-view --seconds 8 --vx 0.5` → `fell=False` 且 `dx` 明显为正（米级）  
- [ ] `bash scripts/run_walk.sh` 后 `ss -tlnp | grep 6008` 有监听，HTTP 可访问  
- [ ] 浏览器中机器人持续向前迈步，不立刻趴下  

### 常见问题

| 现象 | 可能原因 |
| --- | --- |
| 立刻摔倒 | 用了 29-DoF Menagerie 模型；或改了 obs 拼法/尺度 |
| `model.nu != 12` | XML 不是 `g1_12dof` |
| 端口起不来 | 6008 被占用 |
| 走得慢/歪 | 调 `--vx/--vy/--yaw`；或策略本身对侧向命令更敏感 |
| Torch 报错 | 环境缺 `torch`；或 `motion.pt` 路径不对 |

---

## 11. 上游出处与许可注意

- 策略与部署逻辑： [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)（`deploy/pre_train/g1/motion.pt`、`deploy/deploy_mujoco`）  
- 12-DoF 描述：同仓库 `resources/robots/g1_description`  
- 另：Menagerie 的 29-DoF G1 仅用于本项目的「加载展示」，与本行走策略无关  

使用前请自行阅读各仓库 LICENSE；仿真模型 ≠ 整机闭源控制 SDK。

---

## 12. 小结

当前「走路」实现是一条 **标准的 sim2sim 部署链路**：

**预训练 RL 策略（50 Hz）→ 关节目标 → PD 力矩（500 Hz）→ MuJoCo motor →（可选）mjviser@6008。**

它依赖三个对齐：`(模型 DoF/执行器类型)`、`(观测与动作定义)`、`(yaml 中的增益与尺度)`。任一不对齐，表现会从「能走」退化成「秒倒」。
