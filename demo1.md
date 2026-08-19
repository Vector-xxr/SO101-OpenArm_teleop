# Demo 1：SO-101 主臂控制 MuJoCo OpenArm 从臂

## 目标

使用 USB 连接的 SO-101 主臂控制 MuJoCo 中的 OpenArm：

1. 读取 SO-101 的 5 个手臂关节和夹爪位置。
2. 通过正运动学计算 SO-101 夹爪的 6D 位姿。
3. 以启动时的主臂和从臂位姿为参考，计算相对运动。
4. 将相对运动叠加到 OpenArm 末端目标位姿。
5. 使用 `mink` 解算 OpenArm 关节角并在 MuJoCo 中显示。
6. SO-101 夹爪开合量单独映射到 OpenArm 夹爪。

统一的末端坐标轴语义为：

- `+X`：夹爪向前
- `+Y`：夹爪向左
- `+Z`：夹爪向上

## 控制链路

```text
SO101 Present_Position（只读）
  → 关节角低通滤波
  → SO-101 URDF 正运动学
  → TCP 坐标系对齐 R_align = Ry(-90°)
  → 相对位姿 T_rel = T_now ⊖ T_ref
  → T_cmd = T_openarm_home ⊕ scale × T_rel
  → 位姿死区及单步限幅
  → mink FrameTask IK
  → OpenArm MuJoCo qpos
```

程序启动时会锁存：

- 当前 SO-101 TCP：`T_so101_ref`
- 当前 OpenArm TCP：`T_openarm_home`

因此 OpenArm 不会跳到 SO-101 的绝对位置，只跟踪主臂启动后的相对运动。

## 本机硬件检查

环境检查命令：

```bash
conda activate lerobot

python -c "import lerobot, mujoco, mink; \
print('lerobot', lerobot.__file__); \
print('mujoco', mujoco.__version__); \
print('mink ok')"

ls -l /dev/ttyACM* /dev/serial/by-id/
groups | grep -q dialout \
  && echo 'dialout: yes' \
  || echo 'dialout: NO'
```

检查结果：

- LeRobot 可导入。
- MuJoCo 版本为 `3.8.1`。
- `mink` 可导入。
- 当前用户属于 `dialout`。
- 两个 QinHeng/CH343 串口均可读取 STS3215 电机 ID 1–6。

串口对应关系：

| 设备 | by-id | 节点 | 用途 |
|---|---|---|---|
| `5C82107971` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107971-if00` | `/dev/ttyACM1` | SO-101 主臂 |
| `5C82107039` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107039-if00` | `/dev/ttyACM0` | 静止的另一台臂/从臂 |

程序只调用 `sync_read("Present_Position")`，不会向 SO-101 写入
`Goal_Position`。

## 正式运行

```bash
cd /home/vector/work/SO101-umi
conda activate lerobot
export PYTHONPATH=src

python -m so101_umi.teleop \
  --leader-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107971-if00 \
  --scale 2.0 \
  --fps 50
```

看到以下日志后再缓慢移动主臂：

```text
[leader] connected (read-only)
[latch] stored T_so101(0) and OpenArm home TCP
```

正常情况下：

- `qdeg=` 随主臂关节运动变化。
- `so101=` 随 SO-101 TCP 变化。
- `ee=` 随 OpenArm 目标 TCP 变化。
- `pos_err=` 应为有限数值，不能是 `nan`。

停止使用 `Ctrl+C`。

## 无硬件仿真测试

静止的虚拟主臂：

```bash
cd /home/vector/work/SO101-umi
conda activate openarm_teleop1
export PYTHONPATH=src

python -m so101_umi.teleop \
  --sim-leader \
  --sim-static \
  --scale 2.0
```

正弦运动的虚拟主臂：

```bash
python -m so101_umi.teleop \
  --sim-leader \
  --scale 2.0
```

注意：加上 `--sim-leader` 后程序不会读取 USB，实体 SO-101 的运动不会生效。

## 调试过程与修复

### 1. OpenArm 自动来回运动

现象：

- 实体 SO-101 没动，但 OpenArm 一直运动。

原因：

- 使用了 `--sim-leader`，程序内部的正弦假数据正在驱动 OpenArm。

处理：

- 实体主臂控制时删除 `--sim-leader`。
- 只想让虚拟主臂静止时使用 `--sim-leader --sim-static`。

### 2. 实体主臂运动但 OpenArm 不动

现象：

- `ee=` 和 `gripper=` 长时间不变。

原因：

- 最初连接的是 `/dev/ttyACM0`，该总线上的机械臂处于静止状态。

处理：

- 切换到 `/dev/ttyACM1`，即序列号 `5C82107971`。
- 日志增加了 `qdeg` 和 `dq`，便于确认关节是否真的在变化。

### 3. 高频抖动

原因：

- STS3215 编码器的微小读数噪声经过 `scale=2.0` 放大。
- 7 自由度 OpenArm 的 IK 持续追踪微小姿态变化。

处理：

- 主臂关节加入一阶 EMA 低通滤波，默认截止频率 `8 Hz`。
- TCP 加入 `3 mm / 1.7°` 死区。
- OpenArm 姿态权重从 `1.0` 降到 `0.15`。
- 增加 IK 阻尼和姿态偏好。
- 平移单步上限降到 `8 mm`，旋转单步上限降到 `0.06 rad`。

当前默认值：

```yaml
scale: 2.0
fps: 50
joint_lpf_hz: 8.0
deadband_m: 0.003
deadband_rad: 0.03
position_cost: 50.0
orientation_cost: 0.15
damping_cost: 0.2
ik_damping: 0.08
max_ee_step_m: 0.008
max_ee_step_rad: 0.06
```

如仍有姿态抖动，可临时只跟踪位置：

```bash
python -m so101_umi.teleop \
  --leader-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107971-if00 \
  --scale 2.0 \
  --fps 50 \
  --ori-cost 0
```

### 4. MuJoCo 中只剩底座

日志：

```text
WARNING: Nan, Inf or huge value in CTRL
```

原因：

- OpenArm 模型使用力矩执行器，却曾将 IK 关节角直接写入 `ctrl`。
- 数值异常后连杆被仿真动力学甩飞，只剩固定底座。

处理：

- 当前演示改为纯运动学显示：IK 直接更新 `qpos`。
- `ctrl` 保持为零，重力关闭，关节速度清零。
- IK 输入、速度、`qpos` 和输出均加入 NaN/Inf 检查及回退。

### 5. 修复 NaN 后 OpenArm 又冻结在 home

现象：

- SO-101 的 `qdeg` 和 `so101` 在变化，但 `ee` 一直等于 home。
- `clip_step` 出现矩阵乘法溢出。

原因：

- 旋转单步限幅的输出每帧反馈给下一帧。
- 浮点误差使旋转矩阵逐渐失去正交性，反复相乘后误差指数增长。
- 安全保护检测到异常后持续保留上一帧，因此目标被冻结。

处理：

- 新增 `project_rotation()`。
- 每次构造和限幅 6D 位姿时，使用 SVD 将旋转矩阵投影回合法的
  `SO(3)`，保证：

```text
RᵀR = I
det(R) = 1
```

- 增加 5000 次旋转限幅反馈回归测试。
- 完成 500 步虚拟主臂闭环测试，无 NaN，`ee` 可持续运动。

## 验证

运行测试：

```bash
cd /home/vector/work/SO101-umi
export PYTHONPATH=src

python tests/test_se3_fk.py
```

长时间无窗口冒烟测试：

```bash
MUJOCO_GL=egl python -m so101_umi.teleop \
  --sim-leader \
  --no-viewer \
  --steps 500 \
  --fps 50 \
  --scale 2.0
```

验证结果：

- SE(3)、FK、低通、死区和旋转漂移测试通过。
- 500 步 MuJoCo/mink 闭环测试通过。
- `pos_err` 保持有限，未再出现 NaN。

## 当前限制

- 尚未提供正式的 LeRobot SO-101 标定 JSON；当前使用
  `2048 ticks ≈ 0°` 的近似换算。正式使用应通过 `--calibration` 指定标定文件。
- 当前 OpenArm 是纯运动学 MuJoCo 展示，不模拟真实接触和动力学。
- 夹爪使用 `0–100%` 单独映射，不参与手臂 6D IK。
- `scale=2.0` 适合当前演示，但仍应根据任务工作空间调整。
