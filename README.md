# SO101-umi

SO-101 实体 leader 的夹爪 **6D 位姿** → 相对运动（使能时 latch）→ 叠到 OpenArm 仿真 follower TCP → **mink** FrameTask IK → MuJoCo。

```
SO101Leader.get_action()                 # 5 个本体关节 deg + gripper 0–100
  → FK (so101_new_calib.urdf)            # T_so101_native  4×4
  → T_aligned = T_native @ R_align       # +X 前 / +Y 左 / +Z 上
  → T_rel = T(t) ⊖ T(0)                  # p 相减, R(t)@R(0).T
  → T_cmd = T_openarm_home ⊕ (S * T_rel) # 叠到 OpenArm 当前 home TCP
  → mink FrameTask on site ee_aligned
  → MuJoCo qpos / viewer
gripper.pos (0–100) 单独映射到 OpenArm 手指开合，不要和本体关节一起 deg2rad。
```

## 本机 USB 硬件（2026-08-19 实测）

`lsusb` 上有 **两路** QinHeng CH343 / CDC ACM（Feetech / Waveshare 总线舵机转接，SO-101 标配）：

| USB ID | 序列号 | 内核节点 | by-id（推荐） | 权限 |
|---|---|---|---|---|
| `1a86:55d3` | `5C82107039` | `/dev/ttyACM0` `crw-rw---- root:dialout` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107039-if00` | `0660` |
| `1a86:55d3` | `5C82107971` | `/dev/ttyACM1`（另有 `/dev/ttyUSB51` → ACM1）`crwxrwxrwx` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107971-if00` | `0777` |

- 用户 `vector` 在 **dialout** 组；agent 沙箱里 `ls /dev/ttyACM*` 会失败，真实系统上节点存在。
- **两路 handshake 均成功**（电机 ID 1–6 STS3215 都在）。只做了 `Present_Position` 只读，**没有写 Goal_Position**。
- 无 LeRobot calibration JSON（`~/.cache/huggingface/lerobot/calibration` 为空）。原始 ticks（mid=2048 ≈ 0°）：

  - ACM0 / `5C82107039`: pan 1525, lift 2285, elbow 1381, wrist_flex 2820, wrist_roll 2057, gripper 1599
  - ACM1 / `5C82107971`: pan 1666, lift 2352, elbow 1384, wrist_flex 2839, wrist_roll 2061, gripper 2481

- **默认 leader** 写成 ACM0 的 by-id（hub `1-2.1`，先枚举到的那路）。另一路很像 follower，本仓库不要对它写目标位置。若接反，用 `--leader-port` 换。
- 没有真实标定文件时，程序用 **ticks 中位 2048 = 0°** 近似；请把 LeRobot 的 `calibration/teleoperators/so101_leader/<id>.json` 传给 `--calibration`。

## 怎么跑

依赖：`numpy`、`mujoco>=3.8.1`、`mink`、`qpsolvers[daqp]`。实体臂还要本机的 LeRobot（`conda activate lerobot`）。

本机已有环境：

- 仿真 / mink：`conda activate openarm_teleop1`（已装 mujoco + mink）
- 读 leader：`conda activate lerobot`（已装 lerobot + scservo；可 `pip install mink`）

```bash
cd /home/vector/work/SO101-umi
export PYTHONPATH=src

# 无 USB / 无显示器：仿真 leader + 有限步（CI / 干跑）
conda run -n openarm_teleop1 python -m so101_umi.teleop \
  --sim-leader --no-viewer --steps 80 --fps 50

# 带 MuJoCo 窗口
python -m so101_umi.teleop --sim-leader --scale 2.0

# 实体 SO-101 leader（只读位置）
conda run -n lerobot python -m so101_umi.teleop \
  --leader-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107039-if00 \
  --scale 2.0 --fps 50
```

常用参数：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--scale` / `-S` | `2.0` | SO-101 工作空间约 30 cm，OpenArm 更大，2–3 较合适 |
| `--fps` | `50` | 控制周期，busy-wait |
| `--ori-cost` | `1.0` | mink 姿态权重（位置默认 50）。SO-101 腕部自由度少，姿态不要太大 |
| `--pos-cost` | `50` | 位置权重 |
| `--max-ee-step` | `0.02` m | 每 tick EE 平移限幅 |
| `--wrist-roll-offset-deg` | `0` | 加在 `wrist_roll` 上再 FK |
| `--so101-align` | `ry-90` | 见下方坐标系 |
| `--calibration` | 无 | LeRobot JSON |
| `--sim-leader` | off | 正弦 dummy 关节 |
| `--no-viewer` | off | 无窗口 |

## 坐标系（必须对齐）

约定（leader FK 对齐后，以及 OpenArm `ee_aligned`）：

```
+X  forward  出夹爪 / 朝物体
+Y  left     左
+Z  up       上
```

### Native SO-101 TCP（`gripper_frame_link`）

URDF：`assets/so101/so101_new_calib.urdf`（TheRobotStudio `so101_new_calib.urdf`）。

`gripper_frame_joint`：parent `gripper_link`，`xyz=(-0.0079, -0.0002, -0.0981)`，`rpy=(0, π, 0)`。  
Ry(π) 把父系 **-Z** 映成 TCP **+Z**，原点在指尖附近 → **native +Z = 朝物体（forward）**。  
native +Y = `gripper_link` +Y（张合方向 / 左）。native +X = 右手系剩下的轴。

### Native OpenArm TCP（`ee_native`）

模型：OpenArm **v1 单臂** `assets/openarm/openarm.xml`（不是 bimanual）。  
TCP 在 `openarm_link7` 上，指尖中心 `(0, 0.0015, 0.1151)`，姿态 = link7。  
手指沿 ±Y 滑动、沿 +Z 伸出 → **native +Z = forward，native +Y = left**。

### R_align

两边 native 都是 Z-forward / Y-left。要变成 X-forward / Y-left / Z-up：

```
R_align = Ry(-90°) =

  [[ 0,  0, -1],
   [ 0,  1,  0],
   [ 1,  0,  0]]

T_aligned = T_native @ R_align
```

- SO-101：代码里 `--so101-align ry-90`（可改 `identity` / `rz90` …）
- OpenArm：MJCF 里 `ee_aligned` **已经乘了同一个 quat** `0.707 0 -0.707 0`。mink FrameTask 跟踪的是 **对齐后的** site，follower 侧不再二次乘 R_align。

相对运动（与 umi-vista `replay.py` 相同）在对齐后的 TCP 上计算，再叠到 OpenArm home：

```
p_rel = p(t) - p(0)
R_rel = R(t) @ R(0).T
p_cmd = p_home + S * p_rel
R_cmd = R_rel @ R_home
```

启动 / enable 时同时记下 `T_so101(0)` 和当前 OpenArm TCP，follower 不会跳。

## 安全

- Leader：**只读** `Present_Position`，不写 `Goal_Position`，`disconnect(disable_torque=False)`。
- IK：mink `ConfigurationLimit` + 每 tick EE 步长裁剪。
- 接错 USB 时用 by-id；不要把本程序的 port 指到正在闭环的 follower。

## 布局

```
SO101-umi/
  config/default.yaml      # 默认 by-id 端口、scale、权重
  assets/so101/            # URDF（无 STL，FK 不需要）
  assets/openarm/          # v1 单臂 MJCF + visual meshes
  src/so101_umi/
    leader.py              # USB 只读 / --sim-leader
    so101_fk.py            # numpy URDF FK（不用 Placo IK）
    se3.py                 # ⊖ ⊕ scale clip
    frames.py              # R_align
    openarm_ik.py          # mink FrameTask
    teleop.py              # 30–50 Hz busy-wait 循环
```

## 标定笔记

1. 用 LeRobot 标定 leader，保存 JSON，`--calibration` 指向它。  
2. 若相对运动方向反了：试 `--so101-align ry90` 或 `rz90`。  
3. 腕滚零位不对：`--wrist-roll-offset-deg`。  
4. 工作空间过大/过小：`--scale 1.5`～`3`。  
5. OpenArm 跟不住姿态：把 `--ori-cost` 降到 `0.2`，先保证位置。
