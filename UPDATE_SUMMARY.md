# SO101-umi 本次更新总结

## 更新目标

使用实体 SO-101 主臂遥操作 MuJoCo 中的 OpenArm 右臂，保持左臂固定，并统一两台机械臂的 TCP 坐标语义。系统支持运行时重新记录 home 位姿，避免暂停、恢复时发生跳变。

## 主要更新

### OpenArm 模型与 IK

- 使用 OpenArm 双臂 MuJoCo 模型及官方 OBJ 视觉网格。
- 仅对右臂 7 个关节进行 Mink IK 解算，左臂保持启动姿态。
- 以 `openarm_right_link7` 为跟踪对象，在代码中添加：
  - 沿 link7 `+Z` 方向的 `234.3 mm` TCP 偏移；
  - `Ry(-90°)` TCP 旋转变换。
- 坐标变换均在代码中完成，不修改原始 URDF/MJCF。
- 增加关节限位、阻尼、姿态任务、步长限制和数值异常保护。

### 世界坐标相对运动映射

SO-101 按下 `p` 时记录末端 home 位姿，之后计算相对于该 home 的空间运动，并叠加到 OpenArm 当前 TCP 世界位姿：

```text
Δp = p_so_now - p_so_home
ΔR = R_so_now · R_so_home⁻¹

p_open_cmd = p_open_home + scale · Δp
R_open_cmd = ΔR · R_open_home
```

平移和旋转均表达在世界坐标中，不再依赖 OpenArm 记录 home 时的局部 TCP 朝向，因此重新 home 后不会出现 X/Z 轴随末端姿态交换的问题。

统一坐标语义为：

- `+X`：前方；
- `+Y`：左方；
- `+Z`：上方。

### 遥操作状态机

- `p`：记录 SO-101 和 OpenArm 当前 TCP 位姿并开始遥操作。
- `q`：暂停遥操作，OpenArm 保持当前位置。
- 再次按 `p`：以双方当前位置重新记录 home 并平滑恢复。
- `Ctrl+C`：退出。
- MuJoCo 仿真窗口获得焦点后可直接使用 `p/q`，无需切回终端。

### SO-101 硬件兼容

- 通过 Feetech 总线只读获取 SO-101 关节位置。
- 若仅缺少夹爪电机 ID 6，系统继续读取机身关节并将夹爪保持在默认开度。
- 其他机身电机缺失时仍终止运行，避免使用不完整关节数据。
- 支持 LeRobot 标定文件；未提供时使用电机中点作为零位。

### 控制稳定性

- 关节数据使用 EMA 低通滤波。
- 对末端平移和旋转设置死区及单帧最大步长。
- 旋转矩阵投影回 SO(3)，避免长期运行产生数值漂移。
- IK 失败或出现非有限值时恢复到安全状态。

## 涉及文件

- `config/default.yaml`：默认硬件、TCP、IK及滤波参数。
- `src/so101_umi/config.py`：配置数据结构。
- `src/so101_umi/frames.py`：TCP坐标对齐预设。
- `src/so101_umi/se3.py`：世界坐标相对位姿计算与叠加。
- `src/so101_umi/openarm_ik.py`：OpenArm模型、虚拟TCP和Mink IK。
- `src/so101_umi/leader.py`：SO-101电机读取及缺少夹爪时的降级处理。
- `src/so101_umi/teleop.py`：遥操作主循环和键盘状态机。
- `tests/test_se3_fk.py`：坐标映射、FK和稳定性测试。
- `README.md`：安装、运行及操作说明。

## 运行命令

```bash
cd /home/vector/work/SO101-umi
export PYTHONPATH=src
conda activate lerobot

python -m so101_umi.teleop \
  --leader-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107971-if00 \
  --scale 2.0 \
  --fps 50
```

## 当前结果

- OpenArm 右臂能够跟随 SO-101 的相对 6D 末端运动。
- 左臂保持固定。
- 世界坐标方向已统一。
- 暂停和重新记录 home 时不会发生位姿跳变。
- 调试期间加入的临时日志代码已删除。
