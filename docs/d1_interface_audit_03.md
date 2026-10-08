# D1 flat_lab 部署接口审计 03

这份审计描述当前同事提供的 `flat_lab.onnx` 与原 C++ 部署代码。它不是该权重训练过程的复现证明，也不是新自训接口的全部规范。2026-10-09 完成了 **22 个固定输入案例、19 次真实 C++ ONNX 推理，物理步数为 0**。

原生 harness 编译、链接未经修改的 `FSMState_RL.cpp`、`FSMState.cpp`、`onnx_inferrer.cpp`，实际调用 `enter()`、`update_observations()`、`update_forward()`、`run()`。它只抑制后台线程，由代理 inferrer 在调用真实 ONNX 后终止循环，使每次只执行一次推理。另一侧使用独立 NumPy 四元数向量旋转、观察构造和 Python ONNX Runtime。两侧数值检查均通过。

证据入口是 [report.json](../evidence/interface_audit_03/golden_run_01/report.json)，可直接读取的配置是 [schema.json](../evidence/interface_audit_03/golden_run_01/schema.json)。固定 IMU、关节、命令输入在 [fixtures.json](../evidence/interface_audit_03/golden_run_01/fixtures.json)，原 C++ 的每次完整输入、输出和低层命令在 [native.stdout](../evidence/interface_audit_03/golden_run_01/native.stdout)。C++ 使用 ORT **1.10.0**，独立 Python 侧使用 **1.23.2**，仅 CPU 推理。所读取生产源码前后 SHA 均一致。

权重 SHA256：`f8a721d4f7f05777db91a70608992ad2a86fc373982758937ba2fb3d9919d21b`。原部署输入名为 `nn_input0`，float32 `[1,57]`；历史输入名为 `nn_input1`，float32 `[1,10,57]`；输出名为 `nn_output`，float32 `[1,16]`。将历史展平时按 10 行连续排列，共 570 个数。

| 当前 57 维切片（左闭右开） | 内容及量纲 |
| --- | --- |
| `[0:3]` | IMU/body 角速度，`g_filtered = .97*g_current + .03*g_previous_filtered`，再乘 `.25` |
| `[3:6]` | 机体坐标下单位重力方向 `R_body_to_world(q).T @ [0,0,-1]` |
| `[6:9]` | `[vx,vy,wz]` 先乘 gain、加 compensation，再分别夹到 `[-1,1]`，再乘 `[2,2,.25]` |
| `[9:25]` | 所有关节先转 float32，四个轮关节位置设为 0，再减默认姿态；倍率 1 |
| `[25:41]` | 16 个关节速度，含轮速，乘 `.05` |
| `[41:57]` | 上一次原始策略输出，未经过 action scale；原程序没有 action clip |

观察里没有 base 线速度、绝对世界位置、地形高度图或手臂状态。`lin_vel_scale=2` 虽在配置中，但这里命令倍率实际来自 `RLParameters.commands_scale`。当前 `update_control_parameters()` 没有读取 YAML 的 `commands_scale/commands_gain`；当前使用结构体默认值 `[2,2,.25]` 和 `[1,1,1]`，compensation 为零。直接修改 YAML 的这两个字段不会自动改变当前 loader 的行为。

关节顺序完全由当前 `controllers.yaml` 的 `joints` 确定：

| 序号 | 关节 |
| --- | --- |
| 0,1,2,3 | `FL_hip_joint, FL_thigh_joint, FL_calf_joint, FL_foot_joint` |
| 4,5,6,7 | `FR_hip_joint, FR_thigh_joint, FR_calf_joint, FR_foot_joint` |
| 8,9,10,11 | `RL_hip_joint, RL_thigh_joint, RL_calf_joint, RL_foot_joint` |
| 12,13,14,15 | `RR_hip_joint, RR_thigh_joint, RR_calf_joint, RR_foot_joint` |

轮索引为 `[3,7,11,15]`，hip 索引为 `[0,4,8,12]`。当前 `reindex/re_sign` 均为空；没有隐式左右腿重新排序或取反。`flat_lab` 默认姿态为 `[0,.8,-1.5,0] × 4`。不要借用 `flat.onnx` 的 hip ±.1、后腿 thigh=1.0 配置。

MuJoCo `trunk_imu` site 位于 base 原点、无额外旋转。`gyro` 传感器值直接进入 low_state 的 gyro，保持 site/body 轴。MuJoCo `framequat` 的 wxyz 经 bridge 转成 ROS xyzw，控制器再重排成 wxyz；不是对四元数再做一次物理旋转。`orientation_tools.h` 先构造常规旋转矩阵，再转置；因此重力是世界向下转到 body。golden 案例实证：roll=+90° 时重力约 `[0,-1,0]`，pitch=+90° 时约 `[1,0,0]`；q 与 −q 结果一致。实际使用应提供归一化四元数；原函数不会替调用者归一化。

历史按**最旧到最新**排列。`enter()` 将 observations 的滤波 gyro、last_actions 等清零，调用一次 `update_observations()`，然后把这份完整观察复制十次。每次推理又调用一次观察更新，把当前观察和**更新前的十帧历史**交给 ONNX；随后才把当前观察 append 到历史尾部。历史不含本次推理的新动作。由于进入状态已有一次观察更新，非零初始 gyro 在第一次推理时已经经过第二次滤波更新。没有将十帧初始化成全零，也没有把当前观察重复追加两次。

`P_V` 的原始输出映射如下，其中 `a` 为未经夹紧的 ONNX 输出，q/dq 为最近一次 bridge read 的关节状态：

| 关节 | 动作目标 | PD 原始力矩 |
| --- | --- | --- |
| 12 个腿关节 | `q_target = q0 + .25*a`，`dq_target=0`，kp=60，kd=1.5 | `60*(q_target-q) - 1.5*dq` |
| 4 个轮关节 | `q_target=0`，`dq_target=5*a`，kp=0，kd=.5 | `.5*(5*a-dq)` |

FSM 本身只填写 position/velocity/kp/kd/effort 命令，feed-forward effort 为零。`mujoco_system.cpp:152–166` 将上述项相加，再取 URDF effort 限值与 command-interface effort 范围的交集。展开后的 URDF 腿为 **±80 Nm**、轮为 **±12 Nm**；command-interface 范围 ±100，因此最终仍是 ±80/±12。YAML 中轮的 `torque_limit=20` 没有在该 RL→MuJoCo 路径执行限幅。bridge 将结果写入 **`qfrc_applied`**，并非写 motor `ctrl`；MJCF 的 `actuatorfrcrange` 不是这条路径的软件夹紧机制。观察到的 joint-state effort 也来自 `qfrc_applied`，不是电机实测力矩。

上述**FSM 命令映射**已在真实 C++ 函数中通过常规和 ±100 人工动作核验；**bridge 的 PD 求和及 clamp**在此次任务里是源代码/URDF 静态核对和离线派生参考，没有冒充实际调用 ROS bridge。派生数值保存在 [bridge_torque_reference.json](../evidence/interface_audit_03/golden_run_01/bridge_torque_reference.json)。

原观察与动作都没有通用 clip。fixture 的大 gyro/关节速度输入实际进入 ONNX 超过 100，实际 ONNX 输出也可明显超出 `[-1,1]`。正常固定案例的两侧最大绝对差不超过约 `1.58e-5`；刻意远离物理状态的极端案例输出达到约 7906，最大绝对差约 `.00293`。统一判据是 `atol=3e-5, rtol=3e-5`，此极端值的相对差仍很小。通过的是固定输入下的接口一致性，不是这些极端状态可运行的物理资格。

原场景未覆盖 timestep，MuJoCo 3.3 源码 `engine_io.c:120` 的默认值为 **.002 s**。ROS manager 请求 400 Hz（.0025 s），bridge 在离散仿真时刻检查 `sim_period >= control_period`，因此原 dt 下实际每两个 native step 读状态/更新控制一次，即 **250 Hz**。bridge write 每 native step 调用，即 500 Hz，但 PD 使用的是最近一次 read 缓存的 q/dq。20 ms 推理来自 `RLParameters` 默认值和 `gettimeofday/absoluteWait`，是**独立墙钟线程**的名义 50 Hz，不保证每十个物理步准确推理一次；暂停或慢仿真也不能直接等同于冻结观察历史。此处未测试线程调度、ROS QoS 或运行时数值竞态。

复位语义必须区分。冷构造将 `action_vec_` 置零；原 `enter()` 清 observations 并重建历史，**没有清 `action_vec_`**，热重入到首次新推理前 `run()` 可沿用旧动作。本次实际 C++ warm-enter fixture 证实历史的 last_actions 已为零，而用于 run 的旧动作仍为 ±100。`exit()` 只是暂停更新。GUI reset 在 `simulate.cc:1954` 调用 `mj_resetData/mj_forward`，没有重建 FSM、历史、动作或 bridge 的上次控制时刻。源代码上因此还存在时间回退后等待控制时刻追平的可能；本次没有运行 GUI reset 来验证该推断。GUI 复位不等于物理自救。

另外，当前 `RemoteControlData::zero()` 对 twist/pose 三维向量写的是 `.Zero()` 表达式而非赋值；离线 harness 显式填写全部输入，避免依赖未初始化值。原多线程中的 action、state、暂停标志没有在所审代码里展示完备同步。新训练/部署接口应明确初始化和时钟约定，而不是复制这些偶发行为。

合同 03 的新自训接口明确改为动作 `clip[-1,1]`、reset 清 action/gyro/history/延迟队列、同步 sim-time 推理。这些是**新接口行为**，不能用原 warm-enter 残留来判定新接口错误。对正常范围观察构造、顺序、P_V 命令映射仍可逐项与这里的原生 fixture 比较，reset 与超界动作要按新合同另验。根代理将导出输入名 `obs/history`、输出名 `actions`；它们也不同于本次旧 ONNX 的 `nn_input0/nn_input1/nn_output`。

**低速合同的量纲上限：**轮半径 .087 m，轮 action scale=5 rad/s。若新动作夹到 `[-1,1]`，纯滚动轮速目标上限约 `.087*5=.435 m/s`。合同 03 只要求 ±.3 m/s，仍可使用这个版本；它不能据此宣称实现 1.5 m/s。1.5 m/s 对应约 17.24 rad/s，后续应另外定义如 wheel-scale=20 的接口版本、重新训练与验证，并保留实际力矩约束。把旧 flat_lab 权重直接塞进新 clip runtime 后所得结果，不能冒称未改动的旧部署 baseline。

ZIP 中被发现指向带 G2 手臂的 `DDT_D1_CFG` 是 78 observations / 23 actions。其维度、机器人自由度不符，本审计不能把它认证为这个纯 D1 57+570 / 16-action 权重的训练来源。后续纯 D1 自训环境需要独立版本、配置、随机种子和训练回执。

可以用下面命令在新证据目录重跑零物理审计。脚本拒绝覆盖已有目录：

```bash
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B /home/lyh/ddt_ros2_ws/tools/audit_d1_interface.py --output /home/lyh/ddt_ros2_ws/evidence/interface_audit_03/golden_run_02
```

曾实际调用 `claude --model opus --effort high` 做仅 Read/Grep/Glob 的独立复核；回执中的实际模型是 `claude-opus-5`。该调用达到 4 轮上限，返回 `error_max_turns`，没有完成审阅，因此不把它算作审阅通过。原始回执保存在 [opus_review.json](../evidence/interface_audit_03/opus_review.json)。本文结论来自上述源码审计和已运行的原生数值核验。
