# 合同 04 实际结果与能力边界

更新：2026-10-11。本机 MuJoCo 3.3.0 仿真，**未进行实机验证**。合同 03 的结果见[合同 03 结果](career_execution_results_03.md)，保持只读，本文只追加不改写。

## 交付结论

合同 04 执行了两次有界开发训练（04a、04b），各一次、各 1,048,576 transitions，均 COMPLETE 且输入身份未变。两次**控制资格均为 0/3，整体不合格**；正式三臂×三 seed 未放行、未执行。

同时交付了可复用的工程能力：修复了 03 遗留的 ROS 启动环境缺陷（导入预检 11/11 PASS）、建立了通用的冻结脚本注入入口、把零积分静态检查从 25 项扩到 30 项（其中 2 次在训练前拦下了真实缺陷）、并完成了一次决定性的接口物理测量。

**单条任务的最好成绩出现在 04b 的 reverse_stop：8 项门槛通过 7 项**，含真正通过（非度量抵消）的停车位移门，且经独立 C++ 运行器复现到 8 位小数。这是单条任务成绩，**不是整体资格通过**。

## 两次训练的唯一改动与结果

04a 与 04b 是**同胞而非前后继**：两者都在同一个冻结的 03d 基线上**各加一项奖励**，差别只在被测量。因此两者对比隔离了"被测量选择"这一个变量。

| 版本 | 唯一改动 | 训练秒数 | transitions | native | 资格 | 提前失败 |
|---|---|---:|---:|---:|---:|---:|
| 04a | 零令段"离锚点净 XY 位移"的 Huber 惩罚（w=−20, knee=0.05, cap=0.10） | 814.54 | 1,048,576 | 10,485,760 | 0/3 | 0 |
| 04b | 零令段"四轮动作均方幅值"惩罚（w=−10, sat=0.25） | 844.43 | 1,048,576 | 10,485,760 | 0/3 | **1** |

两次均 seed 101、arm B、8 env、从头随机初始化，无 teacher、无行为克隆、无热启动、无真值线速度观测、无 PID、无动作后处理。最小可用内存 5.05 GiB（04a）/ 3.53 GiB（04b），门槛 3 GiB。

### 固定资格评测逐项（seed 7001、delay 0、三条 12 s）

| 版本 / 任务 | vx MAE | wz MAE | 末秒水平速度 | 末秒\|wz\| | 停车位移 | roll/pitch 峰值 | 资格 |
|---|---:|---:|---:|---:|---:|---:|---|
| 门槛 | ≤.06 | ≤.08 | ≤.02 | ≤.03 | ≤.05 | ≤.20 | 三条全过 |
| 03d / forward | .00317 | — | .00451 | .00064 | .08058 | .05298 | FAIL |
| 03d / reverse | .03992 | — | .00225 | .00251 | .01634 | .05878 | PASS |
| 03d / turn | — | .10288 | .01494 | .00959 | .06076 | .03872 | FAIL |
| 04a / forward | .02297 | — | **.02542** | .02407 | .07555 | .09571 | FAIL |
| 04a / reverse | .01765 | — | **.02918** | .02639 | **.07781** | .11314 | FAIL |
| 04a / turn | — | .11179 | **.02262** | **.03061** | **.03485** | .07896 | FAIL |
| 04b / forward | **.06802** | — | **.00138** | .00054 | .07530 | .15793 | FAIL |
| 04b / reverse | **.06950** | — | **.00256** | .00079 | **.02776** | .08414 | FAIL（7/8 通过） |
| 04b / turn | — | .18203 | .36449 | .16634 | 未测得 | .55157 | **跌倒 TERMINATED** |

粗体为相对 03d 的显著变化。04a/04b 三条均无力矩饱和；04a 三条均跑满 12 s；04b 的 forward/reverse 跑满 12 s，**turn 在 5,580 native（11.16 s）处因跌倒终止**（峰值倾角 0.6127 rad = 35°）。

### 预注册预测判定

两次合同都在训练前冻结了预测，事后未修改。

**04a**（回执 `pilot04a_prediction_verdict_01`，主判据 P1 为 turn_stop 位移）：

| | 预测 | 03d | 04a | 判定 |
|---|---|---:|---:|---|
| **P1 主判据** | turn 位移 < 0.05 | .06076 | **.03485** | **MET** |
| P2 | forward 位移 < 0.05 | .08058 | .07555 | NOT_MET |
| P3 | 三条蠕动 \|vx\| < 0.005 | .0045/.0022/.0145 | .0029/**.0204**/.0044 | NOT_MET |
| P4 | reverse 位移 < 0.05 | .01634 | **.07781** | NOT_MET |
| P5（事前声明不预测） | turn wz MAE ≤ .08 | .10288 | .11179 | 未达标 |

假设按主判据 **SUPPORTED**，资格 **0/3 FAIL**。

**04b**（回执 `pilot04b_prediction_verdict_01`，主判据 P1 为三条同时满足位移与末秒速度两门）：

| | 预测 | 03d | 04a | 04b | 判定 |
|---|---|---:|---:|---:|---|
| **P1 主判据** | 三条同时 位移<.05 且 末秒速度<.02 | 位移 2 条败 | 速度 3 条全败 | reverse 两项全达标；forward 位移败；turn 跌倒 | **NOT_MET** |
| P2 | 零令 mean(a²) < .02 | .0652/.0548 | .1736 | **.00099/.00098**/.0236 | NOT_MET（完成的两条达标） |
| P3 | 零令净轮速目标 std < .20 rad/s | .060/.034 | **.976** | **.0019/.0022**/.110 | NOT_MET（完成的两条达标） |
| P4 | forward 8–9 s 位移 < .045 | .04981 | .08535 | .05280 | NOT_MET |
| P5（事前声明不预测） | turn wz MAE ≤ .08 | .10288 | .11179 | .18203 | 未达标 |

假设 **REJECTED**，资格 **0/3 FAIL**，含 1 次跌倒。

## 已查明的机制（按证据强度排序）

### M1 奖励里从来没有被考核量的量纲（事实）

03b/03c/03d 的 11 个奖励项全部是**瞬时量**（`training/d1_rl/env.py:515-527`、`task03c.py:123-130`），而失败门槛 `stop_displacement ≤ 0.05 m` 考的是 8→12 s 的四秒积分位移。此外轮动作幅值**不受任何约束**：`leg_pose_deviation` 按 `task03c.py:24-25,113` 显式排除索引 3/7/11/15（四个轮），`action_rate` 只罚变化量（实测贡献 −0.0001 ~ −0.009 /s），`torque_sum_squared` 权重 −1e-5。

`standstill_tracking = exp(−v²/0.0004)`（σ=0.02 m/s）在 v > 0.05 m/s 时数值上为零（v=0.1965 时 1.2e-42），故在整个刹车过程中不提供任何梯度；训练中实测该项全程仅 +0.004 ~ +0.010 /s，即满值的 0.4–1.0 %。

### M2 零命令下策略的输出不是"静止"（事实，跨 5 个版本一致）

| 零令 11–12 s | mean(a²)/tick | 逐轮\|·\|均值 rad/s | 净轮速目标 rad/s | 净标准差 rad/s |
|---|---:|---:|---:|---:|
| 03c（三条） | — | — | +0.42 ~ +0.80 | — |
| 03d reverse | .05481 | 1.027 | +0.347 | .034 |
| 03d forward | .06516 | 1.127 | +0.460 | .060 |
| 04a turn | **.17363** | **1.632** | +0.115 | **.976** |
| 04b forward | **.00099** | 0.114 | −0.027 | **.0019** |
| 04b reverse | **.00098** | 0.136 | −0.108 | **.0022** |

6/6 的 03c/03d 轨迹净偏置为正，造成 0.0022–0.0598 m/s 的持续前爬。**04b 把 mean(a²) 压低 65–175 倍，把净偏置的符号翻转为负（即轻微反向制动），并把抖振标准差相对 04a 降低约 500 倍。**

### M3 04a 的退化是"被测量不可观测"导致的（事实 + 推断）

57 维观测（`interface.py:81-85`）= 滤波陀螺×0.25 (3) + 机体重力方向 (3) + 命令×[2,2,0.25] (3) + 关节偏移 (16，**轮项强制为 0**) + 关节速度×0.05 (16，含轮) + 上次动作 (16)。**没有基座位置，也没有基座线速度**；历史仅 10 帧 × 20 ms = 200 ms。

因此 04a 要求最小化的 ‖xy − anchor_xy‖ 对 actor **不可观测**（锚点在 4 秒之前设下）。实测后果：

| 11–12 s | 03d forward | 04a turn |
|---|---:|---:|
| vx 主频 / 幅值 | 1.0 Hz / ~0 | **12.0 Hz** / 0.0192 |
| wz 主频 / 幅值 | 19 Hz / .0004 | **12.0 Hz** / .0297 |
| 振荡/漂移比 | 0.1 | **4.4** |
| vx 反转次数/秒 | 0.0 | **24.0** |
| 路径/欧氏位移 | 1.00 | **4.48** |
| 质心摆幅 | — | **1.08 mm**（峰 3.36 mm） |

12 Hz 远低于 50 Hz 策略率的 Nyquist（25 Hz），周期 83 ms ≈ 4.2 个策略 tick；净轮速目标标准差相对 03d 放大 16–29 倍 ⇒ **振荡是策略主动指令出来的**，不是物理或采样假象。wz 的 12 Hz 分量 rms 0.0312 rad/s 正是 `stop_last_second_abs_wz = 0.0306 > 0.03` 的直接来源。

代价核算（用 04a 自己的权重）解释了策略为何这样选：单向蠕动约 −1.83 /s，极限环约 −0.89 /s —— **振荡更便宜，是奖励设计让它成为最优解**。推断部分：高频抖振是在无位置反馈下保持"时间平均位置不动"的唯一无记忆实现方式。

### M4 03d 反向任务的 PASS 有相当成分来自欧氏度量抵消（事实）

| 03d | 欧氏位移 | 路径长度 | 比值 | 沿航向反转 |
|---|---:|---:|---:|---:|
| forward（C++ 实测） | .0805836 | .0966600 | 1.199 | **0** 次 |
| reverse（C++ 实测） | .0163405 | .0694190 | **4.248** | 1 次 |

欧氏口径差 4.93 倍，路径口径只差 1.39 倍。倒退先向后冲 0.0262 m，再被同一个正向偏置推回约 0.042 m，净剩 0.0163 m。相比之下 **04b reverse 的路径/欧氏比为 1.73**（.04358 / .02516），不依赖抵消。

### M5 零轮动作的被动刹停距离远小于门槛（直接测量）

这是合同 04b 第 8 条在"P2/P3 达标而 P1 不成立"分支下强制要求的测量，由[附录 A](career_execution_contract_04b_addendum_a.md) 登记执行。

接口算术预测（`schema03d.json`、`robot.xml`：轮 kp=0、kd=0.5 N·m/(rad/s)、限 12 N·m、r=0.087 m；整车 48.15 kg，含四轮惯量估 51.5 kg；地面 plane condim=3 与轮 condim=1 取 max ⇒ 摩擦生效，μ=max(0.4,1.0)=1.0）：a = −5.13·v，τ = 0.195 s。

**开环探针**（350 native，回执 `passive_braking_probe_01`）：腿动作固定为 0（默认站姿）时，整车 60 ms 内从 0.457 m 沉到 0.432 m，0.7 s 时倾角 0.284 rad、高度 0.324 m，被**非法体支撑终止**，从未进入干净滚动状态。⇒ 算术背后"刚体直立、四轮均载"的前提在这台机器上不成立；默认腿姿在 kp=60 下撑不住整车。

**策略在环探针**（1,650 native，回执 `zero_wheel_braking_probe_02`）：腿由 04b 策略正常控制，**只把四个轮动作强制为 0**：

| 指标 | 实测 |
|---|---:|
| 释放速度 | 0.0883 m/s |
| **刹到 \|v\| < 0.02 m/s 的距离 / 时间** | **0.00733 m / 0.16 s** |
| 同速下 G6 算术预测 | 0.01720 m |
| 1.28 s 滑行总欧氏位移 | 0.00411 m |
| 滑行路径长度 | 0.01966 m |
| 判定 | **`zero_wheel_action_alone_meets_gate: true`** |

**被动阻尼实际比算术预测强约 2.3 倍**，刹停距离远在 0.05 m 门槛以内。

两个必须声明的限制：① 释放速度 0.0883 m/s 只有门槛相关速度（0.1965 m/s）的 45 %，线性外推约 0.016 m，**是外推不是实测**；② 速度降到 0.0076 m/s 后**反弹到 0.0296 m/s** 并在 0.015–0.025 区间游走到 0.70 s 才真正停住 —— 该回弹段是**强制覆盖造成的动作不一致**（腿仍按"轮会被驱动"的预期输出），不能据此断言腿是独立漂移源；报告中的拟合 τ=0.322 s 同样被这段污染，**不是阻尼时间常数**。

**此前一个基于离线拟合的估计（τ=0.84–1.63 s，即阻尼比预测弱 4–8 倍）被本测量推翻** —— 那次拟合取的正是 04b 轨迹中的回弹/游走段，不是自由滑行。该错误估计不作为任何结论依据。

### M6 04b 失败的直接定位（事实）

既然零轮动作足以在门槛内刹停（M5），04b 的 forward 位移失败（0.0753 m，其中 **0.0528 m = 70 % 发生在 8–9 s**）就不是阻尼不足，而是**策略在刹车窗口内并未把轮动作归零**：

| 04b forward 零令段 | mean(a²)/tick | 净轮速目标 rad/s | 该秒欧氏位移 |
|---|---:|---:|---:|
| 8–9 s | **.02208** | **+0.4596** | **.05280** |
| 9–10 s | .00139 | +0.0446 | .01404 |
| 10–11 s | .00102 | −0.0126 | .00398 |
| 11–12 s | .00099 | −0.0273 | .00130 |

8–9 s 的 mean(a²) 是 11–12 s 的 **22 倍**。策略只在"已经停住之后"才归零。⇒ 后续合同应针对**命令切零后的前若干 tick**，而不是整个零令段。

### M7 转向增益不对称跨四个版本一致（事实 + 已声明的局限）

拟合 `achieved = g·cmd + b`：

| 版本 | 正向增益 | 负向增益 | g | b (rad/s) | wz MAE |
|---|---:|---:|---:|---:|---:|
| 03c | 0.419 | 0.726 | 0.572 | −0.031 | .11666 |
| 03d | 0.486 | 0.768 | 0.627 | −0.028 | .10288 |
| 04a | — | — | — | — | .11179 |
| 04b | — | — | — | — | .18203（跌倒） |

即**不是"正向增益低"，而是对称的低增益叠加一个恒定负偏置**。四个版本均未达标，且 04a/04b 的两项改动都**事前声明过不针对它**。

**局限（不可用现有数据消除）**：固定评测把方向与顺序混在一起 —— 正向窗口 3–5 s 紧接静止（命令跳变 +0.2），负向窗口 6–8 s 紧接符号反转（跳变 −0.4）。每个方向只有一条轨迹，无 run-to-run 方差。全部为单 seed 开发诊断，**不是正式多 seed 改进证明**。

### M8 04b 的两个新退化（事实，机制未确定）

- **驱动相 vx 欠调**：`stable_vx_mae` 由 03d 的 .00317/.03992、04a 的 .02297/.01765 退到 **.06802/.06950**，越过 0.06 门槛。4–8 s 实测 vx 为 +0.1320/−0.1305（指令 ±0.2），欠调约 34 %。新项仅在零命令段生效，故这是**间接影响**（同一网络），具体机制未确定。
- **turn 任务跌倒**：03d、04a 均跑满 12 s，04b 在 11.16 s 跌倒，峰值 roll/pitch 0.5516 rad。10–11 s 已出现 0.1359 m/s 的速度与 0.1145 m 的位移，终止前已失控。机制未确定。

## 工程验证与跨语言一致性

- **ROS 启动环境已修复**：新入口 `runtime04/launch_env04.sh`（只做 source，不覆盖 `PYTHONPATH`）+ `runtime04/preflight_ros_imports04.py`，**11/11 PASS**，`rpyutils` 解析到 `/opt/ros/humble/lib/python3.10/site-packages/`，即 03 丢掉的那个路径（`runtime03/prepare_pilot_deployment03.py:79` 只设 `local/lib/python3.10/dist-packages`）。回执显式记录 `physics_steps_executed = 0`、`ros_nodes_created = 0`、**`watchdog_validated = false`** —— 导入通过不等于 watchdog 通过。
- **零积分静态检查**：04a 25/25 PASS，04b 30/30 PASS，均 `native_integration_steps_executed = 0`。**两次在训练前拦下真实缺陷**：04a 的第一次启动因 `factory04` 递归崩溃（0 积分，回执 `pilot04a_launch_failure_01.json`），随后新增 `injected_entry_resolves_without_recursion`；04b 的第一版饱和点 0.09 会把实测区间（04a 0.174、初始化 0.135）放进梯度平区，被新增的 `gradient_live_over_the_measured_range` 抓住（失败回执 `task04b_static_01` 原样保留）。
- **ONNX 黄金比对**：04a 最大误差 4.768e-7，04b 4.768e-7，均 64 例、真实 C++ ORT 1.10.0、0 物理步、容差 1e-5。
- **独立 C++ 部署**：各 1 次 12 s / 6,000 native。04a turn_stop 与 Python 复现到 ~5e-9（位移 .034852480 vs .034852475，J .56291903 vs .56291904）；04b reverse_stop 与 Python 复现到 8 位小数全同。两次结构检查均 24/24 PASS。推理 p99 为 0.0252 ms（04a）/ 0.0248 ms（04b）。**部署一致性不等于控制资格。**
- **通用注入入口** `tools04/run_frozen_entry.py`：先校验冻结脚本 SHA256 等于 03d release 记录值，再以 `runpy` 作为 `__main__` 执行并注入指定 factory。`tools/train_d1_ppo.py`、`tools/export_d1_ppo.py`、`tools/evaluate_d1_ppo.py` 全程**零改动**；`factory04b` 以委托链扩展 `factory04`，后者亦零改动（其 SHA 已写入 04a 冻结回执）。

## 预算与计步（各池分别记账，失败步也计入）

1 transition = 1 策略周期 = 10 native。静态 `mj_forward` 不是积分。

| 池 | 上限 | 实用 | 说明 |
|---|---:|---:|---|
| 04a 开发训练 | 10,485,760 | **10,485,760** | 1/1 次，已耗尽 |
| 04a Python 资格 | 18,000 | **18,000** | 三条 12 s |
| 04a C++ 部署 | 18,000 | **6,000** | 1/3 次（turn_stop），余 12,000 未用 |
| 04a 工程/预检 | 2,000 | 0 | 未用（黄金比对为 0 物理步） |
| 04b 开发训练 | 10,485,760 | **10,485,760** | 1/1 次，已耗尽 |
| 04b Python 资格 | 18,000 | **17,580** | 一条因跌倒提前终止 |
| 04b C++ 部署 | 18,000 | **6,000** | 1/3 次（reverse_stop），余 12,000 未用 |
| 04b 工程/预检 | 2,000 | **2,000** | 已耗尽：350 开环探针 + ≤1,650 因工具缺陷丢失（见下） |
| 附录 A 冒烟 | 20 | **20** | 回执写入验证 |
| 附录 A 测量 | 1,650 | **1,650** | 零轮动作刹停距离 |
| 正式 B/R/M × 3 seed | 0 | 0 | **未放行、未执行** |
| 正式独立评测 | 0 | 0 | **未执行** |

**合同 04 新增合计：2,097,152 transitions / 21,022,770 native。**

**如实记录的预算损失**：策略在环探针的第 1 次尝试物理循环已跑完、`env.close()` 已完成，随后在构造回执时因 `Path.relative_to` 对相对路径抛 `ValueError` 而崩溃，**输出目录尚未创建、计数器值随进程丢失**。实际消耗未知，按计划上限 1,650 保守计入 04b 工程池并视为耗尽。这是工具缺陷造成的损失，**不是测量结果**，不据此声称任何物理结论。缺陷已修（新增 `relative()` 回退为绝对路径），并在重跑前先用 20 native 冒烟验证回执写入。

历史累计：合同 03 为 3,145,728 transitions / 31,891,350 native。**全项目累计 5,242,880 transitions / 52,914,120 native**，历史数字不改写。

## 证据入口

| 项目 | 04a | 04b |
|---|---|---|
| 事前冻结回执 | [pilot04a_release_02.json](../evidence/career_contract_04/pilot04a_release_02.json) | [pilot04b_release_01.json](../evidence/career_contract_04/pilot04b_release_01.json) |
| 零积分静态检查 | [task04a_static_02](../evidence/career_contract_04/task04a_static_02/report.json) | [task04b_static_03](../evidence/career_contract_04/task04b_static_03/report.json) |
| 训练结果 | [pilot04a_02/result.json](../evidence/career_contract_04/pilot04a_02/result.json) | [pilot04b_01/result.json](../evidence/career_contract_04/pilot04b_01/result.json) |
| 导出与黄金 | [export](../evidence/career_contract_04/pilot04a_export_01/metadata.json) / [golden](../evidence/career_contract_04/pilot04a_cxx_golden_01/report.json) | [export](../evidence/career_contract_04/pilot04b_export_01/metadata.json) / [golden](../evidence/career_contract_04/pilot04b_cxx_golden_01/report.json) |
| Python 资格 | [summary](../evidence/career_contract_04/pilot04a_python_eval_01/summary.json) | [summary](../evidence/career_contract_04/pilot04b_python_eval_01/summary.json) |
| 预测判定 | [verdict](../evidence/career_contract_04/pilot04a_prediction_verdict_01/report.json) | [verdict](../evidence/career_contract_04/pilot04b_prediction_verdict_01/report.json) |
| 独立 C++ 评分 | [turn_stop](../evidence/career_contract_04/pilot04a_deployment_plan_01/score03_turn_stop/report.json) | [reverse_stop](../evidence/career_contract_04/pilot04b_deployment_plan_01/score02_reverse_stop/report.json) |

诊断与工程回执：

- [离线轨迹诊断 03c/03d](../evidence/career_contract_04/offline_trace_diagnosis_01/report.json) / [04a](../evidence/career_contract_04/offline_trace_diagnosis_04a_01/report.json)
- [奖励信号探针](../evidence/career_contract_04/reward_signal_probe_01/report.json)：沿冻结轨迹重算 03d 各奖励项与速度敏感度
- [独立 C++ 轨迹审计](../evidence/career_contract_04/cxx_trace_audit_01/report.json)：独立只读智能体对两条 03d 真实轨迹的逐 native 步审计，独立复现评分到 1e-10
- [停车振荡刻画](../evidence/career_contract_04/stop_oscillation_04a_01/report.json)
- [ROS 导入预检](../evidence/career_contract_04/ros_import_preflight_01/report.json)
- [开环刹车探针](../evidence/career_contract_04/passive_braking_probe_01/report.json) / [策略在环零轮动作探针](../evidence/career_contract_04/zero_wheel_braking_probe_02/report.json)
- [04a 启动失败回执](../evidence/career_contract_04/pilot04a_launch_failure_01.json)：0 积分、未扣训练额度
- 合同：[04a](career_execution_contract_04a.md) / [04b](career_execution_contract_04b.md) / [04b 附录 A](career_execution_contract_04b_addendum_a.md)

证据与权重保留本机。源码仓库不包含同事完整模型/权重及大型轨迹。

## 未完成与停止条件

按合同 04a §8 与 04b §8：两次 pilot 额度均已耗尽，**不启动第三次、不换 seed、不挑中间 checkpoint、不调权重重跑**。三条任务未全过 ⇒ **不启动正式 B/R/M × 3 seed**。

仍未完成：

- 平地低速三条任务的整体资格（当前最好为单条 7/8 项）
- 转向增益不对称（四个版本一致未达标，两次改动均事前声明不针对它）
- 驱动相 vx 跟踪与 turn 任务稳定性（04b 的两个新退化，机制未确定）
- 真实 ROS 停发后的物理停稳（watchdog）—— 导入预检已通过，物理验收未做
- 自训策略的丰富地形通过、真实 15 mm 台阶跨越、持续 A/D 侧移、1.5 m/s
- 正式三臂×三 seed、公平延迟改进/消融结论、独立评测
- 物理自救（复位不是自救）、实机验证

下一阶段的证据已指向明确方向（M6：针对命令切零后的前若干 tick，而非整个零令段），但**需要新的可证伪假设与新的有界合同**；当前结果只支持本文列出的已验证工程链路与局部机制结论。
