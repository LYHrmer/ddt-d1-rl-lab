# 2026-10-11 续接：DDT D1 自训与部署（合同 04 收尾）

独立私有仓库：[LYHrmer/ddt-d1-rl-lab](https://github.com/LYHrmer/ddt-d1-rl-lab)，默认分支 `main`，传输沿用已验证的 SSH 443。

合同 04 的两次有界训练（04a、04b）均已完成，**控制资格均为 0/3，整体不合格**，04b 另有 1 次跌倒。正式三臂×三 seed 未放行、未执行。先读本文、[合同 04 结果](career_execution_results_04.md)、[合同 04a](career_execution_contract_04a.md)、[合同 04b](career_execution_contract_04b.md)、[04b 附录 A](career_execution_contract_04b_addendum_a.md)，以及 [合同 04 总账](../evidence/career_contract_04/final_delivery_01.json)；合同 03 的记录见[合同 03 结果](career_execution_results_03.md)与[其总账](../evidence/career_contract_03/final_delivery_01.json)，保持只读。**不要因为换窗口就重新启动训练。**

## 必须按证据表述的现状

五个版本（03b/03c/03d/04a/04b）**都没有通过整体控制资格**。逐版本固定任务通过数：0/3、0/3、1/3、0/3、0/3。

- **03d** 只有 reverse_stop 通过，而合同 04 的离线诊断查明：那次 PASS 有相当成分来自**欧氏度量抵消**（欧氏位移差 4.93 倍，路径长度只差 1.39 倍；倒退先向后冲 0.0262 m 再被正向偏置推回 0.042 m）。
- **04a**（零令段位移锚定惩罚）：主判据 P1 成立（turn 位移 0.06076 → 0.03485），零令净轮速偏置 +0.460 → +0.115 rad/s，**主动刹车首次出现**（负轮速目标 tick 占比 0 % → 45.8 %）。但引入 **12 Hz 指令级抖振**（净轮速目标标准差 0.060 → 0.976 rad/s），三条任务的末秒水平速度门由全部通过变为全部失败。资格 0/3。
- **04b**（零令段轮动作均方幅值惩罚）：mean(a²) 降 65–175 倍、抖振标准差相对 04a 降约 500 倍（P2/P3 在完成的两条上达标），**reverse_stop 通过 8 项门槛中的 7 项**且位移门是真正通过（路径/欧氏比 1.73）。但 **P1 主判据不成立 ⇒ 假设 REJECTED**；新增两个退化：驱动相 vx MAE 退到 0.068/0.070（门槛 0.06）、turn 任务在 11.16 s **跌倒**。资格 0/3。

训练 COMPLETE、输入身份未变、跨语言数值一致、没有跌倒、单项指标改善、回报上涨 —— **以上任何一项都不等于控制资格通过**。

## 合同 04 最重要的一条机制结论

合同 04b 第 8 条强制的接口物理测量（回执 `zero_wheel_braking_probe_02`）给出：**零轮动作下的被动刹停距离远在门槛以内**（从 0.0883 m/s 刹到 \|v\|<0.02 用 0.00733 m / 0.16 s，比接口算术预测的 0.01720 m 还强约 2.3 倍）。

因此 **04b 的位移失败不是阻尼不足，而是策略在刹车窗口内并没有把轮动作归零**：forward 的 8–9 s 实测 mean(a²)=0.02208、净轮速目标 +0.4596 rad/s，是 11–12 s（0.00099）的 **22 倍**，而该窗口贡献了总位移的 70 %（0.0528 / 0.0753 m）。**策略只在"已经停住之后"才归零。**

⇒ 下一个合同的证据方向：针对**命令切零后的前若干 tick**，而不是整个零令段。但这仍然需要**新的可证伪假设与新的有界合同**，不得直接追加训练。

两个必须随附的限制：释放速度 0.0883 m/s 只有门槛相关速度（0.1965）的 45 %，外推到门槛速度约 0.016 m，**是外推不是实测**；刹停后速度反弹到 0.0296 m/s 那一段是**强制覆盖造成的动作不一致**（腿仍按"轮会被驱动"的预期输出），不能据此断言腿是独立漂移源，报告里的拟合 τ=0.322 s 也不是阻尼时间常数。另外，此前一个基于离线拟合的估计（τ=0.84–1.63 s，阻尼比预测弱 4–8 倍）**已被本测量推翻**，不作为任何依据。

## 另一条关键结论：先问"策略看得见吗"

57 维观测（`training/d1_rl/interface.py:81-85`）= 滤波陀螺×0.25(3) + 机体重力方向(3) + 命令×[2,2,0.25](3) + 关节偏移(16，**轮项强制为 0**) + 关节速度×0.05(16，含轮) + 上次动作(16)，**没有基座位置，也没有基座线速度**，历史只有 10 帧 × 20 ms = 200 ms。

04a 要求最小化的"离锚点位移"**对 actor 不可观测**，于是 PPO 找到了唯一的无记忆代理行为 —— 12 Hz 抖振（质心摆幅仅 1.08 mm，但速度幅值约 0.02 m/s）。设计奖励前必须确认被测量可观测。

## 工作区与操作边界

- 工作区 `/home/lyh/ddt_ros2_ws`。训练 Python 依赖在 `.python-deps`，MuJoCo 3.3.0；系统默认 MuJoCo 版本不同，不能替换。C++ 为 MuJoCo3.3.0/ORT1.10.0。
- `/home/lyh/wheel-legged-control-lab`、`wheel-legged-control-lab-work`、合同 02 记录、**77 份冻结文件**及 `results/my_course_drive_01`（9 份）保持**只读**。本轮已逐一校验：77/77、9/9 哈希一致，旧仓库 HEAD `9a10a3df`。
- 合同 03 的 **22 项冻结输入**本轮多次校验，22/22 一致。`tools/train_d1_ppo.py`、`tools/export_d1_ppo.py`、`tools/evaluate_d1_ppo.py`、`training/d1_rl/factory03.py` 等**全程零改动**。
- `training/d1_rl/factory04.py`、`tools04/train_d1_ppo04.py`、`tools04/check_task04a_static.py` 的 SHA 已写入 04a 冻结回执（`pilot04a_release_02.json`），**不得修改**；新增任务请用**委托链**（如 `factory04b` 扩展 `factory04`）而不是改上游。
- 每条 shell 命令加 `rtk`，需原始输出用 `rtk proxy`。本机 `rtk` 无 `proxy` 子命令文档但透传可用。
- 不得重跑旧批次、换 seed、挑中间 checkpoint 或覆盖失败证据。各池实际 transitions/native 分别计账，失败步也计入。

## 入口（合同 04 新增）

| 用途 | 入口 |
|---|---|
| 完整加载 ROS 环境（只 source，不覆盖 `PYTHONPATH`） | `runtime04/ros_env04.sh` |
| 唯一 ROS 启动入口（`--print-env` 打印解析后身份） | `runtime04/launch_env04.sh` |
| 零物理依赖预检（11/11 PASS，显式记录 `watchdog_validated: false`） | `runtime04/preflight_ros_imports04.py` |
| 通用冻结脚本注入入口（校验 SHA → 注入 factory → `runpy` 作 `__main__`） | `tools04/run_frozen_entry.py --factory d1_rl.factory04b --script tools/train_d1_ppo.py -- …` |
| 零积分静态检查 | `tools04/check_task04a_static.py`（25 项）/ `check_task04b_static.py`（30 项） |
| 事前冻结回执 | `tools04/release_pilot04a.py` / `release_pilot04b.py` |
| 预测判定（与资格分开记账） | `tools04/verify_predictions04a.py` / `verify_predictions04b.py` |
| 离线诊断（0 物理步） | `tools04/analyze_pilot_traces04.py`、`reward_signal_probe04.py`、`characterize_stop_oscillation04.py`、`audit_cxx_trace04.py` |
| 接口物理探针 | `tools04/probe_passive_braking04.py`（开环）/ `probe_zero_wheel_braking04.py`（策略在环、仅轮归零） |
| 总账（从回执重算，不抄文档） | `tools04/build_final_ledger04.py` |
| 部署变体与计划（含 04a/04b） | `runtime04/deployment_variants04.py`、`prepare_pilot_deployment04.py` |
| 运行期 schema | `runtime04/schema04a.json`、`runtime04/schema04b.json`（仅换训练配置路径/SHA，控制接口与 03 相同） |

`runtime03/build/d1_runtime03` 已冻结（SHA `62134e7c…`），**无需重建**。

## 关键 SHA（完整清单以各 release/protocol 为准）

| 产物 | SHA-256 |
|---|---|
| `training/configs/pilot04a.json` | `042d15a1f9108b90b5e2bad5ce0ba89673cf16a7a0c6e534171858c02cb2ddd6` |
| `training/configs/pilot04b.json` | `9d855063fb654f1d9bf0ae4d392f7f70294c328c5e944ef79f8e93902fb82e0a` |
| `runtime04/schema04a.json` | `3a3a0b88f3f2b754ecff38083dc080259346827bcf109ea7aaf6bca4ec6e475c` |
| `runtime04/schema04b.json` | `ffca72c5b8942b79ab33c49fff4c5d4ae9ad06bdb699e41f340f156e20a29f6e` |
| 04a `policy.onnx` | `f8ecc20eb8720713bc067630394ce1a48ea192da586c1a5ef3d7f8a76f4f2d3b` |
| 04b `policy.onnx` | `f8328e286a4d566422f32d85f8cb58b680d27b442b42d085dd2f51c98a4b5c8b` |
| `runtime03/build/d1_runtime03` | `62134e7c94fafaefcdee9f8325191b8f621cb3eb51f9ed2a167d68198469975d` |

## 预算与计步（合同 04，各池分别记账）

| 池 | 上限 | 实用 |
|---|---:|---:|
| 04a 训练 | 10,485,760 | **10,485,760**（1/1，耗尽） |
| 04a Python 资格 | 18,000 | **18,000** |
| 04a C++ 部署 | 18,000 | **6,000**（1/3 次，余 12,000 未用） |
| 04a 工程/预检 | 2,000 | 0 |
| 04b 训练 | 10,485,760 | **10,485,760**（1/1，耗尽） |
| 04b Python 资格 | 18,000 | **17,580**（一条因跌倒提前终止） |
| 04b C++ 部署 | 18,000 | **6,000**（1/3 次，余 12,000 未用） |
| 04b 工程/预检 | 2,000 | **2,000**（耗尽；见下） |
| 附录 A | 1,670 | **1,670** |
| 正式 B/R/M×3seed | 0 | 0（**未放行**） |

**合同 04 新增：2,097,152 transitions / 21,022,770 native。全项目累计 5,242,880 transitions / 52,914,120 native**（合同 03 为 3,145,728 / 31,891,350，不改写）。

**如实记录的两次失败消耗**：

1. 04a 第一次启动因 `factory04` 无限递归崩溃（`factory04` 通过已被注入的 `factory03` 查找 03 回退路径）。崩溃在 `tools/train_d1_ppo.py:140`，**0 物理步、0 transitions、未扣训练额度**；回执 `pilot04a_launch_failure_01.json`，失败目录 `pilot04a_01/` 原样保留（仅一个空 `source_snapshot/`）。修复：`factory04` 在 import 时捕获上游函数引用；静态检查新增 `injected_entry_resolves_without_recursion`。
2. 策略在环探针第 1 次尝试物理循环已跑完、`env.close()` 已完成，随后在构造回执时因 `Path.relative_to` 对相对路径抛异常崩溃，**输出目录尚未创建、计数器随进程丢失**。实际消耗未知，按计划上限 **1,650 native 保守计入** 04b 工程池并视为耗尽。这是工具缺陷造成的损失，**不是测量结果**。修复后先用 20 native 冒烟验证回执写入才重跑。

**未用额度随合同失效，不是后续实验的许可。**

## 下一个窗口的建议顺序

1. **核对状态，不重复启动**：工作树、`origin/main` 与发布回执 commit、宿主进程（`train_d1_ppo|d1_runtime03|mujoco|ros2` 应为空）、22 项 03d 冻结输入、77 份旧冻结文件、`my_course_drive_01` 9 份、04a/04b 两份 release 回执。隔离 PID 命名空间里 `ps`/`/proc` 为空**不能**证明宿主已退出。
2. **先做零物理成本的事**：把 M6（命令切零后前若干 tick 未归零）做成一个可证伪假设，并先用离线数据检查该假设的可观测性 —— 策略能看见"命令刚切零"吗？能，命令在观测里（`命令×[2,2,0.25]`），且有 10 帧历史，所以"刚切零"是可观测的。这一点与 04a 的失败形成对照，值得在新合同里写清。
3. **新合同必备**：新版本号、新目录、训练/评测/部署/工程各自预算、冻结输入 SHA、验收门槛（不移动）、预注册预测与**证伪条件**、失败停止规则。先过零积分静态检查并产出 release 回执，才允许训练。
4. **仍然悬空的 watchdog**：导入预检已过，`prepare_pilot_deployment04.py` 已改为经 `launch_env04.sh` 启动发布器并在 04a 上端到端验证过计划生成，但**真实 ROS 停发后的物理停稳从未执行**。按 04a/04b 合同 §9，它要在平地三条全门通过后以独立附录合同登记。若要提前做纯工程验证，必须新登记一个明确标注"工程检查、非资格试次"的预算。
5. **不要做的事**：追加第三次 04 训练、换 seed、挑中间 checkpoint、调权重重跑后宣布成功、把 0/3 或单条 7/8 写成成功、把未用额度当新许可、修改任何已进冻结回执的文件。

## 尚未完成（不得预写成功）

平地低速三条任务的整体资格（当前最好为单条 7/8 项）、转向增益不对称（03c/03d/04a/04b 四版本一致未达标，且 04a/04b 均事前声明不针对它）、04b 的两个新退化（驱动相 vx 跟踪、turn 稳定性，机制未确定）、真实 ROS 停发后的物理停稳、自训策略的丰富地形通过与真实 15 mm 台阶跨越、持续 A/D 侧移、1.5 m/s、正式 B/R/M×3seed 与独立评测、公平延迟改进/消融结论、物理自救（复位不是自救）、实机验证。独立 GUI 仍未做；**原版 GUI 使用同事策略，不能混算成绩**。
