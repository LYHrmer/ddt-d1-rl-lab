# 2026-10-11 续接：DDT D1 自训与部署（合同 04 收尾）

独立私有仓库：[LYHrmer/ddt-d1-rl-lab](https://github.com/LYHrmer/ddt-d1-rl-lab)，默认分支 `main`，传输沿用已验证的 SSH 443。

合同 04 的两次有界训练（04a、04b）均已完成，**控制资格均为 0/3，整体不合格**，04b 另有 1 次跌倒。正式三臂×三 seed 未放行、未执行。先读本文、[合同 04 结果](career_execution_results_04.md)、[合同 04a](career_execution_contract_04a.md)、[合同 04b](career_execution_contract_04b.md)、[04b 附录 A](career_execution_contract_04b_addendum_a.md)，以及 [合同 04 当前总账](../evidence/career_contract_04/final_delivery_02.json)；[历史总账 01](../evidence/career_contract_04/final_delivery_01.json)原样保留。合同 03 的记录见[合同 03 结果](career_execution_results_03.md)与[其总账](../evidence/career_contract_03/final_delivery_01.json)，保持只读。**不要因为换窗口就重新启动训练。**

## 必须按证据表述的现状

五个版本（03b/03c/03d/04a/04b）**都没有通过整体控制资格**。逐版本固定任务通过数：0/3、0/3、1/3、0/3、0/3。

- **03d** 只有 reverse_stop 通过，而合同 04 的离线诊断查明：那次 PASS 有相当成分来自**欧氏度量抵消**（欧氏位移差 4.93 倍，路径长度只差 1.39 倍；倒退先向后冲 0.0262 m 再被正向偏置推回 0.042 m）。
- **04a**（零令段位移锚定惩罚）：主判据 P1 成立（turn 位移 0.06076 → 0.03485），零令净轮速偏置 +0.460 → +0.115 rad/s，**主动刹车首次出现**（负轮速目标 tick 占比 0 % → 45.8 %）。但引入 **12 Hz 指令级抖振**（净轮速目标标准差 0.060 → 0.976 rad/s），三条任务的末秒水平速度门由全部通过变为全部失败。资格 0/3。
- **04b**（零令段轮动作均方幅值惩罚）：mean(a²) 降 65–175 倍、抖振标准差相对 04a 降约 500 倍（P2/P3 在完成的两条上达标），**reverse_stop 通过 7 个适用门槛中的 6 个**且位移门是真正通过（路径/欧氏比 1.73）。但 **P1 主判据不成立 ⇒ 假设 REJECTED**；新增两个退化：驱动相 vx MAE 退到 0.068/0.070（门槛 0.06）、turn 任务在 11.16 s **跌倒**。资格 0/3。

**2026-10-13 勘误**：旧文及历史总账的“7/8 通过”是汇总计数错误。Python 与独立 C++ 原始评分均为 **8 个字段：6 通过、1 失败、1 不适用**（reverse_stop 的稳态 wz 为 `null`）；唯一失败仍为驱动相 vx MAE `0.0694964657 > 0.06`。不能把不适用算通过。六个可比较数值的跨语言最大绝对差为 `3.365e-16`，8 位小数一致，见[勘误回执](../evidence/career_contract_04/handoff_20261013_errata.json)。04a 最终静态检查为 **27/27**（初版 25 项，修复启动递归后新增 2 项），黄金比对最大误差为 **5.364418e-7**；04b 为 30/30、4.768372e-7。预算、资格结论与历史原始回执不变。

训练 COMPLETE、输入身份未变、跨语言数值一致、没有跌倒、单项指标改善、回报上涨 —— **以上任何一项都不等于控制资格通过**。

## 合同 04 最重要的一条机制结论

合同 04b 第 8 条强制的接口物理测量（回执 `zero_wheel_braking_probe_02`）给出：**零轮动作下的被动刹停距离远在门槛以内**（从 0.0883 m/s 刹到 \|v\|<0.02 用 0.00733 m / 0.16 s，比接口算术预测的 0.01720 m 还强约 2.3 倍）。

因此 **04b 的位移失败不是阻尼不足，而是策略在刹车窗口内并没有把轮动作归零**：forward 的 8–9 s 实测 mean(a²)=0.02208、净轮速目标 +0.4596 rad/s，是 11–12 s（0.00099）的 **22 倍**，而该窗口贡献了总位移的 70 %（0.0528 / 0.0753 m）。**策略只在"已经停住之后"才归零。**

⇒ 下一个合同的证据方向：针对**命令切零后的前若干 tick**，而不是整个零令段。但这仍然需要**新的可证伪假设与新的有界合同**，不得直接追加训练。

两个必须随附的限制：释放速度 0.0883 m/s 只有门槛相关速度（0.1965）的 45 %，外推到门槛速度约 0.016 m，**是外推不是实测**；刹停后速度反弹到 0.0296 m/s 那一段是**强制覆盖造成的动作不一致**（腿仍按"轮会被驱动"的预期输出），不能据此断言腿是独立漂移源，报告里的拟合 τ=0.322 s 也不是阻尼时间常数。另外，此前一个基于离线拟合的估计（τ=0.84–1.63 s，阻尼比预测弱 4–8 倍）**已被本测量推翻**，不作为任何依据。

## 另一条关键结论：先问"策略看得见吗"

57 维观测（`training/d1_rl/interface.py:81-85`）= 滤波陀螺×0.25(3) + 机体重力方向(3) + 命令×`[2,2,0.25]`(3) + 关节偏移(16，**轮项强制为 0**) + 关节速度×0.05(16，含轮) + 上次动作(16)，**没有基座位置、基座线速度或地形信息**。另有 10 帧 × 20 ms = 200 ms 历史；单帧 57 加历史打包后实际为 **627→16**。现有 ONNX 分别接收 `obs: [1,57]` 与 `history: [1,10,57]`，输出 `actions: [1,16]`。

04a 要求最小化的“离锚点位移”未直接提供给 actor，有限历史也不覆盖整个锚定段。实测存在 12 Hz 抖振（质心摆幅仅 1.08 mm，但速度幅值约 0.02 m/s）；不可观测目标与退化代理行为之间是机制推断，不能据此证明抖振是唯一可行解。

## 工作区与操作边界

- 工作区 `/home/lyh/ddt_ros2_ws`。训练 Python 依赖在 `.python-deps`，MuJoCo 3.3.0；系统默认 MuJoCo 版本不同，不能替换。C++ 为 MuJoCo3.3.0/ORT1.10.0。
- `/home/lyh/wheel-legged-control-lab`、`wheel-legged-control-lab-work`、合同 02 记录、**77 份冻结文件**及 `results/my_course_drive_01`（9 份）保持**只读**。2026-10-13 逐一校验 77/77、9/9 哈希一致，旧仓库 HEAD `e831dba`；9 份用户结果从未入库、仅此本地一份。
- 三份 release 的冻结输入分别 **22/22、22/22、23/23** 一致。`tools/train_d1_ppo.py`、`tools/export_d1_ppo.py`、`tools/evaluate_d1_ppo.py`、`training/d1_rl/factory03.py` 等**全程零改动**。
- `training/d1_rl/factory04.py`、`tools04/train_d1_ppo04.py`、`tools04/check_task04a_static.py` 的 SHA 已写入 04a 冻结回执（`pilot04a_release_02.json`），**不得修改**；新增任务请用**委托链**（如 `factory04b` 扩展 `factory04`）而不是改上游。
- 每条 shell 命令加 `rtk`，需原始输出用 `rtk proxy`。本机 `rtk` 无 `proxy` 子命令文档但透传可用。
- 不得重跑旧批次、换 seed、挑中间 checkpoint 或覆盖失败证据。各池实际 transitions/native 分别计账，失败步也计入。

旧仓库 `src/wheel_legged_control/experiments.py`（SHA `5c8a7a273fd3219c549c005a7d6996de66600a4fcf54c409b88c4d3cd49a7ed4`）仍冻结。无条件导入 `stable_baselines3` 导致 `[rl]` 实际不可选，以及基准表未调用 `capture_git_provenance`，均为**已查明、未修复**；解冻由仓库所有者决定。

## 2026-10-13 零物理步收尾

本轮接手基线：ddt `4241f52`、旧仓库 `e831dba`。宿主核查看到 PID 1 为 `systemd`、进程输出 590 行，未发现目标训练、运行器、MuJoCo 或 ROS 进程；未重启训练。原指定 pytest 命令基线 **68 passed / 0.27 s**，修订后原命令 **68 passed / 0.38 s**；加强接口签名检查后，在禁止模型构造及物理 API 的护栏下 **68 passed / 0.10 s**，[回执](../evidence/career_contract_04/handoff_20261013_pytest_guard.json)确认未加载模型、0 native。独立 [CI 静态检查回执](../evidence/career_contract_04/handoff_20261013_static_01/report.json)为 **32/32、0 物理步**。

CI 已加入 pytest，push 与 PR 两处触发均覆盖 `tests/**`。清理回执工具已限定回执写入 ddt evidence、拒绝覆盖，修复 B-only 路径及失败回执；17 项 `/tmp` 隔离验证通过，**本轮未执行真实旧仓库清理**。交接提及的三次变异验证是用户报告的既往验证，本轮没有对冻结输入做变异。

代码提交 `71b58c6b4332a25c353a3b528cdd881a546769ad` 已经 SSH 443 上传；[GitHub CI 38348049696](https://github.com/LYHrmer/ddt-d1-rl-lab/actions/runs/38348049696) **PASS**，日志确认 **68 passed / 0.55 s、32/32、0 physics**，见[代码 CI 回执](../evidence/career_contract_04/handoff_20261013_ci_code.json)。代码提交前 `precommit_02` 的三 release、77/77、9/9、32 项静态与当时 229 个本地链接均通过；首轮链接核验仅因命令向量误作 Markdown 链接失败，已修正，并非保护哈希失败。文档收尾提交的身份与最终 CI 单独记录在本地 `evidence/career_contract_04/handoff_20261013_closeout.json`。

## 入口（合同 04 新增）

| 用途 | 入口 |
|---|---|
| 完整加载 ROS 环境（只 source，不覆盖 `PYTHONPATH`） | `runtime04/ros_env04.sh` |
| 唯一 ROS 启动入口（`--print-env` 打印解析后身份） | `runtime04/launch_env04.sh` |
| 零物理依赖预检（11/11 PASS，显式记录 `watchdog_validated: false`） | `runtime04/preflight_ros_imports04.py` |
| 通用冻结脚本注入入口（校验 SHA → 注入 factory → `runpy` 作 `__main__`） | `tools04/run_frozen_entry.py --factory d1_rl.factory04b --script tools/train_d1_ppo.py -- …` |
| 零积分静态检查 | `tools04/check_task04a_static.py`（27 项）/ `check_task04b_static.py`（30 项） |
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

2026-10-13 收尾新增 **0 transitions / 0 native**；当前总账 `final_delivery_02.json` 仅纠正门槛计数，以上预算不变。

**如实记录的两次失败消耗**：

1. 04a 第一次启动因 `factory04` 无限递归崩溃（`factory04` 通过已被注入的 `factory03` 查找 03 回退路径）。崩溃在 `tools/train_d1_ppo.py:140`，**0 物理步、0 transitions、未扣训练额度**；回执 `pilot04a_launch_failure_01.json`，失败目录 `pilot04a_01/` 原样保留（仅一个空 `source_snapshot/`）。修复：`factory04` 在 import 时捕获上游函数引用；静态检查新增 `injected_entry_resolves_without_recursion`。
2. 策略在环探针第 1 次尝试物理循环已跑完、`env.close()` 已完成，随后在构造回执时因 `Path.relative_to` 对相对路径抛异常崩溃，**输出目录尚未创建、计数器随进程丢失**。实际消耗未知，按计划上限 **1,650 native 保守计入** 04b 工程池并视为耗尽。这是工具缺陷造成的损失，**不是测量结果**。修复后先用 20 native 冒烟验证回执写入才重跑。

**未用额度随合同失效，不是后续实验的许可。**

## 下一个窗口的建议顺序

1. **核对状态，不重复启动**：工作树、`origin/main` 与发布回执 commit、宿主进程、三份 release 输入哈希（22/22、22/22、23/23）、77 份旧冻结文件、`my_course_drive_01` 9 份。每次提交前还须完成 32 项 CI 静态检查与文档链接核验，不可遗漏 77 份保护检查。隔离 PID 命名空间里 `ps`/`/proc` 为空**不能**证明宿主已退出。
2. **先做零物理成本的事**：把 M6（命令切零后前若干 tick 未归零）做成一个可证伪假设，并先用离线数据检查该假设的可观测性 —— 策略能看见"命令刚切零"吗？能，命令在观测里（`命令×[2,2,0.25]`），且有 10 帧历史，所以"刚切零"是可观测的。这一点与 04a 的失败形成对照，值得在新合同里写清。
3. **新合同必备**：新版本号、新目录、训练/评测/部署/工程各自预算、冻结输入 SHA、验收门槛（不移动）、预注册预测与**证伪条件**、失败停止规则。先过零积分静态检查并产出 release 回执，才允许训练。
4. **仍然悬空的 watchdog**：导入预检已过，`prepare_pilot_deployment04.py` 已改为经 `launch_env04.sh` 启动发布器并在 04a 上端到端验证过计划生成，但**真实 ROS 停发后的物理停稳从未执行**。按 04a/04b 合同 §9，它要在平地三条全门通过后以独立附录合同登记。若要提前做纯工程验证，必须新登记一个明确标注"工程检查、非资格试次"的预算。
5. **独立候选，尚未验证**：非对称 actor-critic 可让 critic 从训练 `qpos` 读取基座位置/位移，actor 保持现有 57 单帧与 10 帧历史输入。它可能改善价值估计对位移目标的表达，但不保证消除抖振或通过资格；须另立假设与合同，并重新验证导出形状、黄金比对及部署一致性。
6. **不要做的事**：启动第六次训练、换 seed、挑中间 checkpoint、调权重重跑后宣布成功、把 0/3 或单条 6/7 适用项写成成功、把未用额度当新许可、修改任何已进冻结回执的文件。

## 尚未完成（不得预写成功）

平地低速三条任务的整体资格（当前最好为单条 6/7 适用项）、转向增益不对称（03c/03d/04a/04b 四版本一致未达标，且 04a/04b 均事前声明不针对它）、04b 的两个新退化（驱动相 vx 跟踪、turn 稳定性，机制未确定）、真实 ROS 停发后的物理停稳、自训策略的丰富地形通过与真实 15 mm 台阶跨越、持续 A/D 侧移、1.5 m/s、正式 B/R/M×3seed 与独立评测、公平延迟改进/消融结论、物理自救（复位不是自救）、实机验证。独立 GUI 仍未做；**原版 GUI 使用同事策略，不能混算成绩**。

## 2026-10-13 第二轮接手：四项工程优化已完成，研究合同尚未开始

本节追加，不改写上文。接手基线 ddt `203009d`、旧仓库 `e831dba`，均与远端 0/0；宿主无训练/运行器/MuJoCo/ROS 进程；未启动任何训练，**0 新增物理步、0 transitions**。

### 已完成并提交（两个提交：工具与测试，随后一个文档提交；提交哈希以 `git log` 为准）

1. **保护核验固化为正式工具** `tools04/check_handoff_integrity.py`：三 release 输入（22/22、22/22、23/23）、旧仓库 77 份冻结文件、9 份用户结果（精确文件集且未入库）、旧仓库受保护工作树、32 项静态检查、文档本地链接；任何读取/结构/哈希/检查错误都失败并写新回执，回执只能写到 `evidence/` 下的新目录。审查初稿修了两处：符号链接检查会因 root 之上的祖先目录误报；`command()` 硬依赖会话约定的 `rtk` 可执行文件。真实环境只读运行 PASS（`handoff_integrity_20261013_01/_02/_03`）。
2. **清理回执的 17 项隔离测试并入常规 CI**：CI 的 `pytest tests` 自动收集，无需改 workflow。
3. **零模型零积分护栏，实际执行而非声明**：`tests/conftest.py`（运行时，替换 8 个入口）+ `tests/test_static_guard.py`（`ast` 扫描，只依赖 `tests/_forbidden.py` 名单，不 import conftest）。两层互补是**测出来的**：运行时层在场时，补丁安装前就持有原函数的代码真的推进了物理，没被拦，已固化为回归测试。
4. **总账回归测试** `tests/test_ledger04.py`：严格区分 true/false/null，失败说明由原始指标与冻结门槛派生，拒绝格式错误；`overall` 由新增纯函数 `qualification_verdict` 从计数派生，不再是字面量 `'FAIL'`。只新增字段、不改旧字段名；对真实回执重建后无丢失字段，`overall` 仍为 `FAIL`。

测试 **68 → 269**（`tests/` 共 7 个文件）。本机全部执行、0 跳过；无 `evidence/` 的 `git archive` 克隆 + 只装 CI 声明的 4 个包：**264 通过、5 跳过、0 失败**，静态检查 32/32。

### 本轮留下的几条经验，下一个窗口请沿用

- **干净 venv ≠ 干净克隆。** 前者隔离依赖，不隔离文件系统；只有 `git archive` 导出的克隆才能暴露对被 gitignore 文件的隐性依赖。凡是读 `evidence/` 的测试都用 `tests/_workstation.py` 的 `needs_workstation_evidence`，判据是回执是否存在，不是是否在 CI。
- **测试全绿不等于测试有效。** 对每个新工具做变异验证；存活变异要分清是**测试缺口**（夹具没把单一失败源隔离，被另一道防线掩盖）还是**等价变异**（死代码，应删而不是补测试）。
- **提交信息和文档里的数字，从输出取，不手抄。** 本轮自己两次手抄出错（提交信息里的存活变异数与分类、文档里的测试总数和"不依赖本机"论断），都在提交前被复核发现并修正。
- **提交前的固定清单**：全套测试 + `check_handoff_integrity.py`（含 77 份冻结文件）+ 旧仓库状态 + 在无 `evidence/` 的克隆里再跑一遍。

### 公开页面上一处遗漏的更正

README 与教程第 0 章仍写着"8 项门槛通过 7 项"（把不适用的 `null` 当成通过）。Codex 的勘误只改了结果文档和总账。本轮已更正为"7 项适用门槛通过 6 项"并附更正记录。

### 研究合同：**尚未开始**，这是下一阶段的第一步

任务要求在四项工程优化之后，准备针对**命令切零后短窗口**的新合同，再按它完成**一次**有界训练。截至本节，**这一步没有做**：没有新版本号、没有新目录、没有新合同文件、没有预注册预测、没有 release 回执，也没有启动训练。不要把"工程优化已完成"读成"可以训练了"。

接着做的顺序：

1. **先做零新增物理步的离线分析**（只重算冻结轨迹）：比较命令切零后前 5 / 10 个 tick 的动作归零延迟、净轮速目标、位移贡献，以及 10 帧历史（约 200 ms）对切零事件的覆盖。已有证据：`zero_wheel_braking_probe_02` 表明零轮动作被动刹停只需约 0.00733 m / 0.16 s（外推到 0.1965 m/s 约 0.016 m，是外推不是实测）；04b forward 的 8–9 s `mean(a²)=0.02208`、净轮速目标 +0.4596 rad/s，是 11–12 s 的 22 倍，且该窗口贡献约 70% 位移。
2. **据此预注册一个短窗口假设**，检验"切零后早期动作未及时归零"，**不要**继续加整个零令段的惩罚，也不要假定阻尼不足。窗口长度与可观测性必须论证：命令在 57 维观测里，10 帧历史只覆盖约 200 ms。
3. **登记新版本号、新目录、新有界合同**：训练/评测/部署/工程各自预算、冻结输入 SHA、不移动的验收门槛、预注册预测与证伪条件、失败停止规则；先过零积分静态检查并生成事前冻结回执，才允许启动**唯一一次**训练。旧合同的未用余额不是许可。
4. 新实现走新的任务与 factory 委托链（例如 `factory04c` 扩展 `factory04b`），用 `tools04/run_frozen_entry.py` 做 SHA 校验与 `runpy` 注入；**不改**任何已进冻结回执的文件，特别是 `factory04.py`、`train_d1_ppo04.py`、`check_task04a_static.py`。Actor 每帧保持 57 维，导出接口保持 `obs[1,57]`、`history[1,10,57]` → `actions[1,16]`。
5. **非对称 actor-critic 保留为独立候选，不要和短窗口机制混进同一次实验**，也不能声称给 critic 特权信息就必然解决 actor 不可观测问题。

### 仍然不变

五个版本全部未通过整体控制资格（0/3、0/3、1/3、0/3、0/3），正式三臂×三 seed 未放行；本轮工程优化不构成任何控制能力证据。旧仓库 77 份冻结文件（含 `experiments.py`，SHA `5c8a7a273fd3219c549c005a7d6996de66600a4fcf54c409b88c4d3cd49a7ed4`）和 `results/my_course_drive_01` 9 份用户结果（只存在本地）严格只读。
