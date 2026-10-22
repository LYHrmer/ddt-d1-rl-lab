# 求职向概览：DDT D1 强化学习控制实验（机器人控制 / 部署方向）

> 给招聘方/面试官的 2 分钟入口。所有数字都能在 `docs/` 与 `reports/` 找到对应运行记录。

## 一句话

面向 16 执行器轮足机器人的**纯 PPO 训练 → ONNX 导出 → C++/ONNX Runtime 实时闭环 → ROS2 部署链**，用固定 36 + 36 场矩阵、逐拍 SHA 校验和有界资源治理保证可复现。

## 60 秒版本

- 决策 100 Hz（每拍 5 个 2 ms 物理子步）、C++ 单线程推理，**每拍 20 ms 截止**；超时即拒绝该动作并停止仿真，不静默继续。
- 33 参考整形候选在 **formal 36/36 + 独立 holdout 36/36** 通过原阈值；代价是综合 J 相对 25 基线**变差 5.89%**，如实保留。
- 34 在真实 C++ 服务器上执行同一候选：**72 场轨迹与 Python 参考逐位相同**，43,200 次 C++ 推理全部返回；零物理重放 43,200 次推理 59 秒完成。
- 全部实验在单机、无 GPU、任务组 3 GiB 硬限额下完成；失败与部分运行原样保留，不续跑。

## 控制与部署栈

![D1 实时部署链](docs/assets/d1_deploy_chain.svg)

关键约束：每拍 5 个 2 ms 物理子步；单拍推理超过 20 ms 就拒绝该动作并终止；每个回合新建 C++ 进程，状态不跨回合延续。

## 可核验结果

| 结果 | 数字 | 证据 |
|---|---|---|
| 参考整形资格 | formal **36/36** + holdout **36/36**，全部满 600 拍，停车全过 | `docs/reference_shape33_results_20261018.md` |
| C++ 部署等价 | 72 场轨迹与 Python33 **逐位相同**；43,200 次 C++ 推理全返回；最大动作差 0.0 | `docs/cpp_reference34_results_20261018.md` |
| 可重复重放 | 全新 C++ 进程、零物理，43,200 次推理 **59 秒**，与保存动作逐位相同 | `docs/cpp_reference34_usage.md` |
| 纯自训基线 | 25 random 在原开发矩阵 **33/36**；默认模型仍是 16 control（27/36），未替换 | `README.md` 一页结论 |

## 控制/部署技能映射

| 技能 | 在本仓库的落点 |
|---|---|
| 实时 C++ 推理服务 | `runtime34/actor_server34.cpp`：JSONL 协议、单线程 ORT、每拍 SHA、进程级状态复位 |
| ONNX 导出与数值等价 | `tools33/`、`tools34/`：Python/PyTorch 与 C++ ORT 逐拍对比，门槛 1e-5，实测 0 |
| 控制时序与截止时间 | 100 Hz 决策 / 500 Hz 物理；单拍 20 ms 截止，超时拒绝动作 |
| 执行器限幅与保护 | PD 目标、力矩上限、动作裁剪、看门狗、急停语义 |
| 离线轨迹审计 | 逐 native 记录 qpos/qvel/命令/issued/executed/力矩，独立读回重算 |
| 资源受限实验 | 4 GiB 准入、2 GiB 运行保留、cgroup 2/3 GiB、无 swap、Nice10 |
| 可复现工程 | SHA 来源闭包、CI、失败与 partial 原样保留、不续跑 |

## 三个可深挖的面试故事

### 1. 把"部署一致"做成可测试的契约

- **问题**：Python 跑通不等于 C++ 部署后行为一致；有状态参考整形与十帧历史最容易在移植时走样。
- **做法**：C++ 内部完成整形与历史；每拍对完整 627 维有效输入做 SHA，与 Python 独立实现比对；每回合新建进程；模型、源码、ORT 库、适配器逐一 SHA 固定。
- **结果**：72 场、43,200 拍动作差为 0，轨迹 SHA 与 Python 逐条相同；零物理重放 59 秒可复核。
- **可追问**：是"容差内"还是"逐位"？——逐位；门槛仍为 1e-5，实测为 0。

### 2. 资格提升不等于性能提升：5.89% 的取舍

- **背景**：延迟鲁棒性是主要失败来源（60 ms 转向）。
- **做法**：固定 25 权重，只改偏航参考的输入整形（增益 1.5、τ=0.1 s 因果平滑、真实有效历史），不改评分与物理命令。
- **结果**：formal 与 holdout 都 36/36，但综合 J 从 0.215939 变到 0.228668（**+5.89%**）。
- **处理**：保留取舍说明、不替换默认模型、不把"通过率提升"写成"性能提升"。
- **可追问**：为什么不用它？——它解决资格问题，不是综合指标；是否接受这个取舍应由使用者决定。

### 3. 让实验在小机器上可审计

- **约束**：单机、无 GPU、不能停用户其他程序。
- **做法**：4 GiB 有效内存持续 10 秒才准入、运行保留 2 GiB、任务组 MemoryHigh 2 GiB / MemoryMax 3 GiB / swap 0、Nice 10、单线程；每个实验独立 release/输出目录，绑定来源 SHA。
- **结果**：50+ 场级实验可在数分钟内完成并留下完整回执；C++ 资格 239 秒；所有中断留下可复核的失败记录。
- **可追问**：资源不够怎么办？——让监控按预设终止，保留 partial 并如实记录，不用 partial 冒充完整候选。

## 简历可直接用的条目（控制/部署方向）

- 实现轮足机器人纯 PPO 训练→ONNX→C++/ONNX Runtime 的实时部署闭环：决策 100 Hz、物理 500 Hz、单线程每拍 20 ms 截止；72 场评测轨迹与 Python 参考逐位相同（每拍 627 维输入 SHA 校验），零物理重放 43,200 次推理 59 秒完成。
- 设计因果参考整形（增益 1.5、τ=0.1 s、10 帧有效历史），使 formal 与独立 holdout 均 36/36 通过；同时如实记录综合 J 相对基线变差 5.89% 的取舍，默认模型未替换。
- 在 3 GiB 内存硬限额、无 GPU 的单机环境建立有界训练/评测流水线（4 GiB 准入、2 GiB 保留、失败保留、不续跑 partial），支撑 50+ 场级实验的可复现执行。
- 实现逐 native 轨迹审计与独立读回：记录 qpos/qvel、命令、issued/executed 动作、五子步力矩与饱和掩码，用于部署路径与参考实现的逐位核对。

## 边界（建议主动说）

- 全部为 MuJoCo 仿真；未在本仓库版本上做实机部署。
- 完整延迟资格（60 ms 转向）仍未通过；默认模型未替换。
- 机器人模型、mesh 与原始策略来自与合作者共同开发的资源包；`src/ddt_ros2_control` 为 D1 的 ROS2 控制工作区，本仓库工作集中在训练、导出、C++ 部署验证与评测纪律。
- 部分历史运行是 partial/失败，均原样保留并标注，不作为候选。

## 推荐阅读路径（给面试官）

1. 本页 + README 的控制/部署栈图；
2. `docs/cpp_reference34_results_20261018.md`（逐位等价与重放）；
3. `docs/cpp_reference34_usage.md`（协议、时序与复现命令）；
4. `runtime34/actor_server34.cpp` 与 `tools34/run_cpp_reference34.py`（实现）；
5. `docs/reference_shape33_results_20261018.md`（资格与取舍）。

## English summary

Pure-PPO training for a 16-actuator wheel-legged robot, exported to ONNX and executed by a single-threaded C++/ONNX Runtime actor at 100 Hz with a per-tick 20 ms deadline over 500 Hz physics. The reference-shaping candidate passes formal 36/36 and an independent holdout 36/36, at the documented cost of a 5.89% worse aggregate J. The C++ path reproduces all 72 evaluation traces bit-for-bit (per-tick SHA over 627 inputs), and a zero-physics replay re-runs 43,200 inferences in 59 s. All experiments ran on a single CPU machine under a 3 GiB cgroup cap; failures and partial runs are retained, never resumed.
