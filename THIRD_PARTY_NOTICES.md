# 第三方资产与上游许可证声明

本仓库发布**自写代码与文档**（MIT，见 [LICENSE](LICENSE)），以及第 7 节注明来源和许可的少量改编代码。D1 机器人模型、mesh、URDF/xacro、原配 ONNX 策略、训练权重、依赖安装目录与大型逐步轨迹**均不在本仓库内**，必须另行从提供方取得。仅凭 clone 无法启动机器人。

下表是本仓库每一类文件的来源与许可状态，逐项核对过（155 个受跟踪文件）。

## 1. 完全自写（MIT）

| 路径 | 内容 |
|---|---|
| `training/d1_rl/`（除第 7 节的改编方法） | 环境、接口、任务 03c/03d/04a/04b、工厂与委托链，以及本地候选模块 |
| `training/configs/` | pilot03b/03c/03d/04a/04b 配置与接口 schema |
| `tools/`、`tools04/` | 训练/导出/评测/静态检查/离线诊断/探针/总账工具 |
| `runtime03/`（除下述上游外）、`runtime04/` | 独立 C++ 运行器、ROS 启动入口、部署计划、运行期 schema |
| `docs/`、`reports/` | 合同、结果、教程、发布摘要 |
| `src/.../mujoco_ros2_control/{include,src}/step_evidence.{hpp,cpp}` | 本项目**新增**的有界 native 步证据记录器；不属于原 ZIP 的 211 项，无上游版权头 |
| `src/.../d1_description/mujoco/scene_*.xml`（4 个，23–122 行） | 本项目自写的 MuJoCo 场景。它们通过 `<include file="robot.xml"/>` **引用**原机器人模型，但**不嵌入**任何 mesh/STL/URDF 内容 |
| `src/.../training_course_03_assets/`、`video_course_assets/` | 由 `tools/generate_training_course.py`、`tools/generate_video_course.py` 生成的地形高度图与清单 |

## 2. 被引用但**未**再分发

| 资产 | 状态 |
|---|---|
| `robot.xml`、`d1_description` 的 mesh / STL / URDF / xacro | **不在本仓库**。场景 XML 只按文件名引用它。SHA256 记录在 `source_archive_receipt.json` 与各 release 回执中，供核对身份 |
| 原配 ONNX 策略与同事权重 | **不在本仓库**。本项目自训权重也不在（`.gitignore` 排除 `*.onnx`、`*.zip`） |
| `workspace2.zip` 原始包（211 个原文件） | **不在本仓库**。整包 SHA256 `fd0bb8b3…`，见 `tools/upstream_reproduction03.md` |
| 大型 evidence、逐 native 轨迹、录像 | **不在本仓库**（`.gitignore` 排除 `/evidence/`）。唯一例外见第 4 节 |

## 3. 以最小 diff 上下文形式**逐字摘录**的上游

`tools/upstream_reproduction03.patch` 是对本机改动的 unified diff。unified diff 必须携带原文上下文行才能可靠应用，因此它包含上游文件的少量逐字片段（共 219 行上下文、32 个 hunk，对应 188 行本项目新增）。按来源分三类：

| 上游 | 许可证 | 逐字上下文 | 文件 |
|---|---|---:|---|
| `mujoco_ros2_control` | **MIT**，Copyright (c) 2025 Sangtaek Lee | 100 行 | 该包的 6 个源/构建文件 |
| MuJoCo `simulate` 示例 | **Apache-2.0**，Copyright 2021 DeepMind Technologies Limited | 84 行 | `mujoco_sim_ros2/src/{simulate.h,simulate.cc,main.cc}` |
| `rl_controller`、`d1_description` | 见下 | 35 行 | 3 个文件 |

**前两类**的许可证均要求在再分发时保留版权与许可声明，本节即为该声明。Apache-2.0 全文见 <http://www.apache.org/licenses/LICENSE-2.0>；MIT 条文与本仓库 [LICENSE](LICENSE) 相同，版权归上述作者所有，**不归本仓库作者**。

**第三类**（`controller/rl_controller/launch/sim_mujoco.launch.py`、`urdfs/d1_description/CMakeLists.txt`、`urdfs/d1_description/package.xml`）的 `package.xml` 仍写着 `TODO: License declaration`，没有正式的许可证声明。这三个文件及 D1 机器人模型属于本仓库作者与合作者**共同开发**的工作，在合作者知情的情况下使用；此处摘录的 35 行上下文是应用补丁所需的最小量。`tools/upstream_reproduction03.md` 另以散文完整描述了这三处改动（只列本项目自己新增的行与定位锚点），便于不依赖 diff 手工施加。

**无论哪一类，机器人模型、mesh、URDF/xacro、原配 ONNX 策略与训练权重都不在本仓库内**（见第 2 节）。

## 4. 唯一的 evidence 例外

`evidence/interface_audit_03/golden_run_01/schema.json`（4,828 B）—— 本项目自写的受审计观测/动作接口 schema，被每个训练配置以 `audited_interface_schema` 引用，属复现必需的小型配置。首次提交于 `a6779a4`。**不含权重、不含同事资产。**

## 5. 运行期依赖（不随仓库分发）

Ubuntu 22.04 / ROS 2 Humble、MuJoCo 3.3.0（Apache-2.0）、C++ ONNX Runtime 1.10.0（MIT）、PyTorch、Stable-Baselines3、Gymnasium、NumPy、ONNX、onnxruntime、psutil。各自许可证以其分发为准；除第 7 节的方法改编外，本仓库不分发这些依赖的源码或二进制。清单见 `training/DEPENDENCIES03.md`。

## 6. 成绩归属边界

- 原项目 GUI 入口使用**同事的策略**；本项目自训策略的独立 C++ 验证与原版演示**分开记录，不可混算**。
- 旧仓库 [wheel-legged-control-lab](https://github.com/LYHrmer/wheel-legged-control-lab) 是**独立的另一条 git 历史**（根提交不同、无共同祖先）。两套模型的成绩不可拼接。该仓库中 77 份冻结文件与用户结果在本项目中仅作**只读哈希校验**对象，本项目从未修改它们。

发现任何归属或许可问题，请提 issue；本仓库会立即移除相应内容。

## 7. Stable-Baselines3 PPO 更新方法的改编

`training/d1_rl/velocity_policy15.py` 的 `AuxiliaryVelocityPPO.train()` 改编自 [Stable-Baselines3 v2.7.0 的 PPO.train](https://github.com/DLR-RM/stable-baselines3/blob/v2.7.0/stable_baselines3/ppo/ppo.py)，Copyright (c) 2019 Antonin Raffin，MIT 许可。本地仅支持声明的 CPU/连续动作/字典观测合同，并在 PPO 总损失加入速度监督 MSE；并非完全自写算法实现。完整版权与许可条文保存在 [licenses/stable-baselines3-MIT.txt](licenses/stable-baselines3-MIT.txt)。
