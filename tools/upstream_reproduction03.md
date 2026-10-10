# 原项目本机改动的独立补丁

`upstream_reproduction03.patch` 是从用户提供的原始 `workspace2.zip` 与本机提取树逐文件比较生成的 **unified text diff**。原 ZIP SHA256 为 `fd0bb8b361a4b63efbe758242d9d6a00d290c52e0dd6e3eed961595c7eacf6a4`，身份来自 `source_archive_receipt.json`，生成时已重新计算 ZIP 整体 SHA，并核对其中 211 个原始文件的逐项 SHA。原项目代码和模型仍归原提供方；这个补丁只表达本机改动，不把原源码、机器人、网格或 ONNX 权重作为新作品重新发布。生成时本文件和补丁仅在本地，未发布；没有将补丁再次应用到当前源码。

2026-10-09发布补充：该文本差异随自写源码上传至独立私有仓库 [ddt-d1-rl-lab](https://github.com/LYHrmer/ddt-d1-rl-lab)。原同事模型、mesh、权重和完整原文件仍未包含。生成回执与原补丁SHA保持不变。

补丁 SHA256：`366572b3f2fdcccee5da6a2ca826143abe2ad3bb00c3781df554a84ab350e85a`。它改变 12 个原项目文本文件，约 196 行增加、13 行删除；没有二进制 diff。清单如下：

| 路径（相对 `src/ddt_ros2_control/`） | 本机改动用途 |
|---|---|
| `controller/rl_controller/launch/sim_mujoco.launch.py` | 选择场景、配置有界 native 证据路径与上限 |
| `simulation/mujoco_bridge/mujoco_ros2_control/CMakeLists.txt` | 加入证据记录器与几何消息依赖 |
| `simulation/mujoco_bridge/mujoco_ros2_control/src/mujoco_ros2_control_plugin.cpp` | 发布同时间基准的基座位姿/速度并记录 native 步 |
| `simulation/mujoco_bridge/mujoco_ros2_control/src/mujoco_ros2_control.cpp` | 实际控制更新计数 |
| `simulation/mujoco_bridge/mujoco_ros2_control/include/mujoco_ros2_control/mujoco_ros2_control.hpp` | 上述计数接口 |
| `simulation/mujoco_bridge/mujoco_ros2_control/include/mujoco_ros2_control/mujoco_ros2_control_plugin.hpp` | 观测发布与证据记录器声明 |
| `simulation/mujoco_bridge/mujoco_ros2_control/package.xml` | 消息依赖声明 |
| `simulation/mujoco_bridge/mujoco_sim_ros2/src/simulate.h` | viewer 的有界步数接口 |
| `simulation/mujoco_bridge/mujoco_sim_ros2/src/simulate.cc` | viewer 有界步数实现 |
| `simulation/mujoco_bridge/mujoco_sim_ros2/src/main.cc` | 物理步计数/安全停止入口 |
| `urdfs/d1_description/CMakeLists.txt` | 安装自建 MuJoCo 场景所需资源 |
| `urdfs/d1_description/package.xml` | 描述包依赖声明 |

补丁中的 `step_evidence.hpp/.cpp` 是本轮**新增**文件，不属于原 ZIP 的 211 项，因此作为两个单独的新源码文件列入本轮源码包，不假装成原项目原件。地形 XML 和高度图也单独列入。原机器人 XML、STL、URDF、原 ONNX 策略、第三方依赖和大型 evidence 都不在源码包中；仅拿源码包不能启动机器人。

在**另一个干净工作目录**重建时，先从原提供方取得并校验 `workspace2.zip`，按 `source_archive_receipt.json` 的相对布局把原包 `workspace2/DDT_Lab/ddt_ros2_control/` 放到工作区的 `src/ddt_ros2_control/`。再把本轮源码包按其路径放入该工作区，先预检、经人工复核后应用补丁：

```bash
rtk proxy sha256sum /path/to/workspace2.zip
rtk proxy patch --dry-run -p1 -i tools/upstream_reproduction03.patch
rtk proxy patch -p1 -i tools/upstream_reproduction03.patch
```

这些命令需在目标工作区根目录运行，不能在当前已改动的树上重复应用。此补丁已在临时目录对从原 ZIP 提取的 12 个原件试应用，结果逐文件 SHA 与当前工作区对应文件一致；这个检查不等于编译、物理或策略能力验证。补丁生成器是 `tools/generate_upstream_patch03.py`：不加 `--output` 只读列清单和计划 SHA；显式 `--output` 才写一个全新 patch 路径，并拒绝任何二进制变更。

## 2026-10-11 公开边界核对与手工施加说明

公开前逐项核对了完整补丁里逐字上下文行的来源（unified diff 必须携带原文上下文才能可靠应用）：

| 上游 | 许可证 | 逐字上下文 | 可否再分发 |
|---|---|---:|---|
| `mujoco_ros2_control`（6 文件） | **MIT**, Copyright (c) 2025 Sangtaek Lee | 100 行 | 可，须保留版权与许可声明 |
| MuJoCo `simulate` 示例（3 文件） | **Apache-2.0**, Copyright 2021 DeepMind Technologies Limited | 84 行 | 可，须保留声明 |
| `rl_controller` + `d1_description`（3 文件） | **未声明**（`package.xml` 为 `TODO: License declaration`） | 35 行 | 与合作者共同开发，知情使用 |

前两类的版权与许可声明已补入 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)（这是其许可证的再分发要求）。第三类连同 D1 机器人模型属于本仓库作者与合作者共同开发的工作。

发布物仍是**一份**完整补丁 `tools/upstream_reproduction03.patch`，SHA256 `366572b3f2fdcccee5da6a2ca826143abe2ad3bb00c3781df554a84ab350e85a` 与本文上方记录一致，未重新生成、未改写。无论如何，**机器人模型、mesh、URDF/xacro、原 ONNX 策略与训练权重都不在仓库内**。

### 那 3 个文件：改动的散文描述（可不依赖 diff 手工施加）

下面只给出**本项目自己新增的行**与定位锚点，不含上游原文上下文。在取得原 `workspace2.zip` 后按此手工施加，效果与补丁中对应 hunk 等价；需要核对改动意图而不想读 diff 时也更直观。

**`controller/rl_controller/launch/sim_mujoco.launch.py`** —— 让场景与证据参数可由命令行指定，替换原先硬编码的场景路径。

1. 在 `launch_setup(context, ...)` 内、与 `robot_name` / `ns` 等同一组局部变量之后，新增三行读取：

   ```python
   scene = LaunchConfiguration("scene").perform(context)
   evidence_path = LaunchConfiguration("evidence_path").perform(context)
   evidence_max_native_steps = int(LaunchConfiguration("evidence_max_native_steps").perform(context))
   ```

2. 在 `mujoco_sim` 节点的 `parameters=[...]` 列表里，把原来的 `{"model_file": "mujoco/scene.xml"}` 一行**替换**为三行：

   ```python
   {"model_file": "mujoco/" + scene},
   {"evidence_path": evidence_path},
   {"evidence_max_native_steps": evidence_max_native_steps},
   ```

3. 在声明 launch 参数的 `declared_arguments` 段落里追加三个参数（`scene` 沿用该文件既有的 `declared_arguments.append(launch.actions.DeclareLaunchArgument(...))` 多行写法，另两个用紧凑写法）：

   ```python
   declared_arguments.append(
       launch.actions.DeclareLaunchArgument(
           "scene",
           default_value="scene.xml",
           description="MuJoCo scene filename inside the robot package's mujoco directory",
       )
   )
   declared_arguments.append(launch.actions.DeclareLaunchArgument(
       "evidence_path", default_value="",
       description="Optional exclusive native-step JSONL evidence path"))
   declared_arguments.append(launch.actions.DeclareLaunchArgument(
       "evidence_max_native_steps", default_value="0",
       description="Optional physical-step limit; zero preserves interactive playback"))
   ```

**`urdfs/d1_description/CMakeLists.txt`** —— 在 `install(DIRECTORY ...)` 的目录列表里，把被注释掉的 `# meshes` 一行改为生效的 `meshes`，使自建场景引用的资源随包安装。

**`urdfs/d1_description/package.xml`** —— 在依赖段落末尾追加两行，声明共享材质/宏所在的描述包：

```xml
<!-- robot.xacro includes the shared D1H materials and robot macros. -->
<exec_depend         condition="$ROS_VERSION == 2">d1h_description</exec_depend>
```

本次核对不改变补丁内容、任何已记录的 SHA、回执或实验结果；施加命令仍是上文那两条 `--dry-run` 后再应用。
