# 原项目本机改动的独立补丁

`upstream_reproduction03.patch` 是从用户提供的原始 `workspace2.zip` 与本机提取树逐文件比较生成的 **unified text diff**。原 ZIP SHA256 为 `fd0bb8b361a4b63efbe758242d9d6a00d290c52e0dd6e3eed961595c7eacf6a4`，身份来自 `source_archive_receipt.json`，生成时已重新计算 ZIP 整体 SHA，并核对其中 211 个原始文件的逐项 SHA。原项目代码和模型仍归原提供方；这个补丁只表达本机改动，不把原源码、机器人、网格或 ONNX 权重作为新作品重新发布。本文件和补丁目前仅在本地，**未发布，也未应用到当前源码**。

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
