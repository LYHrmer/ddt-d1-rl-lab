# 本机 D1 RL / MuJoCo 复现

工作空间：`/home/lyh/ddt_ros2_ws`。机器人模型来自同事 ZIP，部署使用 ROS2 Humble + ONNX Runtime 1.10.0 + 源码编译的 MuJoCo 3.3.0。最初阶段复现同事的现成 ONNX；2026-10-09 已另建纯 D1 PPO 训练、独立 C++ 部署和评测链路，从随机初始化训练自己的策略。无需 Isaac Sim。下面原版交互入口仍使用同事策略，两条路线的成绩分开记录。

## 丰富场地与自训教程

[独立复现教程](docs/tutorial/00_复现教程.md)包含场地操作、观测/动作/控制时基、训练、导出及评测。新增分区场地包括20米加速道、5/15/30/50毫米单障碍、3/6/10度坡道、阶梯、起伏路、碎板、绕障和窄门。12项静态检查通过；地形通关仍待实测。

启动丰富场地仍使用下文的原版键盘控制：

```bash
rtk proxy /home/lyh/ddt_ros2_ws/run_d1_training_course.sh
```

自己的训练路径为 `training/d1_rl` → `tools/train_d1_ppo.py` → `tools/export_d1_ppo.py` → `runtime03`。03b、03c、03d三次有界pilot均完成seed101、1,048,576条经验的随机初始化训练和ONNX/C++一致性核验。最终03d三条任务无跌倒，后退停车通过全部门槛；前进跟踪误差约0.003m/s但停车位移8.1cm超限，转向仍未达标，因此整体仅1/3通过。三轮额度已用完，正式九次训练未放行。[实际结果与证据](docs/career_execution_results_03.md)分别记录训练完成、数值一致性与控制门槛。低速范围±0.3m/s，1.5m/s、侧移、跳跃和地形通关均未验证。

独立C++也复现了[倒退停车通过的回放](evidence/career_contract_03/pilot03d_reverse_video_01/runtime_trace.mp4)与[前进停车超限的回放](evidence/career_contract_03/pilot03d_forward_video_02/runtime_trace.mp4)。两段均由已有实测状态离线渲染，未增加物理试次；视频显示单任务资格，不能代表整套策略通过。ROS2命令超时检查因发布器环境导入错误未启动仿真，仍未验证。

本轮已清理48.34MiB可再生编译对象与Python缓存，保留模型、权重、运行库和全部证据；历史清理回收量另列，见[清理说明](docs/storage_cleanup_03.md)。

本目录已建立独立本地Git，尚无remote；同事源码、模型/权重、依赖与大轨迹不进入源码历史。小型自写源码包依赖原 `workspace2.zip`，路径与依赖限制见 [DEPENDENCIES03](training/DEPENDENCIES03.md)。旧 `wheel-legged-control-lab` 仓库保持独立。

## 启动

终端一（原生 MuJoCo 窗口，视频外观近似场地）：

```bash
rtk proxy /home/lyh/ddt_ros2_ws/run_d1_mujoco.sh scene:=scene_video_reconstruction.xml
```

终端二（保持焦点在此终端输入机器人控制键）：

```bash
rtk proxy /home/lyh/ddt_ros2_ws/run_d1_keyboard.sh
```

先按 `7`，等约 10 秒站起，再按 `1` 进入推荐的 `flat_lab.onnx` RL。只打开 viewer 时初始为 idle。`0` 对应另一套原版 `flat.onnx`，本机实测存在零速后退和较弱跟踪，因此当前推荐 `1`。两者各用自己的原配配置；换策略请完整重启仿真后重新站起，不交叉混用参数。

| 键（键盘终端内） | 含义 |
|---|---|
| `7` | 从 idle 执行站起 |
| `1` | 从站起状态进入推荐的 flat_lab.onnx RL |
| `0` | 从站起状态进入 flat.onnx，保留用于对照 |
| `w` / `s` | 增加 / 减少前向速度请求 |
| `a` / `d` | 增加 / 减少转向角速度请求 |
| `←` / `→` | 增加 / 减少侧向速度请求；本次未验证有效持续侧移 |
| `r` | 将全部速度请求清零 |
| `8` | 退出 RL、执行蹲下 |

这些是原包的按键映射。速度请求按次累加，松键不会自动归零；按 `r` 停令。键盘显示的通用 `5: jump` 不表示 D1 已配置或验证跳跃。`6` 不能作为当前 RL 状态的退出键。退出前先 `r`，再 `8`，最后在两个终端分别 Ctrl+C。

键盘终端中的 `state machine now` 实际显示最近请求的状态，不是控制器回传状态；新开键盘终端可能显示 idle，而已运行的仿真仍在 RL。实际状态以 launch 日志的 `FSM state changed` 为准。键盘节点的启动、`r` 零速和 Ctrl+C 退出已实际检查。

本工作空间脚本统一使用 `ROS_DOMAIN_ID=84`、`ROS_LOCALHOST_ONLY=1`。不要同时启动两套仿真；也不要让键盘节点与自动测试程序同时发送控制命令。

需要原 ZIP 场景时，终端一改用：

```bash
rtk proxy /home/lyh/ddt_ros2_ws/run_d1_mujoco.sh
```

## 画面与场地

视频完整场地没有随 ZIP 提供，用户确认没有额外资源。新增场地按参考帧估计生成，包括彩块、台阶、平台、长坡、圆柱、碎板和两个真实碰撞高度场；尺寸和摆放不等于同事原始参数。原 `scene.xml`、`robot.xml`、STL 和策略权重保留。

MuJoCo 窗口内 `Tab` 隐藏/显示左面板，`Shift+Tab` 隐藏/显示右面板；可用鼠标拖动和滚轮调整相机，或在 Option → Font 调整字体。机器人控制键在第二个终端输入。

保持 viewer 实时速率；原推理线程按墙钟 50 Hz 运行，修改仿真播放速度会改变控制时基。切换场地请结束并重新启动 launch，避免原桥插件在 viewer Reload 后引用旧模型。

## 已有证据与边界

- MuJoCo 3.3.0 源码构建、D1/D1H 描述包、RL 控制器、键盘节点、MuJoCo 桥接全部编译成功。
- 原生 viewer 实际启动，两个 ONNX 载入；首测日志明确记录 `idle → transform_up → rl_flat`。
- 原场景有界测试 23.515 秒完成：站起 10 秒、RL 零速 5 秒、前进 0.2 m/s 请求 3 秒、零速 5 秒；IMU 和关节各 5841 条，RL 阶段横滚/俯仰绝对值均小于 0.0195 rad。见 `evidence/rl_smoke_stock_01.json`、`evidence/stock_rl_stand.png`、`logs/stock_launch_01.log`。
- 上述首测没有世界位移记录，不能据此认证速度跟踪、停车精度或可靠越障。场地存在也不代表策略已通过每种地形。
- 新增只读世界位姿/速度发布后，在近似场地空旷区分别完整重启并运行两套原配策略。原 `flat` 在 0.2 m/s 请求下末 1 秒平均 vx 为 0.1063 m/s，最终零速阶段仍为 -0.0661 m/s，存在持续后退偏置。
- 推荐的 `flat_lab` 完成同一 45 秒指令序列，总计 45.583 秒。前进请求 0.2 m/s：观测窗口机体前向位移 0.5400 m，末 1 秒平均 vx=0.1847 m/s；随后零速阶段末 vx=0.0010 m/s，最终零速末 vx=-0.0007 m/s。左右转向各请求 ±0.2 rad/s、3 秒，实测 yaw 变化 +0.6282/-0.5928 rad。仿真/墙钟比例约 1；日志确认进入 `rl_flat_lab`。这是一次短时低速验证，不是全工况认证。
- `flat_lab` 侧移请求 ±0.1 m/s、3 秒，只得到约 +0.0099/-0.0537 m 横向位移，末段侧向速度接近零，且负向段伴随约 0.105 rad 航向变化；持续侧移能力未通过。本轮没有验证楼梯、坡道、粗糙地形跨越或 1.5 m/s 速度。
- 两策略原始证据：`evidence/motion_video_course_01.json`、`evidence/motion_video_course_flat_lab_01.json`。真值来自 `base_link` 的世界位姿及原有 `frame_vel` 传感器，25 Hz 发布、同时间戳配对；不修改策略输入或控制输出。
- 转向仍伴随小幅平移，负转向后的零速段又偏航约 0.0864 rad，不能称为精确原地转向。实际 Astra ultra 独立复核见 [策略对比说明](evidence/policy_comparison_20261008.md)；2026-10-08 交付时进程的库和文件身份见 `evidence/runtime_delivery_20261008.json`，其中旧 PID 不是当前运行状态。
- 配置写 400 Hz，但当前 2 ms 物理步长下观测/控制实测约 250 Hz。策略、控制时基和原有力矩限幅问题未在复现过程中顺手修改。

源包逐文件身份见 `source_archive_receipt.json`；新增场地估计参数见 `src/ddt_ros2_control/urdfs/d1_description/mujoco/video_course_assets/layout_estimated.json`。安装使用符号链接，`build/` 和 MuJoCo 构建依赖仍被运行/重建引用，不要直接删除。

## 2026-10-09 新平面对照与台阶前置资格

已完成实际 Astra ultra 制定的合同 02 的唯一新平面对照，真实 15 mm 台阶的静态几何/碰撞预检通过。平面对照执行了 15,164 个 native 步、7,581 次实际控制更新，启动至关闭共 35.254 秒；证据记录逐步连续、ROS 与 native 真值一致，输入哈希不变。

动作序列完成，但资格为 **FAIL**：0.2 m/s 请求的稳态实测约 0.18452 m/s，直行世界 x 增加 1.82709 m，停车末段前向绝对速度约 0.000906 m/s；世界横向偏移峰值 0.124465 m 超出预定 0.10 m 门槛。其他检查通过。起始 yaw 约 3.1°，必须分开分析初始航向造成的世界轨迹倾斜和实际机体侧向误差，不能把全部偏移称为侧滑。

按原合同，到此结束；两次台阶试次均未启动，不改阈值、不重跑、不宣布台阶已通过。详见 [合同结果](evidence/step15_qualification_02_summary.md) 和 `evidence/step_contract_02/plane_01/evaluation.json`。当前没有保留运行中的 viewer，需要交互时使用上方启动命令。

本轮新增记录器默认关闭，仅专用验证显式启用；记录器、计步上限与退出清理均已编译并在上述一次试次中使用。6 项离线评测测试、4 项守护测试通过；未增加训练。求职方向与下一步优先级见 [项目路线](CAREER_ROADMAP.md)。

## 磁盘清理

已回收 1,082,970,112 字节（约 1.01 GiB）：保留全部对象的 Git 压缩、15 个经 SHA 核对的失败下载残片和两个标准测试/lint 缓存。77 份冻结输入、Git refs/reflogs/status、用户结果及完整历史归档保持不变。回执：`evidence/disk_cleanup_20261008.json`。

第二轮另外回收 855,945,216 字节（约 0.797 GiB）：对 127 个单链接 MJB 做内容不变的稀疏存储优化，并清理 306 个可再生且未跟踪的源码字节码缓存。MJB 的路径、内容 SHA、长度、inode、权限和 mtime 不变，物理占用及 ctime 改变；没有删除历史模型、实验记录或完整归档。两轮总回收 1,938,915,328 字节（约 **1.806 GiB**）；这是清理操作释放量，不抵扣后续新建证据。第二轮回执：`evidence/disk_cleanup_round2_20261008.json`。
