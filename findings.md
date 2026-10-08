# 实际发现

- 用户已确认录屏是 MuJoCo 原生 viewer；直接复用现成策略，不需要 Isaac Sim 或训练。
- 压缩包包含 305 个 ddt_ros2_control 子树条目，无 AGENTS.md/CLAUDE.md。
- 当前实际环境已有 ROS2 Humble ros-base、ros2-controllers、colcon 和多数开发库；ros2-control 元包、libglfw3-dev 未安装。与此前环境快照不同，复用现有安装并补齐缺项。
- /opt/ros/humble/setup.bash 存在；DISPLAY=:1，XAUTHORITY=/run/user/1000/gdm/Xauthority。
- /usr/lib 和 /usr/local/lib 未发现 libonnxruntime。
- README 要求 MuJoCo 3.3.0、ONNX Runtime 1.10.0，并明确 D1 mesh 复制与 CMake 安装项修复。

- 已解压 211 个文件（68,856,605 字节）到 src/ddt_ros2_control，原始逐文件 SHA256 见 source_archive_receipt.json。
- D1H meshes 已复制到 D1，D1 CMake meshes 安装项已启用。
- MuJoCo 3.3.0 已 clone，commit 8de9b4e4e373caf01035c6c13efd363347d89032；colcon 正确认出通用 CMake mujoco 包。
- ONNX Runtime 1.10.0 官方 tgz 已下载解压，待系统安装。
- 自动打开 gnome-terminal 失败（Failed to get screen），已请用户在本机终端执行准备好的 sudo 安装脚本。

- 并行核对：旧 Git 3d771c8 为非RL功能/显示参考；并非录屏完整场地，且历史人工体验未通过。旧仓库保持只读。
- 原装 scene.xml 缺少视频的5个彩色块、长坡/平台、圆柱、三类粗糙地块；已请求用户给额外场地资源路径。参考帧保存 evidence/reference/。
- d1_description xacro 引用 d1h_description，后续需显式编译后者。两ONNX元数据匹配57维当前观测、10×57历史、16动作。默认FSM为idle，站起后才能进入rl_0/rl_1。
- 首次 MuJoCo 配置因沙箱 CMake 子进程无法解析 github.com 失败；已按权限机制在沙箱外重启同一构建。

- 2026-10-09 更新：依赖、编译和原配策略复现均已完成。当前推荐flat_lab；世界真值与逐native接触证据已加入，均不进入原策略观测。
- 合同02静态15mm单台阶几何合格；唯一平面对照世界横向偏移0.124465m超过0.10m，故FAIL并停止台阶试次。实际步数15164、控制更新7581，未新增训练。
- 该轨迹起始yaw3.095°，固定起始航向横向误差25.625mm；57维原观测没有世界y或目标航向误差。机体系速度跟踪与世界路径跟踪应分层设计，不能把世界偏移直接归咎为RL侧滑或奖励问题。
- 预直行检查与阶段stamp之间有约0.10s旧样本标记延迟，下一合同需修复发令/采样时间标记；原证据和FAIL不变。
- 求职路线经实际Astra ultra复核，优先补齐自己的训练—导出—部署闭环、一项等预算多seed改进，以及离线接口一致性/运行时故障验证；具体见CAREER_ROADMAP.md。
