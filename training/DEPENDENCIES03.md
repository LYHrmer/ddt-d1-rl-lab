# 本轮依赖身份

这是本机实际运行栈的版本记录，不能代替完整系统镜像。

| 组件 | 本机版本 / 位置 |
|---|---|
| Ubuntu | 22.04，ROS2 Humble |
| Python | `/usr/bin/python3`，3.10 |
| MuJoCo Python | 3.3.0，项目 `.python-deps` |
| MuJoCo C++ | 源码3.3.0，项目 `install/mujoco` |
| Gymnasium | 1.2.0，项目 `.python-deps` |
| stable-baselines3 | 2.7.0，项目 `.python-deps` |
| ONNX | 1.18.0，项目 `.python-deps` |
| PyTorch | 2.12.1+cu130，本机用户安装；本轮PPO使用CPU |
| ONNX Runtime Python | 1.23.2，本机用户安装 |
| ONNX Runtime C++ | 1.10.0，`/usr/lib` 与 `/usr/include` |
| NumPy / psutil | 2.2.6 / 5.9.0 |

运行每次 `protocol.json` 的版本和源码SHA是对应实验的依据。安装与版本检查命令见[教程03](../docs/tutorial/03_训练导出与评测.md)；C++系统依赖与构建见[README_LOCAL](../README_LOCAL.md)。

当前shell入口、部分CMake路径和runtime schema使用 `/home/lyh/ddt_ros2_ws`。源码包不包含第三方模型、STL、同事ONNX、CUDA/PyTorch依赖或已安装系统库；需要原 `workspace2.zip` 和安装步骤才能复现。更换机器、用户名或安装前缀时须调整路径并重新验证身份，不能直接沿用本机的已通过回执。
