# 独立 C++ actor 闭环 19

本轮代码准备与测试只使用假策略、假进程和保存数组，不运行 ORT session、机器人、积分或训练。18 的原输入与单次执行目录继续冻结。

19 只有在 18 整轮 `COMPLETE`、重新计算的胜出 arm 与外层记录相同，且该 arm 的原 36 场及另 36 场 holdout 全部通过时才允许执行。准入会重核阶段产物 SHA、实际计数、原指标、最终 checkpoint、完整六 Gemm 导出及本地 C++ 构建回执。18 没有胜出策略时，19 不运行。

```bash
rtk proxy python3 -B tools19/run_cpp_closedloop19.py
```

默认只打印计划，不导入推理库、不创建输出。新 19 源码提交后，根代理在独立资源窗口中才能明确执行：

```bash
rtk proxy python3 -B tools19/run_cpp_closedloop19.py --execute
```

唯一新目录是 `evidence/cpp_closedloop19/execution_01`，拒绝覆盖、重试与续跑。新预算上限为 21,600 policy transitions、216,000 native steps、108,000 PD updates；额外 64 个黄金 fixture 只执行神经推理，不增加物理积分。C++ 请求总上限是 21,664。

先验证导出中原有 64 个 fixture：当前 57 维观测加十个 previous-only 57 维历史帧，比较 C++、保存的 Torch/Python ORT 动作及当前 Python ORT，绝对误差保持 `1e-5`。随后运行原 3 任务 × 4 延迟 × 3 reset seeds 的 36 场。每一拍的 C++ 动作是唯一的物理驱动；Python ORT 只读取完全相同的实际 627 维输入作为数值 oracle，不另跑一套环境。没有额外训练。

每例保存原正式 NPZ、原指标和独立推理 NPZ，后者含实际 627 维输入、C++/Python ORT 16 维输出及误差。worker 与父进程都从文件重核数值记录，要求 C++ 输出与原轨迹 `issued_action` 逐元素相等，并重新计算原指标。完整矩阵采齐表示工程完成；策略早倒仍按原指标记失败并收齐其余场，不补拍。数值分歧、资源、来源或工程异常立即停止后续工作。资格要求继续使用原门槛，不能由 `COMPLETE` 或单纯 exit 0 推导。

CPU/BLAS/ORT 均单线程；启动有效 RAM 至少 6 GiB，运行保留 3 GiB，1 秒采样，不宣称内存独占。整体 wall 上限 900 秒，worker 840 秒、外层 880 秒停止并保留有界清理时间。外层只清理自己启动的 worker 及其后代，结合 Linux birth tick 和 subreaper 处理独立 session 的 C++ 子进程。C++ 必须真实返回 `CLOSED`、Run 计数闭合且清理完成。

墙钟与资源轮询不能抢占正在阻塞的单个原生调用；外层监督可终止自己启动的进程。迟到的正常退出、清理后超时、没有拿到子进程句柄的启动异常同样记为未完成。已有完整或中途 native/PD 计数保留；构造期间失去计数则写未知，不补成零。

本机 `runtime19/build/` 保存一次 O1 编译的同一二进制、完整编译命令、依赖 SHA、耗时和 RSS 回执；构建产物不进 Git。编译成功、假测试和已有独立 C++64 数值结果均不能替代本轮实际闭环证据，也不授予地形、实机或 sim-to-real 资格。
