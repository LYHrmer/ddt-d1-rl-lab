# D1 C++34 离屏演示录制

35 只新增独立展示入口，使用已完成 25 random 的同一 ONNX、已完成 34 的同一 C++ 二进制和原始 03d 环境。它不执行一次性的 `run_cpp_reference34.py`，不修改默认策略，不重写冻结源，不产生训练、候选排名或新的资格结论。

旧 `trace.npz` 只存基座位置姿态等指标；`raw_actor627` 中四个轮关节位置被观测接口置零，且缺少最后一次控制后的完整观测。因此不能从它们严格恢复全体关节的同步姿态。新增捕获仅在原 `reset()`、`step()` 返回后复制 `qpos`、`qvel`、`data.time`，不额外积分、不改变动作。

预算固定为 `forward_stop` 和 `turn_stop`、0 延迟、reset seed 1001，每场 600 次 C++ 推理、6000 native steps、3000 PD updates，共 1200/12000/6000。每场保存 601 组状态，含 0 和 12 秒边界；构造、reset 和观测的 `mj_forward` 单独计数。这个 seed 只用于演示，不用于排序。

`tools35/record_cpp_showcase35.py --check-only` 可只读核验来源。执行前检查已完成 34 报告、全部源与环境来源、构建输入、C++ 二进制、两场归档 trace/sidecar，以及完整 25 checkpoint/export 链。任何来源改变直接拒绝，不重新绑定清单。

实际执行由调用方放入独立 systemd 用户单元，设置 Nice=10、MemoryHigh=2G、MemoryMax=3G、MemorySwapMax=0，并将以下变量都设为 1：`OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`、`NUMEXPR_NUM_THREADS`、`VECLIB_MAXIMUM_THREADS`、`LP_NUM_THREADS`。`--owned-scope` 必须是该单元的真实 cgroup 路径。内部准入要求有效可用内存至少 4 GiB，运行每秒检查至少保留 2 GiB，最长 600 秒；不操作其他进程。

```bash
python3 -B tools35/record_cpp_showcase35.py --output /new/recording \
  --owned-scope /user.slice/user-1000.slice/user@1000.service/app.slice/OWNED.service

MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless \
  python3 -B tools35/render_showcase35.py \
  --recording /new/recording/forward_stop --output /new/render_forward \
  --owned-scope /user.slice/user-1000.slice/user@1000.service/app.slice/OWNED-RENDER.service
```

录制与渲染在不同单元中串行运行。每个输出目录必须不存在；失败、partial、PNG 和报告全部保留，入口不覆盖、不续跑。每场保存 `model.mjb`、`states.npz`、逐状态刷新落盘的 `states.jsonl`、`actor_io.npz`、`trace.npz`、`manifest.json`、`report.json`，完整父记录和单场记录都通过后才允许渲染。SIGTERM 转为异常后关闭本任务 C++ actor，并保留完整控制边界以前的状态；若中断发生于 reset/step 内，实际总计标为未知，另存已完成前缀与原环境的 attempted counters，不把 partial 报成零或完整候选。

渲染使用 CPU 软件 EGL（`LIBGL_ALWAYS_SOFTWARE=1`、`EGL_PLATFORM=surfaceless`）、512×288、20 fps，固定最近邻选择状态（最大采样时间偏移 10 ms），保留从开始到最终停车的 241 个 PNG。GIF 使用同一 96 色全局调色板和固定 4×4 Bayer 抖动，完整覆盖源记录的 12 秒，最终状态另保持 50 ms，总播放 12.05 秒。渲染副本中将地面设为无反光浅灰，加入世界固定 0.5 m 网格以显示位移；网格只存在于 `mjvScene`，无碰撞、不进入物理模型。只调用 `mj_forward`，没有策略调用或 `mj_step`。超过 5,000,000 字节即记失败，不截去失败段或裁短时域；调色板不改变保存状态或 PNG。

`manifest.json` 绑定模型、状态、动作输入输出与 trace 的 SHA；渲染报告另记录 PNG、GIF SHA、尺寸、相机、采样索引和实际调用计数。README/PORTFOLIO 只能将 GIF 标为该策略的仿真演示；性能数字继续引用已归档的 [34 结果](cpp_reference34_results_20261018.md)。

## 2026-10-22 实际交付

零物理重放完成 1,200 次 C++ 请求，与保存动作逐位相同，最大差 0。独立录制两场均满 600 拍，共 1,200 policy / 12,000 native / 6,000 PD；包括构造、reset 和观测在内共 1,206 次 `mj_forward`，reset 2 次。渲染另执行 482 次 `mj_forward`，策略调用与积分步均为 0，保留 482 张完整 PNG。

| GIF | 字节 | 帧数 | 播放时长 |
|---|---:|---:|---:|
| [forward_stop](assets/d1_cpp34_forward_stop_20261022.gif) | 551,003 | 241 | 12.05 s |
| [turn_stop](assets/d1_cpp34_turn_stop_20261022.gif) | 1,006,299 | 241 | 12.05 s |

四个重放/录制/渲染单元均完整退出、进程组清空，记录的 memory high/max/OOM 事件均为 0；最低有效可用内存 5.392 GiB，最大组峰值 0.877 GiB。逐阶段时间、来源 SHA、模型与状态 SHA、资源回执见[紧凑报告](../reports/showcase35_results_20261022.json)。新增训练为 0，不重跑 34 的一次性资格矩阵，不产生新的控制性能结论。

**展示口径勘误：** 旧 README/PORTFOLIO/部署图曾把其他路径的 100 Hz、5 个物理子步和 20 ms 截止混写为 34 的属性。34 的冻结环境实际是策略 50 Hz、PD 250 Hz、物理 500 Hz；20 ms 是策略周期，`CppReferenceActor` 的默认单次请求超时为 2 s。当前展示页已纠正，原代码与历史回执未改。33/34 已通过其固定 formal/holdout 矩阵，仍保留综合 J 相对 25 变差 5.89%、默认未替换、ROS2 适配及实机资格未验证的边界。
