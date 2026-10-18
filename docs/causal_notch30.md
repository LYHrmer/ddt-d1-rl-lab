# D1 30：固定策略的 5 Hz 因果动作陷波实验

原 25 random 已完成 36 个任务，33 个通过。剩余三个 60 ms 转向任务的负转向平均欠幅仍低于 0.08 rad/s，但波动使 MAE 超标。29 的固定策略约束减少波动，同时增大平均欠幅，最终 31/36。下一项保留原 25 完整权重，只检验动作中的窄带波动是否能在闭环中消除。

唯一候选：50 Hz 控制下，中心 5 Hz、Q=2 的二阶因果陷波器，统一用于全部 16 个动作和全部任务。使用 [W3C Audio EQ Cookbook 的 notch 系数](https://www.w3.org/TR/audio-eq-cookbook/#notch)，DC 与 Nyquist 增益为 1；这不构成机器人闭环稳定性的证明。首动作初始化恒定前史，每个 episode 独立清空；float64 线性递推状态保留未裁剪值，输出裁至原动作边界后转 float32，再进入原环境 FIFO。原 actor 的动作历史记录实际发出的滤波动作。

不训练、不改奖励、不修改原指标。复用 SHA 绑定的完整 25 random ONNX，每拍一次 actor 调用。固定完整 formal 矩阵为三任务 × 四延迟 × 三 seed，每例 600 拍；仅全部通过才运行独立 seeds 2001/2002/2003 的 36 例 holdout。最多 43,200 policy / 432,000 native / 216,000 PD，训练 0。不得按案例启停滤波、搜索参数、续跑或重试本次输出。

每例保存原物理轨迹，以及 raw actor / float64 linear / float32 issued 三组旁路记录。读回精确重放因果递推，核对实际 issued 和整段 FIFO，并使用原 evaluator 从轨迹重新计算指标。任务失败仍收齐矩阵；异常保留实际计数或未知值及已知前缀。

运行仅允许注册的 `d1-causal-notch30-owned01-20261018-01.service`，Nice10、单线程、MemoryHigh=2 GiB、MemoryMax=3 GiB、SwapMax=0、OOMPolicy=kill。有效可用内存 4 GiB 持续 10 秒才准入，最多等待 300 秒；运行保留 2 GiB，执行上限 900 秒，外部 unit 上限 1210 秒。使用新的 plan/output，不改历史执行。默认命令只显示计划；提交实现后 `python3 -B tools30/run_notch30.py --prepare`，再于指定 unit 中 `--execute`。

原 CPP64 只验证网络，不证明新增滤波状态机。即使 Python formal/holdout 都通过，仍需与新控制路径相称的 C++ 闭环验证；本实验不自动替换默认模型。
