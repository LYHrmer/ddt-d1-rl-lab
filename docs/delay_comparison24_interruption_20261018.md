# 24 内存中止核查（2026-10-18）

本轮 `INCOMPLETE` 的直接原因是训练器的 **4 GiB 运行内存门**。这不是 KL 假设的失败判定，也不是已观察到的 RPC 错位、异常或信号终止。旧 24 的源码、计划、输出和唯一执行记录全部保留；本次只读核查没有加载模型或执行物理，也没有重试。

机器可读证据和 20 个原始输入 SHA 见 [核查报告](../reports/delay_comparison24_interruption_20261018.json)。原入口为 `evidence/delay_comparison24/execution_01`。

| 项目 | 实际证据 |
| --- | --- |
| trainer 状态 | `RAM_RESERVE_BELOW_4_GIB`，`error=null` |
| 运行内存线 | 4,294,967,296 B |
| 最低有效可用内存 | 4,294,717,440 B，比门槛低 249,856 B（244 KiB） |
| 首个停止样本 | 训练器启动后 56.510175 s，`phase=training`；`after_learn` 再次采到同值 |
| 耗时 | trainer 57.877554 s，父监督器 61.488580 s，均远小于 1,800/5,400 s 时限 |
| 实际新增计数 | 47,316 policy / 473,160 native / 236,580 PD；另有 50 resets、47,420 forward calls |
| 清理 | trainer 与父回执均为 `cleanup_complete=true`；RPC 无污染，自有清理无剩余 PID、无清理信号 |
| 后续阶段 | random、导出、CPP、原 36 例和 holdout 均未启动 |

`tools24/train_delay24.py:54` 的 `Watchdog24.poll` 在实际有效内存低于计划的 `reserve_bytes` 时记录该停止原因并令 callback 返回 false。父监督器的 `ValueError: child24 failed/late` 是通用失败标签；本轮实际耗时不支持超时解释。父进程和训练器各自每秒采样、计时起点不同，父采样未捕获同一瞬间的越线，不否定训练器持久化的停止样本。标准输出没有训练异常回溯；Matplotlib 的 Axes3D 警告不是停止原因。

临近停止的训练器样本从 52.198 s 的 6,669,701,120 B 降到 53.203 s 的 5,098,909,696 B，继而在 56.510 s 越线。记录只有主机可用 RAM，没有当时各进程的 PSS/USS。因此不能确定这段内存下降由哪个进程贡献，更不能把四个 worker 的 RSS 相加解释成四份私有 Torch 内存。退出后的 transient unit 已 inactive/MainPID=0；已卸载单元返回的默认属性不能证明历史优先级或退出码。

四份 `env_XX.monitor.csv.counts.json` 都已关闭，每个 worker 分别是 11,829 policy / 118,290 native / 59,145 PD，与 result、父回执逐项相加一致。完整 ledger 有 11 个 4,096 步 rollout，共 45,056 步；其后另收集了 2,260 步不完整 rollout。SB3 的 `collect_rollouts` 在 callback false 时退出，`learn` 随即跳出、不调用该批 `train`（`.python-deps/stable_baselines3/common/on_policy_algorithm.py:226`、`:324`）。所以 47,316 是收集/物理交互计数，不应全部称为已进入 PPO 更新的样本。

仅用 ZIP 内 JSON 元数据核对，`partial_policy.zip` 的 `num_timesteps=1,095,892`、`_n_updates=1,035`，初始分别为 1,048,576、1,024。后者表示进入的 epoch 计数，并非 optimizer step 次数。11 次完整 rollout 与新增 11 个 epoch 计数一致；不能据此推断策略合格或 KL 终态全局有界。`progress.csv` 在当前 rollout 的更新之前写入，最后一行也不能冒充最终 checkpoint 的全部训练指标。

`rpc_guard24` 为 `dirty=false, closed=true, waiting=false, inflight=null`，13 次元数据请求全部返回且通过验证；自有清理 `remaining_pids=[]`。四个 worker 的延迟均为 zero，独立延迟流的记录 seed 为 181–184，四桶累计只有 delay 0，总 reset 数 50。原计划、protocol、result 和父报告的 source proof 相等，20 项源码的当前 SHA 与冻结提交 `b7d138964752f3314e197cd6dbb08f1197bead2b` 的 git blob SHA 全部一致；计划、protocol、partial checkpoint SHA 也逐项匹配。

本轮只产生不可评测的 `partial_policy.zip`，没有 `policy.zip`。random 未开始，也没有任何资格结果，因而不支持 zero/random 的效果比较或 target-KL 对策略质量的成败结论。默认策略仍为原完整 control 基线，24 partial 不进入候选池。

下一轮若采用单环境，应建立独立 source、预算和输出，而非重启或续训 24。可考虑 `DummyVecEnv` 单进程、`n_steps=4096`，保留每个 rollout 共 4,096 transitions、batch 256、4 epochs、LR 1e-4 和 target-KL .005。加载时必须将单环境和 `n_steps=4096` 传给 `AuxiliaryVelocityPPO.load`，让其先重建 buffer 再恢复原完整 control 和 Adam；仅修改 `model.n_steps` 不够。

这种布局删除额外 worker/forkserver/IPC 进程和三个环境实例，节省额外进程私有页、模型数据及队列，但原 forkserver 已共享预加载的只读页，不能预报具体能省多少 GiB。4,096×(627+3)×4 B 的观测数据仍为 10,321,920 B（约 9.84 MiB），不会随环境数下降。单环境也改变 GAE 边界、课程轨迹和向量采样，所以只能比较新 scope 内的 zero/random 两组，不能称其相对 24 仅改变了延迟。继续保留 6 GiB 启动、4 GiB 运行、Nice 10 和有界单次执行；PSS/USS 阶段快照可在新 scope 内验证内存节省机制。
