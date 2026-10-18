# D1 26：内存保护中止记录（2026-10-18）

26 的唯一 attempt 已闭账为 `INCOMPLETE`。第一组 control 在 4 GiB 有效内存停止线下退出；没有启动 yaw3，也没有导出、CPP64、正式评测或 holdout。本记录不改变旧现场、不允许续训或重跑，不能据此评价提高 yaw 奖励是否有效。

## 实际消耗与更新边界

实际消耗为 **1,302 policy transitions / 13,020 native steps / 6,510 PD updates**，父报告的 actual 与 known 完全一致。单 worker 另记 resets=2、forward_calls=1307。累计 `num_timesteps` 从 1,310,720 到 1,312,022；未完成首个 4,096 步 rollout，首 buffer 配对证明尚不存在。

保存的是 `partial_policy.zip`，没有最终 `policy.zip`，不能进入评测。原25 random完整基线与该 partial 的 `policy.pth`、`policy.optimizer.pth` 各自 SHA 完全相同，`_n_updates` 均为1088，故 **PPO 更新增量为0**。这不等于没有物理消耗：上述1,302步确实已执行。control 的原/训练累计奖励均为95.97693316395218，额外奖励0；第二组没有可比较数据。

## 停止与资源证据

计划启动门为6 GiB、运行停止线为4 GiB，均未更改。父初始样本为7,402,946,560 B。trainer 在自身 elapsed=12.300189s 记录有效内存 **4,189,859,840 B** 并写入 `RAM_RESERVE_BELOW_4_GIB`；parent 在 elapsed=17.604801s 另记录 **4,133,515,264 B**，以 `RAM_BELOW_4_GIB` 停止整轮。这是两个真实采样值，时钟起点不同，不能直接相减当作清理耗时。父总墙钟为19.743143s，子总墙钟为15.003480s。

trainer 的同一进程 PSS 三次快照约为0.932 / 1.082 / 1.084 GB（加载前/加载后/关闭后），RSS约0.946 / 1.096 / 1.101 GB。它们不是峰值或连续曲线，不能用host可用内存下降或总RSS断言本项目泄漏。原样保留 `cgroup_remaining_bytes=null`，没有补0。

根线程先前只读观察到外部 Python PID810620 的RSS约3.9GB并有NVIDIA命名线程；本项目没有启动GPU任务。此审计在2026-10-18 06:07:05 UTC再次只读检查宿主机 `/proc/810620` 时该项已不存在，无法核回其cwd、argv项目名或启动时刻。没有同时刻的外部RSS序列，**内存压力具体来源仍为UNKNOWN**；不能据进程或线程名绝对归因，也未读取环境变量或发送信号。

## 清理、来源与资格范围

父/子均记 cleanup_complete=true，子guard为CLOSED、dirty=false、无等待/在途调用，3次元数据调用均完整返回并校验；env子进程数为0。根线程宿主机systemd观察确认 unit inactive、MainPID=0；独立后续读取也未见trainer PID812014。trainer的 error=null 表示本次是保护性停止；父error仍如实为 `ValueError: RAM below 4 GiB or unknown`，具体stop_sample已证是低内存而非未知读数。

本次计划的 **35项来源SHA** 与现值、父子来源完全一致，Git身份 `34f77d8cc3523e34f4e19e5ca679fedaab6733f8`。所有11个实际输出文件和计划SHA均完整登记在[机器报告](../reports/yaw_reward26_interruption_20261018.json)。主要身份：

| 工件 | SHA-256 |
|---|---|
| plan | `8c54f7d337aff471b29f4a1c7dc50a08e75dd755dd6da96d2b8160041047b708` |
| parent report | `3be5dc4e8850b3a6ab0472bbd12a5dfe6cc386d889c800b6e5aee07abfbe9132` |
| training result | `4549c85d2480e2f058a2a9111bf2186678867dfaf859ab473e2a2717e37b5c8f` |
| partial ZIP | `72d789eab6a63a569206e92a7b23871aa34958150a2f56db05430d8f7dc24af2` |

两组比较、胜者、reward效应估计均为空；原36/holdout未执行，报告中的false不是已经观察到任务失败。仿真候选、C++闭环及ROS实机资格均不成立。已消费的预算不能补写成0，未执行余额不能用于本attempt重试。

## 下一独立轮的一项准入建议

保留内部 **6 GiB启动门和4 GiB停止线**，下一独立27版拟在外层增加稳定准入：有效内存须≥8 GiB持续60秒，每5秒只读采样；任何低值或未知读数重计稳定窗口，最多等待300秒，低于4 GiB立即停止。未达标只给NOT_READY，不claim唯一训练执行目录或spawn；通过后在唯一父流程spawn前再验一次。此修订仅降低瞬时准入误判，不能保证后续独占内存，其他任务随后增长时仍须按4 GiB停止。本记录不执行该方案，也不修改冻结26。
