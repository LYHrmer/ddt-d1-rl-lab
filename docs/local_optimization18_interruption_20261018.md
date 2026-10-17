# 18 单次训练中断闭账（2026-10-18）

`evidence/local_optimization18/execution_01` 已停止，原回执为 `STOPPED` / `Interrupted: SIGTERM`，耗时 1400.841 秒。最终总计数为未知，整轮没有选胜结果或完整延迟资格。旧现场、冻结的 12 份源码和计划不改写，不补造缺失的随机训练 `result.json`。本记录不授权重跑、续训或使用 partial 权重。

## 已完成与未知的边界

| 阶段 | 已完成新增 policy transitions | 结果 |
| --- | ---: | --- |
| preflight_zero | 32,768 | COMPLETE，8 个 rollout；预检权重不用于正式训练 |
| preflight_random | 32,768 | COMPLETE，8 个 rollout；预检权重不用于正式训练 |
| train_zero | 1,048,576 | COMPLETE，256 个 rollout；从初始 1,048,576 到最终 2,097,152 |
| train_random | 最终未知 | INCOMPLETE，缺少最终结果回执 |

原父回执确认的已闭账训练新增量为 1,114,112；加上基线探针 271 次，已闭账物理阶段小计为 1,114,383 policy / 11,143,830 native / 5,571,915 PD。这些是完成阶段的小计，不能表示整轮最终用量。

随机训练的 ledger 保存了 62 个完整 rollout，可靠前缀为 253,952 policy / 2,539,520 native / 1,269,760 PD。四个 worker 的 `running` 快照各为 64,128 policy，合计 256,512 / 2,565,120 / 1,282,560。原监视器每 128 个 worker policy transitions 发布一次快照，这些数值可作为已有工作量的下界，不能替代最终计数。`partial_policy.zip` 存在，但没有完成训练或资格凭据。

## 已证实的失败链

父进程 571948 的信号处理器记录了 SIGTERM，但没有保存发送者 PID；来源为 **UNKNOWN**。工具调用没有显式 timeout 参数，根代理没有主动发送 terminate 或 Ctrl-C。1400.841 秒耗时和工具 exit 143 不能证明存在固定 20 分钟时限，也不能据此归因于用户操作。

父清理记录向随机训练器 590894 发了 SIGINT。随机训练 [error.txt](../evidence/local_optimization18/execution_01/train_random/error.txt) 显示中断发生于 SB3 `SubprocVecEnv.step_wait()` 的 `remote.recv()` 反序列化过程。随后 [stdout](../evidence/local_optimization18/execution_01/train_random.stdout.log) 明确显示保存结果失败：`TypeError: Object of type ndarray is not JSON serializable`。因此最终 `result.json` 不存在，原父回执的 `actual_counts: null` 必须保留。

源码支持的解释是：`step_wait()` 直到接收完全部响应才清除 `waiting`；中断后的 finalization 又通过同一通道调用 `env_method("counters")` 和 `env_method("get_delay_state18")`。该协议没有响应类型标签，遗留 step 响应可能被当成元数据。当前代码先赋值 `workers` / `delays`，再验证，验证失败后仍将原候选值放进结果。包含观测数组的错位响应会使 JSON 保存失败。实际失败的完整候选对象没有保存下来，无法从现有字节确定数组究竟位于哪一个结果字段；这部分保留为源码推断。

## 资源、guard 与清理

父采样最低为 3,401,994,240 bytes，发生在 `train_zero`；仍高于旧 3 GiB 门。随机训练内部最低 5,549,309,952 bytes，最后一条为 6,146,445,312 bytes，远未到其 5400 秒时限。现有采样不能证明 RAM 触发了本次停止，也不能排除未采到的瞬时变化。

新 `resource_guard20 --execute` 实际返回 `NO_RUNNING_EXPECTED_TARGET`，`target: null`；没有创建 `resource_override_02`，也没有发信号。该结果来自根代理保存的工具执行观察，源码在这个返回分支之前不会调用 `monitor()`。不能把后加的 4 GiB guard 写成本次 SIGTERM 的来源。

父回执 `cleanup_complete: true`。本次只读 `/proc` 核查时，父进程、随机训练器及四个已记录 worker PID 均不存在，训练器 PGID 590894 没有存活成员。这是核查时点的清理证据，不补充未知的最终物理计数。

## 新版本必须先修与复现

旧18继续封闭。新训练入口应在 step 接收被中断或 `vec.waiting` 为真时，把该 RPC 通道视为失去同步，不再用它取得新的计数或 delay 元数据；最终计数未知时明确写 null。计数与 delay 候选必须先通过严格结构和 JSON 可写性检查，再赋给回执字段。失败候选不能残留，单纯把数组转成列表不能证明响应身份正确。

任何新训练前，必须用假进程复现：接收中断且遗留含数组 step 响应、错误计数和 delay 候选、RPC 超时、清理失败、checkpoint 保存失败及重复终止。所有路径均应落下有效失败回执，保留计数边界并有界清理自己的进程。新的长期执行还需独立源码、预算与输出目录，保持启动 6 GiB / 运行保留 4 GiB，并通过 dummy 进程验证受限 user-systemd 生命周期；这些改进不能反推旧 SIGTERM 的发送者。

## 工件身份

完整 SHA 清单和原回执字段见 [机器可读审计](../reports/local_optimization18_interruption_20261018.json)。关键身份：

- 原计划：`d1ecfb1026665b41e40da2b029724e6ee77b3b58dcb860eccc37bab712951411`
- 原父回执：`aee841b96acb3c1ed14e95af263ad3676e42c310d70e4ebd89a7caa852f74595`
- 完成的 zero 权重：`d79dbdffa71a23a5c8debb95a562aab810cb6a80e31d97afac8fbd492daddce8`
- 随机 partial 权重：`5b88af32673ef20445fca93e25856e6a761c122c984aa0b49f193c62858389cd`
- 随机失败 stdout：`11c32d3dadc9cf8c27d1324fd34bd77938488a0edaff04c183c1884858ab9d98`

本次仅阅读文件、计算 SHA 并写入两份独立审计记录；没有模型加载、物理积分、测试、信号、训练或旧工件修改。zero 完整 checkpoint 的后续验收必须另用有界协议，不能据其训练完成宣称整轮18或 C++ 闭环19通过。
