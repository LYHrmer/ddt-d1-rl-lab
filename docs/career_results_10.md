# 合同 10：环境例外后的训练因内存保护未完成

**最终状态（2026-10-17）：`_02` 正式训练完成 3,879,424 / 8,388,608 transitions（46.25%）后，由冻结训练器的 `RAM_RESERVE_BELOW_3_GIB` 门停止。** 只有中间权重，没有正式最终策略；未导出、未做 C++ 比对、资格评测或训练回合重放。Q1–Q5 不可评估，资格为 `NOT_EVALUATED`，不是 0/3。结果见[独立 `_02` 最终摘要](../reports/contract10_amended_summary.json)及[终态快照](../reports/contract10_execution_status.json)。本次尝试已关闭，不重跑或续训。

## 原始 `_01`：零步骤 IPC 失败

合同 10 已在完整冻结、962 项测试和 CI 通过、内存达到 7 GiB 后实际启动一次 E0。失败发生在创建 forkserver 监听 socket：`PermissionError: [Errno 1] Operation not permitted`。监督器确认退出非零、清理完成和冻结输入未变，立即停止流水线；没有进入正式训练、导出或资格评测。

原始训练回执记录 `trained_transitions=0`，`actual_counts=null`、`per_worker_counts=null`。这些 null 原样保留。traceback 的失败位置位于 forkserver/首个环境 worker 创建之前，结合源码控制流，对物理消耗作零值推导；该推导与实际计数器观测分开列在[合同 10 摘要](../reports/contract10_summary.json)。Q1–Q5 全部不可评估，资格为 `NOT_EVALUATED`，不是 0/3。

后续零训练 socket 探针在沙箱内复现相同 bind 失败，在单独批准的沙箱外执行中通过，确认了进程通信权限差异。它不证明模型或控制合格，也不授权重新训练。

原合同 10 没有自动重跑。随后准备的[环境修订](career_execution_contract_10_environment_amendment.md)针对这次有证据的零步骤环境失败提出一次明确例外：保留 `_01`，使用新身份和 `_02` 目录，在沙箱外执行同一科学方案，合计预算不增加。只有在所有者明确批准具体执行命令后才启动新身份。

## 环境例外批准后的独立 `_02`

所有者随后明确批准了包含环境例外说明的沙箱外执行命令。新适配入口登记授权并冻结 72 项输入，release SHA 为 `d3a0ae0a2ba986ec4e9b5f3a2fa2dc58f159c601312885575d0543afe9e56e9c`。真实标准库 forkserver＋Pipe 准入通过，进程组清理完整，模型和物理步数为 0。

第一次内存准入约 5.09 GiB，写入 NOT_READY，没有消耗训练尝试。后续准入达到 7.15 GiB，E0 `_02` 完整完成 32,768 transitions / 327,680 native，8 个 rollout 全部等价、退出 0、计数一致、清理完成。正式训练在约 7.72 GiB 可用内存下启动，约 51.5 分钟后停止，训练阶段平均约 1,256 transitions/s。

正式训练的 8 个 worker 均正常关闭，最终计数逐值一致：3,879,424 transitions / 38,794,240 native / 19,397,120 PD updates。947 个完整 rollout 与冻结基线均无差异，但不足 E1 要求的 1,173 个，故等价性为 `INCOMPLETE`、E1 未达标；末尾额外 512 transitions 不计入完整 rollout。流水线按冻结顺序进入未完成判定，跳过后续模型操作。

## 停止原因与采样限制

训练器状态为 `RAM_RESERVE_BELOW_3_GIB`，退出码 1，`error=null`。监督器的 `RESIDUAL_PROCESS_GROUP` 是 trainer leader 退出后，对自身进程组剩余辅助进程执行 SIGINT 清理的记录；`cleanup_complete=true`。它不替代训练器的 RAM 停止原因，也不表示留下了未清理进程。

训练器保存的最低可用内存 3,289,874,432 bytes（3.063934 GiB）只来自 rollout 末采样。内部每 64 次回调的 RAM 检查可以直接停止训练，但没有持久化触发读数。监督器约每 0.5 秒采样，共 6,172 条，最低为 3,226,914,816 bytes（3.005299 GiB），没有捕获低于 3 GiB 的值，`stop_sample=null`。因此有 RAM 停止状态和代码路径证据，但不能补造确切触发读数或持续时间。

进程树 USS 首值约 2.784 GiB、峰值 2.817 GiB，增加约 34.14 MiB；这些记录不支持将停止归因于训练器持续大幅增长的独占内存，也不足以识别具体外部负载。RSS 汇总峰值约 6.384 GiB 包含跨进程重复计入的共享页，不能据此宣称内存泄漏。7 GiB 启动准入不提供运行期间的内存独占保证。

## 闭账与保留

E0 与正式训练合计 **3,912,192 transitions / 39,121,920 native / 19,560,960 PD updates**；项目累计 **22,416,384 transitions / 224,684,030 native**。journal 的 `actual_complete=true` 仅表示实际计数齐全，不表示训练完成。72 项冻结输入、16 个 worker 最终计数和 journal 绑定的 81 个产物均已独立复核。

原 `_01` 的失败、null 原始计数与零步骤推导保持分开。`_02` 只保留 `partial_policy.zip`，其 SHA 见最终摘要；不挑选中间权重替代最终资格。剩余 4,509,184 transitions / 45,109,840 native 是未消耗上限，不构成再次训练授权。本次没有新 attempt、重试或续训。
