# 合同 10：E0 因沙箱 IPC 权限失败

合同 10 已在完整冻结、962 项测试和 CI 通过、内存达到 7 GiB 后实际启动一次 E0。失败发生在创建 forkserver 监听 socket：`PermissionError: [Errno 1] Operation not permitted`。监督器确认退出非零、清理完成和冻结输入未变，立即停止流水线；没有进入正式训练、导出或资格评测。

原始训练回执记录 `trained_transitions=0`，`actual_counts=null`、`per_worker_counts=null`。这些 null 原样保留。traceback 的失败位置位于 forkserver/首个环境 worker 创建之前，结合源码控制流，对物理消耗作零值推导；该推导与实际计数器观测分开列在[合同 10 摘要](../reports/contract10_summary.json)。Q1–Q5 全部不可评估，资格为 `NOT_EVALUATED`，不是 0/3。

后续零训练 socket 探针在沙箱内复现相同 bind 失败，在单独批准的沙箱外执行中通过，确认了进程通信权限差异。它不证明模型或控制合格，也不授权重新训练。

原合同 10 的一次尝试规则保持有效，没有自动重跑。已准备[环境修订草案](career_execution_contract_10_environment_amendment.md)，请求对这次有证据的零步骤环境失败给予一次明确例外：保留 `_01`，使用新身份和 `_02` 目录，在沙箱外执行同一科学方案，合计预算不增加。该草案必须得到所有者明确批准后才可执行。
