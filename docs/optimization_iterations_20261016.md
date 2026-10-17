# 两仓三轮优化交付

本轮针对两个仓库交付三轮功能改动，两仓各轮源提交的 CI 均通过。D1 的获准训练尝试已按未完成规则闭账：首次 E0 为零步骤 IPC 失败，明确批准环境例外后，`_02` 在完成 46.25% 时因 RAM 门停止。工程交付已完成；新策略的控制资格仍未评测，不能宣称策略变好。

| 轮次 | D1 | wheel-legged-control-lab | 实际验证 |
|---|---|---|---|
| V1 工程可靠性 | 单样本导出匹配实际部署推理路径；严格核对训练完成、退出和资源收尾 | 源码身份清单覆盖工作树；日志缺控制 tick 拒绝计指标 | D1 769 tests，真实 Python/C++ 64 例数值链；wheel 51 项相关测试、24 案例 216 摘要保持 |
| V2 部署与运行诊断 | 工件身份和纯 ROS 导入预检；失败也保存结构化回执 | v27 显式环境迁移；拒绝 host exit0 但 worker 未就绪/未执行的假成功 | ROS 11 项通过，0 节点/物理步；4,072 项输入匹配，wheel 68 项相关测试 |
| V3 实验验证 | 授权→静态→冻结→E0→正式训练→导出/评测→判定的独立执行链；持久阶段记录禁止重试 | v28 完整且相同时域的配对比较；缺失/失败为 null，整训练种子组不取幸存均值 | D1 962 tests、历史 9,163 回合兼容性复算；wheel 23 tests、33 配对和三组均值逐值保持 |

对应源提交：D1 `6b80739`、`4e5b010`、`84719ad`；wheel `79ba9df`、`41153a8`、`a42d561`。两仓三轮源提交的 CI 均通过；wheel 第三轮为 [39016472370](https://github.com/LYHrmer/wheel-legged-control-lab/actions/runs/39016472370)。合同 10 失败闭账另由 `8133718` 记录。

详细证据：

- [D1 导出与监督修复](engineering_optimization_10.md)、[部署预检](deployment_preflight_11.md)、[冻结执行链](contract10_execution_pipeline.md)、[合同 10 失败结果](career_results_10.md)。
- wheel [源码和日志完整性](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/docs/source_and_metrics_integrity_20261016.md)、[v27 环境与完成门](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/docs/b22_runtime_v27_20261016.md)、[v28 比较规则](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/docs/continuous_comparison_v28_20261016.md)。

环境修订代码另见 `0784d45`，CI [39017796910](https://github.com/LYHrmer/ddt-d1-rl-lab/actions/runs/39017796910) 通过；本机 999 tests 通过，干净源码归档 970 通过、29 项因未分发历史资产而跳过，新增 36 项均通过。修订经所有者对具体执行命令明确批准后冻结 72 项输入。E0 `_02` 完整通过；正式训练完成 3,879,424 transitions，947 个完整 rollout 无比较差异但不足 E1 要求，Q1–Q5 不可评估。清理及最终计数完整，没有执行导出、资格评测或重放。

历史冻结代码、失败回执和用户结果保留；72 项冻结输入和 journal 81 项产物复核一致。合同 10 新消耗 3,912,192 transitions / 39,121,920 native，详见[最终摘要](../reports/contract10_amended_summary.json)。三轮功能及其验证已交付；后续训练需新的实验方案与预算，不沿用本次未消耗余额。
