# 两仓三轮优化交付

本轮针对两个仓库交付三轮功能改动。工程通过与控制资格分开记录：D1 新策略尚未完成训练，合同 10 首次 E0 因沙箱 IPC 权限失败停止，不能宣称策略变好或资格通过。

| 轮次 | D1 | wheel-legged-control-lab | 实际验证 |
|---|---|---|---|
| V1 工程可靠性 | 单样本导出匹配实际部署推理路径；严格核对训练完成、退出和资源收尾 | 源码身份清单覆盖工作树；日志缺控制 tick 拒绝计指标 | D1 769 tests，真实 Python/C++ 64 例数值链；wheel 51 项相关测试、24 案例 216 摘要保持 |
| V2 部署与运行诊断 | 工件身份和纯 ROS 导入预检；失败也保存结构化回执 | v27 显式环境迁移；拒绝 host exit0 但 worker 未就绪/未执行的假成功 | ROS 11 项通过，0 节点/物理步；4,072 项输入匹配，wheel 68 项相关测试 |
| V3 实验验证 | 授权→静态→冻结→E0→正式训练→导出/评测→判定的独立执行链；持久阶段记录禁止重试 | v28 完整且相同时域的配对比较；缺失/失败为 null，整训练种子组不取幸存均值 | D1 962 tests、历史 9,163 回合兼容性复算；wheel 23 tests、33 配对和三组均值逐值保持 |

对应源提交：D1 `6b80739`、`4e5b010`、`84719ad`；wheel `79ba9df`、`41153a8`、`a42d561`。两仓三轮源提交的 CI 均通过；wheel 第三轮为 [39016472370](https://github.com/LYHrmer/wheel-legged-control-lab/actions/runs/39016472370)。合同 10 失败闭账另由 `8133718` 记录。

详细证据：

- [D1 导出与监督修复](engineering_optimization_10.md)、[部署预检](deployment_preflight_11.md)、[冻结执行链](contract10_execution_pipeline.md)、[合同 10 失败结果](career_results_10.md)。
- wheel [源码和日志完整性](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/docs/source_and_metrics_integrity_20261016.md)、[v27 环境与完成门](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/docs/b22_runtime_v27_20261016.md)、[v28 比较规则](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/docs/continuous_comparison_v28_20261016.md)。

历史冻结代码、失败回执和用户结果保留。后续若对零步骤环境失败给予一次例外，须按[环境修订草案](career_execution_contract_10_environment_amendment.md)另行明确批准；原预算批准和 socket 探针批准不代替该决定。
