# 导出与训练监督的工程修复

本轮没有新增训练 transitions 或物理步。合同 09 保持未完成；合同 08 的原始导出 FAIL、资格 1/3 保留。新增代码提供未来实验的入口，并在新目录完成数值导出检查，不把工程通过算作控制资格。

## 导出：比较实际的单样本路径

冻结导出器将 SB3 的 64 样本批量推理，与逐条运行的 Torch actor 比较。用合同 08 的同一完整最终权重和同一固定输入复现：批量 SB3 对逐条 Torch 的最大差为 `1.2814998626708984e-6`，超过原 `1e-6` 门；只把 SB3 改成逐条推理，差值为 **0**。

这与 [PyTorch 2.12 的数值说明](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html#batched-computations-or-slice-computations)一致：批量和逐条计算可以因浮点运算顺序产生差异。定位来自上述单变量实测，不能据此推断任意模型的差异都可忽略。

新 [export_single_sample10.py](../tools10/export_single_sample10.py) 校验旧导出器 SHA 后复用原实现，只把传给原一致性检查的 SB3 输出切换为逐样本预测。输入仍为原来的 64 个固定 fixtures；**SB3/Torch ≤ `1e-6`、Torch/ORT ≤ `1e-5` 均不放宽**。批量与逐条的差异单独报告；形状错误或非有限预测仍拒绝。新 sidecar 绑定新入口、旧导出器、checkpoint 和原 metadata 的 SHA，不修改旧导出文件。

本次实际结果见[紧凑摘要](../reports/export_engineering10_summary.json)：

| 检查 | 结果 |
|---|---|
| SB3 单样本 / Torch | 最大绝对差 0，PASS |
| Torch / Python ORT 1.23.2 | `1.3113021850585938e-6`，PASS |
| Torch / C++ ORT 1.10.0 | 64/64，最大差 `1.3113021850585938e-6`，PASS |
| 新 ONNX SHA | `d74b897e2d005a36985b9b70571a220d8e4f58189ec108f4e304a529d7eec290` |
| 新训练、MuJoCo 模型构造或物理积分 | 均未执行；只加载策略并推理 |

使用冻结的 `runtime03/build/d1_runtime03`，没有重建二进制。完整本地证据在 `evidence/engineering10/export_single_sample_01/` 和 `cxx_single_sample_golden_01/`。

```bash
rtk proxy env PYTHONPATH="$PWD/.python-deps" python3 -B tools10/export_single_sample10.py \
  --checkpoint evidence/career_contract_08/scale08_train_01/policy.zip \
  --config training/configs/pilot03d.json --output evidence/engineering10/export_single_sample_01
rtk proxy runtime03/build/d1_runtime03 --schema runtime03/schema03d.json \
  --onnx-golden evidence/engineering10/export_single_sample_01/golden_fixtures.json \
  --policy evidence/engineering10/export_single_sample_01/policy.onnx \
  --expected-policy-sha256 d74b897e2d005a36985b9b70571a220d8e4f58189ec108f4e304a529d7eec290 \
  --output evidence/engineering10/cxx_single_sample_golden_01
```

这是已执行命令的记录，输出均拒绝覆盖。重做工程检查应使用新目录；不会修改合同 08 的冻结评测或把合同 09 的中间权重拿来导出。

## 监督：等价性通过不能掩盖失败退出

零物理步 dummy 进程复现了旧入口的两项遗漏：预检只查等价性，可能接受失败的训练回执；训练器写 COMPLETE 但退出码非零时，启动器仍可能返回成功。旧脚本已冻结，本次没有编辑。

新 [supervise_scale10.py](../tools10/supervise_scale10.py) 只允许新合同 10 的独占输出：

- 预检必须属于同一 release，原始 result 与监督回执一致，训练完整、worker 账本一致、退出码为 0 且清理完成，才可进入正式训练。
- 正式训练的成功必须同时满足退出码、完整计数、输入 SHA 和等价性；缺失计数保持未知，不能记作 0。
- 启动前检查宿主 MemAvailable 和可见 cgroup 硬上限余量。NOT_READY 单独记录，不创建唯一 spawn 目录。
- 运行中记录独立资源采样；触发停止的采样和实际最小值留在 sidecar。它不能恢复冻结训练器曾遗漏的内部采样。
- 先向本轮训练 leader 发 SIGINT，让它向仍存活的 worker 收集末次账本并关闭；超宽限再对自建进程组发 SIGTERM、SIGKILL。遗留 worker 同样使本轮 FAIL。dummy 测试验证不会影响另一个独立 session。
- 监督器自身收到 SIGINT/SIGTERM 时也进入有期限清理并写失败回执；启动过程和清理宽限期的信号竞态均有真实假进程回归。SIGKILL 无法被程序捕获，不承诺在强杀宿主时留下末次回执。

资源观察不等于资源预留。09 的训练 USS 首末只增约 31 MiB，同期宿主可用内存减少约 811 MiB，更支持外部负载挤压，而非大规模训练泄漏；具体外部进程未被历史数据确定。7 GiB 准入线也不能保证外部程序不再占内存。

[合同 10 草案](career_execution_contract_10.md)及[机器可读计划](../training/configs/contract10_supervisor_draft.json)保留 8 worker、seed 101、熵系数 0 和原 3 GiB 停止门，另列新预算与较严格的准入条件。草案被监督器在资源检查/启动之前拒绝，不能当作执行许可。准备时实测可用内存约 4.79 GiB，未达到 7 GiB 准入线；这是一次采样，不能代替执行前重检。当前没有 FROZEN release，**本轮未启动 E0 或正式训练**。

本机 D1 全套 **769 项测试通过**，包括 36 项新监督假进程测试；32 项静态检查、历史冻结输入、77 项保护文件和 9 份用户结果核验通过。三个新工具的 6 项隔离变异均被捕获。轮足仓库的日志和源码清单改进另有 51 项定向测试与全仓 lint 通过，见其[工程说明](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/docs/source_and_metrics_integrity_20261016.md)。
