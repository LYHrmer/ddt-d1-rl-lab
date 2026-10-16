# 合同 09：训练触发内存停止门，实验未完成

**结论：未完成，Q1–Q5 均不可判定，资格未评测。** 去掉熵奖励能否抑制动作噪声、能否阻止前进退化，仍未得到本合同的完整证据。事前冻结的[合同 09](career_execution_contract_09.md)保持原字节，不改预测、预算或停止规则。

可在 GitHub 直接阅读的[紧凑摘要](../reports/contract09_summary.json)由 `tools09/build_summary09.py` 从原始回执生成，不含权重或第三方模型。所有原始数据与中间权重留在本机 `evidence/career_contract_09/`。

## 实际执行到哪里

| 步骤 | 结果 |
|---|---|
| 静态检查、事前 release | 已完成，冻结 15 项输入 |
| E0 预检 | COMPLETE，32,768 transitions；首个 rollout 与 03d 相同 |
| 正式训练 E1 | 首个 rollout 等价性 PASS |
| 正式训练完成门 | **未通过：RAM_RESERVE_BELOW_3_GIB** |
| 正式训练实际消耗 | 4,807,168 / 8,388,608 transitions（57.3059%） |
| 导出、C++ 黄金比对、资格评测 | 未执行 |
| 训练回合重放 | 未执行；计划训练最后 1/4 尚未到达 |
| 关闭判定 | E1 MET，TRAINING NOT_MET，Q1–Q5 NOT_EVALUABLE |

正式训练于 **2026-10-16 14:16:09 北京时间**结束，约运行 61.3 分钟，训练器退出码为 1。Claude 的最后一次回执读取被 API 402 阻断；这不改变此前已经发生的训练停止。后台通知里的退出码 0 来自最后执行日志读取的外层 shell，不能替代训练回执。

只有 `scale09_train_01/partial_policy.zip`，没有完整训练才会写出的 `policy.zip`。不把中间权重当最终策略，不导出、不补资格评测、不重跑、不续训。E1 只说明初始策略的第一轮数据相同，不能说明训练已经完成。

回执记录的最小可用内存是 3,307,237,376 bytes，约 3.08 GiB。这与 3 GiB 停止门不矛盾：冻结训练器每 64 次 step callback 检查一次内存，但只有 rollout 结束时才更新回执的最小值；step 检查触发停止的那次采样没有单独保存。因此只报告停止原因和已有采样，不编造触发瞬间的内存数值，也不修改冻结训练器补记历史数据。

## 判定工具的事后修补

结果前提交的 `tools09/verify_scale09.py`（`1429550`）只检查 E1 是否通过，未检查正式训练是否完成。接手后对这个遗漏做保守修补：只有正式 scale、训练器成功退出、未被启动器中断、状态 COMPLETE、输入不变且 transitions/native 均达到冻结的 8 倍预算，才读取评测与重放并判 Q1–Q5。未完成时，任务通过数写 `null`，资格写 `NOT_EVALUATED`，不是 `0/3`。

这是看到中断后的工具修补，不声称事前已具备；冻结预测和阈值没有变化。新增回归测试覆盖 RAM 与 wall 停止、只有等价性回执、预检冒充正式训练、退出码失败和计数不一致，并保留合同 08 的完整训练回归。关闭命令不加载策略、不构造模型、0 新物理步。

```bash
rtk proxy env PYTHONPATH="$PWD/.python-deps" python3 -B tools09/verify_scale09.py \
  --launcher-report evidence/career_contract_09/scale09_train_01_launcher/report.json \
  --output evidence/career_contract_09/scale09_verdict_01
rtk proxy env PYTHONPATH="$PWD/.python-deps" python3 -B tools09/build_summary09.py
```

以上是本次已经执行的记录，输出均拒绝覆盖。复核时把 `--output` 换成新路径，不重跑训练或评测。摘要工具核对 15 项 release 输入、18 项训练 protocol 输入、launcher 与 trainer 回执一致性、worker 计数及历史总账。

本机全套 **715 项测试通过**，32 项零积分静态检查通过，原 03/04 三份 release、77 项旧仓库保护文件与 9 份用户结果均通过核验。对新完成门和摘要工具做了 10 处隔离内存变异，最终 **10/10 被测试捕获**，没有修改仓库里的冻结文件。首轮 9/10：资格状态与通过数原先同时被破坏，掩盖了一个单字段测试缺口；拆分两个损坏回执案例后复验通过。原始首轮与复验日志均保留在本机 `scale09_closeout_checks_01/`。

## 预算

| 池 | 实用 transitions | 实用 native |
|---|---:|---:|
| E0 | 32,768 | 327,680 |
| 正式训练（含失败前消耗） | 4,807,168 | 48,071,680 |
| 资格、导出比对与本次关闭 | 0 | 0 |
| **合同 09 合计** | **4,839,936** | **48,399,360** |

合同 09 的上限为 8,421,376 transitions / 84,231,760 native；剩余额度随合同关闭失效。以合同 08 的累计 13,664,256 / 137,162,750 为起点，全项目当前累计 **18,504,192 transitions / 185,562,110 native**。最后的 `live_status.json` 早于停止，少记了最后一个未完成 rollout 的 2,560 transitions；总账使用最终 worker 计数。

## 后续边界

本轮没有关于熵假设的支持或反证，完整低速资格仍未通过。再次研究这一问题须登记新编号、新目录、新预算，并事前确定资源条件和停止规则。合同 09 的冻结文件、原始记录和中间权重保持原样。不会用同一合同重新训练、减少 worker 或改内存门来补结果。
