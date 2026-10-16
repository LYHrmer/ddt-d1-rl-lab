# 2026-10-16 续接：合同 09 已按未完成关闭

先读[合同 09 结果](career_execution_results_09.md)和[紧凑摘要](../reports/contract09_summary.json)。最新项目总账为 **18,504,192 transitions / 185,562,110 native**。以前的[交接文档](claude_handoff_20261011.md)保留历史过程，本页记录当前状态。

## 两个仓库

- `/home/lyh/ddt_ros2_ws` → `LYHrmer/ddt-d1-rl-lab`：接手时 HEAD `1429550`。Claude 的合同 09 正式训练在 2026-10-16 14:16 因 3 GiB RAM 门停止，实际 4,807,168 / 8,388,608 transitions，状态 `RAM_RESERVE_BELOW_3_GIB`。E0/E1 等价性均通过，训练未完成；Q1–Q5 不可判定，资格未执行。只有中间权重，**不得重跑或续训，也不得用它补评最终资格**。
- `/home/lyh/wheel-legged-control-lab`：HEAD `1d8177c`，两项工程修复已推送。基础安装缺少 `[rl]` 时基准可降级至 LQR/MPC，每次运行写 provenance。该提交的 [CI 已通过](https://github.com/LYHrmer/wheel-legged-control-lab/actions/runs/38780024034)，本次未更改其源码或重跑历史实验。

## 本次实际完成

1. 核验停止回执、冻结输入及训练 protocol，按最终 worker 计数记入失败消耗。
2. 修补 `tools09/verify_scale09.py`：E1 通过但训练未完成时，跳过所有评测读取，输出 `NOT_EVALUABLE` / `NOT_EVALUATED`。
3. 保存本地 `evidence/career_contract_09/scale09_verdict_01/verdict.json`；生成 GitHub 可读的 `reports/contract09_summary.json`；更新 README 与结果文档。本次 0 新 transitions / 0 新 native。

验证：D1 全套 715 项测试通过，32 项静态检查、历史冻结输入、77 项保护文件、9 份用户结果与本地文档链接均通过；10/10 隔离变异被捕获。轮足最新修复的 7 项测试和 lint 复核通过。

## 继续工作前

运行状态、远端、冻结身份要重新核对。不能因看到“后台命令完成”就假定训练 COMPLETE；读取 `result.json` 和 launcher 的 `trainer_returncode`。同样不能把 `live_status.json` 最后一条当最终账本。

冻结的[合同 09](career_execution_contract_09.md)仍是原文，状态由 release 与新关闭回执共同确定。原 03/04/05M/08 输入保持冻结；轮足仓库的 77 项保护清单中，只有 `experiments.py` 经所有者合法解冻并钉在新 SHA，其他 76 项与 9 项用户结果仍受保护。保护工具为 `tools04/check_handoff_integrity.py`；合同 09 摘要另核对其 release/protocol。

下一次训练须先另立合同与预算，并确定资源条件；不能搬用合同 09 剩余额度。此前六次完整训练均未通过平地整体资格。可靠越障、持续侧移、高速、watchdog 和实机仍未验收。
