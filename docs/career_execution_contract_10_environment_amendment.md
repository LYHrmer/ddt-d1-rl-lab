# 合同 10 环境修订草案：零步骤 IPC 失败的单次例外

**状态：待所有者明确批准；未获重试授权，不可据此启动训练。** 原合同 10、62 项冻结输入和 `_01` 失败回执保持原样。批准零训练 socket 探针不等于批准本修订。

## 已发生的环境失败

`scale10_preflight_01` 在 CPython forkserver 创建 Unix socket 监听地址时触发 `PermissionError: [Errno 1] Operation not permitted`，trainer 非零退出。监督回执显示清理完成、输入未变、0 个 rollout；正式训练、导出、资格评测均未启动。原始 `actual_counts` 和 `per_worker_counts` 为 null，保留此事实，不将其改成伪造的计数器回执。

真实 traceback 停在 `forkserver.ensure_running()` 的 `listener.bind(address)`，位于 forkserver 子进程创建和环境 worker 运行之前；结合独立进程组、无模型/回合输出和零 trained transitions，对本次实际训练/物理消耗作有来源的零值推导。原始观测值与推导值分别记录；如果证据检查不能证明该前置失败，就不能按此例外继续。

零训练探针已在沙箱内复现同一 bind 失败，在单独批准的沙箱外执行中通过。后续须先完成真实标准库 forkserver/pipe 往返探针，禁止加载模型或积分；该探针成功、有效可用内存重新达到 7 GiB，才允许新的训练 spawn。

## 所请求的唯一例外

请求所有者明确允许：将已证明为零步骤的 `_01` 环境失败登记为一次例外，在沙箱外按同一冻结科学方案、新的 `_02` 输出从头执行一次。它不恢复权重、不改变 seed、worker、PPO、奖励、阈值或预算，也不修改原合同/冻结回执。除这一明确列出的环境失败外，不新增一般性的失败重试权。

两次合计仍受原已批准上限约束：**8,421,376 transitions / 84,231,760 native**，其中资格池 18,000 native。`_01` 零消耗的推导必须事前通过审核；新执行仍是 E0 32,768、训练 8,388,608、最终 checkpoint 三任务评测。资源停止门维持 3 GiB，运行不完整或任何新失败都不续训、不再重试。

## 新版本身份和执行边界

新适配层及测试、本修订、所有者批准记录、原 release/失败回执和原 62 项 SHA 一起绑定到独立 `_02` release。旧源码只通过受限上下文复用；若需要映射固定 release/输出路径，进入和退出均校验并恢复，禁止改写旧源码或原字节。新 journal 使用独立路径，已开始阶段不得重复执行，NOT_READY 仍不消耗 spawn。

默认入口只打印计划。只有所有者明确批准包含环境例外说明的具体沙箱外执行命令后，显式批准参数才可登记授权、生成新 release 并执行。工具权限批准必须同时明确覆盖这一单次例外；不能把其他命令或零训练探针的批准沿用到训练。

后续顺序和科学判定与原合同 10 相同：E0 → 正式训练/E1 → 导出/C++ 数值链 → 同一最终 checkpoint 资格评测 → 零积分重放 → 判定和总账。原 `_01` 的错误和判定状态继续公开保留。

## 待批准的具体入口

[适配层](../tools13/contract10_environment_amendment.py)及[36 项合成回归](../tests/test_contract10_environment_amendment.py)已完成。正常调用不带执行参数，只读展示 PLAN_ONLY，不生成所有者记录、02 release 或训练目录。

下面命令只有得到所有者明确批准后才能在沙箱外执行；显式参数表示本次环境例外的授权，不声称程序能读取宿主审批界面：

```bash
rtk proxy /usr/bin/python3 -B tools13/contract10_environment_amendment.py \
  --authorise-environment-exception --execute
```

本机全套 999 项测试通过；干净源码副本没有本地 evidence、模型或私有依赖目录，970 项通过、29 项因缺少历史资产而跳过，其中新增 36 项核心测试全部通过。独立审查复现并验证了 IPC 中断清理修复，确认无自有残留进程。旧 62 项输入、77 项保护文件和 9 份用户结果核验通过。这些检查只完成工程准入准备，尚未生成真实 02 release 或启动训练。
