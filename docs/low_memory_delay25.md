# 25 单环境延迟对照准备

本 scope 是新的独立本机实验：从原完整 control 的同一组 policy 与 Adam 状态分别启动 zero 和 uniform 0..3 两组，保留 LR 1e-4、target-KL .005，改用单进程 `DummyVecEnv`。它不重试、不续训 24，不使用任何旧 partial 权重；当前文档与代码不代表已经执行或取得策略改进。

24 已因 4 GiB 运行内存门中止，见 [独立中止核查](delay_comparison24_interruption_20261018.md)。减少环境数旨在减少额外进程私有页、环境模型数据与 IPC；原 forkserver 已共享预加载的只读页，不能把四个 RSS 相加当四份私有 Torch，也不能预先保证能省多少内存。

## 固定科学与数值范围

两组均从原完整 control checkpoint `428ff0ce9e8e88e10dabf94801a57b88de7e76d07b42800553b69c7f590e4eff` 及其完整 Adam 恢复，初始步数 1,048,576、原 `_n_updates=1024`，seed 181、env 1、aux 0。每组新增 262,144 transitions；每 rollout 为 1×4,096，共 64 次，batch 256、4 epochs。其余 PPO、观测、奖励、课程、网络和 FIFO 执行时序沿用冻结实现，组间只改变每 episode 的训练延迟模式。zero 也保留延迟状态记录。

实际加载必须通过 `AuxiliaryVelocityPPO.load(base, env=DummyVecEnv1, seed=181, device='cpu', force_reset=True, n_steps=4096)`。SB3 在 `_setup_model` 前写入新 env 数与 `n_steps`，重建空的 `DictRolloutBuffer`，随后恢复 policy 与 Adam。运行时核对 buffer 的 shape/dtype、空指针、无旧 `_last_obs`、原 policy/Adam/update SHA，再只设置 LR 与 KL，证明 optimizer momentum 和 policy 未改变。旧 18/22 的四环境加载 validator 不用于新布局。

target-KL .005 沿用本地 SB3 PPO 语义：某 minibatch 近似 KL **严格大于 .0075** 时，跳过该批梯度更新并结束该 rollout 的剩余 epochs。它不回滚此前 minibatch，也不保证最终全局 KL 上限；epoch 计数不能当作 optimizer step 数。

单环境改变了向量采样、rank 轨迹和 GAE 边界。即使总 rollout 仍为 4,096，也不能称 25 对 24 只有延迟一个变量；只允许在新 25 的两组之间作同 seed 的候选筛选。没有多训练 seed 的显著性结论。观测 buffer 主要项 4,096×(627+3)×4 B 仍约 9.84 MiB，不因环境数下降而缩小。

## 独立入口与边界

新增计划固定为 `evidence/low_memory_delay25/plan_01.json`，输出固定为 `evidence/low_memory_delay25/execution_01`。源码必须先提交并逐 git blob 核验；另外单列并绑定本地 SB3 的 loader、buffer、DummyVecEnv 等依赖源码 SHA，未分发的 `.python-deps` 字节不冒充 Git 源码。旧完整 baseline 的 provenance 仍通过原只读接口验证。

默认命令仅给计划摘要，0 NN/0 physics：

```bash
rtk proxy python3 -B tools25/common25.py --plan
rtk proxy python3 -B tools25/run_delay_comparison25.py
```

源码稳定并提交后，根线程准备新计划，经外部 Nice 10、systemd user transient unit 的单次监督入口串行执行。准备过程本身不创建 unit、不安装依赖、不自动降环境数。后续固定命令为：

```bash
rtk proxy python3 -B tools25/common25.py --prepare
rtk proxy python3 -B tools25/run_delay_comparison25.py --execute
```

训练和各后续 stage 都需要当前父监督器的 PID、随机 nonce、计划 SHA、固定 argv 和前序完成记录。固定输出存在即拒绝，不允许续跑或覆盖。启动门 6 GiB，运行门 4 GiB，CPU 单线程，Nice 至少 10；父整轮 5,400 s，每组训练 1,800 s、导出 180 s、CPP 90 s、每个评测矩阵 300 s。父拥有新 session，任何异常均有界清理其自有进程树，外层 unit 还需限制总时长与 cgroup 生命周期。

整轮最初不足 6 GiB 仍返回 `NOT_READY`，不占输出或 attempt。整轮开始后，只在上一阶段已清理且无活子进程、尚未写入下一阶段 STARTED/nonce 时，若 RAM 在 4–6 GiB 之间，允许每 5 s 仅采样、记录 `admission_waits` 与检查父时限，最多等 300 s。到达 6 GiB 后重新核对源码、计划和既有工件，再唯一启动该阶段；低于 4 GiB、读值未知、等待耗尽、总 5,400 s 耗尽或终止信号均停止。等待全部算入父总时长，不计入尚未启动的子阶段时限。子进程自己的 6 GiB 门不变；如果准入后发生资源竞态导致子进程拒绝，整轮即停，绝不重试该 stage。

单进程运行时没有环境子进程、pipe 或 forkserver。metadata 仍执行严格顺序与 schema 校验：pending step、dirty、并发调用或超时之后不得继续读取候选计数/延迟状态；只有 JSON-ready 的验证结果可写入精确事实。失败时实际计数未知保持 null，合法完整前缀另记 known，绝不把 unknown 当 0。原健康路径的 wrapper 顺序和逐拍数值计算保持。

## 计数、资格与筛选

顺序为两组训练，然后各自 export→CPP64→原 36 例；只有某组原 36 全过，才对该组额外执行 36 holdout（seed 2001/2002/2003），不要求两组都过。最多新增：

| 部分 | policy | native | PD |
| --- | ---: | ---: | ---: |
| 两组训练 | 524,288 | 5,242,880 | 2,621,440 |
| 两组原 36 例 | 43,200 | 432,000 | 216,000 |
| 两组有条件 holdout | 43,200 | 432,000 | 216,000 |
| 独立总上限 | 610,688 | 6,106,880 | 3,053,440 |

导出和 CPP64 只有数值推理，不计物理；CPP64 对齐不是 CPP 闭环。正式与 holdout 使用既定 Python ORT 单物理执行及旧阈值，合法提前终止照原指标归档，不当成工程失败，也不因短时域低 J 声称改善。比较另列共同完整时域配对及所有 36 例（含失败）的指标。

只有新增训练完整、1 worker 精确计数完整、最终权重保存成功、输入未变、清理完成及单进程 guard 已安全关闭，才可导出 `policy.zip`。其余仅 `partial_policy.zip`，不可评测。每个非训练回执必须桥接本 arm 的完成训练、checkpoint、protocol、计划与新源码；评测再桥接本 arm 的 ONNX、fixtures、导出与 CPP 回执 SHA，另一组或旧模型不能仅改标签进入。

两组实际初始状态与运行库版本必须完全相同。只有自身原 36 与 holdout 均全过的最终权重可以进入候选筛选，按原 36 平均 J 选择、精确同分选 zero；不自动替换现有基线，不宣称实机资格。任何内存中止也只能解释为资源中止，不能替代延迟/KL效果的资格结论。

## 本轮验证边界

只运行纯 fake/metadata 测试：不加载真实权重、不创建神经网络或机器人、不执行 MuJoCo 积分。针对性回归覆盖新布局/依赖漂移、原 Adam 身份、64 rollout 与单 worker 分母、4 GiB 资源中止、构造期所有权恢复、候选计数污染、dirty/pending 元数据拒绝、清理失败、nonce/单次输出、跨组工件桥以及合法短任务与条件 holdout。真实资源节省和策略效果必须等新的有界执行才可报告。
