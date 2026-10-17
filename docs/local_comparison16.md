# 本机低并行有界对照：准备方案 16

状态：准备方案，未批准新预算、未启动预检或训练。本文把[速度监督工程候选](velocity_candidate15.md)接到有界训练与固定评测入口。旧合同 09/10 保持关闭，中间权重不续训、未消耗余额不复用。

## 比较问题与固定条件

问题是：在相同表示与预算下，增加速度监督是否同时改善跟踪和停车？两组均使用候选 15 的历史编码器、当前 57D＋估计速度 3D 的 actor，以及原 627D critic；都保留编码器的 PPO 梯度。唯一组间科学变量为 `aux_coef`：control 为 0，velocity 为 1。真值标签不输入动作或价值网络。

环境继续使用未改动的 `pilot03d.json`，arm B、seed 101、随机初始化，不使用教师、旧权重、warm start、VecNormalize 或特权 critic。固定 PPO 为每次更新 4,096 transitions、batch 256、4 epochs、学习率 3e-4、gamma .99、GAE .95、clip .2、entropy 0、value coefficient .5、grad norm .5、初始 log std -1。

[准备配置](../training/configs/local_comparison16.json)默认选择 4 个环境，每环境 n_steps=1,024；也可在准备时明确选择 2 个环境、n_steps=2,048。同一准备包的两组与预检/正式训练必须使用同一环境数。正式前可以另行形成 2 环境准备包；不会在失败后自动降并行重跑，也不要求新轨迹复现历史 8 环境。

## 新预算上限

| 阶段 | 每组 transitions | 两组合计 | 两组合计 native | 每组进程时限 |
|---|---:|---:|---:|---:|
| 有界预检 | 32,768 | 65,536 | 655,360 | 900 s |
| 正式开发训练 | 1,048,576 | 2,097,152 | 20,971,520 | 5,400 s |
| 固定评测 | 1,800 | 3,600 | 36,000 | 300 s |
| 总计 | 1,083,144 | **2,166,288** | **21,662,880** | 见下文 |

PD 上限合计 10,831,440。每个 policy transition 为 10 native/5 PD；异常发生在区间中间时，保留实际尝试计数，不补成完整区间。任何预检、数值失败前的物理尝试都占本轮预算；不存在免费的实训内存测试。输出计数未知时保持 `null`，不能按退出码猜零。

表中训练预算是探索性开发预算，不保证足够学出合格策略。顺序为 control 预检、velocity 预检、control 正式、velocity 正式、control 评测、velocity 评测、只读比较。两组不同时运行。六个物理阶段时限合计上限 13,200 s（3 h 40 min），只读比较另限 60 s，另有事先固定的进程启动与有限清理宽限；不是承诺耗时。实际吞吐从预检报告，正式上限不能运行中延长。

## 资源门与停止规则

- CPU 执行，Torch/BLAS 单线程，无 GUI、无 GPU 训练任务；不关闭用户应用。
- 启动前有效可用内存至少 **6 GiB**，运行中至少保留 **3 GiB**。有效值取主机 `MemAvailable` 与当前可见 cgroup-v2 全部祖先硬限制剩余额度的最小值，不将 swap 当作内存。未知或读取失败时拒绝执行/停止。
- 新方案的 6 GiB 是待批准的启动条件，不是对合同 10 的 7 GiB 条件做回改，也不是已证明足够的内存预留。预检仍可能不通过。
- 监督器每秒观察资源，训练 callback 也检查；保留具体停止样本与阶段。操作系统采样不是内存预留，不保证捕获采样间瞬态或防止所有 OOM。
- 达到墙钟上限、RAM 门失败、信号中断、子进程失败、计数不匹配、来源变化、预检前缀不等价或清理不完整等工程失败，均停止整轮并记未完成；不进入下一阶段，不重试/续训，不以 partial 权重评测。
- 只向本次新建的进程组发有界 INT→TERM→KILL；保留历史输出，存在的执行目录不覆盖。

每组正式训练的前八个完整 rollout 必须与该组自己的预检一致，比较计数、分阶段奖励与策略状态摘要，不与历史 8 环境日志作等价要求。跨组还检查首次 PPO 更新前的共同初始化与首个 rollout。ledger 的策略摘要在当前 rollout 的 PPO 更新前记录，不能当作该次更新后的最终 checkpoint。

## 固定评测与判定

仅正式训练精确完成并通过计数、清理、来源与最终 checkpoint 身份检查后，才评测该组 `policy.zip`。预检权重和 `partial_policy.zip` 均不可使用。两组统一使用候选 SB3 CPU 确定性预测，标签填零且不影响控制；没有把一组 checkpoint 与另一组 ONNX 后端混比。

直接复用旧 evaluator 的 pilot 三序列、reset seed 7001、delay 0、每序列最长 12 s。沿用原轨迹指标、缺区间惩罚和资格阈值：稳定 vx MAE≤0.06 m/s、wz MAE≤0.08 rad/s，末秒水平速度≤0.02 m/s、|wz|≤0.03 rad/s，8–12 s 位移≤0.05 m、峰值 roll/pitch≤0.20 rad。短回合不会因为少测了困难阶段而得到“改善”。比较前按 trace SHA、逐步实际命令与来源重新核算。

策略提前跌倒是任务失败的观察结果，区别于程序/资源/计数错误。若三个计划 case 均已合法采完、实际计数与 trace 一致且不超过上限、来源及清理等检查通过，评测阶段可以标记流程 `COMPLETE`，同时用 `full_horizons_complete=false` 和原失败标志保留短回合。它不会阻止另一组的评测，避免只采到 control 而没有 velocity。最终比较仍要求六个完整回合；不足时只保留两组原结果，评分与改善结论均缺省为未完成，不外推缺失阶段。

预先规定两个汇总分数，均越低越好：

1. 跟踪分数：forward/reverse 的稳定 vx MAE 各除以 0.06，turn 的稳定 wz MAE 除以 0.08，三项等权平均。窗口保持原评测定义，不事后挑选。
2. 停车分数：每项任务的末秒水平速度/0.02、末秒 |wz|/0.03、8–12 s 位移/0.05 三项平均，再对三任务等权平均。

仅当六个回合全部完整、均无跌倒/非法支撑，velocity 的原阈值达标任务数不少于 control，且两个汇总分数各相对下降至少 5%，才记 `BOTH_IMPROVED_EXPLORATORY`。其他情况标明取舍、未展示收益或未完成；两组所有单项指标和资格标志均保留，便于查看具体任务的退化。control 分数为零时不能声称有正的相对改善。

这是单训练种子、单 reset seed、零延迟下的探索性比较，不是显著性或泛化结论，也不是 C++/实机部署资格。1M steps 的对照结果不能直接替代历史 8M 训练，更不能把预算差异归因于辅助监督。

## 准备、授权与执行边界

[共享工具](../tools16/contract16.py)记录新来源闭包及旧 72 项输入；准备包状态为 `PREPARED_NOT_AUTHORIZED`。新入口、测试与方案文档必须先提交且没有未提交修改，加载时也核对该提交中的实际文件字节。静态检查 PASS 只说明准备字段与来源一致，不能启动训练。无新 owner 批准记录时，trainer/evaluator/顺序器均拒绝执行。

准备命令不加载机器人或产生物理步，输出路径必须不存在：

```bash
rtk proxy python3 -B tools16/contract16.py prepare --envs 4 --output evidence/local_comparison16/prepared_01.json
rtk proxy python3 -B tools16/contract16.py check --release evidence/local_comparison16/prepared_01.json --output evidence/local_comparison16/static_01.json
rtk proxy python3 -B tools16/run_comparison16.py --release evidence/local_comparison16/prepared_01.json --plan
```

准备阶段只写未批准的授权模板。未来必须取得明确的新预算/环境数批准，记录 owner 原话与来源，才可通过 `freeze` 生成 `AUTHORIZED_FROZEN` release。不得自行将模板改成批准记录；这不是旧预算的重试许可。真正执行还需当场重新检查有效内存准入。

训练入口是 [train_comparison16.py](../tools16/train_comparison16.py)，评测与比较分别是 [evaluate_comparison16.py](../tools16/evaluate_comparison16.py)、[compare_results16.py](../tools16/compare_results16.py)；总顺序器是 [run_comparison16.py](../tools16/run_comparison16.py)。实际训练、评测必须由顺序器按同一新 release 运行；独立入口也会验证固定阶段路径与前置身份，不能绕过单次预算。源码不变与可回退方式是保留原冻结版本和输出，新尝试只写独立目录。
