# D1 28：共同输入策略漂移的实际结果

28已真实COMPLETE：3次weights-only读取、3次216行actor batch，共648样本actor前向、18次Linear batch，耗时1.102316秒；新增训练和物理均为0。它确认两个27候选在这一固定输入集上均已偏离完整random25，**偏离几乎全部来自动作分布均值**。这支持优先检验固定基线锚定约束，不支持直接继续加wz权重、缩小log_std或增加训练步数。策略退化的因果尚未证明，默认策略未替换，27无eligible/holdout/新候选C++闭环的结果保持。

本页由[11KB紧凑摘要](../reports/policy_drift28_results_20261018.json)支持；仅保存数据分析脚本位于本地工作目录`codex_workspace/local_optimization18_20261018/analyze_policy_drift28_saved.py`。脚本只读取小JSON与627558字节的已有NPZ，重算闭式公式；没有导入Torch/SB3/Gym/MuJoCo/ORT，没有读取权重、执行NN、训练或物理。NPZ解压payload626144字节，分析进程峰值RSS32,669,696字节，低于128MiB。

## 实际来源和分母

源码commit为`75ee3e323c5328bc38763fac170c1a7f55ca3294`，现场固定为`/home/lyh/ddt_ros2_ws/evidence/policy_drift28/execution_01`。report SHA为`fa9ad64ac9b2685d17eb62a276b9a0eb2347d2f5cf0d800fb75f4ac5871ede1a`，distributions SHA为`1229d9e1d8b90a14af296acf4110b69ae16622a4217219baeb8a8fc44d0eb9e9`；本次分析重哈并匹配report记录。完整输入/源码身份继续由原28回执绑定，分析没有改写它们。

输入是原23旧control闭环保存的18个完整例：forward/reverse/turn ×0/20ms ×1001/1002/1003；每例固定12拍，drive/stop各6拍，共216个float32/627D输入，每个task×window组36行。原输入来自**旧control的状态分布**，不是25/27的on-policy分布，也不包含40/60ms。各组都完整纳入，不按输出、失败或KL值筛选。

三个原始策略为完整random25 `873cd14f…cdb4f22`、27 control `be727124…4c733ba`、27 yaw3 `b42dec19…c5cab57`。读取真实16维log_std，并执行完整627→3编码器及60→16actor；critic与辅助监督损失不执行。648是样本数，实际actor函数是3次batch调用，不能写成648次调用。

## KL主要是均值变化

计算的是**未clip的对角Gaussian分布**`KL(base25 || new)`。令a/b为base/new的log_std，逐输入精确分为：

`mean_part = Σ (μ_base−μ_new)² / (2 exp(2b))`

`variance_part = Σ [b−a + 0.5 expm1(2(a−b))]`

两部分之和与实际28report的mean KL在1e−12内相符。log_std不依赖观测，所以variance_part对216行相同；这不是将平均std代入公式。clip后确定性mean差异另算，不能把Gaussian KL称为clip后动作分布KL。

| 共同输入指标，216行/3456动作元素 | control27 | yaw3_27 |
|---|---:|---:|
| mean KL(base25→new) | 0.762465067 | 0.717618403 |
| 均值变化分量 | 0.762248403 | 0.717455817 |
| 方差变化分量 | 0.000216665 | 0.000162587 |
| 均值变化占比 | 99.9716% | 99.9773% |
| raw mean绝对差均值 | 0.064812868 | 0.057521229 |
| clipmean绝对差均值 | 0.064394730 | 0.057192356 |
| log_std绝对差均值，16维 | 0.003286570 | 0.002744079 |

方差变化不是这批共同输入KL的主要来源。clip只小幅缩小均值差，不能把观察到的偏离解释为clip之前的变化全部被动作裁剪消除。rawmean到达±1端点的元素为base25 27/3456、control53/3456、yaw3 42/3456；这是固定输入的动作端点统计，**不是力矩饱和率**，也不能与原闭环21600拍的计数混用。

| task×window，每格36行 | control KL | yaw3 KL |
|---|---:|---:|
| forward drive | .937749 | .917495 |
| forward stop | .775049 | .553320 |
| reverse drive | .730493 | .960551 |
| reverse stop | .785451 | .635806 |
| turn drive | .581022 | .662879 |
| turn stop | .765028 | .575659 |

yaw3整体KL较control低，但reverse/turn的drive组反而更高，因此一个平均KL不能给策略排性能名次，更不能说明哪个阶段的漂移导致哪个失败。以action输出的零基索引看，yaw3的第2维独占mean-shift分量约0.30484；本稿不凭索引猜测机器人关节语义，完整16维分量保留在摘要。

原27训练CSV平均approx_KL约.0036/.0035与本次.76/.72不是同一统计：前者是当次rollout旧策略到当前更新的估计，后者是固定base与final在另一组有限共同输入上的完整对角Gaussian KL。`target_kl=.005`只管局部更新，并不承诺固定基线累计KL上限。此处可以确认局部与固定参照的约束不同；不能相除称“违反KL门多少倍”，也不能由跨分布的数值大小证明性能因果。

## 能确认的边界

原36资格仍为base25 33/36、control27 24/36、yaw3 18/36；后两者无eligible，未执行holdout。所有27原36均完整12秒。yaw3平均转向误差改善只相对本轮control，两个候选都弱于base25；28不是新任务评测，也不改变这些结论。27的CPP64是数值对齐，未取得完整新模型C++闭环资格。

本结果支持“优先限制相对固定base的偏移”作为一个可证伪候选；仍无法区分轨迹/课程暴露、不同seed、继续学习自身或reward加权在性能下降中的独立贡献。对未采样的40/60ms状态、高维全部观测分布、转向梯度信号或失败因果，结论均不能从216输入外推。

## 下一项只拟一个变量：固定base KL锚定

建议新的独立小配对：都从完整random25及Adam起步，seed183、DummyVecEnv1、4096拍rollout、uniform0..3延迟、原03d课程/obs/action/reward（wz权重2）、LR1e−4、target_kl.005、aux0保持相同。唯一臂差为`β=0`与`β=1`，加入：

`loss = 原PPO loss + β × mean_states KL(π_fixed_base || π_current)`

这里的π_fixed_base始终是873cd14f…的真实actor Gaussian，不是上一个rollout策略。方向、按维求和/按样本取平均与28一致；β1是预注册工程试验值，不宣称本次数据已推导出最优系数。此方案不叠加wz3、熵修改、课程修改或更多候选系数。

工程上用新模块接PPO总loss，不改15/27冻结源。每rollout用保存的实际4096个actor输入对冻结base作一次batch预测，缓存mean/log_std并与buffer行严格对齐，教师不反传；当前mean/log_std从原训练同一次分布forward取得，不能暗中再多跑一次actor。新增项要能反传到现有encoder与actor，原target_kl局部早停语义保留。β0必须与原PPO数值路径等价；两臂初始policy/Adam/版本相同，首rollout非reward六数组SHA相同。不能把已有aux velocity_target当作教师，或让真值进入actor。

建议每臂仅32768新步（8个rollout），合计65536新训练步；各一次完整原36开发回归，最多43200评测步。**初始机制范围总上限108736 policy /1087360 native /543680 PD**，0额外holdout、0真实CPP36；不按中间checkpoint挑模型、不重试已启动arm。每臂最终保留原导出与CPP64数值门，正式36仍必须如实标为Python ORT。单臂训练≤600秒，整轮≤1800秒，按用户最新资源偏好采用4GiB冷启动/2GiB运行保留、owned3GiB、Nice10和串行运行；这是方案上限，尚未新建计划或执行。

额外教师actor明确上限32768样本/8次4096-row batch（仅β1臂；β0不做无用教师前向）。训练既有actor/critic及early-stop实际调用继续计账，不能拿“物理步不变”称额外NN为0。若复测28固定集，另登记3次216-row batch/648样本；各臂原64fixture数值链也单列，不能藏进教师数。缓存不复制环境、不添加物理步，也不增加rollout长度。

先用纯合成反例验证β0等价、删掉anchor项会使“仅anchor更新”测试红灯、教师参数无梯度、variance/mean KL方向及buffer对应、首批公平、最终计数/清理。正式结果须同时看有限共同输入KL和全36原任务；**KL下降但任务继续退化就是此方案的负结果**，不能只用KL改善宣布完成。未过原36者不消耗holdout、不取得资格、不替换默认策略；若后续确有完整36合格者，再按独立明确范围使用保留的holdout和候选实际C++闭环，不继承旧baseline的资格。

以上仅技术设计，不是29源码、预算执行或新能力证明。当前下一项已完成的事实止于28零训练/零物理审计。
