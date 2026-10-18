# D1 27修订02：实际结果与保存轨迹诊断

27整轮工程执行为COMPLETE，**两个候选都未通过原36资格**。control为24/36，yaw3为18/36，均低于完整基线random25的33/36。无eligible、selected_arm=null、holdout_36_pass=false、simulation_candidate_qualified=false；两份holdout按`SKIPPED_NO_QUALIFIED_CANDIDATE`跳过，没有选择替代臂，条件性C++闭环接续不执行。默认策略未改变。

本页及[紧凑摘要](../reports/yaw_reward27_results_20261018.json)只分析已有数据；完整本地诊断另保留逐例窗口和全部输入SHA，位置和身份已登记在摘要中。原执行计划、现场回执和旧结果保留。源码为`e4382a00744f6a24ac27682bfd6f90b457054e0e`，真实根为`/home/lyh/ddt_ros2_ws/evidence/yaw_reward27/execution_02`。本地分析脚本`/home/lyh/codex_workspace/local_optimization18_20261018/analyze_yaw27_saved.py`仅stdlib/NumPy，输入逐文件stat/SHA后读取；没有加载Torch/ORT/MuJoCo、策略或环境，没有新推理、训练、积分或测试。

## 完成的是哪些执行

- outer为COMPLETE，868.7925秒；parent为COMPLETE，851.5029秒，error=null、cleanup_complete=true，actual=known=**305344 policy /3053440 native /1526720 PD**。
- 两组都从完整random25 ZIP `873cd14ff67e7d399472d97a94e082f7a97d068b00db414986443e50fcdb4f22`及Adam起步，1310720→1441792，各新增131072；各32个4096步rollout。其余43200步来自两组完整36正式评测；不是训练或holdout。
- 两组initial_state整对象（policy/Adam/序列化成员/版本/seed变更/单env布局）相等；first-rollout六个非reward数组身份完全相等，联合SHA为`d9c7002dfc5e27190c5be962151928fd51a648e4fe87aee8238e3439f060ef1a`，两份first-rollout报告SHA也相同。它证明首次更新前配对，不证明更新后轨迹或整轮观测相同。
- 两个final都实际导出并通过64个C++数值fixture门，绝对误差低于原1e-5。正式36仍是Python ORT动作驱动；**CPP64不是新模型的C++闭环**。`cpp_closed_loop_qualified=false`保持。
- 两组正式36全部600拍/12秒，没有提前终止；失败不能归为工程未采齐或RAM中止。两组训练最终源未变、计数闭合、关闭成功。

## 固定36结果与具体退化

每格分母均为9（前进、倒车、转向各3个reset seed），原阈值保持。

| 延迟 | 完整random25 | 27 control，wz权重2 | 27 yaw3，权重3 |
|---|---:|---:|---:|
| 0ms | 9/9 | 6/9 | 6/9 |
| 20ms | 9/9 | 6/9 | 6/9 |
| 40ms | 9/9 | 6/9 | 3/9 |
| 60ms | 6/9 | 6/9 | 3/9 |
| 合计 | 33/36 | 24/36 | 18/36 |
| 完整12秒 | 36/36 | 36/36 | 36/36 |
| 全36平均J | 0.215939 | 0.232595 | 0.250864 |

control失败项恰为12个turn的稳定wz误差；yaw3也是12个turn误差，并额外产生forward40/60ms的6个停车净位移失败。reverse的12例在两组均通过。控制组同原reward继续训练也退化，因此不能把全部退化归因于wz2→3；但新两组之间真实配对支持比较该额外加权的有限效果。

转向稳定指标对应6–8秒反向段，目标−0.2rad/s。每格为三个seed的均值，门为单例MAE≤0.08：

| 延迟 | base25 MAE | control MAE | yaw3 MAE | control实际wz均值 | yaw3实际wz均值 |
|---|---:|---:|---:|---:|---:|
| 0ms | .066315 | .119733 | .103787 | −.08191 | −.10153 |
| 20ms | .072124 | .122001 | .107157 | −.07919 | −.09674 |
| 40ms | .077598 | .123211 | .108919 | −.07764 | −.09353 |
| 60ms | .089495 | .125172 | .113447 | −.07588 | −.09098 |

yaw3相对本轮control降低了转向MAE，但仍显著欠跟踪，并没有恢复base25的低延迟转向能力。control/yaw3在60ms反向段std均约.049rad/s，反而低于base25的.0713；所以不能沿用旧基线60ms“振荡更大”叙述解释本轮全部退化。主要新现象是**0ms已经出现的反向幅值不足**，并非只在3拍延迟下发生。

yaw3前进停车净位移均值由0/20ms的.04251/.04776m增至40/60ms的**.05272/.05851m**，三个seed均超原.05m门。control对应40/60ms为.02449/.02839m，base25为.03582/.04009m。这是加权后的任务取舍，不能只汇报转向误差改善或训练reward变大。

## 实际训练分布：观测到什么，不能推出什么

`Watchdog.on_step`按实际pre-step课程给每拍标签；每rollout严格4096，以下分母均为实际policy ticks，不是episode hazard。没有从命令随机数重生成缺失数据。

| 已保存训练tick比例 | base25 random，262144步 | control，131072步 | yaw3，131072步 |
|---|---:|---:|---:|
| translation drive | 25.41% | 29.01% | 29.04% |
| turn drive | 25.88% | 21.70% | 21.70% |
| 切零后前25拍 | 4.97% | 5.02% | 5.03% |
| 初始/后续静止 | 43.74% | 44.27% | 44.22% |

新两组实际turn ticks均28446；control的translation为38024、yaw3为38068，零指令约49.3%。课程配置仍是episode translation/turn各0.5，turn每段100～150拍、translation每段200～300拍；**episode模式概率不是实际tick比例**。base25用seed181，新的两组共同seed182；“同seed”只对本轮两组成立，跨轮不是同一命令/动作随机流，训练长度和阶段也不同。

delay直方图是**reset次数**：control 30/31/33/38（共132次），yaw3 30/31/34/38（共133次）。不能把它当每拍延迟暴露分母。它符合0..3均匀抽样规则，但有限样本并非恰好四等份。

wz加权作用于全部训练拍，不只转向。yaw3新增reward总和2076.29459，与`.02 × raw_wz_tracking_sum`一致；按保存phase contributions分解，约80.5%的额外正向reward发生在wz目标为0的translation/静止/切零拍。这个结果说明**奖励改动没有专门集中于转向问题**，不证明这些拍贡献了80.5%的policy gradient，也不能从reward均值推导PPO信噪比或确定失败因果。

## PPO漂移与饱和候选

1. **累积策略漂移是当前最值得先验证的假设。** 原reward的control继续32个rollout后也丢掉全部12个turn通过；两组初始与首批非reward数据相同，版本/输入和清理均正确。CSV可见31条更新统计：平均approx_KL为control .003601、yaw3 .003474，平均clip_fraction为.0386/.0361，std约.2545。`velocity_policy15.py::train`的KL是本批rollout旧策略与当前更新间的量，超`.0075`只跳该批并停止余epoch，不约束相对完整random25的累计KL，也不回滚；小局部KL不保证32轮后保留既有能力。这里**尚未测同输入base→final KL**，不能把假设写成已证明机制。
2. **实际课程暴露是混杂候选，不是已证实代码bug。** 新turn tick比例较base低约4.18个百分点，reward增量大量落在非turn拍；但seed/长度不同，单轮训练不能区分随机课程偏移、长期续训和reward之间各自作用。没有依据先改观测维度、delay FIFO或直接堆更多训练步。
3. **动作端点裁剪增多，但没有力矩饱和证据。** 108份正式NPZ（每模型36）SHA均与case报告匹配；每组分母345600个issued-action元素，达到±1的数量为base25 1502、control 2439、yaw3 4216。yaw3前进尤其明显：2917/115200，base25为0。三组每组21600个policy interval的`tau_saturation_count>0`均为0；env源码的`_last_saturation`是该interval所有PD更新的OR。端点裁剪与力矩限幅是不同概念；闭环轨迹也已不同，不能把裁剪当独立因果。
4. **观测裁剪/隐藏输入问题记为NOT_OBSERVED。** 这些正式trace未保存actor627输入，first-rollout只保存六数组SHA，没有全轮raw观测；本轮不能凭动作日志计算观测饱和率。初始配对与既有FIFO核验不能替代整轮观测审计，也没有出现足够证据支持更改观测schema。

CSV在下一rollout更新前dump，32轮只留下31条train统计，最后一次更新不保证在progress.csv中。不能由这些平均值断言所有batch都低于KL门、未发生early-stop，或捏造实际Adam步数。没有中间checkpoint的独立能力评测，不能定位“第几轮开始退化”，也不事后挑中途权重。

## 只推荐一项下一步

**等Wheel42重任务结束后，先做一次零物理的固定基线参照策略漂移审计，不立即再开训练或继续加wz权重。**

固定使用原23已保存0/20ms的18个完整case，各取预先登记的12个drive/stop拍（例如index125/150/175/325/350/375及425/450/475/500/525/575），共216个真实float32/627D输入。分别对完整random25、27 control final、27 yaw3 final作相同输入的actor分布前向，最多**648次actor forward，0新MuJoCo步/0训练**；比较原始mean、clip后mean、log_std和闭式对角高斯`KL(base25 || new)`。不用最终reward、更多评测seed或holdout来挑权重。

这一步直接检验“局部KL小但相对完整基线已漂移”，并能区分mean变化、variance变化与clip后动作变化。原23输入来自旧control轨迹，必须标为固定诊断参考集，不称25/27的on-policy分布、性能资格或因果证明。若该证据支持明显base偏离，再为单一基线锚定约束设计新的有界实验；这份诊断不预先确定系数或执行新训练。若不支持，则不把同一个漂移叙述强套到下一轮。

## 资源与审计边界

2480条保存resource JSONL记录（包含不同进程对同unit的重复采样）中，已读到的kernel `memory.peak`最大为**1,090,940,928B，约1.016GiB**；`memory.current`采样最大1,090,154,496B。它们不是unit退出后重新取得的最终完整peak，也不是把各进程RSS相加。high/max/oom/oom_kill等保存事件最大值均0；已读最小外部有效可用量4,342,136,832B，高于4GiB。没有证据把性能退化归为本轮RAM保护或OOM。

分析逐份NPZ打开，单文件最大解压payload205248B；连同数组交换和小型派生量保守估计不到1MiB，实测分析进程自身peakRSS约36～37MB，低于128MiB限制。输入JSON/CSV/ledger/资源日志及108份trace的路径/bytes/SHA保存在机器摘要中；SHA检查和描述性数组统计不等于本次重新执行完整frozen evaluator/native审计。实际来源、全阶段计数和完整资格仍由原27父/子终态回执约束。

新CPP接续的前提已失败，因此不执行此前已登记在本地工作目录的条件性方案。旧17/23原control范围、25 random33/36、原默认策略与27所有失败记录继续保留；工程COMPLETE不替代完整资格。
