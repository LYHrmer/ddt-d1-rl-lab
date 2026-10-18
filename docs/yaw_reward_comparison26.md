# D1 26：等预算 yaw 奖励配对筛查

25 random已完成原36例评测，33/36通过；60ms转向三例仍未过yaw稳态MAE≤0.08rad/s。保存轨迹表现为欠跟踪叠加波动，不能据此判定奖励权重不足、继续训练必然有效，或直接放大动作增益。

26只检验一个候选：在相同随机延迟训练中，提高yaw tracking权重能否比继续原奖励训练更好。它是独立的新预算和新输出，不修改、续写或重跑25及其他旧attempt。

## 两个 final-only 候选

两组均独立加载25 random完整最终checkpoint和Adam，模型ZIP SHA为`873cd14ff67e7d399472d97a94e082f7a97d068b00db414986443e50fcdb4f22`。原ZIP的policy/optimizer序列化条目也独立绑定；不能拿25的初始Adam冒充其最终Adam。

| 项目 | control | yaw3 |
|---|---|---|
| 新增训练 | 131072 transitions | 131072 transitions |
| yaw tracking权重 | 2.0 | 3.0 |
| delay | 每episode独立均匀0/1/2/3 ticks | 相同规则 |
| 新训练seed | 182 | 182 |
| 向量与rollout | 单进程DummyVecEnv，1env×4096 | 相同 |
| PPO | LR=1e-4、target_kl=.005、aux=0，其余原参数 | 相同 |

累计`num_timesteps`从1310720到1441792，`reset_num_timesteps=False`。两组保留同一完整policy、Adam状态、LR和KL设置；没有重新初始化或把LR/KL再改一次。只使用预算结束后的最终checkpoint，partial不能进入导出或评测，不选最佳中途模型。

权重变化作用于**所有policy ticks**的`wz_tracking`项，包括wz命令为0的平移与停车，不能称为只对转向阶段加奖励。原03d命令、幅值0.1～0.3rad/s、双向2～3s保持、reset、观测、16维动作映射、PD/native时序及其他奖励项保持原实现。新训练wrapper在Monitor内更新真实PPO reward、wz贡献和总和，并累计实际差额；不是只改日志或评测命令。

相同初始policy/Adam及版本须由两组实际收据相互核对。首次rollout在PPO更新前对已有buffer中的观测actor/target、动作、episode starts、value和log-prob做非奖励哈希，两组须一致；不要求刻意不同的reward或returns一致，也不增加推理或物理来制造配对证明。

## 固定顺序与唯一 holdout

顺序为两组train，然后control的export/CPP64/formal36，再yaw3的export/CPP64/formal36。每个阶段都是新独占目录、一次spawn，并由父report的当前stage、来源、nonce和精确argv授权；单独拿计划调用训练或评测不能跳过顺序。

先收齐两组开发36例。合法自然早倒是任务结果，仍收齐另一组；导入、来源、数值、计数、清理或资源异常是工程失败，立即停止整轮。

只有本组开发36例全部600拍、完整矩阵且通过所有原阈值，才进入eligible集合。在eligible者中按开发36例mean J最小选一个，精确相等选control。零eligible则不做holdout。选举写入父report后保持不变，不能使用holdout挑选winner。

只有选中组可以执行一次holdout36，reset seeds固定2001/2002/2003；另一组槽位明确跳过。holdout失败后结束，不换另一组、不换seed、不重训或改奖励重试。最终仿真候选须开发36与唯一holdout36均全过；工程`COMPLETE`不代表资格通过，也不自动替换现用模型。

原36例已经用于多轮开发，只能作为开发/回归集。准备期只读核对本仓及原16 replacement仓的`evidence/reports`中seed/reset-seed实际case记录，并检查保存trace：未找到2001/2002/2003的已执行评测case或对应NPZ；21、22、25的holdout回执均为跳过，24未到该阶段。这个结论只覆盖现存已登记工件，不证明其他目录或未归档实验从未用过这些数字。

因此它们是保留的一组reset验证，**不是隐藏seed或盲测**：数值早已公开于旧计划，命令结构也与开发集相同，不能宣称未见命令分布泛化。若发现既往执行/据其调参的记录，应在启动前重新登记独立验证方案，不能执行后换seed。本组只有一个共同训练seed；三个评测reset seeds不能代替多训练seed重复实验。

## 参考结果与反证

父报告绑定原17 baseline及25 random正式report和保存trace，按原纯指标函数离线核验。比较同时保留实际失败数、完整时域数量、含原失败罚分的all36 J，以及四组/逐参考共同完整case集合的J和明确分母。不能用早倒短回合降低误差，或把含巨大失败罚分的改善混称为正常跟踪同幅度改善。这些参考说明不新增、放宽或替代原资格门。

若control同样改善，就不能把改善归因于加yaw权重；若yaw3仅拉近均值却增加波动、早倒、平移或停车退化，就不支持该候选。训练reward升高本身不能作为成功。开发集未过门时，不因分数“接近”而扩大预算或开放holdout。

## 预算、资源与关闭

最大新物理预算为：两组训练262144，加两套formal43200，加唯一holdout21600，共 **326944 policy transitions / 3269440 native steps / 1634720 PD updates**。holdout未执行的预算不能转为额外训练或重试。

每组CPP64是同一最终权重ONNX的独立数值验证，物理计数0，使用原1e-6/1e-5误差门。正式与holdout仍由Python ORT驱动原MuJoCo环境。本合同**不包含真正C++在环36例**，不宣称完整C++闭环、旧19资格、ROS发布或实机资格。若以后需要真实C++在环，必须另行固定新来源和预算，不能暗用本合同余量。

CPU单线程、单env、串行阶段、Nice10；启动/阶段准入至少6GiB有效内存，运行中每秒监测4GiB停止线。阶段间没有活child且原阶段清理完成时，4～6GiB可最多等300秒，每5秒采样；低于4GiB、未知内存、超时或信号立即停止。等待计入总3600秒墙钟，源码与旧工件读回后、stage claim前再次验总时间。子进程拒绝或退出后不能重新spawn。

父进程仅清理自己创建的session/进程组和adopted descendants，使用有界停止；没有handle的spawn异常也扫描自己的adopted树。真实部分计数按最小来源身份与typed counters保留，缺失计数继续unknown/null，不能补0或整rollout。阶段事实用于幂等重建总数，避免清理/写报告期间的信号重复收费。

默认命令仅显示计划，不加载NN、创建环境或执行训练：

```bash
rtk proxy python3 -B tools26/run_yaw_reward26.py
```

实际执行前必须提交并核对本合同全部来源、独立准备固定`evidence/yaw_reward26/plan_01.json`，再通过有明确总时限、Nice10及owned-cgroup清理的独立user transient unit启动`--execute`。固定输出为`evidence/yaw_reward26/execution_01/{control,yaw3}/{train,export,cpp,formal,holdout}`，已存在目录不得覆盖。本说明是合同，不是执行或通过收据；实际状态只认对应report和SHA。
