# 24：KL保护下的零延迟／随机延迟续训对照

本方案处于源码准备阶段，尚未训练或评测。旧09、10、18、22等尝试保持封闭；不得重跑、续接partial或覆盖旧权重。

22已完成真实工程步骤，但原36仅18/36通过，零延迟倒车停车位移超过原.05m门。实际账本中倒车／前进运动tick约各半，没有依据先提高倒车运动比例。本轮先在两组共同增加显式KL停止条件，再比较延迟分布。

## 固定科学范围

- 两arm分别从原16完整control checkpoint `428ff0ce9e8e88e10dabf94801a57b88de7e76d07b42800553b69c7f590e4eff`及原Adam独立加载；不接22或另一arm的最终权重。
- 共同设置：4env、seed181、n_steps1024、rollout4096、batch256、最多4epochs、LR1e-4、aux0、原03d命令／奖励／接口。每arm仅新增262144 transitions，原计数1048576→1310720，`reset_num_timesteps=False`，final-only。
- `zero`为固定0tick延迟；`random`每episode从0、1、2、3tick均匀抽样。抽样沿冻结18的独立stream18018，不额外消耗命令或reset随机流。
- 两arm唯一差异为延迟模式；起始policy、Adam及运行库版本必须相同。没有要求延迟不同的物理轨迹或回合终止时间相同。
- 先通过冻结22的纯数值加载函数恢复原policy／Adam并将LR改为1e-4，再显式验证`target_kl=None`并设置`.005`。记录KL前后policy和完整optimizer SHA不变、原Adam非LR部分不变、None→.005。

真实PPO代码在当前minibatch估计KL严格大于`1.5*.005=.0075`时，跳过该批backward／optimizer并结束剩余epochs。它不撤销此前更新，也不保证最终模型相对原模型的全局KL≤.005或≤.0075；因此仍需原停车资格门。目标KL不会添加累计停车位移信息，本轮未预设它一定有效。

两arm之间是同起点、同预算、同seed的延迟模式对照。与22相比共同新增了KL条件，不能把24相对22的变化都称为随机延迟的因果收益。单个训练seed不构成统计显著性或机器人部署证据。

## 顺序、门与预算

固定顺序为：train_zero、train_random、export_zero、cpp_zero、formal_zero、export_random、cpp_random、formal_random、holdout_zero、holdout_random。

两个原36矩阵都会收齐合法任务结果；短回合自然任务失败不阻止另一arm，来源／资源／进程／归档等工程异常立即停止整轮。每arm只有原36全部通过才运行自己的新36 holdout；未通过则显式跳过，不能补预算或改门。

| 范围 | policy transitions | native | PD |
|---|---:|---:|---:|
| 每arm训练 | 262144 | 2621440 | 1310720 |
| 每arm原36，上限 | 21600 | 216000 | 108000 |
| 每arm条件holdout36，上限 | 21600 | 216000 | 108000 |
| 两arm合计上限 | **610688** | **6106880** | **3053440** |

各arm导出及64行C++黄金数值核验为0新增物理步，不增加训练预算。导出仍为完整encoder+actor图，opset11/IR7，SB3／Torch阈值1e-6、Torch／ORT与C++阈值1e-5不放宽。新24自行核最终模型1310720步、LR、KL、checkpoint与本arm来源，不走旧22资格入口。

正式评测使用一套原MuJoCo环境和导出ONNX的Python ORT推理；这里的C++阶段仅为64行数值核验，不能称为C++驱动物理闭环。原36／holdout均保留固定场景、命令、阈值和10native／5PD比例，并复用原saved-trace验证数学。

全轮工程COMPLETE与科学通过分开。资格要求该arm原36及holdout36均通过；零延迟9/9无退化是必要条件。只有满足两门的最终checkpoint可进入本轮选择：按原36平均J更低者选，数值完全相同固定优先zero。没有满足条件者则selected_arm=null。只比较两个预注册final，不挑中间checkpoint、不自动替换默认模型。报告另保留两组完整矩阵和共同完整配对J，早退不能因累计误差较少被当作改善。

## 本机资源与生命周期

CPU单线程数值／学习；全程一次只运行一个训练或评测子进程。每阶段启动有效内存至少6GiB，运行保留4GiB，每秒只读采样。训练每arm1800秒，导出180秒、C++90秒、每次36评测300秒；父总墙钟5400秒，启动、验证和清理都计入。

执行前须已有外层串行调度与Nice至少10；建议沿已用systemd用户transient单元运行，外层有界wall及自有cgroup兜底。代码不会为了腾内存结束其他用户程序，也不会降低资源门或切换env数。低启动内存只返回NOT_READY，不占唯一execution目录；一旦claim输出或spawn，失败尝试不退款、不重试。

父进程使用自有session、subreaper及birth-tick身份清理，只清理自身创建的树，INT／TERM／KILL均有有限宽限。Popen期间收到信号也会拿到handle后处理，无法取得handle仍执行自有后代清理；清理或晚发现超时都不能报COMPLETE。

训练复用22的单飞RPC guard：未完成step／dirty通道不发送metadata RPC；先验证worker计数／delay数据为JSON安全值再赋值，避免ndarray污染失败回执。未知总计数保留null，可信已完成前缀单列；从唯一stage事实幂等重建总账，不能把缺失当零或重复累加。

## 新身份与操作入口

源码闭包包括新七文件及实际复用的冻结数值、运行与清理依赖；prepare要求这些字节已提交并与git对象一致。新plan还绑定旧18plan、原完整control／protocol及已完成zero初始状态来源，但不要求旧18总尝试COMPLETE。

输出固定为`evidence/delay_comparison24/execution_01/{zero,random}/{train,export,cpp,formal,holdout}`。父journal以train_zero等唯一stage名绑定plan SHA、source、子命令和专用nonce；独立子入口没有当前父凭据不能启动。所有非train回执再绑定本arm实际训练checkpoint／protocol／result身份；formal／holdout另绑定本armexport report、ONNX、fixtures及cpp report四SHA。每阶段与末尾重核全部已完成工件。

默认只列计划，不prepare、不读模型、不spawn：

```bash
rtk proxy python3 -B tools24/common24.py --plan
rtk proxy python3 -B tools24/run_delay_comparison24.py
```

源码与purefake检查稳定、提交且根线程复核后，才可准备新plan；本文件不代表prepare或实际执行已经发生：

```bash
rtk proxy python3 -B tools24/common24.py --prepare
rtk proxy python3 -B tools24/run_delay_comparison24.py --execute
```

本轮开发验证仅使用fake模型／进程／内存及已有工件元数据，没有实际模型加载、ORT session、训练、机器人构造或物理步。实际结果只能由后续新回执给出。
