# 本机延迟鲁棒性优化18：新范围与固定执行

本轮目标来自所有者的新指令：**把这些优化点，在这台电脑能力之内的都完成**。此前17完成工程收尾但完整延迟资格未过，本轮继续实际改进；不是旧09/10/16/17运行的重试，也不使用旧预算余额。本文记录执行者在新授权范围内选择的本机操作上限，不能把它写成已完成或资格通过回执。

## 诊断与单变量候选

16的control实际使用`arm=B`，每回合动作延迟固定0。配置里的R分支没有启用，且原R范围不含60ms。17的36条原轨迹逐拍符合`applied[t]=clip(issued[t-delay])`，未发现FIFO错拍；40/60ms转向在反向之前已出现持续高频振荡。这支持检验延迟训练分布，尚不能证明因果机制或称为代码bug。

新`D1Delay18Env`只覆盖每回合延迟选择：`zero`固定0，`uniform_0_3`独立均匀采0/20/40/60ms。独立SeedSequence使用`[seed,rank,episode_index,18018]`，不消费旧命令或重置随机流。固定评测延迟优先。原03d配置、任务、奖励、P_V动作、历史/627D输入、终止线、PD/物理时钟均不改。

两组均从**已完成**的16 control最终checkpoint及Adam状态独立加载，SHA为`428ff0ce9e8e88e10dabf94801a57b88de7e76d07b42800553b69c7f590e4eff`。两组新seed181、4环境、每环境1024步/合计4096一轮，全部原PPO参数和aux=0保持。保留原已训练1,048,576与本轮新步数两个字段，`reset_num_timesteps=False`；每组正式结束的累计num_timesteps应为2,097,152。预检权重丢弃，正式组重新加载同一原最终权重。跨组核初始policy/Adam相同；同组预检与正式前八轮核确定性全等，不能要求不同延迟组的轨迹全等。

## 本机总上限

已核验i7-14650HX/24逻辑CPU，约15.25GiB总RAM、当前约7GiB可用、可见cgroup无限额，108GiB磁盘余量；GPU可见但此CPU小网络管线不依赖CUDA。历史4环境双组2,166,288新步约34分40秒、最低4.935GiB有效RAM。预计新训练35–45分钟级，实际受机器负载影响。

| 阶段 | 新policy上限 | native上限 | 子进程墙钟上限 |
|---|---:|---:|---:|
| 原权重最小失败反馈：turn/60ms/seed1001 | 600 | 6,000 | 120s |
| 两组独立预检，各32,768 | 65,536 | 655,360 | 各900s |
| 两组独立正式微调，各1,048,576 | 2,097,152 | 20,971,520 | 各5,400s |
| 两组完整图导出和C++64 | 0 | 0 | 各180s/90s |
| 两组各原36例 | 43,200 | 432,000 | 各600s |
| 合格候选额外36例，reset seed2001/2002/2003 | 21,600 | 216,000 | 600s |
| 合计上限 | **2,228,088** | **22,280,880** | 单阶段另有30s进程退出宽限 |

PD总上限11,140,440；正常比例为每policy10native/5PD，提前终止按实际计数。所有重任务串行，Torch/BLAS各1线程，启动有效RAM至少6GiB、运行保留3GiB；有效值取host MemAvailable与可见cgroup祖先硬余量最小值，不把swap当RAM。每秒采样，资源/时限/源码/未知计数/工程错误停止，不自动重试或续跑；只清理本次创建的进程组。每个stage只有一次，输出拒绝覆盖。

原权重探针是新预算下的独立反馈，不改17结果。任务跌倒是评测失败，保留并采齐预定矩阵；工程错误停止。候选只有原36例完整且所有原参考阈值通过才进入额外36例。两者均合格时按平均原J较低选择，精确同分优先zero。没有候选合格时跳过额外资格，不补跑、不把流程COMPLETE解释成优化目标完成。

## 来源、入口与结果边界

所有新源码先提交，再生成包含Git blob与逐文件SHA的新固定计划。计划重新核16完成训练与原86输入/工件/资产，以及17原56工件；各已完成新stage保存工件SHA，前后核验。最终模型必须来自本轮完成正式组，不能把预检或partial权重作为合格模型。导出包含encoder/actor/clip完整6Gemm图，保持64组SB3/Torch/ORT数值阈值和C++冻结运行时；数值验证不运行物理或ROS。

```bash
rtk proxy python3 -B tools18/common18.py --prepare evidence/local_optimization18/plan_01.json
rtk proxy python3 -B tools18/run_optimization18.py plan --plan evidence/local_optimization18/plan_01.json
rtk proxy python3 -B tools18/run_optimization18.py execute --plan evidence/local_optimization18/plan_01.json
```

执行前两条只读/准备不会加载模型或运行物理。第三条是唯一新固定执行，使用原本机`.python-deps`和`training`环境。Git保留小回执，模型/轨迹在本机`evidence/local_optimization18/execution_01/`。保留旧冻结源、旧权重和旧失败记录，不覆盖任何旧运行。

正式任务仍为forward/reverse/turn_stop，各12s、0/20/40/60ms、reset1001/1002/1003共36例，使用原SHA绑定的命令/指标/阈值。额外36例只换reset种子；单次warm-start训练种子181，不构成独立多训练种子统计。结果不外推实机、地形或ROS；wheel有自己的完整任务和优化预算。

所借鉴的训练/CPU原则可参阅[既有上游源码审查](upstream_reference_review_20261017.md)和[SB3官方PPO说明](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)。本轮不迁移外部框架或复制外部权重，是否改善以本机实际失败反馈、原完整矩阵和额外reset验证为准。
