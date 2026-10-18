# D1 27：同一科学问题的独立资源修订

26 在 control 的第1,302个新增policy步触发4 GiB保护，首个rollout未完成，policy/Adam序列化字节仍与25 random基线相同，PPO更新增量0。它没有形成两组比较，不能当作yaw奖励的负结果。27是新来源、新计划、新输出和独立预算；不读取26 partial、不续训或重跑26现场，也不把旧余额转入新预算。

## 保留的科学与数值实现

入口逐SHA锁定冻结26的 common、runtime、trainer、runner 四份生产源，再编译到独立私有模块。只允许清单中的format、固定路径、nonce、首buffer文件名和内部import绑定变化；命令路由到27唯一入口。没有全局字符串替换，没有改旧模块全局变量，也没有复制一套训练实现。私有模块的 `__file__` 保留真实26来源路径，计划同时保存原SHA和派生AST SHA；不能把它说成改写后的26源文件。

来源闭包包含原全部35文件和27入口、测试、本文共38文件；正式准备必须先提交并逐项匹配Git字节。旧72项冻结不变。26只作为冻结实现来源，训练基线始终是25 random完整ZIP `873cd14ff67e7d399472d97a94e082f7a97d068b00db414986443e50fcdb4f22`，同时绑定其最终Adam和原协议。

两组control/yaw3各新增131072步：同一完整policy+Adam、seed182、DummyVecEnv单env、4096步rollout、LR1e-4、target_kl=.005、aux0、均匀episode delay0..3。唯一科学变量仍是所有policy ticks的wz tracking权重2→3，命令、其他reward、动作、观测与物理实现不变。初始state及首rollout非reward六数组须实际相等；只选final，partial不能评测。

原36全收集，task failure不阻止另一组收集，工程失败停止。只有自己原36全部完整且通过原阈值才eligible；eligible中mean J最小选一个，精确tie选control。只有该组可看唯一holdout36（reset seeds2001/2002/2003），失败不换组、不换seed；无eligible则不执行holdout。保留原17及25 random的共同完整时域J比较。这些公开reset值不是隐藏seed/盲测，也不代表未见命令分布。

独立最大预算为 **326944 policy / 3269440 native / 1634720 PD**（两train262144、两formal43200、唯一holdout21600）。两组各CPP64仅数值验证、0物理；formal/holdout仍用Python ORT，不能称真正C++闭环、旧19资格或ROS实机资格。不会自动替换模型。

## 只新增启动前的稳定准入

实际执行先创建独立 `evidence/yaw_reward27/admissions/` 下的新侧录，不claim训练输出或启动训练worker。有效内存须 **≥8 GiB持续60秒，每5秒采样，最多等300秒**。4～8 GiB重计稳定窗口，低于4 GiB或未知立即拒绝；信号也停止。达标后重新核来源/计划并读取新内存值，仍须≥8 GiB，才写一次性ticket、唯一spawn父流程。

ticket绑定计划SHA、38项来源、采样SHA、精确argv、nonce和外层PID；私有父入口检查PPID及实际连续采样，不可只凭授权计划直接跳过准入。侧录失败只给NOT_READY/失败回执，不占训练execution目录。每次显式准入可有一个新只读采样侧录；训练目录一旦存在仍绝不再spawn或重试。

达标后的父流程仍使用原 **6 GiB阶段门/4 GiB运行停止线**、阶段间最多300秒有界等待、单线程CPU/Nice10/串行和单次阶段。父总墙钟3600秒；包含启动前准入的外层总界3900秒，外部user transient unit应给3930秒并清理自身cgroup。外层每秒只读监测4 GiB并有限清理自己创建的子树，包含Popen尚未返回handle的异常；不杀用户应用。任何退出/清理/来源或计数错误都不能改报COMPLETE。

60秒稳定观察不能保证未来内存独占；外部负载随后增长时仍可能正确停止。不要为提高完成率降低4 GiB停止线。准入采样、ticket、stdout、父运行采样及外层回执与真实训练回执分开保存，未知计数保持null。

## 冷默认与实际入口

```bash
rtk proxy python3 -B tools27/run_yaw_reward27.py
```

默认仅PLAN_ONLY，不准备计划、不加载NN/ORT、不构造环境。根线程在源码提交后才可显式 `--prepare`，固定计划 `evidence/yaw_reward27/plan_01.json`；实际由新user transient unit调用 `--execute`。训练根固定 `evidence/yaw_reward27/execution_01/{control,yaw3}/{train,export,cpp,formal,holdout}`，不能覆盖。本文及纯假测试不是实际执行收据；本轮实现阶段没有prepare、训练或物理。
