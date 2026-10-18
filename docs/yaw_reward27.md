# D1 27：同一科学预算的资源修订02

修订02采用用户最新要求：**6GiB有效外部余量连续10秒**即可启动（5秒采样，至少3个样本），本任务整组进程设**3GiB硬上限**。内部新阶段冷启动仍6GiB、运行仍保留4GiB。没有新增科学预算、改变训练参数或执行模型；本次仅准备源码与纯假测试。

## 原01完整保留、02独占执行

旧提交`14934a40ee74baa3391aecf9faf3663f61ba0e15`、`evidence/yaw_reward27/plan_01.json`和三份`admissions/`原件不变。三次均为启动前`NOT_READY`，`spawn_attempted=false`、`execution_started=false`、清理完成，known为0、actual为null；没有`execution_01`，没有训练或PPO更新。第三次301.5428秒、61样本、最低5,098,098,688B；报告SHA`341bf87dc1b798ed6c3616675dcb88a2fdb9f9a9e235eee6c03317ad1c7f8f72`。

源码提交后显式prepare才写`superseded_01_by_02.json`，将01登记为`SUPERSEDED_WITHOUT_EXECUTION`，不修改原计划和回执。新计划固定`evidence/yaw_reward27/plan_02.json`、输出固定`execution_02`、准入侧录为`admissions_02/`。prepare、计划加载及每阶段都要求旧execution不存在、原计划与三份report/samples的SHA及no-spawn事实仍相符。新入口拒绝01路径；新nonce、v2回执、来源和`resource_revision=owned_cgroup_v2`分开记录。02执行目录一旦存在禁止重跑或续跑。

总cap仍为**326944 policy /3269440 native /1634720 PD**，是01尚未启动的同一科学问题，`additional_scientific_budget=0`，不能把01和02相加为两份预算。26只完成1302新增步且0PPO更新的partial不用；基线始终为完整25 random及Adam，不借用09/10余额。

## 科学与数值保持

基线ZIP为`873cd14ff67e7d399472d97a94e082f7a97d068b00db414986443e50fcdb4f22`。两组control/yaw3同seed182、Dummy1、n_steps4096、LR1e-4、target_KL.005、aux0、每episode均匀delay0..3，各新增131072，唯一科学变量为wz tracking权重2→3。观测、动作、其他reward、课程、物理和原阈值不改。初始policy/Adam与首rollout非reward六数组须实际相等；唯一final、partial不可评。

原36全采齐，合法任务失败不筛掉另一组，工程失败停止。只有自己原36完整且全部过原门才eligible；eligible中meanJ最小，exact tie选control；只允许一个winner使用2001/2/3的36例，失败不换臂。无eligible则跳过holdout。保留原17与25 random共同完整J比较；不声称隐藏seed、未见命令分布或自动替换默认。

各CPP64仅数值对齐，formal/holdout仍由Python ORT驱动物理，不是新增真实C++闭环、旧19资格或ROS实机资格。

四份26生产源SHA继续钉住并编译到独立私有namespace，`__file__`保留原26真实路径。有限AST映射只涉及02身份/路径/nonce与一处资源语义：**trainer冷入口仍6GiB，但Torch导入后已开始使用本任务内存，改查4GiB运行线**。这个after-import节点按唯一表达式和消息精确匹配；逆向恢复后全AST等于原件。无全局字符串替换，无原module globals修改。

## 两份内存账与真实硬限制

旧read_memory取所有祖先remaining的最小值；若直接给本任务3GiB上限，原6/4GiB检查会必然拒绝。新stdlib guard显式拆开：

- `effective_available_bytes=min(host MemAvailable, external ancestor remaining)`，只排除**精确登记的自有unit一层**；其他祖先限额保留，不将未知当无限。
- `owned`单列同unit整棵子树的current/high/max/remaining/swap、inode、memory.events、kernel peak和sampled max。没有kernel peak时记录null，不能把三次selfRSS采样写成整轮峰值。

固定scope为`/user.slice/user-<uid>.slice/user@<uid>.service/app.slice/d1-yaw-reward27-owned02-20261018-01.service`。实际self cgroup须精确相符，内核文件须为**MemoryHigh=2048MiB、MemoryMax=3072MiB、MemorySwapMax=0、memory.oom.group=1**。systemd还须设置`OOMPolicy=kill`、`KillMode=control-group`、Nice10/CPUWeight25、单线程及`Slice=app.slice`。不接受只在命令行请求而未落实的memory.max；cgroup路径、控制项、读数异常或OOM事件均拒绝，无旧reader fallback。

guard在私有common模块执行后、下游trainer/runner函数定义前绑定，确保common alias及train.run、runner.execute、run_stage默认参数均读新接口。准备/默认入口不要求在unit内；实际outer/parent/阶段所有memory读数都必须通过该同unit校验。setsid不会把C++子进程移出unit预算。

高水位可能触发回收与变慢，硬上限可能导致OOM；本协议不保证完成，也不保证别的程序不增长。宿主/外层祖先低于4GiB仍停止自己的进程，不终止用户应用。若整个unit被内核杀死，unit内收尾可能来不及写回执；外部记录systemd Result/退出码和cgroup事件，实际未知保持null、已持久化known前缀单列，不能因unit退出填成0或重试。

## 准入、时限与执行接口

准入每5秒采样，连续10秒及末次均≥6GiB，至少3样本，最多等300秒；低于6重计窗口，低于4/未知/信号立即停止。达标后重新验来源/旧未执行证明，再采样并验证有限连续间隔，才写ticket并唯一spawn父。ticket绑定新计划/来源/样本SHA、精确argv、nonce与PPID。

阶段间仅在无活子进程、上阶段清理完且下一请求未建立时，允许4～6GiB最多300秒有界等待；child冷入口自己拒绝后不重试。父3600秒、含初始准入外层3900秒，user unit外界3930秒并有界清理；所有来源核验、等待、保存读回与清理计入时间。

```bash
rtk proxy python3 -B tools27/run_yaw_reward27.py
```

默认`PLAN_ONLY`、0模型/0物理/不写计划或输出。源码提交后根线程显式`--prepare`写plan02和新的supersession小记录；实际串行unit运行`--execute`。本次未prepare、未执行，也未改变旧01/26或其他冻结输出。

## 必要检查

修订后入口38项pure/fake与共享guard31项，共**69项通过**；critical Ruff通过。覆盖逆AST数值不变、冷6/导入后4、所有私有默认reader正确捕获、10秒3样本、延迟来源/信号/Popen句柄前异常/清理、旧三回执变化或未知拒用、01路径拒绝、02独占及同cap不增。guard用临时cgroup文件验证自身层排除、祖先余量仍生效、实际3GiB/2GiB/swap/oom-group、错scope/控制项漂移/OOM/未知读数失败。它们不替代真实unit、模型、训练或物理验证。
