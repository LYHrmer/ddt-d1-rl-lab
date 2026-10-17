# 七份外部资料对两个仓库的启示

阅读日期：2026-10-17。实际阅读了六个 GitHub 仓库的主说明及有关观测、动作、执行器、训练配置和部署状态机的关键源码，并读取 RoboMaster 文章正文。追加的 Wheel-Legged-Gym 还追到了编码器、PPO 更新与导出路径；robot_lab 追到了轮足任务覆盖后的配置。下面把源码事实与本项目的改进建议分开；没有运行外部模型、仿真或训练，也没有修改现有冻结输入。

这些资料有用，尤其是部署故障处理、执行器建模和训练任务的组织方法。它们没有证明换用某套框架、某个奖励权重或某个现成策略就能让 D1 通过资格。当前 D1 的低速跟踪与停车问题，仍需在本机模型上验证。

## 读到了什么

| 资料及本次固定版本 | 有直接依据的内容 | 对本项目的用途 |
|---|---|---|
| [quadruped_robot_lab](https://github.com/Taojunfeng123/quadruped_robot_lab/tree/bc25831f3a07d44110099c38f1215b27875cfefa) | 周期停走事件、命令课程、执行器网络；另有区分腿位置动作与轮速度动作的 Tita 配置 | 借鉴阶段划分、执行器辨识方法；四足步态奖励和参数需按机体重新设计 |
| [XYEGA 部署](https://github.com/chushanxiaodaoshi/XYEGA_RM2026_WheelLeg_Infatry_RLdeploy/tree/fcfdd3959be5c9b00893ec0701a591ddbf55b830) | 推理与低层控制分频、观测历史、位置/速度目标映射、实车最终力矩仲裁；仓库同时含伪代码与实车源码 | 适合检查低层每周期的故障处理与模式切换；不能把伪代码直接视为完备驱动 |
| [SCUT sim2sim 部署](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/tree/dd367bf78c7e393d811595edeb4affe253156c95) | 明确的 35D→6D 合同、八关节顺序、500 Hz 控制/50 Hz 推理、有限值失败处理和串口状态超时 | 可作为部署接口说明与异常注入用例的参考，ONNX 与本地策略不兼容 |
| [SCUT 训练](https://github.com/scutrobotlab/wheeled-legged_RL/tree/1a26356d6f591734d4f6f666f5cead1a6925da67) | V14 环境及其继承配置、关节类型与执行器配置、观测/动作处理和扰动选项 | 追溯训练端与部署端的实际配置，区分已定义选项、被覆盖选项和真正启用的分支 |
| [HYNova 16DOF 文章](https://bbs.robomaster.com/article/1946179?source=1) | mjlab 训练、独立 MuJoCo 回放、Flat/Rough/Crawl 任务及部署接口检查；4×(3 腿关节＋1 轮)=16 执行器 | 在结构和 50 Hz 策略/2 ms 物理时钟上最接近 D1，适合后续深入对照；同为 16DOF 不代表动力学或权重可互换 |
| [Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym/tree/c354431e5633eb98d9663fa3fbe444201d2de383) | 默认序列策略、历史编码器的速度监督、VMC 动作映射；旧 Isaac Gym Preview 4 栈 | 可提出独立的速度估计辅助任务；借用前需审查完整导出图、历史与噪声索引 |
| [robot_lab](https://github.com/fan-ziqi/robot_lab/tree/500399ed75f510aeaff28705a8ce736c514dbec3) | Tita/Go2W 的非对称观测、按机器人覆盖奖励/课程、腿轮动作分类、导出版本分支 | 为训练信息分工和有效配置审计提供参考；不是本地 D1 的直接适配包 |

文章作者为 Sulcunfu，原站元数据显示发布于北京时间 2026-08-28 17:22，更新于 08-31 22:28。正文关联的[训练与仿真仓库](https://github.com/wusi321/RC_Legged_Training_Simulation)和[实机控制仓库](https://github.com/Dichen33/RC_Legged_Control)是进一步阅读入口；本次没有据其展示效果认定本地机器人已具备同等能力。

## 现有能力与值得补的部分

D1 已有严格的关节顺序、轮索引、默认角、P_V 动作、裁剪、滤波、历史排列和时钟校验；Python/C++ 数值与回放检查也已做过。其输入是当前 57 维加十个过去帧，共 627 维，过去窗口为 200 ms。策略 50 Hz、PD 250 Hz、物理 500 Hz。不能把“补关节映射”或“增加历史帧”直接算作本轮新发现。[本地接口](../training/d1_rl/interface.py)、[冻结配置](../training/configs/pilot03d.json)

wheel 仓库已有 100 Hz 控制、2 ms 执行器通道，且明确区分状态延迟和动作延迟；也已做增益/一阶响应/延迟辨识、补偿对照，以及低维动作参数化的固定预算消融。合成执行器的拟合成功不等于真机辨识完成，力矩误差下降也曾伴随速度误差略增；不能重新提出“一阶延迟补偿”便宣称填补了空白。[执行器对照](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/docs/actuator_transfer.md)、[状态接口](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/docs/state_estimation.md)、[动作消融](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/docs/action_parameterization.md)

### 优先一：把部署预检扩展到离线故障行为

这是工程上最确定的增量。首次阅读时，D1 的 ROS 预检只验证了工件与纯导入。随后已新增[实际发布入口的启动/故障回执修复及 43 项虚拟 ROS 测试](watchdog_publisher14.md)，覆盖初始化失败、陈旧 truth、终点与清理；这仍不等于真实 DDS 或物理停稳验证。wheel 的运行完成门验证了 worker 就绪、实际步数和清理，也不等于验证了所有低层停机行为。

XYEGA 在每个低层周期进行最终仲裁，推理降频不降低仲裁频率。其正常 Stop 分支清的是 RL 力矩；大倾角保护有更高优先级，仍可能输出大腿制动力矩。SCUT 则在无效观测/输出时清理控制输出并请求 IDLE，另有串口状态超时禁止发送控制包的边界。这些是不同层级的动作，不能统称为“停止”。[XYEGA 实车仲裁](https://github.com/chushanxiaodaoshi/XYEGA_RM2026_WheelLeg_Infatry_RLdeploy/blob/fcfdd3959be5c9b00893ec0701a591ddbf55b830/source_code/User/Application/chassis/chassis.cpp#L1385)、[SCUT 失败处理](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/dd367bf78c7e393d811595edeb4affe253156c95/src/controllers/template_ros2_controller/src/fsm/states/state_rl.cpp#L296)、[接口与超时](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/dd367bf78c7e393d811595edeb4affe253156c95/PARAMETERS.md)

建议先用虚拟时钟和传感器序列注入 NaN、陈旧状态、指令中断、推理失败和模式重入，核对输出与恢复历史。分别声明“用户给零速度”“动作归零”“保持姿态并制动”“禁能/零力矩”的本机语义。D1 中动作归零仍会产生默认姿态 PD 与轮阻尼，不能直接等同零力矩。第一步不需要机器人、ROS 节点或物理积分；实际停稳能力仍需另行验证。

### 优先二：先检查阶段学习信号，再选择下一项训练改动

D1 已有正反成对的运动和停车指令，不缺一个通用的周期 stop 函数。更有价值的是沿现有日志，把前进、后退、转向分别拆成运动、切零制动、静止阶段，检查各奖励项的贡献、可观测性与失败位置。这样能判断 episode 总回报是否掩盖资格任务中的退化。[已有成对指令](../training/configs/pilot03d.json)

四足仓库提供周期启停和奖励驱动课程的思路，但其启停函数只清零 x/y 指令，保留 yaw；所谓 stop 补偿是在整回合奖励阈值上乘运动时间比例，并非真正逐步排除停止段。因此建议借鉴课程结构，在已有任务上设计按子任务分别判断的升级条件，而不复制其函数或阈值。课程变化是否能改善 D1 资格是待测假设。[启停函数](https://github.com/Taojunfeng123/quadruped_robot_lab/blob/bc25831f3a07d44110099c38f1215b27875cfefa/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/events.py#L274)、[课程实现](https://github.com/Taojunfeng123/quadruped_robot_lab/blob/bc25831f3a07d44110099c38f1215b27875cfefa/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/curriculums.py#L158)

这一步随后已落实为[完整历史日志的阶段诊断](phase_learning14.md)：仅复用完整训练及已有资格回执，缺失阶段保留缺失；训练期逐阶段奖励没有记录，不能事后补造。没有挑选合同 09/10 的中间权重代替最终评测。若随后改变课程或观测，应分别建立单变量新实验，保留固定资格任务与原阈值。现有熵假设仍未验证，不能从两次内存停止得出熵系数无效。

还值得单独考察**非对称 actor/critic**。SCUT 当前 Flat 配置启用了额外特权观测，环境确实为 critic 构造包含真值线速度等量的独立输入，actor 仍使用受扰动的本体观测。D1 可形成一个独立候选：保持部署 actor 的 627D 合同，用仅训练可见的状态辅助价值估计。这可能改善价值学习，但不会自动补足 actor 的可观测性，也未证明能解决停车；当前源码更不能证明发布 ONNX 当时使用了完全相同的训练配置。[当前配置](https://github.com/scutrobotlab/wheeled-legged_RL/blob/1a26356d6f591734d4f6f666f5cead1a6925da67/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/env_cfg.py)、[独立 critic 数据流](https://github.com/scutrobotlab/wheeled-legged_RL/blob/1a26356d6f591734d4f6f666f5cead1a6925da67/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env.py)

对 wheel，更相关的是检查已有观测没有显式携带的控制器内部状态。随后新增的[控制器状态诊断](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/docs/controller_state_v29_20261017.md)读取了完整连续任务的 45D 分支日志，发现策略垂向动作持续裁剪、转向初段 PI 与当前 yaw 方向相反等描述性现象；基线也有后者，不能归因于 PPO。距离参考和完整 actor 帧未记录，不能据此重建状态混叠。该日志不是 B22 的 99D 分支；不直接修改既有 checkpoint 输入，也不重复已完成的 Fx/Fz 低维消融。

### 优先三：对齐执行器的实际语义，再考虑增加模型复杂度

应比较的是“网络输出→裁剪/滤波→延迟→位置或速度目标→PD→机构映射/补偿→限幅→实际力矩”的完整顺序。特别要记录延迟所在时钟与毫秒值，以及观测里的 last_action 究竟代表刚发布动作、滤波动作还是已执行动作。

XYEGA 的 Spin 伪代码示例展示了一个容易混淆的区别：执行动作可延迟一个 500 Hz 周期，last_action 仍保存当前裁剪后的发布动作。它的历史先追加当前帧，再进行推理；D1 则明确只堆叠过去十帧。都可以是合法合同，但不可互换。[动作与历史伪代码](https://github.com/chushanxiaodaoshi/XYEGA_RM2026_WheelLeg_Infatry_RLdeploy/blob/fcfdd3959be5c9b00893ec0701a591ddbf55b830/pseudocode/math_core.cpp#L193)

四足项目的执行器网络利用位置误差和速度历史预测力矩；文件定义了弹簧补偿子类，但最终 `MY_DOG_CFG` 使用普通 `ActuatorNetMLPCfg`。本机应使用独立记录验证所借鉴的动态模型，不能照搬网络权重、弹簧系数或把定义存在当成实际启用；本次未核实上游辨识数据或独立验证集。[执行器配置](https://github.com/Taojunfeng123/quadruped_robot_lab/blob/bc25831f3a07d44110099c38f1215b27875cfefa/source/robot_lab/robot_lab/assets/my_dog.py)

对 D1，先核对已有轨迹中的目标、饱和与时序；对 wheel，先复用既有通道与独立验证集，只有残差证据支持时再考虑速度相关力矩包络或额外动态。执行器更复杂本身不证明仿真资格会更好，更不应以外部参数替代本机辨识。

SCUT 提供了一个值得警惕的具体例子：部署 xacro 同时填入 `tau=0.014` 和二阶参数，实际 C++ 优先选择 `natural_frequency>0` 的二阶分支，因此不能称为“14 ms 一阶电机”。该分支还有限幅、变化率和速度相关 droop；训练端默认资产却是 IdealPD。训练物理/低层为 200 Hz、部署低层为 500 Hz，腿/轮限幅也有差异。这里的价值是提示我们逐项核对有效配置，而非证明这两仓已经完全对齐。[部署动态分支](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/dd367bf78c7e393d811595edeb4affe253156c95/src/interfaces/mujoco_bridge/src/mujoco_system.cpp)、[部署配置](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/dd367bf78c7e393d811595edeb4affe253156c95/src/resources/robot_descriptions/wheelbipe_V14/xacro/ros2control.xacro)、[训练资产](https://github.com/scutrobotlab/wheeled-legged_RL/blob/1a26356d6f591734d4f6f666f5cead1a6925da67/source/agent_world/agent_world/assets/wheelbipe_V14_2.py)

## 追加两份资料带来的具体候选

### 历史输入之外，显式监督速度估计

Wheel-Legged-Gym 默认使用 `ActorCriticSequence`：当前本体观测 27D，历史输入 135D，含五帧且包括当前帧。编码器默认输出三个速度估计量，再与当前观测拼接供 actor 使用；PPO 更新中另有优化器，用 critic 输入中的真值线速度监督这三个量。训练时的 actor 路径对 latent 使用 `detach`，因此这不是仅靠策略梯度自己学出速度，也不是只在论文介绍中出现的模块。[默认配置](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/wheel_legged_gym/envs/base/legged_robot_config.py#L271)、[actor 数据流](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/wheel_legged_gym/rsl_rl/modules/actor_critic_sequence.py#L181)、[独立监督损失](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/wheel_legged_gym/rsl_rl/algorithms/ppo.py#L256)

这给 D1 增加了一个有区别的候选：保持部署可取得的 627D 信息来源，用训练期真值监督一个速度估计分支，再让 actor 使用估计结果。需明确标签的坐标系、单位与采样时刻，避免把部署不可取得的真值直接送入 actor。它与前述“仅 critic 多看真值”是两个不同改动，应分别比较，不能一次合并后归因。当前 D1 已有历史，不能把此候选写成补历史；尚无本地结果证明它改善低速跟踪或停车。

借用时必须同时审查导出。上游通用 `export_policy_as_jit` 的非循环分支只复制 `actor_critic.actor`，没有带上默认序列策略的历史编码器；该 helper 不能直接承担完整序列策略的导出。若本地实现此候选，需要导出“627D 输入→编码器→actor”的完整计算图，并在 Python/导出工件/C++ 间比较相同输入的结果、历史重置和必要的归一化。不能只导出最后一个 actor 网络便宣称部署等价。[导出 helper](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/wheel_legged_gym/utils/helpers.py#L244)

### 任务配置要看最终生效值

robot_lab 的 Tita/Go2W Rough 配置为 actor 去掉线速度与地形扫描，critic 保留额外信息，可再次支持非对称观测这一候选的设计依据；Tita Flat 则另外关闭 critic 的地形扫描。接触传感器的历史长度不能当成 actor 的历史堆叠。Tita 的腿静止奖励和轮速度惩罚也可参考，但轮速度惩罚会按命令与机体速度选择分支：运动时只作用于首次离地拍的轮，静止时作用于全部轮，并非统一的切零制动机制，权重不能直接移植。[Tita 观测和奖励](https://github.com/fan-ziqi/robot_lab/blob/500399ed75f510aeaff28705a8ce736c514dbec3/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/ddtrobot_tita/rough_env_cfg.py#L103)、[条件轮速度惩罚](https://github.com/fan-ziqi/robot_lab/blob/500399ed75f510aeaff28705a8ce736c514dbec3/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py#L132)

覆盖顺序会改变结论：Tita 最终关闭了两个命令课程；其 Flat 配置仍使用含 cobblestone 高度噪声的地形生成器，名称不代表严格平面。play 脚本赋值的 `events.push_robot` 与基类已配置的 `randomize_push_robot` 名称不同，也不能只凭注释认定回放关闭了推扰动。这些是固定版本的静态配置观察，未在外部运行中复现；对本项目的启示是冻结最终有效配置及启用项，而不是只保存基类或配置名称。[课程覆盖](https://github.com/fan-ziqi/robot_lab/blob/500399ed75f510aeaff28705a8ce736c514dbec3/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/ddtrobot_tita/rough_env_cfg.py#L253)、[Flat 配置](https://github.com/fan-ziqi/robot_lab/blob/500399ed75f510aeaff28705a8ce736c514dbec3/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/ddtrobot_tita/flat_env_cfg.py#L28)、[play 设置](https://github.com/fan-ziqi/robot_lab/blob/500399ed75f510aeaff28705a8ce736c514dbec3/scripts/reinforcement_learning/rsl_rl/play.py#L133)、[基类事件](https://github.com/fan-ziqi/robot_lab/blob/500399ed75f510aeaff28705a8ce736c514dbec3/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L366)

前一份 quadruped_robot_lab 的 README 已注明基于 robot_lab 二次开发，不能把两份同源实现当成独立效果验证。robot_lab 列表里的 Agibot D1 是四足型号，不是本地 DDT D1 轮足。腿位置/轮速度动作分类在本地已有，不算新增缺口；该仓的 Gazebo/实机部署说明还指向另一个 rl_sar 仓库，不能把说明入口当成本仓已完成的部署验证。[同源说明](https://github.com/Taojunfeng123/quadruped_robot_lab/blob/bc25831f3a07d44110099c38f1215b27875cfefa/README.md#L32)、[robot_lab 说明](https://github.com/fan-ziqi/robot_lab/blob/500399ed75f510aeaff28705a8ce736c514dbec3/README.md)

### VMC 与时间、索引的借鉴边界

Wheel-Legged-Gym 的 VMC 策略输出两腿的角度/长度目标及轮速度，共 6D，经任务空间 PD 和机构几何映射成关节力矩。这是完整控制参数化，不是本地 LQR 上有界叠加 Fx/Fz 的相同实现。wheel 已做低维动作消融；如再研究虚拟坐标目标，需先验证本机机构的雅可比、虚功和左右符号，另立不同假设，不能用上游效果覆盖本地负结果。[VMC 动作与力矩](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/wheel_legged_gym/envs/wheel_legged_vmc/wheel_legged_vmc.py#L350)

其默认物理 200 Hz、策略 100 Hz，五帧含当前帧，均与 D1 的 500/50 Hz 和过去十帧合同不同。源码还可静态看到两处索引风险：27D 观测有三个命令量，噪声掩码只跳过其中两个，后续分组发生偏移；默认 0–10 ms 延迟在 5 ms 时钟下创建长度为 2 的 FIFO，但四舍五入采样可能得到索引 2，再直接索引 FIFO。这里没有运行外部模型来断言实际触发率或训练影响；这些具体边界说明迁移前应测试每个通道、历史顺序和延迟端点。[观测拼接](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/wheel_legged_gym/envs/wheel_legged_vmc/wheel_legged_vmc.py#L285)、[噪声分组](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/wheel_legged_gym/envs/wheel_legged_vmc/wheel_legged_vmc.py#L467)、[FIFO 长度](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/wheel_legged_gym/envs/wheel_legged_vmc/wheel_legged_vmc.py#L671)、[延迟采样](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/wheel_legged_gym/envs/wheel_legged_vmc/wheel_legged_vmc.py#L808)

### 仍按本机 CPU 方案推进

上述算法思路可以在现有 CPU/MuJoCo 栈中独立实现，不要求先迁移整套 Isaac 框架。Wheel-Legged-Gym 的 README 提供 CPU 参数，但没有给出本机低内存运行的证据；旧 Preview 4 与 robot_lab 的 Isaac Lab/Isaac Sim 也不是同一代环境。[Wheel-Legged-Gym 安装与参数](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/README.md)

资源数字需要限定版本：Isaac Lab v2.0 和 v2.2 安装文档建议至少 32 GB RAM、16 GB VRAM；Isaac Sim 5.1 的最低配置同为 32 GB/16 GB，且该版本页面已标记不再受支持。这不是所有旧版本的统一最低值，也不证明小任务绝对不能启动。本机约 16 GB RAM/8 GB VRAM，现阶段迁移会增加依赖与资源不确定性，故继续采用[本机低并行度草案](local_training_plan14_draft.md)，4 环境优先、事前预检决定是否选 2，保留内存门与有界时长。[Lab v2.0 安装建议](https://isaac-sim.github.io/IsaacLab/v2.0.0/source/setup/installation/index.html)、[Lab v2.2 安装建议](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/setup/installation/index.html)、[Sim 5.1 资源要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

## 暂不照搬的做法

- SCUT 的 35D/6D 与 XYEGA 的双轮腿模型，均不能直接替换 D1 的 627D/16D 策略。SCUT 35D 的第 0 项是速度**指令**，不是实测基座速度；不能据此说它解决了 D1 的线速度可观测性问题。[35D 合同](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/dd367bf78c7e393d811595edeb4affe253156c95/PARAMETERS.md)
- XYEGA 实车末端在 RL/LQR 间选择，不构成本地有界残差叠加有效的证据。四足步态、腾空和落足奖励也不能直接移给轮足；同仓 Tita 对腿/轮分别配置动作并关闭若干四足奖励，更值得参考其分类方法。[实车选择](https://github.com/chushanxiaodaoshi/XYEGA_RM2026_WheelLeg_Infatry_RLdeploy/blob/fcfdd3959be5c9b00893ec0701a591ddbf55b830/source_code/User/Application/chassis/chassis.cpp#L852)、[Tita 配置](https://github.com/Taojunfeng123/quadruped_robot_lab/blob/bc25831f3a07d44110099c38f1215b27875cfefa/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/ddtrobot_tita/rough_env_cfg.py)
- 这些资料没有提供可直接修复本机 RAM 停止的内存实测。更换 GPU 训练框架、增加并行数或模型规模不能当作已验证的资源优化。合同 10 的 USS 记录也不支持直接宣称持续内存泄漏。[本次停止证据](career_results_10.md)

建议顺序是：先做无需新物理步的故障测试、接口对照和已有日志分析，再形成一个有区分力的新训练假设。后续训练使用新合同与预算，保留 D1 当前 release 的 72 项冻结输入和全部历史失败；wheel 的原保护清单按已有授权替换规则校验，另保留 9 项用户结果。本次阅读没有启动新实验。
