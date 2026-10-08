# Contract 03c：第二次 pilot 的任务修订

2026-10-09，Astra ultra 审阅。本文在 03b 最终评测失败之后、03c 训练之前登记；它不是成功报告。原 03/03b 合同、源文件、checkpoint 和回执保留。03c 是总计最多三个 pilot 中的第二个，仍从随机初始化 seed 101 开始；禁止 teacher、静力补偿前馈、旧权重微调或选择中间 checkpoint。

## 已确认的失败与证据

- `evidence/career_contract_03/pilot03b_python_eval_01/`：03b 完整训练 1,048,576 transitions 后三条固定任务均未通过。新 ONNX golden 与 500 native Python/C++ 配对已通过，不能把接口一致性当作控制能力通过。
- 前进在 5.96 s 因 tilt 0.60493 rad 终止；非法机体支撑始终 false，三条任务均无力矩饱和。前进 3–5 s 的 RR thigh issued action 约 77% 被截到 +1。实际 C++ `pilot03b_deployment_plan_01/trial01_forward_stop/trace.jsonl` 中，该关节从 2 s 的 1.434 rad 漂到 4 s 的 1.784、终点 2.102；默认值为 0.8。不能据此继续放大 action scale。
- 纯转向 3–5 s 的平均 yaw rate 约 0.0036 rad/s，而命令为 +0.2；全任务稳态 yaw MAE 0.19638。动作并非完全不响应，但未形成持续转向；其全程 `mean((a[t]-a[t-1])²)` 约 0.12558。
- 倒车停车后的 0.5 s 已移动 0.0480 m，4 s 累计 0.0870 m；最后阶段约 0.01 m/s 残余漂移。不是仅因末速门槛误判。

由此选择一个**复合任务修订**：实际关节姿态与整机姿态约束、yaw/静止奖励、显式纯动作命令覆盖。它检验这些目标是否改善任务能力，不声称区分其中各项的因果贡献。训练长度不足仍是备选解释，不能因该解释盲目重跑原 03b。

## 唯一下一轮实现

新增独立 03c 配置及任务子类，继承已验证的 `fast_checks.py`，原 `env.py` 不改。保持 03b 的 interface ID/schema、57 当前观测、10 帧历史、16 动作、腿/轮尺度 .5/5、动作截幅、默认姿态、PD/力矩上限、积分/刷新顺序、reset、终止、20 s episode、RNG 分流、PPO 网络与优化超参全部不变。不得改变 yaw 观测缩放或加入隐藏 PID/控制器。

每次 4 s 命令重采样，使用原独立 command RNG 的一次类别抽样：`u<.2` 为零令；`.2<=u<.5` 为纯前后；`.5<=u<.8` 为纯转向；其余为混合。被激活的 vx/wz 仍分别独立均匀采样 [-.3,.3]，其余轴精确为零；首 1 s 零令不变。原 `zero_probability=.2` 保留。命令采样必须在子类中实现，不能污染旧版本。

以原 reward 各项原始量及“总和最后乘 .02”定义为基础，只冻结以下修改：

| 项 | 03c 权重 / 定义 |
|---|---|
| yaw tracking | 权重 2.0，指数分母 0.04 rad²/s²；原为 1.0 / 0.09 |
| projected gravity xy squared | -6.0；原 -2.0 |
| nominal height error squared | -20.0；原 -3.0；目标仍为原 h_nom |
| mean action increment squared | -0.1；原 -0.01；仍对 16 维 issued clipped action 取均值 |
| 新 `leg_pose_deviation_sum_squared` | 原始量为当前 **实际** 12 个腿关节相对 DEFAULT_Q 的偏差平方和，排除四轮，权重 -0.5；不是动作大小、PD target 或静力补偿误差 |
| 新 `standstill_tracking` | 仅本次 interval command 的 vx、wz 均精确为零时，原始量为 `exp(-(vx_body²+vy_body²)/0.0004 - wz_body²/0.0025)`；否则为 0；权重 +1.0 |

vx tracking 2.0 / 0.04、alive .25、vy squared -2、实际力矩平方 -1e-5、非法支撑 -1 均不变；新增项也只乘一次 .02，并进入原有逐项 reward 日志。使用 interval command，避免在重采样边界把上一段运动误记为停止误差。零命令静止奖励不依赖评测时间或未来命令。

量级依据：忽略 0.2 rad/s yaw 命令时，旧 yaw 项仍获得理想值的 64.1%，03c 为 36.8% 且权重翻倍。0.01 m/s 无偏航残速使新增静止奖励损失约 0.221/秒，而旧 vx 项仅损失约 0.005/秒。实际腿姿态项会惩罚 RR thigh 漂移，不惩罚支撑所需的非零 issued action。增加姿态约束可能拖慢探索；本轮结果决定，不能事后修改权重。

## 开始前检查、预算与硬门

冻结新配置、子类、训练/导出入口和新 C++ schema 的 SHA；schema 仅改配置身份，物理与观测接口不变。零积分检查命令四类/范围/首秒、reward 手算值及仅一次 dt 缩放、实际腿索引排除轮轴、旧 env/config SHA 未变。新实现必须把附加源文件纳入训练与导出的完整 source snapshot。

- 训练仅一次：B 臂、seed 101、8 env、1,048,576 transitions / 10,485,760 native 上限、90 min、3 GiB 可用内存下限；仍按既定规则异常即停。不得自动续训、补步、换 seed 或选择最好 checkpoint。fast 的公平单环境测速为 952.54 / 352.59 = 2.7016 倍，8 env 端到端速度须另记实测。
- 只对完成的最终 checkpoint 导出；保留 Torch/Python ORT/C++ ORT golden、500×2 native 学习策略闭环配对。配对消耗原预检池，开始前登记实际剩余额度，不新建隐形预算。
- Python 仍只跑原三条 12 s 序列、seed7001、delay0，每条最多6000 native，合计18000；不改窗口、阈值或失败处理。C++沿用每pilot最多三次、总45000 native 的独立部署上限，失败诊断也计入。
- 通过条件仍为三条全部完成且无跌倒/非法支撑、vx MAE≤.06、wz MAE≤.08、停车末秒水平速度≤.02、|wz|≤.03、停车位移≤.05、roll/pitch峰值≤.20，及独立部署门通过。接口通过、训练回报上涨或95%回合满20s都不能替代它。
- 若失败，保留全部证据；只剩一次 pilot，必须据新失败轨迹重新登记一个具体假设后才可执行，不自动重跑03c。正式 B/R/M×seeds11/23/37 暂不启动；待完整pilot闭环通过，九次统一使用选定任务与fast实现、等预算从头训练。03b/03c开发比较不算正式三seed改进/消融，开发seed7001不是独立保留集。

本轮仍只资格验证平地低速控制。丰富地形已存在并不代表该策略通过坡道、台阶或起伏路。
