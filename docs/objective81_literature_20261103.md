# 81修订：原始论文核查与采用边界

核查日期：2026-11-03。只读取论文、官方项目/代码与已有存档，没有执行模型、训练或物理仿真。下表“采用”均为本项目的设计判断，不是论文对本仓的保证。

| 原始来源 | 核实内容 | 本次采用 / 暂不采用 |
|---|---|---|
| [SuperMimic，2026，§3、Algorithm A.1、Appendix B/G](https://arxiv.org/pdf/2610.28480) | 学习模型提供代理回报梯度；每轮采集、模型拟合、策略反传和critic更新；速度接口使用自身任务奖励，且仅更新接口encoder。 | 采用任务目标对齐、受限策略更新和真实数据反馈的方向；当前只测一次冻结模型支持的更新，不称完整复现。论文速度奖励仍含正则，本项目Q臂不能冒称同一奖励或同一参数化。 |
| [SHAC：Accelerated Policy Learning with Parallel Differentiable Simulation，ICLR 2022，§3.2](https://short-horizon-actor-critic.github.io/)；[官方actor/critic实现](https://github.com/NVlabs/DiffRL/blob/main/algorithms/shac.py) | 短时域回报加终端价值，官方实现包含TD-λ及目标critic。 | critic值得后续独立研究；当前同一H32窗口内已存在模型/物理有限变化反号，不能预先认定加入critic就会修好。 |
| [Adaptive Horizon Actor-Critic，2024，§4.1](https://arxiv.org/pdf/2405.17784) | 根据刚性接触相关的大Jacobian调整展开长度，而非所有接触都截断。 | 暂不引入自适应H。轮子持续接触、学习模型Jacobian不等于物理刚性证据；变H同时会改变本次问题。 |
| [Policy-Aware Model Learning for Policy Gradient Methods，2021，§3、式7–10](https://arxiv.org/pdf/2003.00030) | 让模型学习目标考虑规划器实际使用的策略梯度，而不仅是预测误差。 | 支持“实际policy方向响应”作为后续模型训练候选。论文的梯度估计和损失不是本方案四尺度FD，不能把一维投影修好称完整Jacobian修好。 |
| [ME-TRPO，ICLR 2018，§5、§6.3](https://arxiv.org/pdf/1802.10592) | 分析策略利用模型误差的问题，以模型集成、策略约束和验证限制退化；使用似然比梯度。 | 保留全部三模型、限制每次改动；不把三模型一致性当物理正确证明，不移植论文的模型投票比例来替换本仓全三模型门。 |
| [MBPO，NeurIPS 2019，§4.2–5](https://papers.neurips.cc/paper_files/paper/2019/file/5faf461eff3099671ad63c6f3f094f7f-Paper.pdf) | 分析模型误差随展开累积，使用从真实数据状态出发的短模型轨迹。 | 支持从真实live快照开始短验证；不据此改H32或套用其SAC/critic与理论保证。 |
| [Do Differentiable Simulators Give Better Policy Gradients?，ICML 2022，§3–4](https://proceedings.mlr.press/v162/suh22b/suh22b.pdf) | 刚性、不连续和有限样本可让一阶估计失效；研究一阶/零阶混合及误差约束。 | 分开计算自检、有限尺度响应与真实候选效果。四尺度确定性FD不是论文的随机零阶无偏估计，不能套用其置信保证。 |
| [后续复核，ICLR 2026，2026-10-24 v2，§4–5、Appendix E](https://arxiv.org/html/2604.18161v2) | 重新检验混合梯度方法，区分不连续偏差和方差控制；某些机器人任务上逐步方差控制比检测偏差更关键。 | 不把当前反号直接归因于接触，也不拿三个模型的离散程度当可靠方差估计。暂不增加DDCG、IVW-H或PPO模块。 |

另核对[TRPO，ICML 2015，§3–6](https://proceedings.mlr.press/v37/schulman15.pdf)与[概率模型信赖域方法，Algorithm 3.1](https://arxiv.org/pdf/1304.2808)：两者支持控制更新幅度、评价实际候选的设计思路；本仓确定性F32策略、行L1界、max/abs指标和接触动力学不满足已核实的同一定理条件。这里称“固定步幅的有界候选试验”，不宣称TRPO、完整信赖域算法或单调改进保证。

Abbeel、Quigley、Ng的[Using Inaccurate Models in Reinforcement Learning，ICML 2006](https://www.robotics.stanford.edu/~ang/papers/icml06-usinginaccuratemodelsinrl.pdf)也讨论模型提出局部改动、真实系统评价，以及导数误差和轨迹基点误差的区别。其导数准确性、光滑性等前提不能视为本仓已满足；本方案不照搬其line search。

## 文献如何实际改变方案

1. **分开方向生成、独立验证和完整验收。** v1在同一开发reset上生成并检验方向，不能证明对新状态有效；v2使用互不重叠的D/V/H角色。
2. **同起点比较目标。** 两臂都用六个相同稳态起点，避免27点与6点造成任务采样混杂。它检验两个完整目标函数，仍不能把收益单独归因于MAE或max算子。
3. **提前固定实际候选。** v1只输出方向诊断；v2在看V之前封存六个固定一步候选，V通过后按已注册规则进入完整H，不靠查看四尺度结果选择步长。
4. **诊断性质与候选效用分开报告。** 全尺度单调性仍原样记录；实际候选能否进入H由原80形式的全三模型短筛查决定。81尚未批准或执行，v1保留，80及更早判定完全不改。
5. **critic、H变化、模型再拟合保持独立。** 先判定本次固定方向和候选效果；后续若需修模型，优先研究它实际服务的策略方向，而非再要求所有45维预测全面改善。

当前论文与本地证据只能支持上述实验设计，不能保证单步达到原5%改善门，也不能推导更大预算训练必然有效。
