# 82：论文方法与已有实现的交叉核对

2026-11-03。只读原文、官方代码和仓库源码；未运行模型、训练或物理。此笔记不新增实验、不修改81判定。

## 必须修正的原推论

81 V 的 Q6 主尺度 `h=.009` 为9/9模型预测下降、6/9物理实际下降，不能直接推出原点策略梯度反号。已存四尺度中，两个方向臂 C6/Q6 都只有 `.009` 出现3个非单调点；`.0045/.00225/.001125` 各9/9满足 `Q(+h)<Q(0)<Q(-h)`。这些是有限尺度观察，不是 h→0 的真实导数认证；不据此选择较小尺度替代原主候选。

对同一固定h，令 `odd=(Qplus-Qminus)/2`，`even=(Qplus+Qminus)/2-Qzero`，则 `Qplus-Qzero=odd+even`。有限中心响应反号、负odd被正even抵消、两端均劣于零点应分开报告。even可含动力学非线性、参数/动作舍入、夹紧和abs/max坐标变换；不能命名为已识别的曲率原因。

## DMO 与旧55的关系

原文：[First Order Model-Based RL through Decoupled Backpropagation，CoRL2025](https://arxiv.org/pdf/2509.00215v2)。关键定位：§3.2，附录A.1.3（PDF第15页）；官方代码 [GradientSwapingFunction](https://github.com/machines-in-motion/DMO_code/blob/master/models/dyn_model.py#L156) 与 [rollout](https://github.com/machines-in-motion/DMO_code/blob/master/algorithms/dmo_shac.py#L1046)。

`forward`给真实下一状态，`backward`接回相同真实前状态/动作处的模型Jacobians。actor逐拍重新计算，奖励在真实轨迹上可微重算，动作历史保留。论文同时定义无critic的DMO-BPTT，不能说DMO必然要求critic，也不能将其GPU吞吐收益外推到本项目CPU协议。

仓库 `tools55/observed55.py:CenteredModel` 已实现 `F(s,a)+detach(s_ref_next-F(s_ref,a_ref))`：固定原点处前向真实，局部反向仍为模型导数。这是同类anchor梯度，不能把“使用DMO”当作尚未尝试的新修复。当前81头参数方向与旧55标量残差方向不同，可有新的诊断对象，但不能据旧方法名称重新启动一轮无判别实验。

结构等价的范围仅在固定原点附近。旧轨迹、actor版本、commands以及history/filter/shaper/FIFO必须匹配。更新actor后反复复用同一future真状态不是新的on-policy DMO rollout。将logged actions直接detach替代重新生成actions，也会切断反馈与策略参数路径。

81 V NPZ具有32×45 post-state、32×627原始观测、issued/applied动作、commands和首拍shaped输入；起点context及pickle的python_state['shaper']保留控制记忆。它在控制链逐拍重建准入后可用于固定anchor只读诊断；H紧凑日志缺同等full45上下文。保存integration状态并不保证可恢复完整MuJoCo derived cache后逐值再仿真。

DMO不能保证纠正模型局部Jacobians、有限大步响应、训练外状态分布或完整H控制门。普通finite difference不能用于检查“每个扰动都重新替换为同一固定日志值”的自定义反向；自检应固定校正并明确它只是局部代理曲线。

## SuperMimic的32维latent不是本仓627→3 encoder

原文：[SuperMimic](https://arxiv.org/pdf/2610.28480)，Appendix B Eq.(A1)、Table A4（PDF第14–15页），主文§4.3/§6。速度接口生成32维tanh latent，由共享decoder结合机器人观测生成全身动作；速度任务梯度不更新decoder，但decoder仍由motion-tracking任务训练。不能把这概括成“decoder在所有训练中永久冻结”。

本仓 `tools52/adapter52.py` / `training/d1_rl/velocity_policy15.py` 的encoder是627→128→64→3的速度估计器；其3维输出拼接57维当前frame供policy网络，再经16维action head。81只更新4个轮输出行共516参数，冻结其余网络。这里的encoder不是论文的goal-to-skill encoder，简单解冻它不是对论文速度接口的等价复现。冻结腿输出权重也不意味着闭环腿动作永不变化：观测改变仍会改变腿输出。

论文结构允许latent经decoder耦合多关节响应；当前头子空间较窄是可检验的结构差异，但现有证据不能证明它就是5%门未过的原因，不能据此直接扩大训练参数。

## 本轮纯读回的边界

`response_crosscheck82.py` 重建81 V所有3模型×3reset×2臂×4尺度×4前缀的Q/C有限响应，读取actual-F32/ideal-F64原始参数storage，但不执行pickle或导入Torch。D只存ideal-F64端点目标；V只存actual-F32参数提升F64的模型目标。D与V状态不同，不存在可用的同条件F32/F64性能干预对照；更没有F64物理结果。

小参数舍入误差、大于多个ULP的实际动作，只能排除“初始动作完全舍入为零”等具体解释，不能排除后续闭环放大，也不能单独证明量化或模型Jacobian造成反号。三模型共享三环境reset，不是九个独立环境seed。局部C是冻结54六项cost，不能替代完整H的J。
