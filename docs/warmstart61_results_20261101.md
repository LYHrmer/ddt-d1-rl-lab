# 61：warmstart干预未发现局部响应反号

**完整执行，判定NO_WITNESS_IN_REGISTERED_INTERVENTION。** 12个固定状态、两臂共24个偏航参考全部通过预定四档稳定门；保留和清零warmstart之间没有一个稳定反号点。这不证明state45在所有数值路径上都是充分状态，但本轮变化量不足以支持把warmstart作为当前反号的主要解释。

干预仅为恢复完整live snapshot后，将22维qacc_warmstart保留或清零；state45、动作、轮相位及其他integration inputs逐值不变，solver选项前后相同，没有插入额外forward或actor调用。四条原训练基线精确复现53，保留臂全部120条物理分支逐值复现60的clean中心。

全部48项配对中心差分的最大绝对变化为 **3.13082893e-10**，最大相对变化为 **2.91123277e-08**。两臂zero下一状态45维最大绝对差为 **6.57316632e-11**。这里的偏航响应本身约为0.0017至0.0185量级（含53112/260约−0.0110）；量级对照是描述性诊断，未增加事后判定门。

模型确实未显式输入warmstart，但“源码存在数值依赖”不能直接推出“它解释了模型反号”。[MuJoCo 3.3.0官方说明](https://mujoco.readthedocs.io/en/3.3.0/computation/index.html#warmstart-acceleration)要求精确重现时保存此状态，也指出其通常对物理量影响很小；本轮针对这12点给出了实际量级。下一步优先检验现有噪声训练轨迹是否覆盖部署策略的状态/动作响应，暂不因猜测增加模型状态维度或改变solver。

预算：**2640policy/26400native/13200PD**，240/240分支完整、120/120原分支精确；零模型、策略、控制更新。CPU9.277秒，进程墙钟10.309秒，含准入监督器21.124秒；峰值内存0.7158GiB，运行最小有效可用3.8937GiB，无内存事件。27项合成测试与Ruff通过；267项源码/输入绑定与全部旧身份前后复核完成，无partial、重试或续跑。

证据：[执行前计划](warmstart61_plan_20261101.md)、[输入/源码绑定](experiments61/warmstart61_preregistered.json)、[全部数字及原始文件SHA](../reports/warmstart61_results_20261101.json)。原始记录保留于 `evidence/warmstart61/execution_01/`；58–60及此前结论保持原样，没有控制性能提升结论。
