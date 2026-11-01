# 61：固定物理状态的warmstart反事实（执行前登记）

2026-11-01。60否定了“当前动作噪声统一解释反号”。代码审计确认该环境零延迟且轮控制无PI积分，但state45仅包含qpos/qvel，未包含MuJoCo的qacc_warmstart。官方说明该量用于约束求解的初值，精确数值重现需要保存它；这并不证明它足以改变当前动作响应方向。[MuJoCo 3.3.0状态说明](https://mujoco.readthedocs.io/en/3.3.0/computation/index.html#warmstart-acceleration)

1. **问题与唯一变量**：在固定原12个训练状态、全部相位、当前clean动作及历史下，仅将22维qacc_warmstart保留原值或清零。不改模型solver、容差、迭代、特征、actor、shaper或冻结源码；干预后不额外mj_forward。检验原state45相同的输入能否出现稳定的相反动作响应，不假定warmstart是59失败的根因。
2. **重演及原分支身份**：原四训练seed53103/53106/53109/53112各600步，noiseσ=.02与seed+900000原RNG不变；每tick单次actor/shaper取得clean，按原clip/cast加noise推进完整基线，逐值复现53。固定110/260/410在actor之后、env.step之前保存live完整MjData。分支只用clean中心；原warmstart臂全部120条分支须逐字段复现60的clean分支，累计计数单列。每臂两个zero只核自身重复，不能要求清零臂等于原臂。
3. **扰动与预算**：沿用58四档绝对δ=3.125e−6/6.25e−6/1.25e−5/2.5e−5及[+1,−1,+1,−1]轮方向、L∞1/L2=2、不乘yaw error，每中心2zero+8扰动各一步。2400基线+240分支=2640policy/26400native/13200PD，cap3000；零模型、策略、控制更新，零动力学模型调用。CPU300秒、墙钟420秒，预计20–40秒；本机重任务串行，Nice10、单线程、MemoryHigh2G/Max3G/swap0，4GiB有效准入、2GiB运行保留。
4. **预定判定**：全部240分支必须完整；每点每臂四轮实际扰动可见、两zero精确，FD先在F64求值。沿用幅值≥1e−5、四档同号、全档极差/最大绝对值≤25%、各档单侧同号且差≤25%的稳定门。某一预定点的两臂均稳定且每档相反，便是该数值执行路径的输入混叠反例，记WITNESS_FOUND；所有12点完整报告，其他未解析点不能抹去已成立反例。全12对稳定且无反号只记NO_WITNESS_IN_REGISTERED_INTERVENTION，不能声称Markov已证明；无反例且有未知则REFERENCE_UNRESOLVED。没有选seed排名、性能改善或理想物理非Markov的结论。
5. **记录与完整性**：保存干预前、干预后和一步后的warmstart；断言state45及实际action不变，验证除warmstart外的完整integration state在赋值前后逐值相同。报告两臂zero下一状态差、全四档FD区间作为描述，不新增通过门。源码/输入先SHA绑定、末尾复核，失败和partial全部保留、不覆盖、不续跑；不改48归档、v27或58–60的原结果。

输入与源码绑定：`docs/experiments61/warmstart61_preregistered.json`。
