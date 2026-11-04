# 83：只读跨任务策略梯度几何审计

本轮只回答：在同一冻结25 actor的516个轮行参数上，三个完整75 control世界模型给出的 Q6 下降方向，是否与前进和停车代理的局部不增约束冲突；同时满足约束时，还保留多少局部偏航下降量。模型可行不等于真实收益。0权重更新、0新物理、0候选部署；不改变75、78、81及任何原控制门的结论。

## 数据与重复工作核查

77处理保存的世界模型一步输出，78融合C/A世界模型输出并保护同输入局部身体速度。这两轮没有计算或投影策略头跨任务梯度；本轮不是其重复。

固定21个开发点，全部使用原25 actor的原始完整开发回合，不用81 V/H或任何旧holdout生成方向：

| 代理 | 保存位置 | seed/tick | 点数 |
|---|---|---|---:|
| yaw Q6 | `evidence/objective81/execution_01/development/turn_stop_{seed}` | 81107..81109 × 175/325 | 6 |
| forward | `evidence/steady_sampling80/execution_01/development/forward_stop_{seed}` | 80101..80103 × 110/260 | 6 |
| stop | 同80开发目录，分别forward/reverse/turn_stop | 80101..80109 × 410 | 9 |

每个目录有完整600拍 `full_or_partial_trace.npz`、`point_arrays.npz`、`integration_context_archive.pkl`、`receipt.json`。先核原development seal、每个成员SHA、回合COMPLETE和600/6000/3000计数，再读取本项目自己生成的pickle。NPZ交叉核对45维状态、57维当前帧、10×57历史、滤波、上一动作、0延迟FIFO、33×3命令。627维actor输入由当前帧和历史组成；原shaper记忆从SHA绑定的pickle恢复。没有调用MuJoCo restore，也不把integration archive当成完整live MjData。

九个stop点是运动结束后的零命令上下文，含不同运动历史；不能称独立冷启动站立。若将来确需冷启动站立，应另封存采集方案，目前不采集。

模型只载入75的 `control_seed64001..64003/final_state_dict.pt`。核完整1000更新回执、原sealed文件SHA和最终mappingSHA。actor为53协议引用的完整25 `policy.zip`，mappingSHA必须为 `8fb00b77019c6bce7247a5f3e9774e63064e48307f620fc56ac7f0ef02a35106`。F32源只复制为F64用于反传；所有源和工作副本权重、buffers、保护参数在前后逐mapping核验，禁止保存新策略。

## 三个精确目标

每点从保存的完整控制上下文，以冻结54闭环数学重新计算actor动作，H=32、theta=0；不替换成logged future actions。

- yaw：严格复用 `tools81.policy81.objective_values` 的 Q6，即每reset正/反两个窗口yaw MAE的较大值，再对三个reset取均值。
- forward：六个前进点各32拍 `|body_vx - cmd_vx|` 的均值。身体速度由预测post-state四元数和世界速度经冻结54逆旋转得到。坐标与 `tools/evaluate_d1_ppo.py:180` 的 `body_vx_m_s` 和原J的vx项一致；只是短窗代理，未冒充完整12秒J。省略正比例常数1/.2不会改变约束半空间。
- stop：九个停车点的冻结54原六项C均值，包含vx/yaw/lateral/tilt/height/action_smoothness。它是明确的停车代理，不等于完整J、所有停车安全阈值或冷启动站立性能。

三个目标在同一actor上分别取梯度，再仅保留四个轮行3/7/11/15的128权重和1偏置。记录abs零点、Q6 max平局与分支、动作夹紧数；这些非光滑性意味着AD给出的局部线性量不能自动当作所有方向的真实方向导数。

## 几何问题与可判定输出

令三个516维梯度为 `g_y, g_f, g_s`，计算唯一欧氏投影：

`d* = argmin_d 0.5 ||d + g_y||²`, subject to `g_f·d <= 0`, `g_s·d <= 0`。

枚举空、forward、stop、两者四个active set，不使用随机PCGrad顺序。约束法向量仅正比例单位化以改善数值；检验primal、dual、stationarity、complementarity，并记录原单位内积。奇异/近奇异不能悄悄正则化；没有数值证书即停止并保留partial。平行冗余或相反法向量可由单active解覆盖。

分别将无约束方向和投影方向除以最大轮行L1，复核约束。报告未约束冲突、投影yaw下降保留比例、各任务方向导数、`.009`理想步的线性变化、归一化后是否仍有可分辨非零下降。零向量总是可行，不能把“可行”本身写成成功；接近零的投影不能经归一化伪装成稳定可用方向。

同时报告每轮行L1≤.01条件下、忽略任务约束的线性yaw下降上界：

`B = .01 * Σ_(四轮行 r) max_j |g_y[r,j]|`。

这是L1/L∞对偶的局部线性上界，包含每行偏置坐标；除以当前Q6可描述局部比例。它不证明真实5%必过，也不证明非线性有限步5%不可能。没有进行端点rollout、F32候选量化或真实导数资格检查。

审计只允许得出以下分支：原方向未冲突，则本D代理没有支持“需投影”的证据；有冲突但投影几乎无可用下降，则当前子空间内该局部方法缺乏依据；三个模型均保留可分辨下降且归一化证书完整，才有理由另行预注册物理对照。三模型是初始化重复，不把它们乘上共享reset当独立环境seed。不选一个最好模型放行。

## 论文对应与边界

[PCGrad原文](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf) §2.2（PDF第3页）指出梯度冲突单独不足以推出有害干扰；§2.3/Algorithm 1（第3–4页）采用随机顺序的逐对投影。本审计是同时满足两个半空间的确定性QP，不称PCGrad复现。其§2.4的理论条件包含凸性、平滑性和受限步长，不能移植为这里含abs/max、动作夹紧和学得接触动力学的物理保证。本轮不估计曲率，也不宣称已经证明论文的三因素机制。

## 执行预算和封存

只创建 `tools83/policy_geometry83.py` 与 `tests/test_policy_geometry83.py`。`build_plan()` 为纯stdlib，生成精确spec及继承81的冻结源闭包和全部D/模型来源绑定；root写入并SHA封存后才运行。CLI：

```text
python3 tools83/policy_geometry83.py --plan <sealed-plan.json> --output <exclusive-output-dir>
```

三模型 × 21点 × 32拍 = **2016 dynamics forward + 2016 actor action_head forward**；复用各点图取三个目标梯度，不重算端点。0物理步、0模型/策略更新、0新回合。上限CPU120秒、wall180秒、产物16MiB；root串行systemd Nice10、单线程、MemoryHigh2G/Max3G/swap0，遵循≥4GiB启动及≥2GiB运行保留。模块内CPU硬限119/120秒和逐forward软限110CPU/170wall，外层负责硬wall及内存准入。

输出包括逐点durable JSON、三个模型完整梯度/投影/KKT/参数来源SHA、所有计数及总report；exclusive路径，不覆盖、不续跑。source和plan前后验证；任何异常保留已完成前缀，总体INCOMPLETE。只有root统一执行合成测试和一次审计。本文件编写时尚未执行模型、物理、训练或pytest。
