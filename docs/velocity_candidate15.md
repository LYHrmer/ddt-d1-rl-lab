# D1 速度监督候选：本地实现与数值验证

状态：工程候选，尚未进行机器人采样或资格评测，不是训练 release。合同 09/10 的失败、中间权重和冻结输入均保持原样。本轮把[上游阅读](upstream_reference_review_20261017.md)中的一个假设落实到现有 CPU/SB3 工程，没有迁移 Isaac 框架。

## 候选改变什么

D1 现有部署输入不含实测线速度，但已有当前 57D 与过去十帧，共 627D。本候选用这份可部署输入估计机体系三维线速度，再供 actor 使用；真值仅在训练损失中出现。它不增加部署传感器，不同时加入特权 critic、奖励或课程改动。

| 部分 | 输入与计算 |
|---|---|
| 速度编码器 | 原 627D → Linear/Tanh 128 → Linear/Tanh 64 → Linear 3，输出单位 m/s |
| actor | 当前 57D 与估计速度 3D 拼为 60D → 两层 128/Tanh → 16D 高斯均值；部署裁剪到 [-1,1] |
| critic | 原 627D → 两层 128/Tanh → 价值；不读取速度真值 |
| 辅助损失 | 三维预测与同拍机体系速度真值的均方误差，乘 `aux_coef` 后加入 PPO 总损失 |

这不是 Wheel-Legged-Gym 的逐行移植。上游训练 actor 使用分离梯度的 latent，并另行监督编码器；本地让编码器继续接收 PPO 策略梯度，辅助项与 PPO 共用一次优化。这样 `aux_coef=0` 时编码器仍可学习，可与正系数版本作同架构比较。最终是否采用这个调整，需要新实验回答。

与旧 627D 直接 MLP 相比，候选也改变了 actor 的表示与网络结构。因此，候选对旧成绩的差异不能全部归因于速度监督。应先比较同架构 `aux_coef=0` 与预先固定的正系数，保持环境数、种子、任务、预算、PPO 参数与评测后端一致；单个种子仍只是开发性证据。

## 标签与观测必须同拍

[VelocityLabelWrapper](../training/d1_rl/velocity_labels15.py) 在底层 `reset()` / `step()` 返回后读取缓存的 `trunk_quat` 和 `frame_vel`，复用已审计的 `inverse_rotate`，把世界线速度旋至机体系。底层 `_observe()` 已刷新过这些缓存；wrapper 不调用额外 forward 或物理积分。

训练观测为 `{'actor': float32[627], 'velocity_target': float32[3]}`，交由 SB3 的 DictRolloutBuffer 成对存储。标签对应当前返回的观测，不能从一次 step 的 info 中取动作执行后速度，去监督缓冲中的动作执行前观测。终点观测与外层自动 reset 后的新观测也必须各自携带自己的标签。

策略特征提取器只读取 `actor`。真值既不进入动作分布，也不进入 value；只有辅助损失读取 `velocity_target`。数值测试会改变真值而保持 actor 输入不变，检查动作和 value 不变，并分别验证辅助损失有有效梯度。字典仅是训练数据容器；对外部署仍使用原 `obs` / `history` 两输入。

## 导出必须包括编码器

[新增导出检查](../tools15/export_velocity15.py) 构建完整的 `obs + history → 627D → encoder → actor → action head → clip` 图。输入保持 `[1,57]` 与 `[1,10,57]`，历史仍是过去十帧、从旧到新；不将真值标签作为 ONNX 输入。

使用固定随机数值 fixture，逐样本比较策略确定性输出、Torch 完整图和 Python ONNX Runtime。保留原来的绝对容差：策略/Torch `1e-6`，Torch/ORT `1e-5`；ONNX opset 11、IR 7，并检查签名及算子。随机图的输出对齐，只证明该图的数值链路，不证明训练完成、速度估计准确或机器人能停稳。

本轮命令行只提供 `--self-check`。生成的是明确标注的随机数值工件，不能作为正式策略导出或资格依据；没有接入旧中间权重，也没有绕过旧导出器的架构检查。未来实际完成的候选 checkpoint 还需新的来源检查、最终权重身份、C++ ORT 对齐与固定资格评测。

## 本轮实际验证

- 新增测试 67 项：标签 20、算法 20、导出 27，全部通过；与接口及零物理静态保护一起运行共 167 项通过。
- 关闭辅助项时，本地更新与未改动的 SB3 2.7.0 `PPO.train()` 对照逐参数一致。另在内存中删除总损失的辅助项，专门的接线测试实际失败；生产文件未被改写。这证明测试能发现辅助项没有接入优化的错误。
- [独立数值回执](../reports/velocity_candidate15_numerical.json)：64 条 batch1 fixture，策略/Torch 最大误差 `0`，Torch/Python ORT 最大误差 `6.51925802230835e-9`；改变标签时 actor/value 输出差异均为 `0`。
- 漏掉编码器、保留合法签名和算子的错误 ONNX 图，也会被数值检查拒绝。新逻辑已纳入零模型/零物理静态扫描，Ruff 关键错误检查通过。

本机回执记录 Torch `2.12.1+cu130`，计算实际在 CPU 上；CI 使用 Torch `2.8.0` CPU 构建，二者不合并为同一运行身份。随机工件的 ONNX 与 fixture 文件 SHA 在回执中；仓库只保存小型 JSON 回执，CI 可重新产生随机数值工件。64 个输入的对齐不构成所有输入的形式化等价证明。

## 验证方式与本机安排

在已有本机依赖下，以下命令只运行合成数据测试和随机图导出，不创建机器人模型或产生机器人 transitions：

```bash
rtk proxy env PYTHONPATH=training:.python-deps OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m pytest tests/test_velocity_labels15.py tests/test_velocity_policy15.py -q
rtk proxy env PYTHONPATH=training:.python-deps OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m pytest tests/test_export_velocity15.py -q
rtk proxy env PYTHONPATH=training:.python-deps OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 tools15/export_velocity15.py --self-check --output /tmp/d1_velocity15_numerical_fixture
```

输出目录必须不存在。测试包含少量合成 minibatch 梯度更新，不能记作机器人训练成果。导出在独立进程开始时配置 Torch/ORT 单线程，避免依赖已经进行计算的进程能重新设置全局 inter-op 线程。CI 为候选单列 CPU Torch 检查；本地串行执行，避免并发 Torch/ORT 占用内存。

后续实际训练沿用[本机低并行准备草案](local_training_plan14_draft.md)：4 环境候选、必要时在正式开始前选择 2，保留内存保护并允许更慢。[准备方案 16](local_comparison16.md)已补齐有界训练入口、新预算提案、预检计数、有效内存准入、正式时长、失败规则和同架构比较顺序；仍需明确批准后才可冻结执行，不沿用合同 09/10 余额。
