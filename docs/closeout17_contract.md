# 两仓收尾 17：固定验收范围

用户在获知两个仓库剩余收尾项后设置目标“完成收尾工作”。本轮据此完成真实 control 权重导出、64 例 C++ 数值对齐、36 个正式评测场景，以及 wheel 仓库既有 v27 GUI 与固定控制演示复验。**不增加训练，不调参，不重开合同 09/10 或已关闭的方案 16。** 验收失败保留真实结果，不自动重试或扩大任务。

## D1 独立执行

只接受[方案 16 执行摘要](../reports/local_comparison16_execution.json)绑定的独立 replacement：`/home/lyh/ddt_comparison16_replacement_20261017`。取 `train_control/policy.zip`，SHA 为 `428ff0ce9e8e88e10dabf94801a57b88de7e76d07b42800553b69c7f590e4eff`。检查整个方案 16 已完成、原 86 项来源、全部登记产物、真实完成计数与最终 checkpoint，不能拿 partial 或随机 fixture 替代。

顺序固定为：真实导出 → C++64 数值对齐 → 正式36场景。新源码先提交，执行前记录实际提交及源码 SHA；旧训练输入和权重保持原字节。新产物写 `evidence/closeout17/execution_01`，不覆盖，不续跑。准备/计划入口不加载模型或产生物理步。只有独立的授权冻结收尾记录才能执行正式评测。

| 阶段 | 最大 policy transitions | 最大 native steps | 最大 PD updates | 子阶段墙钟上限 |
|---|---:|---:|---:|---:|
| 实际完成 checkpoint 导出与 64 例 SB3/Torch/Python ORT | 0 | 0 | 0 | 180 s |
| 同 ONNX 的 C++ ORT 1.10 数值对齐 | 0 | 0 | 0 | 90 s |
| 36 个正式评测 case | 21,600 | 216,000 | 108,000 | 600 s |

两个数值阶段不构造机器人或积分物理；导出包含训练过的完整 encoder、actor 和 clip，保持原 `[1,57]`＋`[1,10,57]` 输入，opset11/IR7。64 样本逐例 batch1，SB3/Torch 最大绝对误差 ≤1e-6，Torch/Python ORT 与 Torch/C++ ORT ≤1e-5。数值对齐不等于 C++ 闭环控制资格。

正式评测使用这个实际导出 ONNX 的 Python ORT CPU 推理，不能换回随机图或别的 checkpoint。逐字复用 SHA 绑定的原 evaluator 场景、回合与指标函数：三个任务 `forward_stop/reverse_stop/turn_stop`，reset seed 为 1001/1002/1003，delay 为 0/20/40/60 ms，共36例，每例最多600个20ms控制区间（12s）。原跟踪/停车/姿态参考阈值不改；分别报告9个零延迟 case 与各延迟的完整性、失败和参考达标数，不以总体均值掩盖失败。

流程完成与参考资格分开：合法提前跌倒仍是该 case 的任务失败，继续收齐预定矩阵；代码、资源、来源、计数或清理错误停止整个新执行。只有36例完整且全部满足原参考阈值，才标记本轮 `all_36_reference_pass=true`；同时保留原指标中只对零延迟定义的 nominal 标志。没有多训练种子的显著性、地形、C++闭环、ROS或实机通过声明。

有效内存沿用取 host MemAvailable 与可见 cgroup-v2 祖先 hard-cap 余量最小值，swap 不算 RAM。阶段启动至少6GiB，执行中保留3GiB；CPU、Torch/BLAS/ORT均单线程。父监督器每秒采样，评测预测回调检查资源与600s时限。子进程各有30s启动/收尾宽限；停止只清本次新建的进程组，以 INT10s→TERM5s→KILL2s有限收尾。未知计数保持 null，所有已尝试物理步占本轮上限，禁止按返回码猜零。

## wheel 独立演示

只复验已存在 v27 的 `script / flat_1p6 / actor B`，使用原99D/16D C18＋残差模型和固定 seed181202；真实桌面显示，不使用 keyboard、新任务、训练或人为接管。上限1600个控制拍（16s仿真）、正常积分8000 native，另保留原编译探针最多2 native。原 host 的180/235/240s门与外层600s门保持；整个命令额外有900s有界监督与同样的6GiB启动/3GiB保留线。

v27 READY只证明输入身份。本轮必须保存真实窗口就绪标识、初始/行驶/最终图、render frames、实际控制计数、性能与 worker/host/supervisor/launch 完成回执，确认不是 exit0 的空跑。新 verifier 独立保存结论，不改原 v27 源码/manifest、历史资格或用户结果。

## 交付与回退

保留旧成功版本、旧失败和中止记录以及方案16两组最终权重；新验证仅写独立目录。数值图或正式评测失败时不提升部署资格。两仓更新当前入口、完成/未通过事项、可追溯小回执与复验说明，相关测试与CI通过后推送。完整权重和原始轨迹按各仓库约定保留本地；收尾不宣称实机完成。
