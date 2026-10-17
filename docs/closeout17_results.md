# 两仓工程收尾与 D1 正式评测结果

本轮收尾的实际执行已完成，**没有新增训练**。D1 的真实 control 导出和 C++ 数值对齐通过，36 个正式场景已采齐，但完整延迟参考资格未通过；wheel 的既有 v27 平地 GUI/控制演示复验通过。保留这些明确边界，不继续调参或加训练。

| 仓库与验收项 | 实际结果 | 证据 |
|---|---|---|
| D1 完成权重的完整 ONNX 导出 | PASS，包含 encoder/actor/clip | [收尾摘要](../reports/closeout17_20261018.json) |
| 同权重 C++ ORT 1.10 数值对齐 | 64/64，最大绝对误差 4.1723e-7 | 同上，原数值阶段0物理步 |
| D1 正式36场景收集 | COMPLETE，27/36满足原参考阈值 | 下表与逐例摘要 |
| D1 全36例参考资格 | **未通过**，40/60ms存在失败 | 不授予完整部署资格 |
| wheel v27真实窗口与固定控制演示 | PASS，1600拍完整、正常8000 native | [wheel结果与入口](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/docs/b22_closeout30_result_20261018.md) |

## D1 固定矩阵与失败位置

使用实际完整导出的 ONNX，通过 Python ORT CPU、单线程确定性推理；环境/配置与方案16相同。沿用原 evaluator 的场景、每例最多12s、指标、缺失区间惩罚和参考阈值。三项任务分别乘 reset seed1001/1002/1003 与0/20/40/60ms延迟，正好36例，不混入训练 seed101 之外的新训练。

| 延迟 | 已观测场景 | 完整12s | 提前终止 | 原参考阈值达标 |
|---|---:|---:|---:|---:|
| 0ms | 9 | 9 | 0 | **9/9** |
| 20ms | 9 | 9 | 0 | **9/9** |
| 40ms | 9 | 6 | 3 | **6/9** |
| 60ms | 9 | 4 | 5 | **3/9** |
| 合计 | **36** | **28** | **8** | **27/36** |

| 任务 | 0ms | 20ms | 40ms | 60ms |
|---|---:|---:|---:|---:|
| forward_stop | 3/3 | 3/3 | 3/3 | 3/3 |
| reverse_stop | 3/3 | 3/3 | 3/3 | 0/3 |
| turn_stop | 3/3 | 3/3 | 0/3 | 0/3 |

八例提前终止均在最后保存状态触发 `fallen=true`，tilt 超过原0.6rad终止线，没有非法触地标志；它们作为任务失败保留，没有重试：

- 40ms转向的三个 seed：分别在7.86、8.32、8.24s终止。
- 60ms转向的三个 seed：分别在5.42、5.40、5.40s终止。
- 60ms后退的 seed1001/1002：分别在11.92、11.66s终止。

另有一例完整运行但未达标：60ms后退 seed1003。稳定 vx MAE 为0.002323m/s，跟踪项达标；停车位移为0.353621m（上限0.05）、末秒水平速度0.238106m/s（上限0.02）、末秒绝对 wz 为0.260569rad/s（上限0.03）、峰值 roll/pitch 为0.306423rad（上限0.20），四项失败。

因此可以报告本次固定三任务、三个 reset seed 下0/20ms各9/9通过；不能把20ms解释成通用安全延迟上界，也不能把36例“流程采齐”写成36例资格通过。单训练种子不构成统计显著性或泛化结论。

## 真实权重与数值链

选择[方案16](local_comparison16_results.md)的 aux_coef=0 control。原 aux_coef=1 对照未展示联合收益，其负结果与原始输出保持不变。本轮没有加载旧合同的 partial 权重，没有使用随机图替代真实模型。

- 训练完成数：1,048,576 transitions，原训练 seed101、4环境，随机初始化且无教师。
- 最终 checkpoint SHA：`428ff0ce9e8e88e10dabf94801a57b88de7e76d07b42800553b69c7f590e4eff`。
- 导出 ONNX SHA：`8957beef884e0837b9fce8e199a21d69dcad4a570a32f8184ae80e8a6719c34e`。
- 图包含六个 Gemm，opset11/IR7；输入为 `obs[1,57]` 与 `history[1,10,57]`，输出为裁剪后的 `actions[1,16]`。历史保持过去十帧、从旧到新，不输入真值标签。
- 真实权重的64个 batch1 fixture：SB3/Torch最大差为0，Torch/Python ORT与Torch/C++ ORT最大差均为`4.172325134277344e-7`，保持原1e-6/1e-5容差。
- C++使用原冻结二进制和 schema、ORT1.10.0；仅数值推理，没有构造机器人、物理积分或ROS节点。

导出阶段原回执仍保留当时的 `cxx_ort_1_10_status=NOT_TESTED` 和未授予资格标志；随后独立的 C++ PASS 与正式评测结果分别保存，不回写旧阶段来制造“当时已通过”。本次没有 C++闭环轨迹、ROS闭环、地形或实机验收。

## 本轮预算与资源

[事前收尾范围](closeout17_contract.md)规定：导出与C++数值阶段0物理步，正式矩阵最多21,600 policy /216,000 native /108,000 PD。实际因为八例合法提前终止，共 **20,011 policy /200,110 native /100,055 PD**；所有计数有归属、没有未知阶段，不补齐短回合或复用旧训练余额。

D1三阶段墙钟约 **98.583s**，父监督器最低观测有效可用内存 **6,791,360,512 bytes（约6.325GiB）**；启动6GiB与运行3GiB门均未触发。全部进程退出0，来源、计数与有限清理检查通过，无残留清理信号。

wheel 是独立模型与独立协议，本轮另计正常8000 native与编译探针2 native，不能把其1.6m/s成绩归给D1纯RL权重或叠加为同一控制结果。其实际 GUI、初始/行驶/最终三图、199帧渲染和停止过程均完成；[独立小回执](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/reports/b22_closeout30_20261018.json)保留来源、资源与清理证据。

## 本机交付位置与只读检查

当前实验候选仅在上述验证范围内交付，完整延迟资格未过，未替换旧ROS或实机控制器。

| 内容 | 本机位置 |
|---|---|
| 原完成训练 | `/home/lyh/ddt_comparison16_replacement_20261017/evidence/local_comparison16/` |
| 本轮来源/授权冻结 | `/home/lyh/ddt_ros2_ws/evidence/closeout17/release_01.json` |
| 实际导出模型与metadata | `/home/lyh/ddt_ros2_ws/evidence/closeout17/execution_01/export/` |
| C++64原回执 | `/home/lyh/ddt_ros2_ws/evidence/closeout17/execution_01/cpp/cpp_native/report.json` |
| 正式36例回执与NPZ | `/home/lyh/ddt_ros2_ws/evidence/closeout17/execution_01/formal/` |

在原本机工作区可只读核验冻结来源并列出已完成的执行计划：

```bash
rtk proxy python3 -B tools17/run_closeout17.py plan --release evidence/closeout17/release_01.json
```

该命令不加载模型或运行物理。已经存在的执行目录禁止覆盖或续跑；日常阅读使用本页和摘要即可。完整模型、第三方资产、原始轨迹和日志按仓库约定保留本地，Git保存源码与小回执。旧09/10、方案16的成功/失败记录和原用户文件均保留。

新源码提交为`44e89688642932a88ee7d4607820b9d389bc66c6`；[两项源码CI](https://github.com/LYHrmer/ddt-d1-rl-lab/actions/runs/39049777483)通过。271项本地定向检查与单独的随机网络完整导出测试通过；实际权重、数值向量、36条原轨迹及来源/计数由独立只读核验复算，不产生额外物理步。

本轮以工程交付和如实完成固定验收收尾。40/60ms鲁棒性、更多训练种子、地形和实机能力作为明确未通过或未验证事项保留，不属于已经完成的性能承诺。
