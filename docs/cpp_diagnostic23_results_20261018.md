# D1 23：真实 C++ 推理闭环结果（2026-10-18）

独立只读审计通过。本轮复用原16已完成 control 权重及17导出 ONNX，真实 C++ ORT 1.10 actor 的动作驱动一套本地物理环境；Python ORT 1.23.2 只对同一实际输入做数值对照。没有新增训练。

这证明该冻结模型在原36场景中的后端闭环等价性。原模型的完整延迟资格仍未通过，`old19_eligibility_granted=false`、`qualification_eligible=false`、`deployment_qualified=false`；本轮不构成 ROS 或实机资格。

## 实际执行与预算

证据目录为 `evidence/cpp_diagnostic23/execution_01`。父回执和 worker 均为 `COMPLETE`、`error=null`、`cleanup_complete=true`，worker 另记 `inputs_unchanged=true`。实际计数为 **20,011 policy / 200,110 native / 100,055 PD**，低于独立上限 21,600 / 216,000 / 108,000；早终止部分按实际值记账，没有补满或重跑。

父流程耗时 **108.511596 s**，worker 106.649812 s，均在预定900/840秒界内。父资源日志109个样本的最低有效可用内存为 **6,940,532,736 B（约6.464 GiB）**；worker 106个样本最低7,573,585,920 B。日志中的可见 cgroup 剩余额均为 null，有效值等于 host MemAvailable；这些是采样最低值，不能推断未采样瞬时最低值或进程峰值RSS。启动线6 GiB、停止保留线4 GiB。父清理回执只扫描自己接管的后代，`remaining={}`、未用额外信号；actor 经 EOF 正常 `CLOSED`、退出0。

## 64 fixture 与实际逐拍对齐

C++流为1条 READY、20,075条连续且唯一的 OK 响应（seq 0–20,074）、1条 CLOSED。attempted/returned/requests 均为20,075，等于 **64 fixture + 20,011实际策略拍**，Python oracle 调用数相同。READY 绑定了实际 binary、完整 ONNX、源码与 ORT 动态库 SHA，CPU单线程。

64个固定 fixture 相对导出时 Torch/SB3 动作的最大绝对误差为 **4.172325134277344e-7**，小于原1e-5门；对 Python ORT 为0。逐拍保存的20,011组627维输入及16维 C++/Python ORT动作均有限、形状和float32类型正确，动作范围[-1,1]，两个后端动作逐字节相同，最大误差 **0**。C++动作还逐字节等于物理轨迹中的 `issued_action`。previous-only十帧历史逐拍移位正确；每例初始历史按原环境重复初始观测，未改变原语义。

## 原36场景与17的公平对照

矩阵保持 forward/reverse/turn stop × 0/20/40/60 ms × seed1001/1002/1003。独立重算冻结原指标、命令、时钟与计数，36例全部精确匹配保存回执；没有修改阈值。

| 指令延迟 | 原阈值通过 | 完整600拍 | 早终止 |
|---|---:|---:|---:|
| 0 ms | 9/9 | 9/9 | 0 |
| 20 ms | 9/9 | 9/9 | 0 |
| 40 ms | 6/9 | 6/9 | 3 |
| 60 ms | 3/9 | 4/9 | 5 |

总计27/36通过、28/36完整。60ms reverse seed1003虽完整，但停车速度、停车角速度、停车位移与姿态峰值仍不合格；其余8例为早终止。

23的36份物理NPZ与17对应文件 **SHA全部相同**。进一步逐例比较19个物理字段，共 **684个数组**：dtype、shape及C-order字节全部相同，所有共有case JSON字段也精确一致。因此这次C++动作链路复现了原17的成功与失败，没有额外证明抗延迟性能改善，也不构成后端速度基准。

## 来源与复核范围

独立核对29项source SHA、78项23工件、62项原17/16来源工件，全部相符。数值证据使用原完成control checkpoint，未使用18/22部分或退化候选。

| 工件 | SHA256 |
|---|---|
| 23父报告 | `7a0aeaa8ce5275fc72a8ec4bbe5dec65c8617d867448a781460c5a35c36c5602` |
| 23 release | `b2d1d7090dc8b4cb68b9d4b787a0f8b757389888b871df1cf024e8548483274b` |
| 原完成control checkpoint | `428ff0ce9e8e88e10dabf94801a57b88de7e76d07b42800553b69c7f590e4eff` |
| 完整actor ONNX | `8957beef884e0837b9fce8e199a21d69dcad4a570a32f8184ae80e8a6719c34e` |
| C++ actor binary | `79d056440ebe276cc705399705226d5387240206b01522ec7fd7b85e4a0904a5` |
| C++逐响应stdout | `f420a24774e54abf75f8b59374afba3dc2387de4e5225706c30a552ab9034772` |

完整工件SHA、逐例数组比较、计数与资源样本核验见 [机器报告](../reports/cpp_diagnostic23_results_20261018.json)。本次复核只读取既有JSON/JSONL/NPZ及文件字节；仅运行NumPy与冻结纯指标函数，没有加载Torch、MuJoCo或ORT，没有新推理、仿真、重放或测试，也没有改写旧23现场或冻结源码。
