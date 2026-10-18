# D1 34：33候选的C++有状态闭环验证结果

**C++部署路径通过：formal 36/36与独立holdout 36/36全部满600拍，原阈值未改，停车全部通过，且与Python33逐位等价。** 机器人执行的动作只来自C++服务器（gain1.5、tau0.1s偏航整形与十帧有效历史在C++内完成，原始627观测直接送入）；Python只是同输入的数值对照，不参与控制。

| 矩阵 | 通过/满时域 | 失败标记 | 平均 J | C++请求 / Python对照 | 最大动作差 | 627有效输入逐位一致 |
|---|---:|---:|---:|---:|---:|---|
| formal | 36/36 / 36/36 | 0 | 0.228668257 | 21600 / 21600 | 0.0 | 是 |
| holdout | 36/36 / 36/36 | 0 | 0.230278735 | 21600 / 21600 | 0.0 | 是 |

停车最差：末秒水平速度formal 0.00187、holdout 0.00212 m/s；停车位移最大0.0403 / 0.0413 m；最大横滚或俯仰0.108 / 0.109 rad。门槛为预注册的原值，没有放宽。

**与Python33的一致性。** 72条轨迹的SHA与已发布Python33逐条完全相同，J的最大绝对差为0；因此C++路径复现的是同一组闭环行为，而不是另一组“恰好也过线”的结果。动作差为0是单线程、同ORT 1.10、同输入的结果，并非容限内近似；门槛仍是≤1e-5。独立读回重新计算了报告点名的216个轨迹/sidecar文件SHA，0个不符。

## 保留的取舍

综合指标没有改变：formal平均J 0.2286682573，相对25 random基线 0.2159389287**仍差5.89%**，只比32好3.92%。相对当前保留的默认17，33/34把formal从27/36提升到36/36，但17的J含失败惩罚，不能把J降幅当作同等的连续控制提升。34验证的是部署链，不能改善也不会掩盖这一取舍；默认模型未被自动替换。

## 规模与来源

- 实际43200 policy / 432000 native / 216000 PD，C++请求43200次全部返回，训练0；整轮239.14秒（含编译与准入）。
- 资源：最低有效可用内存6.24 GiB，任务组采样峰值0.67 GiB，high/max/oom事件均为0；unit退出码0、MainPID=0、cgroup已清空，cleanup_complete=true，输入未变。
- 源 `dd93918165dedb23171cace8dbebeb77265d1e4d`（CI [39136582371](https://github.com/LYHrmer/ddt-d1-rl-lab/actions/runs/39136582371) 成功）；二进制SHA `eb3e0dc8f1500e8c57a9001e3a9bd6a78153dc319d024d27aecd4b2c2df2c3aa`；计划SHA `0453b65ccc07c2ada4a5b2b48ebd0706b59b15f2d72b67bfa867f9c4397b58c3`；顶层report SHA `c5a66490a4fbd92c3f4458792fcde28f65002acb7ee5aca519c79e366d50addb`。
- 完整输出 `evidence/cpp_reference34/execution_01/`（本地保留，不入git）；紧凑机器摘要 [`reports/cpp_reference34_results_20261018.json`](../reports/cpp_reference34_results_20261018.json)，由 `tools34/summarize_cpp_reference34.py` 只读保存文件生成。

## 边界

只覆盖原72场（forward_stop / reverse_stop / turn_stop × 0/20/40/60ms延迟 × 每矩阵3个种子，formal 1001–1003、holdout 2001–2003），不外推到全部场景；这是仿真闭环，不是新增实机结果。

[验证设计](cpp_reference34.md) · [33结果](reference_shape33_results_20261018.md)
