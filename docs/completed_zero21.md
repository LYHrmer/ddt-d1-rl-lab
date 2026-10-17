# 已完成 zero18 的独立验收21

18因监督器收到SIGTERM而停止，发送者尚未确认；原报告保持STOPPED，random的最终计数未知，partial权重不作为候选。zero已经独立完成本轮1,048,576新步，累计2,097,152步。21只验收这份完整最终权重，不恢复18、不使用旧预算余额，也不改变19要求完整18赢家的资格门。

本入口依据所有者持续目标“把这些优化点，在这台电脑能力之内的都完成”，使用独立新输出 `evidence/completed_zero21/execution_01`。新训练为0；默认只显示计划，不读取权重、不创建输出、不加载推理或物理库。

```bash
rtk proxy python3 -B tools21/closeout_completed18.py
```

新源必须先提交。实际执行由根代理安排串行窗口，拟使用有明确RuntimeMaxSec和KillMode=control-group的systemd用户transient unit，让任务独立于交互工具会话运行。代码仍自行采样内存、限定进程时限并清理自有子进程；本轮准备未创建unit或启动真实任务。

```bash
rtk proxy python3 -B tools21/closeout_completed18.py --execute
```

独立release先绑定新三份源/测试/文档、复用19生命周期及其导入helper的Git blob/SHA。旧来源门要求18已停止且清理完成；zero及对应E0的完整工件集合必须与原COMPLETE stage的逐文件SHA、最终receipt一致；核四worker完整计数、初始policy/Adam状态和前八轮确定性字段相等。随后重用18训练身份/模型参数校验、17完整6Gemm图导出及64组SB3/Torch/ORT数值规则、原冻结C++64和03d正式评测纯helper。不会读取random部分权重或伪造旧stage启动凭据。

| 新阶段 | 最大policy/native/PD | 墙钟上限 |
|---|---|---:|
| 完整图导出与64组数值对齐 | 0/0/0 | 180s |
| 原冻结C++64数值核对 | 0/0/0 | 90s（内部进程60s） |
| 原36例：3任务×4延迟×reset1001/1002/1003 | 21,600/216,000/108,000 | 300s |
| 仅当前36例全部达原阈值时，额外reset2001/2002/2003 | 同上 | 300s |
| 整轮硬上限 | **43,200/432,000/216,000** | **900s** |

所有阶段串行，启动有效RAM≥6GiB、运行保留≥4GiB；有效值由既有host/cgroup祖先最小余量算法给出，swap不计。每阶段一个进程、独占新输出、nonce/父PID/固定命令核验；异常、资源未知、内存或时限不足停止，不重试、不续跑。任务自然早停仍保留全部36例、原惩罚及实际分母；原36不全过时额外36记SKIPPED，工程COMPLETE不等于完整延迟资格通过。

运行结果保存在新report、protocol及原始NPZ。父监督器独立重算轨迹指标与计数；未知总数保留null，已知部分另列，不从缺失回执补零。原18的random未知计数仍未知，21不会将项目累计值改成精确总数。

zero仅是额外零延迟训练，21可判断这份权重在固定矩阵的表现，不能证明延迟随机化收益、恢复18的两组对照、或授予地形/ROS/实机资格。后续训练须先看本轮失败反馈及外部SIGTERM诊断，另定独立问题、来源和预算。

合成测试只用临时文件、fake进程/内存与有限计数，不加载模型、ORT、环境或积分：

```bash
rtk proxy env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest --noconftest -o addopts= -q tests/test_closeout_completed18.py
```
