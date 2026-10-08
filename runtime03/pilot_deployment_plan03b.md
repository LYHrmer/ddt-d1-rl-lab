# 03b 自训 pilot 的三次原生部署计划

此文件是事前计划，不是运行结果。子代理只写代码并做零物理检查；主代理统一放行、执行和计步。每次上限12秒仿真、6000个native步，最多三次、合计18000步；失败、提前跌倒、watchdog未触发均消耗本次名额，不能用重跑补齐成功。03b正式自训checkpoint与Python评测、实际C++ ORT1.10黄金检查通过后才由主代理决定释放试次。

冻结原生二进制SHA为 `62134e7c94fafaefcdee9f8325191b8f621cb3eb51f9ed2a167d68198469975d`；runtime03/schema_pilot03b.json SHA为 `a9eba4202468ef0eb2aa52dfde838b8efcb23d8ac25c3067064adc3aed941b55`。该计划没有改runtime源码、binary、训练环境、接口或配置。初态固定seed7001的fresh reset，含qpos/qvel/qacc_warmstart；同一文件传给所有试次，history/动作/FIFO由runtime reset。策略SHA在真实导出后从metadata读取并核验，不填虚构占位哈希。

| 次数 | 模式 | 12秒内的命令 |
| --- | --- | --- |
| 1 | replay，forward_stop | [0,2)零，[2,8)vx=.2，[8,12)零 |
| 2 | replay，turn_stop | [0,2)零，[2,5)wz=.2，[5,8)wz=−.2，[8,12)零 |
| 3 | live，真实ROS停发 | 依据实际truth的simtime：[0,2)每50ms发零，[2,8)每50ms发vx=.2；8秒之后完全停止发布，不主动发零；runtime依靠超过0.5秒无新命令的watchdog请求零速 |

replay必须传 `runtime03/sequences03b/forward_stop.json` 或 `turn_stop.json`。这些JSON与tools/evaluate_d1_ppo.py的完整时间段一致；旧runtime内建同名字符串的停令时刻不同，不用于这两次评测。reverse_stop.json同样提供给离线评分器和后续已授权入口，但本计划不增加第四次reverse部署。

`tools/evaluate_runtime03_trace.py`只读取report、native trace和counter文件。它验证身份、6000步上限、2/4/20ms实际计数/时刻和完整命令序列，每10个native的末帧对应一个Python评测样本，调用原evaluate_d1_ppo.py的trace_metrics函数体。样本的物理时间是区间终点，评分索引是区间起点，与Python保持一致。C++不记录reward，零占位只用于原函数的长度，不作为实测奖励。完整性不通过是INVALID；真实提前失败仍给缺失时段和固定失败惩罚，不靠提前结束降低J。

live使用ROS_DOMAIN_ID=84、ROS_LOCALHOST_ONLY=1，专用 `/d1_selftrained/cmd_vel` 和 `/d1_selftrained/truth`。在两个进程中先启动runtime03/live_watchdog03.py发布器，再启动live runtime；发布器等待实际truth并检查单一发布/订阅端点。它通过真实rclpy发送Twist，记录单调时间、每次发送、停止发送事件以及实际收到的truth。没有MuJoCo调用，但发令会控制独立原生仿真，所以启动前仍须计入主代理释放的那一次物理试次。`--help`不会创建ROS节点。

live的停车位移从**native trace中命令实际由非零变零的区间起点**算到12秒；该起点的位置取本区间积分之前的状态。末1秒水平速度均值须≤.02 m/s，绝对wz均值≤.03 rad/s，以上停车位移≤.05 m，且完整12秒无跌倒/非法承载。同时必须报告8秒停发到实际回零之间的仿真时长、累计路程和净位移，不能隐藏这约0.5秒的继续运动。它不套用固定序列“8秒立即停令”的J或门槛。

冻结runtime没有逐tick记录last-receive单调时间，因此真实试次报告的是“最后发出运动消息→观察到truth命令回零”的外部延迟，不能冒称直接测到内部callback age。内部0.5秒比较逻辑已有独立组件测试；实测另外证明真实ROS停发之后cmd归零并检查物理停车。runtime03/evaluate_watchdog03.py离线汇总两种证据。若策略在8秒前跌倒，应明确“watchdog未触发，试次已消耗”，不允许补跑冒充完整验证。

真实导出和原生黄金回执准备好后，主代理可运行零执行清单生成器：

```bash
rtk proxy /usr/bin/python3 -B /home/lyh/ddt_ros2_ws/runtime03/prepare_pilot_deployment03.py --export 实际导出目录 --onnx-golden-report 实际C++黄金检查report.json --output 全新计划目录
```

生成器验证原生binary/schema、ONNX和训练配置SHA、03b身份、64例ORT1.10黄金通过以及fresh初态，仅生成plan.json、commands.json、commands_for_review.txt，不执行命令。其中的shell命令均以rtk开头，已带独立输出目录、hash和6000步硬上限。主代理仍需逐项执行、合并已消耗账本；不得直接无条件整批重试。每份真实运行/评分回执都保留，不复写旧记录。
