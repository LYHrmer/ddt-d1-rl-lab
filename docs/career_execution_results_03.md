# 合同03实际结果与能力边界

更新：2026-10-09。本机MuJoCo仿真，未进行实机验证。

## 交付结论

丰富地形、随机初始化PPO训练、ONNX导出、独立C++运行器、固定评测、五章教程及可再生文件清理已完成。三次有界开发训练均执行完整，最终03d只在三条开发任务中通过一条；整套低速控制资格未通过，正式三臂×三seed未启动。

## 开发结果

03b→03c改变任务/奖励，03c→03d只改变命令调度。下面是同一开发seed的诊断结果，不是正式独立多seed改进证明。每次从随机初始化训练1,048,576条经验，仅评最终checkpoint。

| 版本 | 训练秒数 | 原生积分步 | 固定任务通过 | 提前跌倒 |
|---|---:|---:|---:|---:|
| 03b | 1326.25 | 10,485,760 | 0/3 | 1 |
| 03c | 801.58 | 10,485,760 | 0/3 | 0 |
| 03d | 794.67 | 10,485,760 | 1/3 | 0 |

| 版本 / 任务 | vx MAE(m/s) | wz MAE(rad/s) | 末秒水平速度(m/s) | 停车位移(m) | roll/pitch峰值(rad) | 资格 |
|---|---:|---:|---:|---:|---:|---|
| 03b / forward_stop | 0.02902 | 不适用 | 未完整测得 | 未完整测得 | 0.59362 | FAIL |
| 03b / reverse_stop | 0.01050 | 不适用 | 0.01218 | 0.08701 | 0.13662 | FAIL |
| 03b / turn_stop | 不适用 | 0.19638 | 0.01314 | 0.03707 | 0.14689 | FAIL |
| 03c / forward_stop | 0.00412 | 不适用 | 0.02187 | 0.09652 | 0.05286 | FAIL |
| 03c / reverse_stop | 0.02798 | 不适用 | 0.02671 | 0.06876 | 0.09940 | FAIL |
| 03c / turn_stop | 不适用 | 0.11666 | 0.05998 | 0.14664 | 0.04994 | FAIL |
| 03d / forward_stop | 0.00317 | 不适用 | 0.00451 | 0.08058 | 0.05298 | FAIL |
| 03d / reverse_stop | 0.03992 | 不适用 | 0.00225 | 0.01634 | 0.05878 | PASS |
| 03d / turn_stop | 不适用 | 0.10288 | 0.01494 | 0.06076 | 0.03872 | FAIL |

03b前进在5.96秒侧倒，该任务局部稳定段误差不能当完整窗口结果。03c/03d各三条均跑满12秒且无跌倒。03d后退停车通过原全部门槛；前进唯一未过的是停车位移0.08058m>0.05m；转向wz MAE取正负两方向较大值（正向0.102882、负向0.046317rad/s），仍有0.10288>0.08rad/s及停车位移0.06076m>0.05m。不能据1/3或单项改善宣称整体可靠。

## 工程验证与计步

- 分级台阶、坡道、阶梯、起伏路、绕障、窄门、加速道已经生成；MuJoCo3.3.0的12项静态检查通过，地图和渲染已交付。策略地形通过待测。
- 57维当前观测、10帧历史、16动作、P_V/PD/力矩限幅及时基已独立审计。三版均通过64例Torch/Python ORT/实际C++ ORT1.10比对及500+500native闭环配对。03d动作误差0，qpos误差约1.55e-15。
- 03c独立12秒C++转向诊断复现Python失败，推理P99为0.020438ms。03d独立C++倒退任务PASS（停车1.63cm，推理P99 0.023405ms），前进任务FAIL（停车8.06cm，P99 0.030430ms），各6000native/12秒。部署一致性不等于控制资格。
- 最后一个ROS2 watchdog槽在发布器导入rclpy时因`rpyutils`搜索路径缺失退出，native运行器尚未启动，新增0native。回执是启动失败，命令超时后的物理停稳未验证；按合同不补跑。
- 预检/工程诊断池实际362,110/655,360 native；三次训练合计31,457,280 native/3,145,728 transitions。Python开发评测分别14,980、18,000、18,000 native，总50,980。独立C++部署实际20,980 native（03b 2,980；03c 6,000；03d 12,000；live 0）。闭环500+500计入预检池；[最终总账](../evidence/career_contract_03/final_delivery_01.json)各池分别记录，无重复计数。
- 本轮删除1057个可重建文件回收50,688,000B（48.34MiB），保护源/模型/权重/库/证据和安装链接不变。[清理回执说明](storage_cleanup_03.md)。

## 证据入口

| 版本 | 完整训练 | 固定Python评测 |
|---|---|---|
| 03b | [result](../evidence/career_contract_03/pilot03b_01/result.json) | [summary](../evidence/career_contract_03/pilot03b_python_eval_01/summary.json) |
| 03c | [result](../evidence/career_contract_03/pilot03c_01/result.json) | [summary](../evidence/career_contract_03/pilot03c_python_eval_01/summary.json) |
| 03d | [result](../evidence/career_contract_03/pilot03d_01/result.json) | [summary](../evidence/career_contract_03/pilot03d_python_eval_01/summary.json) |

- [03d实际部署完成回执](../evidence/career_contract_03/pilot03d_deployment_plan_01/completion_01.json)：实际两条replay和一次启动失败，与原规划区分。
- [03d倒退C++评分](../evidence/career_contract_03/pilot03d_deployment_plan_01/final_score01_reverse_stop/report.json) / [成功任务回放](../evidence/career_contract_03/pilot03d_reverse_video_01/runtime_trace.mp4)
- [03d前进C++评分](../evidence/career_contract_03/pilot03d_deployment_plan_01/final_score02_forward_stop/report.json) / [失败任务回放](../evidence/career_contract_03/pilot03d_forward_video_02/runtime_trace.mp4)
- [ROS2启动失败回执](../evidence/career_contract_03/pilot03d_deployment_plan_01/live_execution_01.json)：未启动物理仿真，不代表watchdog通过。
- [03d闭环一致性](../evidence/career_contract_03/pilot03d_onnx_pair_comparison_01/report.json)
- [03c独立C++评分](../evidence/career_contract_03/pilot03c_deployment_plan_01/score02_turn_stop/report.json)
- [03c失败资格回放](../evidence/career_contract_03/pilot03c_turn_video_01/runtime_trace.mp4)：已有轨迹的离线渲染，HUD标明FAIL，0新增积分。
- [教程](tutorial/00_复现教程.md) / [续接文档](codex_handoff_20261009.md)

证据和权重保留本地。源码ZIP不包含原同事模型/权重及大型轨迹，复算还需原ZIP和相应证据。

## 未完成与停止条件

本合同最多三次pilot，现已全部使用；不启动第四次、补步或把失败模型标为通过。正式B/R/M×3seed、公平延迟改进/消融、ROS命令超时后的物理停稳、新策略丰富地形通过、持续侧移、真实15mm跨越和1.5m/s均未完成。下一阶段需要新的证据假设与有界合同；当前结果只支持局部改善和已实现的工程链路。
