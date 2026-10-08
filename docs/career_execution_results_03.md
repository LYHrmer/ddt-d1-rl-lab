# 合同03实际结果与能力边界

更新：2026-10-09。以下为本机 MuJoCo 仿真，不含实机验证。场景建设、数值一致性、完整训练与控制资格分别报告。

## 已完成的工程交付

- 新增分级台阶、坡道、阶梯、起伏路、绕障、窄门和加速道；MuJoCo3.3.0的12项静态检查通过，场地地图和渲染已交付。未执行新策略地形通关评测。
- 重建纯D1的57维当前观测、10帧历史、16动作与P_V执行器契约；独立Python/C++离线审计和原生轨迹配对通过。
- 完成随机初始化PPO训练、最终checkpoint导出、实际C++ ORT1.10推理与MuJoCo独立部署。低速资格目前未通过。
- 完成五章教程，保留配置、输入身份、预算计步、训练曲线、失败轨迹和回放。

## 开发pilot结果

03b与03c改变了任务/奖励，下面是开发诊断对照，不是正式单因素消融或多seed统计。每轮均从seed101随机初始化训练1,048,576条经验，仅评最终checkpoint。

| 任务版本 | 训练秒数 | 原生步数 | Python固定任务通过 | 提前跌倒 |
|---|---:|---:|---:|---:|
| 03b | 1326.25 | 10,485,760 | 0/3 | 1 |
| 03c | 801.58 | 10,485,760 | 0/3 | 0 |

| 版本 / 固定任务 | 稳态vx MAE(m/s) | 稳态wz MAE(rad/s) | 末秒水平速度(m/s) | 停车位移(m) | roll/pitch峰值(rad) |
|---|---:|---:|---:|---:|---:|
| 03b / forward_stop | 0.02902 | 未完整测得 | 未完整测得 | 未完整测得 | 0.59362 |
| 03b / reverse_stop | 0.01050 | 未完整测得 | 0.01218 | 0.08701 | 0.13662 |
| 03b / turn_stop | 未完整测得 | 0.19638 | 0.01314 | 0.03707 | 0.14689 |
| 03c / forward_stop | 0.00412 | 未完整测得 | 0.02187 | 0.09652 | 0.05286 |
| 03c / reverse_stop | 0.02798 | 未完整测得 | 0.02671 | 0.06876 | 0.09940 |
| 03c / turn_stop | 未完整测得 | 0.11666 | 0.05998 | 0.14664 | 0.04994 |

03b前进在5.96秒侧倒，表中该任务的局部误差不能解释为完整稳定段通过。03c全部跑满12秒且姿态门通过，但所有停车位移仍超过0.05m，转向MAE仍超过0.08rad/s。状态COMPLETE只说明执行完整。

两轮均通过64例Torch/Python ORT/C++ ORT1.10以及500+500 native闭环配对。03c配对动作误差0，qpos误差约4.0e-15；它只证明部署一致，不能修正资格FAIL。03c独立12秒C++转向诊断复现Python失败，推理P99为0.020438ms。

## 可审计证据

- [03b训练](../evidence/career_contract_03/pilot03b_01/result.json) / [03b固定评测](../evidence/career_contract_03/pilot03b_python_eval_01/summary.json)
- [03c训练](../evidence/career_contract_03/pilot03c_01/result.json) / [03c固定评测](../evidence/career_contract_03/pilot03c_python_eval_01/summary.json)
- [03c独立C++评分](../evidence/career_contract_03/pilot03c_deployment_plan_01/score02_turn_stop/report.json)
- [03c闭环一致性](../evidence/career_contract_03/pilot03c_onnx_pair_comparison_01/report.json)
- [教程](tutorial/00_复现教程.md)

证据目录保持本地，源码ZIP不包含大型轨迹或权重；外部读者需取得独立证据包，不能只凭本文复算指标。

## 尚未完成

正式B/R/M×3seed、公平延迟改进/消融、ROS活命令超时后停稳资格、新策略丰富地形通过、持续侧移、真实15mm跨越和1.5m/s均未完成。合同开发期最多3次pilot，已使用2次；第三次须按03c失败轨迹另立具体方案，当前没有自动追加训练。
