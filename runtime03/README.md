# D1 自训策略运行器 03

该目录独立于原 ROS2 bridge/FSM。它只面向合同03的纯 D1 自训策略；不加载旧 `flat_lab` 作为改过 action clip 的对照。轮速度倍率5、动作夹到±1，任务范围仅限±.3 m/s；这不是已验证的跟踪能力，不能声称1.5 m/s。倍率、PD参数与80/12 Nm限制从显式schema读取。旧 `schema_pilot03.json` 保留腿倍率.25的预检身份；03b使用独立schema、腿倍率.5，不复写旧文件。

构建不安装或覆盖旧二进制：

```bash
rtk proxy bash /home/lyh/ddt_ros2_ws/tools/build_runtime03.sh
```

可执行文件在 `runtime03/build/d1_runtime03`。运行时需 `LD_LIBRARY_PATH=/opt/ros/humble/lib:/home/lyh/ddt_ros2_ws/install/mujoco/lib`，也可在已加载原 ROS 环境的终端启动。

零物理检查入口如下，每个输出目录必须不存在：

```bash
rtk proxy env LD_LIBRARY_PATH=/opt/ros/humble/lib:/home/lyh/ddt_ros2_ws/install/mujoco/lib /home/lyh/ddt_ros2_ws/runtime03/build/d1_runtime03 --self-test --output /home/lyh/ddt_ros2_ws/runtime03/self_test_next
rtk proxy python3 -B /home/lyh/ddt_ros2_ws/runtime03/compare_python_interface.py --output /home/lyh/ddt_ros2_ws/runtime03/parity_next
```

`--offline-golden JSON` 使用新运行器的真实纯接口，无模型加载、ROS节点或积分。输入为 `{"delay_ticks":0,"cases":[...]}`。每个 case 给 `op=reset|observe|action`，`gyro[3]`、`quat_wxyz[4]`、`command[3]`、`q[16]`、`dq[16]`，action操作另给 `action[16]`。输出含完整 current/history、滤波、issued/executed动作、最终PD力矩和饱和mask。`offline_parity_01/report.json` 的256个案例按绝对误差≤1e-5比较独立Python实现；当前实测各字段最大差0。

03b使用 `--schema /home/lyh/ddt_ros2_ws/runtime03/schema_pilot03b.json`；SHA256为 `a9eba4202468ef0eb2aa52dfde838b8efcb23d8ac25c3067064adc3aed941b55`。`offline_parity03b_01/report.json` 记录新倍率下256组误差全0；`static_reset03b_01/parity_report.json` 记录显式初态含qacc_warmstart对照全0。这些检查均为0物理步。新schema同时引用旧真实FSM接口审计、新03b训练接口和新训练config的独立SHA。

`--onnx-golden 导出/golden_fixtures.json --policy 导出/policy.onnx --expected-policy-sha256 导出metadata里的SHA --output 新目录` 用实际 **ORT1.10.0** 检查导出器64个fixture，比较 `torch_actions`，判据绝对误差≤1e-5。输入必须为float `obs[1,57]`、`history[1,10,57]`，输出float `actions[1,16]`。这个模式不加载MuJoCo模型，也不启动ROS或积分。

v1预检checkpoint已实际通过这个C++入口：`evidence/career_contract_03/cxx_preflight_export_golden_02/report.json`，64/64通过、最大绝对差6.70552e-08、物理步0。第一次 `_01` 在读取ONNX dtype时失败，原因是ORT1.10临时TypeInfo生命周期；原失败回执保留，代码改为持有owner后才重试。该结果仅证明这份checkpoint的导出数值一致，不证明站立或行驶通过。

实际仿真入口需要 `--model XML --policy ONNX --expected-model-sha256 SHA --expected-policy-sha256 SHA --expected-schema-sha256 SHA --output 新目录`。`--schema` 默认为本目录的schema。运行时同时核验schema引用的原机器人XML、独立接口审计schema、训练config，记录所用身份。主代理统一放行试次、冻结expected SHA并计入预算；本目录的编译或单测结果不代表物理试次获放行或已通过。

| 参数 | 行为 |
| --- | --- |
| `--mode replay` | 默认，按仿真tick内部发令，不启动ROS节点，不受墙钟回调改变 |
| `--mode live` | 实时限速，订阅 `/d1_selftrained/cmd_vel`，vx/wz限制±.3、vy=0；超过.5s单调时间未收令自动请求零速 |
| `--sequence forward_stop` | 默认：0–2s零，2–6s vx=.2，6s起零 |
| `--sequence reverse_stop` | 同上，移动段vx=−.2 |
| `--sequence turn_stop` | 0–2s零，2–5s wz=.2，5–6s零，6–9s wz=−.2，9s起零 |
| `--sequence JSON` | `{"segments":[{"start":0,"end":2,"command":[0,0,0]},...]}`，连续覆盖试次，限定低速合同 |
| `--seconds` | >0、≤30，默认12；按2ms整数步向下取整 |
| `--max-native` | 1–15000；与seconds对应上限取小者；任何失败步仍记录尝试次数 |
| `--delay-ticks` | 0–3个20ms策略tick，先clip再进FIFO，reset队列填delay个零动作 |
| `--seed` | 原生reset随机种子；C++ RNG和NumPy不同，不能仅按相同seed声称初态一致 |
| `--initial-state JSON` | 使用显式qpos[23]/qvel[22]做fresh reset，可接收D1Env.get_initial_state()结果；可选qacc_warmstart[22]在最终静态forward后回填，用于匹配solver初态；不是任意中途状态恢复 |
| `--actions JSON` | replay模式下替代policy，格式 `{"actions":[[16个数],...]}`；每行一个20ms动作，仍clip；行用尽立即停止 |
| `--validate-only` | 验证身份、模型/策略shape、静态初态，只mj_forward，无mj_step/策略rollout/ROS节点 |

时基为2ms物理、4ms PD更新且两物理步保持力矩、20ms同步推理。首次推理直接使用reset帧；每个后续策略边界只更新一次gyro滤波和观察。历史在接受动作后追加本次观察，推理输入中不含当前帧。issued clipped action作为last_action，与延迟后的executed action分开记录。

为了与Python环境的数值流程一致，`mj_forward`只在几何/reset及策略观察边界调用；native日志直接从当前free qpos/qvel计算姿态和机体速度，不通过额外forward改动solver的warmstart。PD使用当前q/dq并显式夹80/12 Nm，写qfrc_applied；ctrl、freebase和其他广义力归零。fall/非法身体承载在策略边界结束该试次；数值NaN/Inf或MuJoCo警告立即停止。

每次运行保存 `initial_state.json`、`sequence.json`、逐native的 `trace.jsonl` 和 `report.json`。trace记录完整qpos/qvel、位置/姿态、body vx/wz、命令、issued/executed动作、夹紧前后力矩、饱和和实际policy/PD/native计数。报告记录运行二进制SHA和初始warmstart是否恢复。`native_step_counter.bin` 是本机字节序的两个uint64：尝试调用mj_step次数、已返回次数；每个调用前后都写入，末尾fsync。它不是仿真步数预测。

`compare_dynamic_traces.py --python-run Python试次目录 --cpp-run C++试次目录 --output 新目录` 只读取既有日志，无任何积分调用。它检查每个native的计数/时间、qpos、qvel以及Python的qfrc_applied[6:22]对C++实际torque，绝对阈值固定1e-5；检查初态文件SHA、model/config/schema身份和结束原因；逐步误差向量保存在 `native_differences.jsonl`。两边在同一步发生相同跌倒可以得到“轨迹对照通过”，这不代表该动作或策略具有稳定站立能力。

报告另记录推理warmup次数、P50/P95/P99/max、实际policy/PD仿真时刻、live墙钟迟到、watchdog过期次数。单次推理超过20ms时拒绝执行该新动作并停止仿真；异常输出/shape也拒绝继续。终止时清动作/队列；正常终止设置受限零动作PD，异常停止清施加力。没有额外“恢复积分”，这些停止逻辑的物理稳定性需要单独有界验证。

live发布 `/d1_selftrained/odometry`（nav_msgs/Odometry，world pose、body twist）及 `/d1_selftrained/truth`（JSON String）。没有启动旧 `/command/cmd_key`，没有改原viewer或添加自救动作。独立GUI/回放由主工程入口提供。

已验证与尚待验证要分开：编译、组件故障注入、Python/C++纯接口、静态reset和v1预检ONNX数值均已有0物理步回执；跨语言轨迹、03b自训checkpoint、watchdog停稳及实时性能，必须由对应实际运行回执支持，不能根据以上单测提前宣称完成。
