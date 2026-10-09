# D1 自训策略运行器 03

该目录独立于原 ROS2 bridge/FSM。它只面向合同03的纯 D1 自训策略；不加载旧 `flat_lab` 作为改过 action clip 的对照。轮速度倍率5、动作夹到±1，任务范围仅限±.3 m/s；这不是已验证的跟踪能力，不能声称1.5 m/s。倍率、PD参数与80/12 Nm限制从显式schema读取。旧 `schema_pilot03.json` 保留腿倍率.25的预检身份；03b/03c/03d均使用腿倍率.5及同一 `d1-selftrained-pilot03b-v2` 控制接口，各自保留独立训练配置和runtime schema。

本次文档更新时（2026-10-09），03b/03c/03d均已完成训练，固定控制评测分别为 **0/3、0/3、1/3通过**。03d只有reverse_stop全门通过，forward_stop停车位移0.080584m、turn_stop的yaw MAE 0.102882rad/s与停车位移0.060764m仍超门槛，整体不合格。三次pilot开发结束，不追加第四次或启动正式九次训练。03d独立C++已完成倒退PASS和前进FAIL两次12s部署，各6,000 native，停车位移分别0.016340523m与0.080583646m。最后一次live尝试在publisher导入阶段退出，原生程序未启动、新增0 native；状态为 **WATCHDOG NOT_VALIDATED_STARTUP_ERROR**，不补跑。实际结果见[执行结果](../docs/career_execution_results_03.md)。

现有 `runtime03/build/d1_runtime03` 已冻结，启动前无需重建。以下命令只用于另行登记的重建；冻结实验期间不要执行。清理过的152个编译对象会在下次构建重新生成，耗时未测，详见[存储清理说明](../docs/storage_cleanup_03.md)。构建不安装或覆盖原 ROS2 bridge/FSM 二进制：

```bash
rtk proxy bash /home/lyh/ddt_ros2_ws/tools/build_runtime03.sh
```

可执行文件在 `runtime03/build/d1_runtime03`。运行时需 `LD_LIBRARY_PATH=/opt/ros/humble/lib:/home/lyh/ddt_ros2_ws/install/mujoco/lib`，也可在已加载原 ROS 环境的终端启动。

建议从 `tools/run_selftrained03.py` 进入。它读取导出目录的 `metadata.json`，按显式 `config_version` 选择下表中的配置，核验模型/接口/config/ONNX/fixture SHA，并自动设置原生程序的库路径。缺失版本、未知版本、配置交叉哈希会被拒绝，不按导出目录名猜测，也不自动把旧策略切换到03d。

| metadata `config_version` | 训练配置 | runtime schema |
| --- | --- | --- |
| `d1-pure-ppo-pilot-03b` | `training/configs/pilot03b.json` | `runtime03/schema_pilot03b.json` |
| `d1-pure-ppo-pilot-03c` | `training/configs/pilot03c.json` | `runtime03/schema03c.json` |
| `d1-pure-ppo-pilot-03d` | `training/configs/pilot03d.json` | `runtime03/schema03d.json` |

下面用已存在的03d导出演示零积分检查。输出路径须**事先不存在**，即使它是空目录也会拒绝；重做时换新编号，保留旧记录，不删除旧目录以绕过保护。`--print-command`不会创建输出目录或启动运行器，`--validate-only`会创建独占检查目录、加载模型并调用静态 `mj_forward`，不调用 `mj_step`。

```bash
rtk proxy python3 -B /home/lyh/ddt_ros2_ws/tools/run_selftrained03.py --export /home/lyh/ddt_ros2_ws/evidence/career_contract_03/pilot03d_export_01 --output /home/lyh/ddt_ros2_ws/results/inspect03d_new_01 --sequence forward_stop --print-command
rtk proxy python3 -B /home/lyh/ddt_ros2_ws/tools/run_selftrained03.py --export /home/lyh/ddt_ros2_ws/evidence/career_contract_03/pilot03d_export_01 --output /home/lyh/ddt_ros2_ws/results/validate03d_new_01 --sequence forward_stop --validate-only
```

03d导出目录的metadata会选中 `schema03d.json`；最终ONNX SHA为 `cf62bb43a53019d85b0dec00258dd261f217516f4a157a7e3826d21deffad48e`。上例不能证明策略已满足控制门槛；此处也未替用户执行这些命令。去掉检查参数会实际运行12s、最多6000 native，必须先由主代理放行并计入合同预算。准备工具 `prepare_pilot_deployment03.py` 只生成命令清单，也不代替放行。

资格评测使用 `runtime03/sequences03b/forward_stop.json`、`reverse_stop.json`、`turn_stop.json`，三者与Python固定评测共用12s窗口，03c/03d继续使用这些文件。目录名中的03b不表示切换策略版本。启动器的 `--sequence forward_stop|reverse_stop|turn_stop` 自动展开为这些JSON的绝对路径：前/倒进均为0–2s零、2–8s ±.2m/s、8–12s零；转向为0–2s零、2–5s +.2rad/s、5–8s −.2rad/s、8–12s零。直接调用原生二进制时应显式传入JSON路径。下表列出的原生内置同名短序列有不同时间窗口，只用于兼容/调试，不能当作这三条资格评测。

零物理检查入口如下，每个输出目录必须不存在：

```bash
rtk proxy env LD_LIBRARY_PATH=/opt/ros/humble/lib:/home/lyh/ddt_ros2_ws/install/mujoco/lib /home/lyh/ddt_ros2_ws/runtime03/build/d1_runtime03 --self-test --output /home/lyh/ddt_ros2_ws/runtime03/self_test_next
rtk proxy python3 -B /home/lyh/ddt_ros2_ws/runtime03/compare_python_interface.py --output /home/lyh/ddt_ros2_ws/runtime03/parity_next
```

`--offline-golden JSON` 使用新运行器的真实纯接口，无模型加载、ROS节点或积分。输入为 `{"delay_ticks":0,"cases":[...]}`。每个 case 给 `op=reset|observe|action`，`gyro[3]`、`quat_wxyz[4]`、`command[3]`、`q[16]`、`dq[16]`，action操作另给 `action[16]`。输出含完整 current/history、滤波、issued/executed动作、最终PD力矩和饱和mask。`offline_parity_01/report.json` 的256个案例按绝对误差≤1e-5比较独立Python实现；当前实测各字段最大差0。

03b使用 `--schema /home/lyh/ddt_ros2_ws/runtime03/schema_pilot03b.json`；SHA256为 `a9eba4202468ef0eb2aa52dfde838b8efcb23d8ac25c3067064adc3aed941b55`。`offline_parity03b_01/report.json` 记录新倍率下256组误差全0；`static_reset03b_01/parity_report.json` 记录显式初态含qacc_warmstart对照全0。这些检查均为0物理步。新schema同时引用旧真实FSM接口审计、新03b训练接口和新训练config的独立SHA。

`--onnx-golden 导出/golden_fixtures.json --policy 导出/policy.onnx --expected-policy-sha256 导出metadata里的SHA --output 新目录` 用实际 **ORT1.10.0** 检查导出器64个fixture，比较 `torch_actions`，判据绝对误差≤1e-5。输入必须为float `obs[1,57]`、`history[1,10,57]`，输出float `actions[1,16]`。这个模式不加载MuJoCo模型，也不启动ROS或积分。

v1预检checkpoint已实际通过这个C++入口：`evidence/career_contract_03/cxx_preflight_export_golden_02/report.json`，64/64通过、最大绝对差6.70552e-08、物理步0。第一次 `_01` 在读取ONNX dtype时失败，原因是ORT1.10临时TypeInfo生命周期；原失败回执保留，代码改为持有owner后才重试。该结果仅证明这份checkpoint的导出数值一致，不证明站立或行驶通过。

直接调用原生仿真入口时需要 `--schema 对应版本JSON --model XML --policy ONNX --expected-model-sha256 SHA --expected-policy-sha256 SHA --expected-schema-sha256 SHA --sequence 固定评测JSON --output 新目录`。原生程序不会依据metadata自动选版本，03b/03c/03d试次均应显式传schema；推荐使用上述启动器完成选择。运行时同时核验schema引用的原机器人XML、独立接口审计schema、训练config，记录所用身份。主代理统一放行试次、冻结expected SHA并计入预算；本目录的编译或单测结果不代表物理试次获放行或已通过。

| 参数 | 行为 |
| --- | --- |
| `--mode replay` | 默认，按仿真tick内部发令，不启动ROS节点，不受墙钟回调改变 |
| `--mode live` | 实时限速，订阅 `/d1_selftrained/cmd_vel`，vx/wz限制±.3、vy=0；超过.5s单调时间未收令自动请求零速 |
| 原生 `--sequence forward_stop` | 内置默认短序列：0–2s零，2–6s vx=.2，6s起零；非固定资格JSON |
| 原生 `--sequence reverse_stop` | 内置短序列：同上，移动段vx=−.2；非固定资格JSON |
| 原生 `--sequence turn_stop` | 内置短序列：0–2s零，2–5s wz=.2，5–6s零，6–9s wz=−.2，9s起零；非固定资格JSON |
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

`live_watchdog03.py` 的 `--drive-vx` 默认为+.2m/s，03d最后一个已批准工程槽登记使用 `--drive-vx -0.1`：0–2s零令、2–8s倒退、8s后不再发送任何命令（包括零令），由原生0.5s watchdog触发。`evaluate_watchdog03.py` 从publisher protocol读取速度，拒绝发送记录或原生指令方向不匹配。停车位移从原生实际非零→零指令边界计算，8s停发到实际回零之间的时间与路程另报；外部发送/接收时间戳不冒充内部精确回调年龄。评分限定 `scope=live_watchdog_engineering_only`、`qualifies_overall_pilot=false`。本次publisher因 `ModuleNotFoundError: rpyutils` 在导入 `rclpy` 时退出，没有创建ROS节点、发布消息或启动原生程序，故没有真实停发或停车评分。`live_execution_01.json` 原始状态为 `INTERRUPTED_OR_FAILED`，工程结论记为 **WATCHDOG NOT_VALIDATED_STARTUP_ERROR**，不能记作物理FAIL，也不能用静态或synthetic测试替代。

启动失败来自不完整的Python路径：原命令只设 `local/lib/python3.10/dist-packages`，但已安装的 `rpyutils` 位于 `/opt/ros/humble/lib/python3.10/site-packages`。零物理诊断中，完整加载 `/opt/ros/humble/setup.bash` 后 `rpyutils`、`rclpy`、`geometry_msgs.msg`、`std_msgs.msg` 四项导入通过，没有创建节点或发布消息。下一合同应从完整ROS环境开始、先做导入预检；本轮不修改冻结命令或补跑live。三轮独立C++部署最终累计20,980 native。

验证结论须对应具体产物和试次。03d训练完成1,048,576 transitions/10,485,760 native、794.6727s，输入身份保持；64例C++ ORT1.10黄金核验最大误差4.7684e-7，Python500+C++500 native闭环配对通过，后者计入预检池（累计362,110）。三条Python资格消耗18,000 native、无跌倒，但仅1/3通过。数值一致不能替代控制资格；独立C++部署和ROS watchdog停车也不能用静态或配对结果代替。所有证据入口见[续接状态](../docs/codex_handoff_20261009.md)。
