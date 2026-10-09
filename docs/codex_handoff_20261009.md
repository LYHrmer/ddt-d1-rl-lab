# 2026-10-09 续接：DDT D1 自训与部署

03d已经完成训练、导出、数值核验和三条Python评测，资格 **1/3通过，整体不合格**。合同03的三次pilot至此结束，不再追加第四次或启动正式九次训练。目标仍是“稳定直行 → 可靠越障 → 提高速度”，本合同只验证平地低速纯前进/倒退/转向，不能声称地形通过、持续侧移或1.5m/s。先读本文、[03d最后一次pilot合同](career_execution_contract_03d_addendum.md)、[实际结果](career_execution_results_03.md)，再核对原始回执；不要因为换窗口重新启动训练。

## 工作区与操作边界

- 当前工作区：`/home/lyh/ddt_ros2_ws`。训练Python依赖在 `.python-deps`，使用MuJoCo3.3.0；系统默认MuJoCo版本不同，不能直接替换。C++为MuJoCo3.3.0/ORT1.10.0。
- `/home/lyh/wheel-legged-control-lab`、`/home/lyh/wheel-legged-control-lab-work`、合同02记录、77份冻结文件及 `my_course_drive_01` 保持只读。禁止重跑旧三个65k、24场G1、旧训练和物理批次；不要把当前DDT结果混入它们的统计。
- 用户要求每条shell命令带 `rtk`，需原始输出时用 `rtk proxy`。实际 Astra ultra 已审阅03d并在训练前放行；继续由主代理集成、统一计步和释放物理试次，可并行交付有界代码/测试，不虚称使用了Claude或某模型。
- 写本文时，DDT本地Git执行 `rtk proxy git remote -v` 输出为空；没有配置remote。不要把旧轮足仓库远端自动套到这个工作区。旧仓库若有已授权上传，遵照其已验证SSH443方式，不能随意打包同事权重或大证据。
- 本轮输入保持冻结，权威完整SHA清单在 `evidence/career_contract_03/pilot03d_release_01.json` 的 `inputs_sha256` 以及 `pilot03d_01/protocol.json`，最终回执确认输入未变。不要修改env、task、factory、train/export/eval、config、schema或二进制以改写本轮结果，不重建。

## 最终状态与停止边界

03b、03c各自从seed101随机初始化完成 **1,048,576 transitions / 10,485,760 native**，最终checkpoint导出、64例Torch/Python ORT/C++ ORT1.10及500+500 native配对均已完成。两者控制资格均 **0/3**。03b前进侧倒；03c三条均跑满12s、姿态改善，但停车位移及转向响应仍失败，C++转向复现失败。COMPLETE、数值一致或回报上涨都不等于控制通过。

**03d为第3次、最后一次pilot，已完成。** 从头随机初始化训练1,048,576 transitions / 10,485,760 native，训练耗时794.6727s，最小可用RAM为4,962,959,360 B，输入身份未变。只改训练命令调度，保证每完整episode两次运动→零及成对正负命令；奖励、控制接口和PPO均沿用03c。没有teacher、行为克隆、旧权重热启动、真值线速度actor输入或PID补位。

最终训练证据在 `evidence/career_contract_03/pilot03d_01/result.json`，同目录保留 `protocol.json`、`live_status.json`、`ledger.jsonl`，外层回执为 `pilot03d_01.watchdog.json`。旧live快照不能覆盖最终COMPLETE回执。接手仍应核对宿主是否有遗留进程；隔离PID命名空间里空的 `/proc` 或 `ps` 不能证明宿主已退出，更不能据此重复发起训练。

最终checkpoint SHA为 `d0bcaa779f1687531dfaad2942f758f885dccf686a9eec06f1903a7eca498ef8`；ONNX SHA为 `cf62bb43a53019d85b0dec00258dd261f217516f4a157a7e3826d21deffad48e`。64例实际C++ ORT1.10黄金检查PASS，最大动作误差4.7684e-7；Python500+C++500 native闭环配对PASS、issued action误差0。这些证明数值一致性，不能替代下述控制结果。

Python三条均完成12s、各6,000 native，无跌倒/非法支撑；只有reverse_stop全门通过。forward_stop停车位移0.080584m超过0.05m；turn_stop稳态yaw MAE为0.102882rad/s、停车位移0.060764m，超过0.08rad/s与0.05m。因此总计 **1/3通过**，未达到三条全部通过的资格门。停止本合同开发，**禁止第四次pilot、换seed/挑中间checkpoint、延长训练或以“正式实验”名义绕过失败**；正式九次训练未放行、未执行。

**03d独立C++部署：** 已完成2次、合计12,000 native。`pilot03d_deployment_plan_01/final_trial01_reverse_stop/` 跑满12s/6,000步，`final_score01_reverse_stop/report.json` 为PASS，停车位移0.016340523m；`final_trial02_forward_stop/` 跑满12s/6,000步，`final_score02_forward_stop/report.json` 为FAIL，停车位移0.080583646m。这两次结果与Python对应任务一致，前进失败不被倒退单项通过抵消。

**最后live槽已关闭：WATCHDOG NOT_VALIDATED_STARTUP_ERROR。** Astra批准的是一次12s、最多6,000 native工程检查，登记序列为0–2s零令、2–8s vx=−0.1m/s、8s后停止发布，由原生0.5s watchdog请求零速。但publisher在导入 `rclpy` 时因 `ModuleNotFoundError: rpyutils` 退出，主程序在原生启动前中止；只有publisher进程被创建，没有ROS节点、消息发布或新物理步。原始回执 `pilot03d_deployment_plan_01/live_execution_01.json` 为 `INTERRUPTED_OR_FAILED`，详见同目录 `live_publisher_stderr_01.txt`。新增0 native，整个开发期C++部署最终为20,980。该结果不是物理FAIL，没有真实watchdog或停车分数；本合同不补跑。配对中的C++500步计入预检池，不混作独立部署。

**下一合同的环境修复线索（本次只做了导入诊断）：** `rpyutils` 已安装于 `/opt/ros/humble/lib/python3.10/site-packages`；失败命令只设置 `/opt/ros/humble/local/lib/python3.10/dist-packages`，遗漏了前者。完整加载 `/opt/ros/humble/setup.bash` 后，用系统Python3 `-B` 导入 `rpyutils`、`rclpy`、`geometry_msgs.msg`、`std_msgs.msg` 全部通过。诊断没有调用 `rclpy.init`、创建节点、发布消息或启动原生程序，物理步为0。下一合同应先加载完整ROS环境并冻结经过导入预检的启动入口，再决定是否登记新的live试次；不修改本轮冻结脚本、命令或失败回执，不以导入成功冒充watchdog验收。

## 独立预算池

1 transition = 1策略周期 = 10 native；静态 `mj_forward` 不是积分，任何额外 `mj_step` 都须计数。各池不互相挪用，历史已耗预算不抹掉。

| 池 | 上限与当前已知计数 |
| --- | --- |
| 预检/工程物理 | 总上限655,360 native；03d的Python500+C++500配对已消耗1,000，累计实用362,110。剩余额度不授权新实验。 |
| 开发训练 | 3次均已完成，每次1,048,576 transitions/10,485,760 native；合计3,145,728 transitions/31,457,280 native。合同开发训练已结束，不能启动第4次。03d为B、seed101、8env，满足90min和可用RAM≥3GiB限制。 |
| Python pilot资格 | 每版本原三条12s、最多18,000 native，seed7001、delay0。03b实用14,980，03c18,000，03d18,000。三轮结果分别0/3、0/3、1/3。 |
| C++ pilot部署 | 每版本最多3次、每次≤30s/15,000 native，版本总≤45,000，开发三版本总≤135,000。03b实际1次/2,980；03c实际1次/6,000；03d物理试次2次/12,000，累计20,980。最后live尝试在publisher导入阶段失败，原生未启动、新增0；该槽关闭且不补跑。 |
| 正式B/R/M×3seed | 未放行、未执行；03d整体资格失败，按合同不启动正式九次。原合同候选预算不是执行许可，失败pilot不是正式3seed统计。 |
| 正式独立评测 | 尚未执行；原合同候选9模型×36条12s=1,944,000 native，另设reset额外步上限100,000；不是可随意使用的开发余额。 |

03d当前放行数值以[release回执](../evidence/career_contract_03/pilot03d_release_01.json)为准；完整停止规则及正式预算条件见[合同03](career_execution_contract_03.md)。

## 已交付及入口

- **地形与教程：** `run_d1_training_course.sh`、`tools/course03_layout.json`、`scene_training_course_03.xml`；12项静态验收和地图/渲染在 `evidence/terrain_course_03/`，五章教程从 [00_复现教程](tutorial/00_复现教程.md) 开始。场地资产存在不等于自训策略通关；原桥现成策略展示与本轮纯自训评测分开。
- **训练/导出/评测：** `run_training03.sh` 固定依赖路径及线程数；实际脚本为 `tools/train_d1_ppo.py`、`export_d1_ppo.py`、`evaluate_d1_ppo.py`。三个pilot均已完成，这些路径用于复核源码/状态，不重复训练、导出或已有物理批次。
- **部署：** [runtime03/README.md](../runtime03/README.md)、`tools/run_selftrained03.py`、`runtime03/prepare_pilot_deployment03.py`。按metadata明确的03b/c/d版本选schema，输出目录必须不存在。资格命令使用 `runtime03/sequences03b/*.json` 与Python同一12s窗口；原生二进制内置短序列不能替代资格试验。`--print-command`和`--validate-only`均不证明策略通过。
- **运行器与对照：** `runtime03/build/d1_runtime03` 已冻结，无需重建；日志逐native记录实际力矩/动作/位姿/计数。`runtime03/compare_dynamic_traces.py` 和 `tools/evaluate_runtime03_trace.py` 只做离线对照/评分。观测为57维当前帧与10帧历史，输出16维动作；腿倍率.5、轮5，2ms物理/4ms PD/20ms策略；轮目标量级约.435m/s，不能声称1.5m/s。ROS专用命令topic `/d1_selftrained/cmd_vel`，最后live尝试因启动环境失败，0.5s停发回退的真实停车仍未验收。
- **清理已完成，不再执行旧白名单：** 本轮删除1,057个再生文件，回收48.34MiB，1,642文件SHA及228软链接保持；历史两轮1.806GiB另计。详见[存储清理说明](storage_cleanup_03.md)和 `evidence/cleanup_regenerable03_execution_01.json`。这不是释放RAM；152个编译对象下次构建需重编，代价未测，不为恢复它们重建冻结二进制。
- **源码包：** `tools/make_source_bundle03.py` 是显式清单，省略 `--output` 仅dry-run。包不含原同事完整模型/mesh/权重及大型实验记录，须另取得原 `workspace2.zip` 并核SHA。未自行新增Git remote或发布。

## 优先证据与身份

训练/最终结果：`evidence/career_contract_03/pilot03{b,c,d}_01/`；三轮固定资格为对应 `pilot03{b,c,d}_python_eval_01/summary.json`，逐条指标见 `episodes.jsonl`；03c汇总 `pilot03c_outcome_01.json`。历史结果只能追加说明，不能改写。

03d事前与零积分证据：`pilot03d_release_01.json`、`task03d_static_02/report.json`、`deployment03d_adapter_01/report.json`、`deployment03d_static_01/{report,parity_report}.json`。真实导出及后续证据：`pilot03d_export_01/metadata.json`、`pilot03d_cxx_golden_01/report.json`、`pilot03d_onnx_pair_comparison_01/report.json`、`pilot03d_python_eval_01/summary.json`；独立C++两次评分为 `pilot03d_deployment_plan_01/final_score0{1_reverse_stop,2_forward_stop}/report.json`。最后live的事前身份为同目录 `live_release_01.json`，启动失败回执为 `live_execution_01.json`；没有原生运行或watchdog评分产物。静态初态、ONNX数值和跨语言一致性各自通过，整体控制资格仍FAIL。

关键SHA（完整源清单仍以release/protocol为准）：

| 产物 | SHA-256 |
| --- | --- |
| `training/configs/pilot03d.json` | `9cd6546e8a8df50c7d578a664003681fc79ddca7fa0a27b80158f0f5b6842bea` |
| `training/d1_rl/task03d.py` | `1780b223ed141eecced07f69d01a6ae5802868093258ff76b0b47383ac8a9e5d` |
| `runtime03/schema03d.json` | `64473031cbf027c950469a3db5a6e39b868f244da2b3f244fa174ab503f64c1c` |
| `runtime03/build/d1_runtime03` | `62134e7c94fafaefcdee9f8325191b8f621cb3eb51f9ed2a167d68198469975d` |

尚未完成：正式B/R/M×3seed、公平延迟改进结论、真实ROS停发后的停车资格、自训策略丰富地形/真实15mm台阶通过、持续A/D侧移、高速及实机验证。没有物理自救实现；复位不是自救。
