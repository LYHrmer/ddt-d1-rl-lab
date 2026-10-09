# 进度

- 2026-10-08：读取权威 README 和学习路线图；确认当前依赖状态和 sudo 交互要求。尚未安装新系统包、编译或启动仿真。

- 已启动 MuJoCo 源码构建，关闭其示例/测试/独立simulate目标；ROS桥接包含自己的原生simulate界面，后续另行编译。
- 并行只读任务：旧Git传统控制基线、RL启动/FSM、录屏场景资源对照。

- MuJoCo 源码构建成功（2分3秒），ctypes 调用 mj_versionString() 实测为3.3.0。
- D1/D1H描述包已编译；补上D1对D1H的实际xacro运行依赖，xacro展开实测成功。ros_utils完成，keyboard_controller正在构建。
- sudo安装仍待用户本机验证；先准备下载包头文件用于用户目录编译，不视为系统安装已完成。

- 用户已完成本机sudo安装；/usr/lib/libonnxruntime.so.1.10.0与官方下载逐字节SHA一致，GLFW开发头存在。
- RL控制器构建通过；其编译使用同一官方1.10头/库；无训练。MuJoCo桥接主程序与控制插件已完成，等待桥接元包结束。
- 用户确认不存在额外场地文件；保留stock场景并新增可选视频近似场地。历史与源包均未含完整录屏场地。

- 全部编译成功；原生 MuJoCo viewer 实际启动，两个 ONNX 已加载，d1_rl_controller active。
- stock 实际运行 transform_up 10 秒、rl_0 零速 5 秒、0.2 m/s 请求 3 秒、零速 5 秒：完成，23.515 秒，IMU/关节各 5841 条，RL 阶段 roll/pitch 绝对值小于 0.0195 rad。FSM 日志确认进入 rl_flat；不是仅加载权重。证据 evidence/rl_smoke_stock_01.json、stock_rl_stand.png、logs/stock_launch_01.log。没有世界位移证据，未认证速度跟踪或停车精度。
- 可选估计场地生成/静态安装验证成功，待实际加载。Git 保全压缩已回收 276447232 字节，16076 对象与 refs/reflogs/status 全保留。

- 近似场地实际 MuJoCo 3.3 mj_loadXML 与原生 viewer 均加载成功。新增桥插件世界 base_pose/frame_vel 只读发布，25Hz，保持机器人和ONNX字节不变。
- 完成 flat 原配真值对照 46.325 秒，发现零速阶段约 -0.066 m/s 持续后退、0.2 请求仅实测约0.106 m/s，侧移不足；未将站姿稳定误报为良好运动能力。
- 完整重启后完成 flat_lab 原配同序列对照 45.583 秒：0.2 请求末段约0.185m/s，观测前进0.540m；转向±0.2rad/s、各3s，yaw变化+0.628/-0.593rad；最终零速末段约-0.0007m/s。推荐策略改为用户按1。侧移未通过；未继续追加训练或越障/高速实验。
- 磁盘清理累计回收1082970112字节（1.01GiB）。77冻结哈希、旧结果、refs/reflogs/status均保留。用户操作见README_LOCAL.md。
- 键盘终端实际启动、r零速、Ctrl+C退出通过。交付时保留当前 MuJoCo viewer（PID267665、logs/video_course_launch_flat_lab_01.log），rl_flat_lab已运行、最后命令零速；自动测试和录屏进程均已结束，未留下键盘节点。进程映射实证使用本工作空间MuJoCo3.3.0与/usr/lib/ONNX Runtime1.10.0。

- 2026-10-09：第二轮清理完成。127个单链接MJB只做零字节区稀疏存储，内容SHA/路径/inode/长度/mtime不变；306个可再生且未跟踪的源码pyc清除。新增释放855945216字节，两轮累计1938915328字节（1.806GiB）。未删除旧模型、实验记录、完整归档或构建依赖。
- 实际gpt-6-astra/ultra制定合同02；旧viewer已关闭。新增plane/15mm单台阶场景、零积分几何预检、默认关闭的逐native接触记录与硬步上限、独立评测和在线守护。编译通过，6项评测合成测试与4项守护合成测试通过；证据模式退出改为先停线程/关闭记录器再销毁模型。
- 合同02唯一plane_01执行：15164 native步，7581实际控制更新，35.253978秒墙钟，30.328秒仿真。序列完成，资格FAIL，唯一失败是世界横向偏移0.124465m超过0.10m。实测稳态机体vx=0.184520m/s，停车末段绝对vx=.000906m/s；四轮承载与其他资格通过。按照合同没有启动step_01/02，没有重跑或放宽阈值。
- 离线分析确认：初始yaw3.095°，末yaw3.946°；固定初始航向的横向末点误差25.625mm。世界124.465mm偏移可分为初始航向前进投影98.887mm、后续航向变化投影23.645mm、机体侧向投影1.933mm。这是运动学分解，不是动力学因果证明；原FAIL不变。
- 阶段标记存在已知限制：pre-drive检查耗时约.10138s，drive起点仍用了检查前位姿stamp，故冻结评测区间10.12仿真秒、实际命令约10.003墙钟秒。保留原证据和判定，下一合同单独修复标记/命令生效对齐；不把当前区间用于精确瞬态比较。
- 本轮交付前重新核验：旧77保护文件和protocol、HEAD/refs/status均不变，本次17项输入hash不变；native完整footer，当前无mujoco_sim进程。回执evidence/step_contract_02_delivery.json。没有启动新的训练、台阶、侧移或高速物理试次。
- 用户询问求职优化，已创建CAREER_ROADMAP.md并经实际Astra ultra审阅：先做零训练接口审计和固定输入一致性检查，再完成自己的单seed训练—ONNX—C++部署闭环，最后选一个可检验改进做等预算至少3seed对照/消融。现成权重继续明确标为同事来源，1.5m/s仍是未验证目标。

## 2026-10-09 合同03预检与导出
- 4env完整16384策略步/163840 native，536.14 transitions/s。8env首次因3GiB RAM保留线退出，0物理；共享导入后8env完整16384策略步/163840 native，628.29 transitions/s，USS峰值3,009,773,568B、可用RAM最低3,410,489,344B。两个完成回执inputs_unchanged=true。
- 当前预检累计327680 native（上限655360）；sandbox socket失败与RAM失败保留。
- preflight checkpoint导出opset11/IR7，Python和实际C++ ORT1.10的64个fixture均通过，max_abs_error=6.70552e-8，0物理。初次C++ dtype生命周期错误修复前失败目录保留。
- 短训练episode平均不足1s，尚未形成站立/跟踪能力。Astra静态支撑分析确认默认站姿下腿倍率.25使前calf所需动作1.212；较低姿态可行，不能称模型完全不能学。新03b腿倍率.5、轮5正在另建schema，原配置保留；先诊断再pilot。

## 03b物理接口放行与首个pilot
- 三条诊断共1630 native：D0零动作430步后terminal_observation ndarray日志序列化失败，原native430行和policy42行保留，不补跑；D1/D2各600步完成固定1.2秒，无跌倒/非法接触/饱和。静力补偿只作诊断，训练仍随机初始化纯PPO。
- 三初态/延迟0、1、2tick，Python与Cpp各200native，共1200步，逐native qpos/qvel/tau误差最大<1.72e-13；三组PASS。累计预检330510/655360 native。
- Astra审核放行03b pilot_01：seed101，B，8env，上限1048576 transitions/10485760native/90分钟，外部deadline5440秒；进程运行中。新learned ONNX的500×2闭环配对留到导出后，从预检池扣除。
- Astra只读核验SB3实际raw动作/logprob、clip执行、TimeLimit bootstrap、GAE跨reset及奖励0.02缩放，未发现阻断性错误。worker base_seed实际由SB3置为101+rank，再与rank/episode/stream形成SeedSequence。最终_n_updates须从checkpoint读取，CSV损失滞后一轮。

## 有界工程剖析
- 两次独立固定支撑profile各640policy/6400native，共12800；预检总343310/655360 native。普通热段576步2.06455s=278.995eps，加载/reset分列。
- cProfile640步2.525s：finite .839s（warning遍历.708）、contact .678s、PD .165s、mj_step .323s。已定位包装层遍历占主导，未改物理。
- Astra同意独立FastChecks subclass，只覆盖有限/警告与接触批量检查；原env/train/config仍冻结。已分配但尚未消耗的动态配对3000+测速6400，上限≤1e-5且≥1.25倍单env增益才集成。
- 本机ddt_ros2_ws已git init main，无remote；尚未git add/commit，不包含同事资产发布。源码allowlist与12项原ZIP文本修复补丁已生成并在临时目录验证，未触碰当前原始资产。

## Pilot03b_01最终结果
- COMPLETE：1048576 transitions、10485760 native、5242880 PD；1326.25s训练、790.63eps，全部冻结输入不变。checkpoint实际_n_updates=1024 epochs。
- 新ONNX SHA6425c19b7ccc569a8829627988e9a602b37e94ab50ac18024a40811dd2094466，Torch/Python/C++ ORT1.10的64例最大差4.1723e-7；500native×两端闭环通过，动作误差0，qvel最大约1.73e-12。
- Python开发评测共14980native，0/3门槛通过：forward5.96s倾角终止；reverse完整，vxMAE .01050但停车位移.08701m>.05；turn完整但wzMAE .19638>.08。独立C++部署slot1复现forward2980native FALL，P99推理.021659ms；仅诊断失败，不启动另外两个试次掩盖失败。
- FastChecks动态3组全状态/观测/奖励/指标0误差。新无并发baseline352.586eps，fast952.543eps，单环境2.70159倍；旧并发baseline比值3.414不用于净增益。工程实际28600native。当前预检总360110/655360（含learned-ONNX1000），pilot训练/评测/C++池单独记账。
- 当前无在跑训练；失败视频渲染使用既有trace，只mj_forward，不增加物理积分。Astra正在根据NPZ与C++全状态诊断制定第二轮开发修订；正式9run仍未放行。

## 03c pilot 2 登记

Astra ultra 复核03c复合任务修订。factory已接线，零积分构造/reset/语法及版本选择检查通过。下一次训练仅B/seed101/8env/1,048,576 transitions，90min上限；源SHA见pilot03c_release_01.json。预检池现360,110/655,360 native，另预留03c ONNX配对1,000 native，尚未消耗；Python18,000和C++45,000各自独立。旧03b失败及源快照保持。

### 03c 接线与独立审阅补充

训练已唯一启动。实际 Astra ultra 的接线审阅在启动后建议将包 __init__.py 和 audited schema 另外纳入身份清单；未改正在运行的源码/protocol，已保存带明确时点的 pilot03c_identity_supplement_01。schema 的 SHA 已由事前配置绑定，__init__.py 与启动前 commit a6779a4 相同，Astra 确认无需重训，结束时复核。保护检查 protected_guard_during03c_01.json：77份旧冻结源码相同、旧HEAD仍9a10a3d、用户my_course_drive_01保持。部署03c入口零积分核验及教程04章已完成；源码ZIP可执行位问题已修复并经实际解压检查。

## 03c 完整训练及固定验收

训练 COMPLETE：1,048,576 transitions / 10,485,760 native，801.58秒，输入SHA全保持；补录身份结束复核PASS。Torch/PythonORT/C++ORT1.10 64例最大误差5.3644e-7；500+500闭环PASS，预检池累计361,110/655,360。Python三条各6000native，无跌倒但资格0/3：前进vxMAE0.004121，停车0.096517m；倒车vxMAE0.027979，停车0.068760m；转向wzMAE0.116664，停车0.146643m。03c独立C++slot1用于turn12秒诊断，6000native，复现相同FAIL，P99推理0.020438ms。未追加Python评测、未开正式九次训练；Astra正在依据全状态审阅仅剩最后1次pilot。

## 构建中间文件与缓存清理完成

03c训练、评测和回放进程全部退出后，宿主只读ps确认无本项目训练/编译/仿真进程（cleanup_regenerable03_host_idle_01.json）。按精确白名单删除1057个再生文件，实际回收50,688,000 B（48.34MiB）；1642保护文件及228软链接均保持，旧77份冻结文件、旧Git状态及用户my_course_drive结果复核一致。见cleanup_regenerable03_execution_01.json。磁盘回收与训练进程退出释放RAM分开，未删除模型/权重/证据/库/可执行文件。后续构建需重编152个翻译单元，耗时未测。源码已提交0a1bb9e（本地Git，无remote）。

## 03d 最后pilot放行

实际Astra ultra最终审阅通过后，登记pilot03d_release_01.json；命令课程为唯一修改，奖励/接口/物理/PPO保持03c。11项静态与64seed调度检查、root入口整合、C++静态初态全PASS/0积分。第三次且最后一次，seed101/B/8env/1,048,576 transitions/90min，从随机初始化开始。预检当前361,110，另预留学习策略配对1000 native；Python18,000、C++45,000独立预算。

## 03d 最终训练、验收与部署关闭

03d COMPLETE：1,048,576 transitions / 10,485,760 native，794.67秒，冻结输入全保持；最后100 episode均长19.53秒不作为控制资格。64例Torch/PythonORT/实际C++ORT1.10最大误差4.7684e-7；500+500配对动作差0、qpos最大1.55e-15，预检累计362,110/655,360 native。

Python三条各6000native，1/3通过：前进vxMAE0.003172、停车0.080584m>0.05m；倒退vxMAE0.039917、停车0.016341m，全部门槛PASS；转向较差方向wzMAE0.102882>0.08rad/s、停车0.060764m>0.05m。无跌倒。独立C++各6000native/12秒，复现倒退PASS与前进FAIL，推理P99分别0.023405/0.030430ms。

最后live槽只启动发布器，import rclpy时因缺rpyutils搜索路径退出；native未启动，0新native/0ROS发布。完整ROS setup后的四项依赖import通过（0节点/0发布/0积分）；本合同不重试，不改冻结命令。watchdog停稳NOT_VALIDATED_STARTUP_ERROR；3个槽为两次物理运行及一次启动失败，不是三次完成。

合同03最终总账 final_delivery_01.json：预检/工程362,110；三次训练31,457,280；Python开发50,980；独立C++部署20,980 native，互斥合计31,891,350。三轮pilot已用完，整体资格FAIL；正式九次训练、地形策略评测和1.5m/s均未运行。实际Astra ultra审阅原始轨迹与数值结论。旧77冻结哈希、旧Git及用户结果最后保护复核不变（protected_guard_final03_01.json）。

## 交付回放与资源关闭

reverse_video_01与forward_video_02（跟随镜头）各391帧，使用已有C++状态+mj_forward，新增0native；root检查末段画面及PASS/FAIL标注。原forward_video_01固定镜头版本保留。media_delivery_01.json绑定视频、评分和预览身份。实际Astra最终审阅17项SHA和31,891,350总步一致，回执astra_final_review_01.json。宿主ps确认无项目训练/仿真/渲染进程（host_resource_final03_01.json），全系统可用RAM8,609,165,312B只作当时状态，不等同本项目释放量；本轮磁盘清理48.34MiB另记。

源码包 dist/d1_contract03_sources_20261009.zip 已生成，111个显式清单文件，359,236B，SHA256 109452dac0d5602928bb95f428833033d07a22014ecb593b50b8d1aa69ab20f7；逐成员hash、长度、Unix可执行权限、CRC与资产排除检查通过。包需另取原workspace2.zip，不含权重/模型/大型证据。本地源码提交不等于GitHub上传，当前remote为空。
