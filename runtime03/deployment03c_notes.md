# 03b / 03c / 03d 部署身份选择

03b、03c与03d的观测、动作倍率、PD、物理/控制周期和reset相同，使用 `d1-selftrained-pilot03b-v2`。它们的训练任务配置不同，因此不能互换配置SHA。本文件保留原名以延续已有链接；03d仅修改训练命令调度，见[03d合同](../docs/career_execution_contract_03d_addendum.md)。

新增 `schema03c.json` 只改变旧03b runtime schema的 `training_config` 和 `training_config_sha256` 两个字段。03c配置SHA为 `562bdc6b26fba7bbb4906f7045ba11cbe70848a3335bd8d5e1659540a3f2b604`，新runtime schema SHA为 `c3019b6737a358ebbdcd8c5cdb43bdd6b5a9783f36deeb31c30c87f8f820a538`。C++源码、binary、旧schema和历史记录均未改动。

03d沿用相同方式新增 `schema03d.json`，相对03c也仅更换上述两个字段。03d配置SHA为 `9cd6546e8a8df50c7d578a664003681fc79ddca7fa0a27b80158f0f5b6842bea`，schema SHA为 `64473031cbf027c950469a3db5a6e39b868f244da2b3f244fa174ab503f64c1c`。现有C++二进制SHA仍为 `62134e7c94fafaefcdee9f8325191b8f621cb3eb51f9ed2a167d68198469975d`，未重建。

`tools/run_selftrained03.py` 与 `runtime03/prepare_pilot_deployment03.py` 共用 `deployment_variants.py`，按导出metadata中的显式 `config_version` 选择配置，拒绝缺失/未知版本和交叉配置哈希。`d1-pure-ppo-pilot-03b`选择schema_pilot03b.json，`d1-pure-ppo-pilot-03c`选择schema03c.json，`d1-pure-ppo-pilot-03d`选择schema03d.json；不根据导出目录名称猜测，旧版本不会被新默认值替换。复制源文件包时必须包含该共享模块。

启动器的 `--print-command` 只验证并打印命令；`--validate-only`调用原生静态加载，无积分。两者都要求 `--output` 路径尚不存在，已有空目录也拒绝，重复检查请换新编号并保留旧结果。准备工具只生成可审阅命令清单，不执行试次，也不增加主代理已经批准的物理预算。默认fresh初态文件来自03b静态reset，因三个版本的reset和控制接口相同而可以复用；这不等于复用03b策略权重。完整终端示例见[运行器说明](README.md)。

资格评测必须使用 `sequences03b/forward_stop.json`、`reverse_stop.json`、`turn_stop.json`，与Python原三条12s序列相同。启动器的短选项会展开成这些JSON；直接调用原生程序也应传JSON路径。原生内置同名短序列的时间窗口不同，不能混作资格试验或比较其得分。

03c初次接入时用真实03b导出回归打印入口及内存metadata验证路由；原记录保留。03d接入时进一步对真实03b、03c导出做打印入口回归，对03d使用无权重的内存metadata验证路由及交叉哈希拒绝。`evidence/career_contract_03/deployment03d_adapter_01/` 与 `deployment03d_static_01/` 记录0物理步验证，03d/03c静态初态只有schema身份不同，其他字段逐值相同。这些结果不包含真实03d策略推理或行驶。

截至本次更新（2026-10-09），03d训练已完成1,048,576 transitions/10,485,760 native，耗时794.6727s、最小可用RAM 4,962,959,360 B，输入身份未变。真实ONNX位于 `evidence/career_contract_03/pilot03d_export_01/`，SHA为 `cf62bb43a53019d85b0dec00258dd261f217516f4a157a7e3826d21deffad48e`；64例实际C++ ORT1.10核验PASS、最大误差4.7684e-7，Python500+C++500 native闭环配对PASS，预检池累计362,110。

03b/03c/03d固定控制资格分别 **0/3、0/3、1/3通过**。03d三条Python任务均完整无跌倒、合计18,000 native；reverse_stop全门通过，forward_stop停车位移0.080584m超限，turn_stop的yaw MAE 0.102882rad/s与停车位移0.060764m超限。整体仍不合格，按合同结束三次pilot开发，不追加第四次或正式九次训练。

03d独立C++已执行两个12s槽，各6,000 native：`pilot03d_deployment_plan_01/final_score01_reverse_stop/report.json` 为PASS，停车位移0.016340523m；`final_score02_forward_stop/report.json` 为FAIL，停车位移0.080583646m。03d独立部署最终合计12,000 native，整个开发期合计20,980。最后live尝试的publisher在导入 `rclpy` 时因 `ModuleNotFoundError: rpyutils` 退出；原生程序未启动，新增0 native，无任何ROS发布。`live_execution_01.json` 原始状态为 `INTERRUPTED_OR_FAILED`，工程结论为 **WATCHDOG NOT_VALIDATED_STARTUP_ERROR**，不是物理FAIL；本合同不补跑。

最后live槽登记的publisher参数为 `--drive-vx -0.1`，计划2–8s倒退、8s后停止发布，原生watchdog仍为0.5s；该指令序列未实际执行。scorer读取publisher protocol内的速度，原停车门槛和窗口不变，结果限定工程scope、`qualifies_overall_pilot=false`。零物理诊断确认完整加载 `/opt/ros/humble/setup.bash` 后ROS依赖导入通过；原命令遗漏已安装 `rpyutils` 所在的 `/opt/ros/humble/lib/python3.10/site-packages`。这只给出下一合同的启动环境修复方向，不证明真实watchdog通过。不把synthetic接口测试当成实测，也不以这个槽替代03d整体资格。闭环配对的500个C++步计入预检池，不混作部署试次。最终部署状态见[续接状态](../docs/codex_handoff_20261009.md)，完整数值见[执行结果](../docs/career_execution_results_03.md)。
