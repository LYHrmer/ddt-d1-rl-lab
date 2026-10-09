# 给同机 Claude 的续接指令

独立私有仓库：[LYHrmer/ddt-d1-rl-lab](https://github.com/LYHrmer/ddt-d1-rl-lab)，默认分支 `main`。发布身份以远端 HEAD 和本机 `evidence/career_contract_03/github_publication_receipt_01.json` 为准。旧交接与总账中的“未上传”是当时状态，保留原记录。

复制下面整段到本机新的 Claude 窗口：

```text
请继续 /home/lyh/ddt_ros2_ws 的 D1 纯强化学习项目，按“稳定直行→可靠越障→提高速度”推进，目标是可复现、可审查的运控求职项目。先完整阅读 docs/codex_handoff_20261009.md、docs/career_execution_results_03.md、docs/career_execution_contract_03d_addendum.md、docs/tutorial/00_复现教程.md，以及 evidence/career_contract_03/final_delivery_01.json、evidence/career_contract_03/github_publication_receipt_01.json；再核对工作树、Git 远端、宿主运行进程、release/protocol 输入哈希和保护回执，不根据隔离进程列表为空就重复启动任务。源码入口为 training/d1_rl、tools/train_d1_ppo.py、tools/export_d1_ppo.py、tools/evaluate_d1_ppo.py 与 runtime03。

现状必须按证据表述：03b/03c/03d 三次训练均已完成，固定低速资格分别 0/3、0/3、1/3。03d 只有倒退通过；独立 C++ 倒退停车 1.63cm 通过，前进停车 8.06cm 超过 5cm，转向正向误差约 0.1029rad/s 超过 0.08rad/s。完整训练、数值一致和没有跌倒均不代表控制全门通过。最后 live 发布器导入 rpyutils 失败，原生未启动、0 新物理步，watchdog 未验证。丰富地形和教程已交付，但自训策略的真实 15mm 台阶、持续侧移、1.5m/s、独立 GUI、物理自救及实机验证均未完成。

先在新启动入口加载完整 ROS Humble 环境，仅做依赖导入检查，不建节点、不发消息。rpyutils 已安装在 /opt/ros/humble/lib/python3.10/site-packages，旧命令漏了该路径；完整 setup 后四项依赖导入已通过。随后离线分析既有轨迹中的前进刹车和正负转向差异，区分事实与假设，只选一项有证据支持的训练改动，写新版本、新输出目录和新有界合同，再执行该合同内的实验。不要直接追加旧训练，也不要把旧合同三次 pilot 的停止边界误读为永久禁止后续训练；新实验需要新预算、冻结身份、验收门槛和失败停止规则。

旧 /home/lyh/wheel-legged-control-lab 及其 work 目录、77 份冻结文件、用户原结果和本轮历史记录保持只读；不得覆盖日志、换 seed 挑 checkpoint 或重跑旧批次来补好成绩。各池实际 transitions/native 分别计账，失败步也计入，保留失败证据，不把 mj_forward 算成积分。平地前进、倒退、转向及独立部署全门通过后，先验证 15mm 单台阶几何/接触和低速跨越，再逐级提速；1.5m/s 仍是目标。

每条 shell 命令加 rtk，需原始输出用 rtk proxy。常规开发、验证和 GitHub 上传已获授权，请直接完成可审查的工作，不反复确认。可让实际 Opus 智能体并行承担有界代码与测试，主代理负责方案、集成、计步和物理验收；不得虚称模型调用。上传只选本项目自有源码和允许的说明，排除同事模型/权重、依赖和大轨迹，使用本项目独立私有远端；清理只处理明确可再生文件，并复核保护哈希。收尾更新教程、结果、总账、续接文档及发布回执，严格区分已验证、假设和未完成事项。
```

## 优先路线与验收

1. **启动环境：** 新入口完整加载 ROS 环境；`rpyutils`、`rclpy`、`geometry_msgs.msg`、`std_msgs.msg` 导入通过并记录环境身份。此阶段必须是零节点、零消息、零积分；导入通过不算 watchdog 通过。
2. **离线诊断：** 比较前进/倒退的停车过程与正/负转向段，检查命令、观察、动作、力矩限幅和既有奖励/命令分布；保存可追溯分析，不提前认定某一根因。只形成一个可证伪假设及其预期指标变化。
3. **新有界合同：** 预先登记版本、输入 SHA、单一改动、seed/checkpoint 规则、训练/评测/部署各自步数与墙钟/RAM 上限、失败停止条件。用新目录记录新实验；旧预算余额不是新许可，不移动原有门槛来宣布通过。
4. **分级物理验收：** 平地固定三任务须完整通过原全部门槛，跨语言核验、独立 C++ 和真实 ROS 停发停车分别验收。之后才登记 15mm 单台阶的几何/接触与低速跨越合同，再制定速度阶梯；每一级保留失败记录，未通过不晋级。

历史最终总账：训练 **31,457,280 native / 3,145,728 transitions**；预检/工程 **362,110 native**；Python 评测 **50,980 native**；独立 C++ 部署 **20,980 native**。总计 **31,891,350 native**，新合同从新增计数开始，不改写上述历史累计。完整证据保留本机；源码仓库不替代权重和物理原始记录。
