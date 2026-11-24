# AME-2 怎样把不完整的地形图交给运控

核对版本为 ETH Zürich 官方 `ame2_minimal` 的 `2f68676`。最值得 D1 借鉴的是让策略同时知道“轮前有多高”和“这个高度有多少依据”。机器人在台沿看见上表面后，视野可能暂时被腿挡住。如果反复复用同一个旧高度会增加置信度，策略就可能把已经结束的台面一直当作接下来的路。论文的学生策略在一组遮挡场景里也把障碍看成较高的平台。[官方版本与发布范围](https://github.com/leggedrobotics/ame2_minimal/tree/2f68676cc5388cc97c128036936d9a3d7c399d19) · [论文的学生模型消融](https://arxiv.org/html/2601.08485v3#S7.SS2)

| 要查的问题 | 作者的实现 | D1 的对应问题 |
|---|---|---|
| 一次扫描产生什么？ | G1 示例用 10 Hz 的模拟 MID-360 扫描，过滤机身和近距离返回。预训练地图网络估计格子的高度及不确定度，按位姿将不同帧对齐。 | D1 的点云存档与在跑策略的八束测距是两套输入。先在有点云的存档验证地图，再讨论接入策略。[传感器与地图更新](https://github.com/leggedrobotics/ame2_minimal/blob/2f68676cc5388cc97c128036936d9a3d7c399d19/ame2/ame2/sensors/livox_neural_map_sensor.py#L16-L27) |
| 遮挡的下一帧要不要覆盖上一帧？ | 根据新旧估计的方差选择高度，避免把同一位置反复预测当作新的独立测量。 | 轮速和 IMU 的位姿误差可能让查询落到台沿另一侧。即使旧回波很确定，也要检查它是否仍与当前轮迹属于同一片表面。[融合实现](https://github.com/leggedrobotics/ame2_minimal/blob/2f68676cc5388cc97c128036936d9a3d7c399d19/ame2/ame2/sensors/neural_mapping_utils.py#L645-L715) |
| 策略能读什么？ | 学生把每格的平面位置、高度和不确定度交给地图编码器，用本体历史参与空间注意力。minimal 的仿真传感器还读取机身和雷达的世界位姿。 | D1 可用带时间戳的测距、关节编码器、轮速和 IMU。以模拟器真值对齐得到的结果应作为定位上界单列。[学生观测](https://github.com/leggedrobotics/ame2_minimal/blob/2f68676cc5388cc97c128036936d9a3d7c399d19/ame2/ame2/tasks/mdp/observations.py#L201-L240) · [姿态来源](https://github.com/leggedrobotics/ame2_minimal/blob/2f68676cc5388cc97c128036936d9a3d7c399d19/ame2/ame2/sensors/livox_neural_map_sensor.py#L238-L299) |

**特权信息在哪一步消失？** G1 示例的学生训练在每个环境重置时有 10% 概率改用加噪的理想地形扫描。播放配置重新构建传感器，该比例取默认的 0。两处都过滤 0.5 m 内的激光回波。D1 要辨别近处的轮前台沿，最小量程需按自己的光心和轮迹实测。[学生训练配置](https://github.com/leggedrobotics/ame2_minimal/blob/2f68676cc5388cc97c128036936d9a3d7c399d19/ame2/ame2/tasks/ame2_env_cfg.py#L777-L793) · [选择理想扫描的实现](https://github.com/leggedrobotics/ame2_minimal/blob/2f68676cc5388cc97c128036936d9a3d7c399d19/ame2/ame2/sensors/livox_neural_map_sensor.py#L123-L162) · [播放配置](https://github.com/leggedrobotics/ame2_minimal/blob/2f68676cc5388cc97c128036936d9a3d7c399d19/ame2/ame2/tasks/ame2_env_cfg.py#L836-L851) · [默认比例](https://github.com/leggedrobotics/ame2_minimal/blob/2f68676cc5388cc97c128036936d9a3d7c399d19/ame2/ame2/sensors/livox_neural_map_sensor.py#L424-L430)

minimal 公开了 G1 教师 PPO、学生蒸馏与强化学习、预训练地图组件和示例权重。该版本把地图训练代码列为后续发布计划。论文实机平台是 ANYmal-D 和 LimX TRON1。仓库主许可证为 GPL-3.0，第三方组件的来源与许可另列。本轮完成了源码和论文对读。闭环试验运行 D1 自己的策略。[官方 README](https://github.com/leggedrobotics/ame2_minimal/blob/2f68676cc5388cc97c128036936d9a3d7c399d19/README.md) · [论文实机部分](https://arxiv.org/html/2601.08485v3#S6.SS2)

## 可用现有存档检验的问题

D1 第 106 轮的 6 次历史补读中，3 次补对，3 次补错。可先固定同批点云、外参、策略和奖励，只改历史补读的规则。仅当轮迹查询在机载里程计的定位误差范围内始终落在同一片有回波支撑的表面时，才采用旧高度。误差范围用单独的标定来源确定并冻结，验收时只用当时已送达的信息预测，事后才用真值计分。按独立轨迹分别报告错误补读、正确补读和未知数量。错误补读减少且正确补读不减才支持继续。这项几何复核可先用存档完成，新增仿真和训练预算均为零。既有 #106 来源已被分析过，独立泛化需要另留数据。[D1 逐包证据及误读原因](terrain_history106.md)

AME-2 给的是地形信息。SuperMimic 式世界模型学的是动作改变后机器人会怎样运动。即使前者把台沿估准，后者仍需要同一起点下改变动作的触阶轨迹，才能检验模型给的训练方向。[SuperMimic 策略更新实现](https://github.com/HybridRobotics/supermimic/blob/698ea39bc901d8fe733b3c8621df3199d241f7e4/src/supermimic/learners/policy_learner.py#L83-L125) · [D1 触阶对照与独立盲测协议](experiments108/contact.json)
