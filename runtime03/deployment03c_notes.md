# 03c 部署身份选择

03c 与 03b 的观测、动作倍率、PD、物理/控制周期和reset相同，仍使用 `d1-selftrained-pilot03b-v2`。它们的训练任务配置不同，因此不能互换配置SHA。

新增 `schema03c.json` 只改变旧03b runtime schema的 `training_config` 和 `training_config_sha256` 两个字段。03c配置SHA为 `562bdc6b26fba7bbb4906f7045ba11cbe70848a3335bd8d5e1659540a3f2b604`，新runtime schema SHA为 `c3019b6737a358ebbdcd8c5cdb43bdd6b5a9783f36deeb31c30c87f8f820a538`。C++源码、binary、旧schema和历史记录均未改动。

`tools/run_selftrained03.py` 与 `runtime03/prepare_pilot_deployment03.py` 共用 `deployment_variants.py`，按导出metadata中的显式 `config_version` 选择配置，拒绝缺失/未知版本和交叉配置哈希。`d1-pure-ppo-pilot-03b`选择schema_pilot03b.json，`d1-pure-ppo-pilot-03c`选择schema03c.json；不根据导出目录名称猜测。复制源文件包时必须包含该共享模块。

启动器的 `--print-command` 只验证并打印命令；`--validate-only`调用原生静态加载，无积分。准备工具只生成可审阅命令清单，不执行试次，也不增加主代理已经批准的物理预算。默认fresh初态文件来自03b静态reset，因两版本的reset和控制接口相同而可以复用；这不等于复用03b策略权重。

此次接入检查使用真实03b导出回归打印入口，并以无权重、仅含版本/哈希的内存字典验证03c路由。实际C++加载新schema和原机器人、构造静态初态通过，物理步为0。接入完成时尚无03c导出，因此没有声称03c策略推理、跟踪或部署通过；后续由主代理对真实03c导出分别完成原生ONNX黄金核验和有界运行。
