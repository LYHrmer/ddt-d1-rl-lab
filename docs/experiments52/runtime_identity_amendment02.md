# 52 版本身份修正，正式执行前

原提交 `cdfb688` 的准入 `admission_01` 在构造环境前被版本检查拒绝：计划误用了 wheel 本地依赖中的 Torch 2.13.0。D1 的实际模块来自 `/home/lyh/.local/lib/python3.10/site-packages/torch`，发行包元数据为2.12.1，`torch/version.py` 为2.12.1+cu130；本次仍只在CPU执行，CUDA_VISIBLE_DEVICES为空。

`model_gradient52_revision02.json` 仅修正这个包版本并增加本附录对应的父计划/失败回执身份。原计划、已提交源码、seed、模型、预算和门槛全部保留不改。首次失败在加载actor与构造环境之前发生：0物理步、0模型更新，原计数字段null表示环境未建立，不能当作一次训练试验。

后续使用全新的 `admission_02`，通过才允许首次正式 `execution_01`。不续跑或覆盖 `admission_01`；两个准入目录的累计物理预算仍不超过原256策略步。运行入口与计划见[五项预注册方案](../model_gradient52_plan_20261031.md)。
