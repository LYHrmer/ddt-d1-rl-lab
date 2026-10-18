# D1 33/34 参考整形：C++ 部署链使用说明

**范围:** 这是 lab 内的显式入口,与默认17、23的先例同级。它不改 ROS2 `rl_controller`、不改 `runtime03` 原生试点链、不替换默认模型,也不授予实机资格。5.89% 的综合 J 取舍(相对25基线)依然成立,见[34结果](cpp_reference34_results_20261018.md)。

## 构成

| 部分 | 位置 | 作用 |
|---|---|---|
| 整形 + ORT 推理 | `runtime34/actor_server34.cpp` | gain 1.5、clip ±0.25(偏航命令输入)、dt 0.02 / tau 0.1 一阶因果平滑、十帧有效历史;每个进程一个回合,新进程清空状态 |
| 传输与身份核验 | `tools34/cpp_reference34.py`(复用 `tools19/cpp_actor19.py`) | JSONL 协议,逐拍核对二进制、模型、源码、ORT 库与适配器 SHA |
| 一次性闭环验证 | `tools34/run_cpp_reference34.py` | 冻结的72场物理闭环(已执行,`single34 attempt only`,不可重跑) |
| **可重复重放** | `tools34/replay_cpp_reference34.py` | 零物理:把保存的原始627输入喂给真实 C++ 进程,要求16个动作与保存值逐位相同 |

## 协议要点

服务器按行读 JSON:`{"format":"d1-cpp-actor19-request-v1","seq":N,"actor":[627个float]}`,`actor` 是**未整形**的原始观测(11帧×57,偏航命令在每帧第8列,已乘0.25)。服务器在内部整形后推理,返回16个已裁剪动作和 `effective_yaw_commands`、`effective_input_sha256`。`seq` 必须从0连续;每进程最多 21664 个请求;模型 SHA、源码 SHA 必须在命令行显式给出,ORT 必须恰为 1.10.0。偏航历史必须是真实的历次输入,首帧要求11帧原始偏航命令全相同(从静止/恒定命令起步);否则服务器拒绝并报错,不会静默继续。

## 复现

```bash
# 单个回合
python3 -B tools34/replay_cpp_reference34.py --stage formal \
  --episode forward_stop_delay0_seed1001 --output /path/to/new_dir

# 全部72场(约1分钟,0物理步)
python3 -B tools34/replay_cpp_reference34.py --output /path/to/new_dir
```

输出目录必须事先不存在。入口会拒绝非 COMPLETE 或未通过资格的执行、被改动的 sidecar、与资格执行不同的模型/二进制。成功时 `report.json` 的 `all_actions_bitwise_equal_to_saved` 为 true,进程以0退出。资源上,建议像其它重任务一样放进受限单元:

```bash
systemd-run --user --collect --wait -p MemoryHigh=2G -p MemoryMax=3G -p MemorySwapMax=0 -p Nice=10 -p CPUWeight=25 \
  python3 -B tools34/replay_cpp_reference34.py --output /path/to/new_dir
```

## 实际重放结果

本机已执行全量重放(`evidence/cpp_reference34/replay_all_01`,单元 `d1-replay34-all-20261018-01`,退出码0):72场共43200次真实 C++ 推理全部返回,与保存的动作**逐位相同**(最大差0.0,不匹配拍数0),每个进程清理完整,耗时58.9秒,0物理步、0机器人模型。机器摘要见 [`reports/cpp_reference34_replay_20261018.json`](../reports/cpp_reference34_replay_20261018.json)。这证明在这台机器上、同一二进制/模型/ORT 下,整形与推理可重复;不是新的闭环物理结果。

## 不在范围内

- 与 ROS2 `rl_controller`(`FSMState_RL`)或 `runtime03` 的集成:前者是与合作者共用的实机控制器,后者的接口(腿倍率0.5、轮倍率5)来自 pilot03d 训练配置,与本策略的兼容性未验证,二者都没有改动。
- 更换默认模型:是否接受 5.89% 的综合 J 取舍、换取 36/36 + 36/36 的完整资格,由使用者决定。
