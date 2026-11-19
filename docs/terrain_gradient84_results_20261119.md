# 84 轮双轮迹地形梯度核验

**原安装位的 924 个闭环测量包中，可计算双轮迹地形梯度的有 0 个。** 静态标定通过 9/9。将测距原点前移 16 cm，在存档捕捉姿态上重新投射射线后，308 帧中有 80 帧可以核验，均通过。这个结果量化了安装位置对可见性的影响。完整任务的控制收益仍待验证。[本仓证据摘要](assets/project84/evidence.json) · [wheel 闭环结果与视频](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/docs/terrain_preview84_results_20261119.md)

## 哪些输入有可计算的导数

| 输入来源 | 总帧数 | 可判且通过 | 未覆盖 | 含非零斜率的帧 | 中心差分对解析斜率的最大绝对误差 |
|---|---:|---:|---:|---:|---:|
| 平地和 ±4° 坡面，各配 3 个静态姿态 | 9 | 9 | 0 | 6 | 2.89 × 10⁻¹² |
| 27 个正式闭环工况，原安装位，每个新包计一次 | 924 | 0 | 924 | 无可判帧 | 无样本 |
| 事后前移 16 cm，9 组去重轨迹的原捕捉姿态 | 308 | 80 | 228 | 39 | 5.62 × 10⁻¹² |

统计单位是新采集帧。任一轮迹的解析斜率绝对值超过 10⁻⁸ 即计入非零帧。静态和前移两组的可判帧中，自动微分与解析斜率的最大绝对差均为 0。单独阻断查询坐标的梯度后，两轮迹的前向高度逐值相同，坐标导数均为 0。正式工况的报告状态为 `NO_COVERAGE`，误差栏没有可用于比较的数值。[静态帧](assets/project84/static_frames.jsonl) · [正式测量包](assets/project84/frames.jsonl) · [事后重采样帧](assets/project84/translated_mount_frames.jsonl)

## 为什么正式运行没有覆盖

三种预瞄输入在同一道路和种子下产生了相同的捕捉轨迹。将 27 个工况按轨迹去重得到 9 组，共 308 个姿态。每个姿态发射 8 束射线，原安装位的 2,464 次首次命中检查中，有 2,276 束先遇到前大腿且距离过近，只有 179 束得到量程内的环境返回。左右轮迹都能在查询位置找到相邻地形点的帧数为 0。[wheel 安装位置审计](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/scripts/terrain_preview84_mount_audit.py) · [跨仓结果](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/docs/terrain_preview84_results_20261119.md)

事后审计固定了存档姿态、查询位置与道路，使用捕捉时的仿真姿态对齐重力方向，只将左右测距原点沿机身前向平移 0.16 m。双轮迹覆盖了 82 帧，其中 80 帧还通过左右坡度一致性条件。在 5,123 个去重控制周期里，同时具备有效地形和纵向力余量的有 12 拍，位于前轮首次触坡前的有 3 拍。这次重采样的新增物理积分步数为 0。前移位置的实装闭环响应尚未测量。[安装复查与资源记录](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/docs/terrain_preview84_results_20261119.md)

## 导数能回答什么

给定采集坐标系下的轮轴纵向位置 `query_x_m`，工具在左右轮迹上分别寻找包围该位置的唯一一对相邻测点，线性插值得到高度。查询点及其两侧各 10⁻⁵ m 都必须位于区间内部。缺测、重叠区间、端点和超出视野的查询会记入未覆盖。工具在 CPU 上用 float64 将解析 `dz/dx` 与中心差分、PyTorch 自动微分逐轨比较。这个量的单位是 m/m，回答轮轴沿已测地形移动一小段时，查询到的地面高度改变多少。[可微查询与核验实现](../tools84/terrain_gradient84.py) · [缺测与梯度测试](../tests/test_terrain_gradient84.py)

策略收益还需要测量另一条因果关系，即改变实际施加的俯仰参考，会怎样影响机器人轨迹和任务评分。六个正式坡道配对均未走完坡脚后 0.4 m 的评分区间，入坡俯仰误差改善率无样本。给奖励项增加权重也无法补出这段物理轨迹。放行学习前，应先在新的配对闭环中证实控制器能在触坡前用上观测和力余量、能通过评分区间，再用同一起点的正负动作微扰核验任务响应。这轮 D1 策略更新数为 0。[wheel 配对验收](https://github.com/LYHrmer/wheel-legged-control-lab/blob/main/docs/terrain_preview84_results_20261119.md)

## 从提交的输入复算

在本仓库根目录、装有 CPU 版 PyTorch 的环境中串行运行以下命令。输出采用独占创建。静态、正式、事后三种输入预期分别得到 `PASS`、`NO_COVERAGE`、`PASS`，其中正式输入对应的 CLI 退出码为 1。每份输出保留逐帧判断、测量序号和输入 SHA-256。

```bash
review_dir=$(mktemp -d)
python -B tools84/terrain_gradient84.py --frames docs/assets/project84/static_frames.jsonl --output "$review_dir/static.json"
python -B tools84/terrain_gradient84.py --frames docs/assets/project84/frames.jsonl --output "$review_dir/formal.json"
python -B tools84/terrain_gradient84.py --frames docs/assets/project84/translated_mount_frames.jsonl --output "$review_dir/translated.json"
```
