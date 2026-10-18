# D1 28：固定共同输入的策略漂移审计

状态：**已COMPLETE，实际648样本actor前向，耗时1.102316秒，0训练/0物理**。见[实际结果与有限输入KL分解](policy_drift28_results_20261018.md)。以下保留执行定义。27 control24/36、yaw3 18/36均弱于完整random25的33/36；无eligible/holdout，不能执行候选C++闭环。28只检验“相邻更新KL小，是否仍已明显偏离固定基线”，不是训练、控制资格或新checkpoint选择。

## 固定输入与权重

使用原23闭环保存的0/20ms完整18例：三任务×两个延迟×三个reset。每例固定tick125/150/175/325/350/375与425/450/475/500/525/575，共**216个实际float32/627D输入**，每例drive/stop各6。输入直接取已保存inference NPZ，按原23已闭合report SHA和逐文件SHA核验，不重新构造观测、环境或物理。

三个policy固定为完整random25（873cd14f…cdb4f22）、27 control final（be727124…4c733ba）、27 yaw3 final（b42dec19…c5cab57）。逐SHA锁定已完成训练result/protocol、checkpoint、同一版本栈和velocity_policy15架构。27的工程终态与无资格身份如实保留；28不要求失败候选获得完整任务资格，正是为了诊断它们。

原23观测来自旧control的闭环，**不是25/27的on-policy分布**。其输入集用于有限同输入诊断，不能解释成新策略整体KL、真实性能、全部课程代表性或失败因果。

## 读取与实际数学

不新增pickle/storage解析器。实际执行时导入Torch与NumPy，并在CPU执行：从固定SB3 ZIP读取`policy.pth`，调用`torch.load(weights_only=True,map_location='cpu')`，拒绝19个float32 CPU tensor之外的键/shape/nonfinite。SB3 `data`与optimizer不反序列化，不import SB3/Gym/MuJoCo/ORT，不构造nn.Module、机器人或环境。

按SHA固定的15架构直接使用6个`torch.nn.functional.linear`及Tanh：encoder627→128→64→3，concat当前57+估计3，policy60→128→128→16。raw mean不做额外归一化/真值输入；clipmean=clip(rawmean,−1,1)，16维log_std从完整state读取，不能用CSV平均std替代。critic权重只随state读取，不执行critic或辅助头。

逐输入计算对角高斯的闭式`KL(base25 || new)`，用float64作统计；另列raw mean、clipmean与log_std差异以及task×drive/stop分组。不用随机Gaussian抽样，不调用旧ONNX、不再跑64fixture。

## 精确预算和输出

- 三次actor **batch函数调用**，每次216行；共**648样本actor forward**。内部合计18次batched linear操作，不把648样本写成648次函数调用。
- 三次weights-only checkpoint读取；0新增训练、0物理、0额外神经校准、0critic前向/0辅助监督损失。编码器的3维速度输出属于actor前向的一部分，已包含在预算中。实际attempt/return与样本行数分别记账。
- 固定新输出`evidence/policy_drift28/execution_01/{report.json,distributions.npz}`，拒覆盖/重试。保存输入216×627及三策略raw mean/clipmean/log_std，来源和输入SHA；失败回执保留已返回前缀，不补造成功或资格。

```bash
rtk proxy python3 -B tools28/audit_policy_drift28.py
```

默认PLAN_ONLY，不读取tensor或导入NumPy/Torch，不创建输出。源码提交且输入复核后，根线程在与轮腿重任务串行的窗口显式`--execute`。全部新source须与当前Git blob相同；已有源不变。120秒整轮界、CPU/BLAS/Torch单线程。

实际unit固定`d1-policy-drift28-20261018-01.service`，`Slice=app.slice`；沿共享guard核实际High2GiB/Max3GiB、swap0、oom.group1，启动外部有效6GiB，导入和运行后保留4GiB。NN不在当前源码准备阶段执行。unit被OOM终止而来不及落盘时由外部终态记录unknown，不重开执行目录。

## 验证与限制

18项纯/合成测试通过；未执行真实NN。测试只用小型合成数组、假Torch对象和ZIP；验证weights_only/CPU参数、精确19tensor、完整6层/216行、闭式KL方向/数值、clip前后区别、真实216位置、exact3/648/18账、失败前缀/来源漂移/时限/低内存与独占输出。默认冷导入单独进程核验。可选本机检查只重哈既有小元数据和18输入文件，不读取tensor或执行NN；本机该检查已通过。CI无历史现场时跳过此一项，其余合成测试仍执行。

没有真实SB3/Torch模型数值对照的新调用：functional actor依赖固定15架构与完整state的直接数学实现，正式审计将报告有限输入输出与版本。若后续需要额外parity探针，必须显式登记它们，不藏进648样本预算。结果只能决定下一项可证伪实验，不据KL大小自动替换模型或授予资格。
