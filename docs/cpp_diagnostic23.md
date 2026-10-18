# 原17 control 的独立 C++ 闭环诊断23

本轮仅验证已完成原16 control、原17完整ONNX在C++ ORT 1.10下实际驱动同一仿真环境时的数值链、调用、资源与清理。0新增训练。原17的36例参考结果为27/36，0/20/40/60ms分别9/9、9/9、6/9、3/9，完整延迟参考资格未通过；本轮不会授予部署、ROS、实机、holdout或旧19资格。

[旧19](cpp_closedloop19.md)的18胜出策略原36及holdout36全部通过门保持原样，当前不满足。23使用独立release、来源、输出与预算，不调用或替换19的资格loader、run_worker或execute。仅复用它的C++传输、有限进程清理、计数与保存轨迹数值核查函数。

## 固定输入与来源

只接受[原17摘要](../reports/closeout17_20261018.json)绑定的原16已完成control（seed101、aux0、4环境、1048576训练步），checkpoint SHA `428ff0ce9e8e88e10dabf94801a57b88de7e76d07b42800553b69c7f590e4eff`。读取原17三阶段登记工件、全部36条既有轨迹和原始数值证明，并重新核对原16完整来源。ONNX必须是该模型的完整encoder→actor→clip图，obs57加previous-only、oldest→newest历史10×57；不加载partial、随机图或22模型。

复用`runtime19/build/actor_server19`前核对原构建回执、二进制、C++源、adapter、ORT库与编译输入全部SHA，不重新编译。新23三个文件、复用19/17和相关16来源都必须已提交且当前字节等于该commit；执行前、执行中入口和闭账重新核来源。旧记录与源文件不修改。

## 唯一新执行

默认命令只打印计划，0神经推理、0机器人模型、0物理步：

```bash
rtk proxy python3 -B tools23/run_cpp_diagnostic23.py --plan-only
```

提交稳定来源并满足资源门后，显式执行入口为：

```bash
rtk proxy python3 -B tools23/run_cpp_diagnostic23.py --execute
```

实际安排由外层systemd用户transient单元以Nice10、低CPU权重运行，避免工具生命周期向训练/仿真进程传递信号。此脚本不创建单元或改变系统设置。固定目录`evidence/cpp_diagnostic23/execution_01`独占；存在即拒绝，不续跑、不重试。低于6GiB启动线返回NOT_READY且不占目录；占目录后任何失败都保留原记录。

父监督器创建新release与nonce，子进程仅能通过同文件隐藏的`--worker`入口在该父PID、活动journal、release SHA和nonce一致时启动。不能直接调用CLI跳过父监督。source提交门不依赖旧19资格，也不依赖22结果。

## 动作、矩阵与计数

一个C++ actor服务先完成64个原17固定样本，与保存的Torch/Python ORT输出及当次Python ORT输出逐例比较，最大绝对误差≤1e-5，0物理步。然后运行forward_stop/reverse_stop/turn_stop × delay0/1/2/3个20ms tick × reset seed1001/1002/1003，共36例，每例最多600个控制区间。场景、命令、指标、阈值、03d环境与历史语义均复用原冻结实现。

每拍实际float32的627维观测同时传给C++和Python ORT；只有C++返回的16维动作进入`env.step`，Python ORT只作同输入oracle、不启动第二个环境。逐例保存原物理NPZ和独立`*_inference.npz`，包含observation(n,627)、cpp_actions(n,16)、python_ort_actions(n,16)、max_absolute_error(n)。父级重新核SHA、dtype/形状/有限值、每拍误差≤1e-5，并要求物理trace.issued_action与cpp_actions精确相等。保存的C++ adapter回执必须与child内嵌回执相等，其模型、二进制、source、adapter、ORT库五项身份接回冻结来源；READY签名/线程/上限/零物理与EOF实际计数另行逐字段复核。

物理上限21600 policy /216000 native /108000 PD；C++和Python ORT各最多21664次推理，其中64次为加载探针。正常完成须C++ CLOSED且requests_completed、inference_attempts、inference_returns、oracle_calls全部精确等于64＋实际policy步数。异常中的尝试/返回及物理计数分别保留，未知为null，不能用exit code猜零。

## 停止与结论

启动有效RAM≥6GiB；运行有效RAM≥4GiB，host可用内存与可见cgroup硬限制余量取最小，swap不算RAM。CPU/BLAS/ORT单线程；父每秒、子预测入口按1s周期检查内存，时限每次检查；整轮900s、worker840s、父880s开始停止，留出有限清理时间。仅清本轮自有后代，采用Linux birth tick避免PID复用误杀；Popen尚未返回句柄也清理自有后代。信号期间保护最终计数与回执，未知清理不算成功。

合法提前跌倒是任务失败，继续收齐36例，不补拍；工程、来源、数值、计数、资源或清理错误则停止。COMPLETE只表示预定诊断工程闭账完整，仍固定diagnostic_only=true、qualification_eligible=false、deployment_qualified=false、old19_eligibility_granted=false。各延迟完整回合、失败、参考通过数和all_36_reference_pass由本轮C++轨迹实际重算；不预填原27/36，不因数值容差声称整条轨迹逐字节等价。原17全资格未通过记录保留，即使本轮某个矩阵通过也不自动升级任何旧资格。

当前仅新增源码、纯fake/NumPy回归与文档；尚未实际运行23的C++服务、模型或物理。
