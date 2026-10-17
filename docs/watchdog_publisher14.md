# Watchdog 发布器 v14

`tools14/watchdog_publisher14.py` 是独立的新 ROS 发令入口。它保留旧试验时序，补齐导入、初始化、节点建立、运行和清理失败时的结构化回执。本次只通过假 ROS 验证，没有创建真实节点、发布真实消息、加载模型或执行物理步。

## 保留的试验语义

- 要求显式 `--execute-live-publisher` 和 `ROS_DOMAIN_ID=84`，使用原来的两个 `/d1_selftrained/` topic。
- 按最后收到的原生 truth 时间发令：0–2 秒发零令、2–8 秒发指定的有限非零 vx、8 秒后完全停发。清理时也不补发零令，以保留真实停发 watchdog 试验的含义。
- 每次发令前，truth 发布者、命令发布者、命令订阅者都必须恰好一个。运动前允许发现过程稳定；运动后端点数改变就结束采集。
- truth 年龄超过 0.25 秒时跳过发令，收到有效新 truth 后恢复。拒绝非有限、重复、回退或越界时间；其余未使用字段保持原样，但无法保存为有限 JSON 的消息不参与发令。
- 收到有效 12 秒终点后，即使原生发布端点已退出，也算采集完成；这不会放松任何发令前的端点检查。ROS 关闭明确记作 `ROS_SHUTDOWN`，不再误记为墙钟超时。

`CAPTURE_COMPLETE` 仅表示收到终点且清理成功。它不证明运动发生、原生轨迹完整、watchdog 触发或机器人实际停住；零运动采集也可能完成。

## 故障回执与资源归属

输出必须使用全新目录，已有目录不覆盖，也不会加载 ROS。创建目录后，可捕获的启动、采集或清理异常都会写入 `report.json`。`protocol.json` 记录新协议和旧源文件身份；事件与 truth 流在导入 ROS 前打开。

入口创建专属 `Context` 和同一 context 的 `SingleThreadedExecutor`，不初始化或关闭调用方默认 context。节点句柄在发布器、订阅器建立前保存，部分建立失败仍会清理。finally 独立尝试 executor 关闭（请求 1 秒超时）、节点销毁、专属 context 关闭和两个流关闭；一个失败不会跳过其余清理。清理失败不能声称成功，原采集终因保留在 `capture_terminal_reason`。

12–30 秒墙钟限制约束采集循环，不能抢占阻塞的 ROS 导入或原生 API。executor 的超时参数也不等于进程硬截止保证。强制杀进程或存储不可写可能阻止最终回执，仍需外部进程监督。

`commands_sent` 统计成功返回的发布调用，后续事件落盘失败也保留该计数；它不证明原生订阅方收到或应用命令。观察者时间戳也不等于原生回调年龄，不能据此声称直接测到内部精确 0.5 秒超时。

## 使用与兼容边界

`python3 -B tools14/watchdog_publisher14.py --help` 只读帮助，不导入 ROS。真实使用须另有物理试次授权，并传入全新输出路径和显式执行参数；本次未执行真实使用。

新协议为 `d1-real-ros-watchdog-probe-v14`，回执为 `d1-watchdog-publisher-receipt-v14`，声明 `legacy_evaluator_compatible=false`、`trial_authorised_by_tool=false`。不能交给旧 `evaluate_watchdog03.py` 冒充历史试验，也不能沿用旧物理额度；物理停车评测还需独立审查、授权和兼容评测器。

## 离线交付证据

`rtk pytest tests/test_watchdog_publisher14.py -q`：43 项通过。测试用假 ROS 和假时钟驱动实际新 `main()` 与接收回调；另一个子进程直接执行脚本 CLI，使用临时假 ROS 包和默认懒加载器，注入 init 失败。覆盖启动各阶段、部分节点建立、流打开和关闭、独立清理失败、发布异常、时序窗口、陈旧后恢复、非法 truth、端点变化、终点退出、ROS 关闭、时钟异常、中断、独立 context、已有目录拒绝、墙钟上限及帮助。

旧脚本保持 SHA256 `814b4101be302cff307c35c45a853209d2ea064583dbaf64eb7799cdc9f8c825`。直接驱动旧 `main()` 已复现 init 抛异常后目录和三个采集文件存在、`report.json` 缺失；新入口为同类故障记录 `ROS_INIT_FAILED` 并清理专属 context。精确场景、结果、CLI 参数和源文件 SHA 见 `reports/watchdog_fault14_summary.json`。
