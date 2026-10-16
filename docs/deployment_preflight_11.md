# 部署预检与失败回执

第二轮新增 [preflight_deployment11.py](../tools11/preflight_deployment11.py)，在运行原生控制之前核验数值工件和 ROS 导入环境。入口接受 `pilot` 和 `scale` 工件，绑定 metadata、单样本导出 sidecar、64 例 C++ 数值回执和当前二进制身份，随后仅执行冻结入口的 `--print-command` 与纯 ROS imports。

旧 live watchdog 的导入/节点构造发生在其回执保护段之前；用注入的 `rclpy` 导入失败可以复现只留下空目录、没有结构化回执。旧文件保持冻结。新预检在导入失败、命令解析失败、非零退出、超时及输入变化时保存失败阶段、错误、stdout/stderr 和退出状态，关闭本轮自建进程组。它不替代 watchdog 本身的物理验证。

本机 `evidence/engineering11/deployment_preflight_01/report.json` 已通过：工程 ONNX 为 `d74b897e2d005a36985b9b70571a220d8e4f58189ec108f4e304a529d7eec290`，运行二进制为 `62134e7c94fafaefcdee9f8325191b8f621cb3eb51f9ed2a167d68198469975d`，11 项 ROS 检查通过、默认 context 未初始化；0 节点、0 物理步、0 原生命令执行。生成的 argv 供后续有预算的试次使用；本轮没有执行 validate-only 或 live，也没有取得控制资格。

```bash
rtk proxy env PYTHONPATH="$PWD/.python-deps" python3 -B tools11/preflight_deployment11.py \
  --export evidence/engineering10/export_single_sample_01 \
  --cpp-report evidence/engineering10/cxx_single_sample_golden_01/report.json \
  --output evidence/engineering11/deployment_preflight_01
```

上面是已执行命令的记录，输出目录拒绝覆盖。22 项假环境回归覆盖工件损坏、解析错误、依赖失败、超时、输入变化、信号竞态和进程清理；测试不会创建模型或节点。
