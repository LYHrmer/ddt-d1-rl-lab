# 存储清理记录 03

2026-10-09（北京时间）本轮仅清理 `/home/lyh/ddt_ros2_ws` 内已核验、未被 Git 跟踪的再生文件，共 **1,057 个，50,688,000 B（48.34 MiB）**。回收量按每个文件删除前的 `st_blocks × 512` 相加，执行后确认全部候选路径已不存在；不以整个磁盘的 `df` 波动代替逐文件核算。详见[执行回执](../evidence/cleanup_regenerable03_execution_01.json)及其[逐文件日志](../evidence/cleanup_regenerable03_execution_01.json.deleted.jsonl)。

| 删除项 | 文件数 | 删除前实占字节 |
| --- | ---: | ---: |
| CMake 编译对象 `.o` | 152 | 41,705,472 |
| 对象依赖文件 `.o.d` | 152 | 2,486,272 |
| 有对应源码的 Python 字节码缓存 | 753 | 6,496,256 |

执行前使用[精确白名单](../evidence/cleanup_regenerable03_allowlist_02.json)逐项检查 SHA-256、inode、长度、mtime、单硬链接和未跟踪状态，确认对应源码与链接产物存在、静态库不是薄归档。执行前后 **1,642 个保护文件 SHA-256 一致，228 条工作区软链接保持**。源码、配置、已安装依赖、现有 `.so` / `.a` / 可执行文件、权重和实验记录均保留；没有再清理旧仓库、77 份冻结文件或 `my_course_drive_01`。

主代理在 **2026-10-08 18:32:28 UTC** 使用宿主命令 `rtk proxy ps -eo pid,ppid,rss,args` 检查相关训练、运行器和构建进程，未发现匹配进程，随后于 18:32:43 UTC 开始执行。独立的[宿主确认回执](../evidence/cleanup_regenerable03_host_idle_01.json)与执行记录中的 `--host-idle-confirmed` 对应。隔离命名空间中的空进程列表、以及带挂载权限警告的 `lsof` 输出，本身不足以证明宿主空闲。

这是**磁盘空间回收**。训练进程自然退出后，其内存由操作系统回收；不能把删除对象或字节码缓存的 48.34 MiB 算成释放 RAM。下次 CMake / colcon 构建需要重新编译 **152 个翻译单元**并重新链接相关目标，Python 也可能再生成缓存；重建耗时未测量，当前已安装程序的运行文件仍保留。冻结实验期间不要为恢复对象文件而重建已冻结二进制。

历史两轮清理另计：第一轮含 Git 压缩回收 **1,082,970,112 B**，第二轮内容不变的 MJB 稀疏化及缓存清理回收 **855,945,216 B**，历史合计 **1,938,915,328 B（1.806 GiB）**，均不包含本轮 48.34 MiB。依据[第一轮回执](../evidence/disk_cleanup_20261008.json)和[第二轮回执](../evidence/disk_cleanup_round2_20261008.json)，不重复计入或重做旧清理。

工具源码为 [`audit_storage_readonly.py`](../runtime03/audit_storage_readonly.py)、[`audit_build_intermediates03.py`](../runtime03/audit_build_intermediates03.py) 与 [`cleanup_regenerable03.py`](../runtime03/cleanup_regenerable03.py)。旧白名单已经消费，不能重复执行；以后如需整理应重新审计、生成新白名单并 dry-run。源码包包含这些工具与本说明，完整执行证据保留在原工作区。
