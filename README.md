# 多人共享对话记忆实验

本仓库保存 GateMem 数据集、九个实验基线及其 GateMem 接入代码，以及我们的 Card–State–Sleep 多用户记忆系统。当前是实验代码，不代表正式实验已经完成或所有上游系统均已完整复现。

## 目录

| 路径 | 内容 |
| --- | --- |
| [`baselines/GateMem/bench/data`](baselines/GateMem/bench/data) | GateMem 完整数据：medical、office、education、household，共 91 个场景、2,218 个检查点 |
| [`baselines/GateMem`](baselines/GateMem) | GateMem 评测器、NaiveRAG、Mem0、A-MEM 和九种基线的适配入口 |
| [`baselines/DG-Mem`](baselines/DG-Mem) | DG-Mem 匿名仓库快照；BM25 检索器也由此提供 |
| [`baselines/EGMemory`](baselines/EGMemory) | EGMemory 匿名仓库快照 |
| [`baselines/MemSyco-Bench/baselines`](baselines/MemSyco-Bench/baselines) | MemGPT、LightMem 和 MemoryBank 等基线实现及其 vendored 组件 |
| [`baselines/LightMem-upstream`](baselines/LightMem-upstream) | LightMem 官方源码快照，供对照；实验使用的实现见上一行 |
| [`baselines/EvoMemState`](baselines/EvoMemState) | 用户指定的参考项目 |
| [`ours`](ours) | 我们的记忆系统、GateMem 接口、配置、测试和固定版本的 MemOS 依赖源码 |
| [`docs`](docs) | 基线来源、适配记录和历史实验准备说明 |

九个基线的名称为 NaiveRAG、BM25、Mem0、A-MEM、MemGPT、LightMem、MemoryBank、DG-Mem、EGMemory。入口在 [`external_baselines.py`](baselines/GateMem/bench/agents/external_baselines.py) 和 GateMem 内置 agent。A-MEM、MemGPT、MemoryBank、LightMem 等存在 benchmark 适配或轻量实现，具体差异见[适配说明](docs/GateMem适配说明与检查结果.md)，不能直接声称均为论文官方完整复现。

EGMemory 的多轮原生读取在本接入中每题共享 `top_k` 条不同记忆节点的预算，默认 20；它仍可能与其他系统的条目长度和 token 不同。当前配置使用 MiniLM 向量及 `noop` 重排序器。详见[EGMemory 检索限制](baselines/GateMem/configs/EGMemory-检索限制说明.md)。

## 环境与实验

按各目录的依赖说明创建独立 Python 环境。我们的方法与状态维护说明见 [`ours/README.md`](ours/README.md) 和 [`ours/docs/联合判断校验与恢复.md`](ours/docs/联合判断校验与恢复.md)。九个基线的 GateMem 适配测试可在 `baselines/GateMem` 下运行：

```powershell
python -m unittest discover -s tests -p 'test_external_baselines.py' -v
python -m unittest discover -s tests -p 'test_egmemory_budget.py' -v
```

正式运行需要独立配置 LLM、中转地址和 `all-MiniLM-L6-v2` 向量模型；DG-Mem 参考配置还需要其指定的重排序模型。密钥通过环境变量或忽略的本地 `.env` 文件提供，不能提交。模型权重、虚拟环境、缓存及历史实验输出不包含在本仓库。实际模型名、服务分组、裁判配置和实验范围应在启动前核验并记录；上述命令只是离线代码检查。

## 来源和许可

上游来源和固定版本见[来源记录](docs/基线清单与获取记录.md)及各子目录；GateMem、MemSyco-Bench、LightMem 和 MemOS 等保留其随附许可证和版权声明。DG-Mem 和 EGMemory 匿名快照中未找到明确的许可证文件，其可复用与再分发条件需要向权利人确认，不能将它们视为自动采用其他目录的许可证。`docs` 中旧日期文件保留为历史记录，当前行为以代码和上述专项说明为准。
