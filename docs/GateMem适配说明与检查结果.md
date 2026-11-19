# 九个基线的 GateMem 适配说明

更新时间：2026-11-19。当前状态为**适配代码、离线检查和 DeepSeek 单点链路检查已完成，全量正式实验尚未完成**。单点检查不能证明完整复现；用户后续明确不进行试运行，之后未再做模型测试。

## 什么是适配

GateMem 给每个系统同一段按时间发生的多人对话，并在其中一些时点提问。每套系统原本接收的输入和输出格式不同，因此需要一层连接代码：开始新案例时清空记忆；逐条送入当时已经发生的对话、身份和角色；到检查点时用提问者和问题检索记忆；调用回答模型；将回答整理为 GateMem 的四种动作及文本，并记录送入模型的记忆。GateMem 随后才能用相同标准计算实用性、越权泄露和遗忘失败。

适配不是替基线增加访问控制或强制删除能力。对没有这些能力的系统，仅把对话中可见的删除请求当作普通输入，不使用隐藏的删除标签直接筛掉记忆，才能反映该系统在此设定下的表现。系统原本有的机制及适配选择需要分别报告。适配也不是把未来对话或标准答案提供给模型：新增适配器明确屏蔽检查点隐藏字段，并测试了未来消息不会进入检索。

## 逐项检查与适配状态

| 基线 | GateMem 参数 | 使用的实现 | 现阶段验证与限制 |
| --- | --- | --- | --- |
| NaiveRAG | `rag_naive` | GateMem 原有实现 | 测试了增量写入与检索；正式运行须固定 `embedding` 或 `tfidf`。类本身默认 TF-IDF，但官方命令行默认 embedding，应显式传参。 |
| BM25 | `bm25` | DG-Mem 中的独立 BM25 检索器 | 已测试真实 GateMem 样本、未来隔离、案例重置。每个检查点从已看到的消息重建词法索引。 |
| Mem0 | `mem0` | GateMem 既有 builtin/upstream | builtin 已用模拟模型和向量完成写入及检索检查；真实提供商未测。须明确指定后端。stub 模式下 Mem0 不生成记忆。上游实现对部分异常采用 warning 后继续，正式运行要检查日志。 |
| A-MEM | `a_mem` | GateMem 的 A-MEM 风格适配实现 | 已测试写入检索，不等同官方原版。类默认 heuristic，GateMem 命令行默认 llm；llm 失败会回退 heuristic，应监控警告。 |
| MemGPT | `memgpt` | MemSyco-Bench 的轻量重实现 | 已改为每个 episode 持续保留核心与归档状态，只处理新增消息。已使用 DeepSeek 做单点连通性测试；尚需检查完整长对话行为。 |
| LightMem | `lightmem` | MemSyco-Bench 适配层及 vendored LightMem | 已改为保留同一个 episode 的记忆存储并仅处理新增消息。两轮输入的 DeepSeek 单点流程已检索到 3 条记忆，Windows 临时数据库关闭顺序已修复；仍需更广泛验证。上游图记忆未实现，本地显式关闭；每次检查点执行 offline update。 |
| MemoryBank | `memorybank` | MemSyco-Bench 适配层及其 vendored 辅助代码 | 已改为累积新增对话和分批摘要、更新整体摘要；这是一种明确的多人增量适配，而非原始 MemoryBank 的完整复现。新版已完成两轮输入的真实单点接口检查，多检查点持续状态仍仅有模拟测试。 |
| DG-Mem | `dg_mem` | 匿名仓库双图检索器 | 已安装 spaCy 和参考重排模型并在 DeepSeek 下完成单点链路；每次有新消息后重建图。当前运行环境为 Python 3.10，而上游 README 要求 3.11+，需单独评估环境差异。 |
| EGMemory | `egmemory` | 匿名仓库 GroupMemory | 已增加原生工具调用回答路径并记录工具返回内容；模拟及 DeepSeek 单点链路测试通过。临时 JSONL 隔离每个案例；本地统一 MiniLM 向量、无重排，与其参考配置不同。 |

MemGPT、LightMem 和 MemoryBank 的适配已改为每个 episode 持续保留状态。LightMem 的 offline update 发生在每个检查点；MemoryBank 使用按新增对话批次的摘要和整体摘要，都是明确的适配选择，不等于完整复现论文。EGMemory 原生问答的格式归一和工具轨迹已实现。真实模型的单点测试不能代替长对话、跨用户授权和删除的验证。

## 文件与运行条件

- 新适配器：`GateMem/bench/agents/external_baselines.py`；注册位置：`GateMem/bench/agents/__init__.py`。
- 测试：从 `GateMem/` 执行 `python -m unittest discover -s tests -p 'test_external_baselines.py' -v`。
- EGMemory 配置示例：`GateMem/configs/egmemory.example.json`，将模型名改为真实名称，设置环境变量 `GATEMEM_EGMEMORY_CONFIG` 指向配置文件，并通过环境变量配置 OpenAI 兼容服务的凭据和地址。示例的 hashing/noop 是轻量配置，非论文参考配置；配置文件中不要保存密钥。
- DG-Mem 默认参考组件，依赖 Python 3.11+、spaCy `en_core_web_sm` 和相应模型；`GATEMEM_DGMEM_MODE=smoke` 是仅供接口测试的降级组件，不用于正式结果。
- MemGPT、LightMem、MemoryBank 依照 `MemSyco-Bench/baselines/configs/` 使用 `MEMORY_*` 环境变量及其上游依赖。LightMem 本地适配层现在默认选择可用的 `cuda:0` 或 CPU，环境变量可显式覆盖；安装依赖时建议使用独立环境，不要直接混装。
- 上游 GateMem 的 stub 回答会读取 `expected_action`，只能用于检查流程，不能用于评估分数。新增适配器即使在 stub 测试时也屏蔽该字段；GateMem 内置三项原有 stub 行为不变。
- 正式运行要固定响应模型、记忆写入模型、嵌入模型、检索数量、提示词和随机参数等，记录差异、模型调用及成本，且先使用单案例串行运行。GateMem 官方的统一提示含领域策略信息，适用于此适配协议，但不能将其误认为每个基线原本自带的权限机制。

## 检查结果

下载时两个匿名 ZIP 内所有文件与解压后的内容逐一 SHA-256 一致。之后为兼容中转站，将 EGMemory 本地副本 `groupmem/llms/openai_compat.py` 的 HTTP User-Agent 改为研究客户端标识；原 ZIP 未修改。MemSyco-Bench Git 对象连通性检查通过；未发现所检查目录中的 Git LFS 指针文件。GateMem、MemSyco-Bench 和 EvoMemState 的版本哈希见 `基线清单与获取记录.md`。九项都找到了相应代码，但 A-MEM、MemGPT、MemoryBank 等的来源与原作者完整仓库不完全相同，不能简单称为九套官方原版。代码的存在或可导入不代表权重、服务、环境依赖及论文复现条件已经齐备。

对 DG-Mem、EGMemory Python 源码做语法编译检查通过；MemSyco-Bench 原版 baselines 编译检查发现上述 LightMem `graph.py` 错误。本地修补后编译检查通过：默认不开图记忆，主动开启会明确报“未实现”，并未伪造图功能。如实验需要图记忆，必须先取得完整实现并验证。已另外拉取未修改的官方仓库 `LightMem官方源码-浅克隆/`，提交 `80211869f55e9c2276a7e7f564356d6cb03b7d8c`；该版本的相同文件也为空类定义。`LightMem官方源码-未完成/` 和 `LightMem官方源码-未完成.zip` 是中断下载的残留，不是完整代码。

2026-11-19 针对 LightMem 补充的五项测试覆盖默认配置、图功能显式报错、设备选择、写入/更新/检索的模拟调用。还在离线模式下使用本机缓存的 LLMLingua-2 模型，通过 `cuda:0` 成功加载并压缩了一段示例文本；随后在 DeepSeek 组完成了小规模接口验证。现在已在本地配置中转站凭据，但模型实际返回名称仍待确认；本机已安装的依赖版本与上游锁定版本不完全一致。其他模型的缓存目录存在不代表权重完整或模型可用。

GateMem 现有四个领域的数据中，所有对话 turn 的 `record_refs` 和 `memory_ops` 均为空。GateMem 的 A-MEM 实现会在这些字段非空时使用它们，因此将来替换或扩展数据时需要重新检查这些字段是否属于系统可以看到的信息。
