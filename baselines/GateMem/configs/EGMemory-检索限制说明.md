# EGMemory 在 GateMem 中的检索限制

GateMem 接入的 EGMemory 原生问答仍使用模型驱动的多轮工具检索，但每个问题共享启动器 `top_k` 指定的累计证据预算（默认 20 个不同的 EGMemory 记忆节点）。普通搜索、短语查找和当前决策工具共享同一个预算；跨轮重复节点只计算一次。日期工具只能读取已返回节点的日期，不得通过未返回节点绕过预算。下一问题重新计数。

预测输出记录 `egmemory_evidence_limit`、`egmemory_evidence_count`、`egmemory_evidence_ids` 和原生工具轨迹，供核对。这里的一条是 EGMemory 存储中的一个节点；不同系统的条目长度、聚合方式、最终 token 和工具调用次数仍可能不同。限定 20 个节点是对 EGMemory 原生读取逻辑的实验适配，不能称为未修改的原生读取结果。当前重排序器为 `noop`，也应在实验配置中披露。
