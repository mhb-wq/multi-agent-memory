# GateMem 正式实验配置（审阅草稿）

## 数据集规模

使用 GateMem 数据集，共 **91 个场景、2,218 个检查点**，包含 20,293 条对话消息。

| 领域 | 场景数 | 检查点数 |
| --- | ---: | ---: |
| 医疗 | 21 | 579 |
| 办公 | 17 | 547 |
| 教育 | 30 | 540 |
| 家庭 | 23 | 552 |
| **合计** | **91** | **2,218** |

## 记忆系统

我们的记忆系统，以及九个基线：NaiveRAG、BM25、Mem0、A-MEM、MemGPT、LightMem、MemoryBank、DG-Mem、EGMemory。

## 向量模型

统一使用 `sentence-transformers/all-MiniLM-L6-v2`，向量维度为 384。BM25 使用词法检索。

## 检索数量

统一设置 `top_k=20`。EGMemory 的多轮检索累计最多返回 20 个不同的记忆节点。

## 大语言模型

分别使用以下三个模型开展实验：

- DeepSeek V4.1 Flash
- GPT-5.6-sol
- GLM-5.3

每组实验中，记忆系统内部的模型调用、最终回答和 Judge 均使用该组模型。最终回答使用 GateMem 官方统一提示词。三个模型通过同一个中转站使用：`https://api.ciyuan.fast/v1`。

## Judge 与结果指标

每个系统在每个模型组下，对每个检查点生成的答案，使用 GateMem 官方 Judge 提示词**独立评判三次**；三次结果分别保存，汇总方式待确定。

最终报告与 GateMem 官方指标保持一致：Utility（U）、Access-Control Violation（A）、Active-Forgetting Failure（F）、Over-Refusal（OR）和 Memory Governance Score（MGS）。其中，`MGS = U × (1 − A) × (1 − F)`。
