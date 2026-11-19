# 公共记忆池存储与检索修正

## 行为

`shared_pool` 现在每个场景/space 只有一个物理 MemCube/Qdrant collection。所有作者、主体、类别共用此索引；这些字段继续作为卡片和 State 元数据保留。不同场景不混存。

写入的 policy 仅由 space 决定，不再由用户、类别或读者列表决定。公共库所有者是稳定的公共库标识，不受首个写入人或查询人影响。内部共享模式不再检查旧 AccessGraph 的分库权限。

检索调用一次原生 GeneralTextMemory.search(query, top_k)，保留原生返回顺序；subject_ids 不再筛掉其他人的记忆。共享路径不调用跨库名次合并，不分配个人配额，不用 State 插入替代证据，不扩展同源卡片，不由说明模型删减候选。时间和状态解释通过附注提供，不在公共检索后额外删除历史卡片。Sleep 一次维护整个空间的待处理状态。

这对齐 MemOS 原生基线的公共库和搜索方式。不同基线各自的抽取、更新和检索算法仍有差别，不声称所有系统实现完全相同。查询继续沿用当前 runner 的“提问人 + 问题”文本。

旧 governed 路径保留兼容；当前实验 shared_pool 路径采用上述逻辑。旧的按用户分库运行目录会明确报错，须在新目录重新回放消息构建公共索引。没有改写旧结果或自动迁移旧索引。

## 验证

- 新增 `tests/test_public_pool.py`：26 条跨作者/类别记忆只创建一个公共库；一次检索保留同一作者的两条最高排名证据；说明模型不选择也不丢弃；重启后仍从同一库查询；场景隔离；旧分库拒绝混用；Sleep 跨人维护一次且保留各自 State。
- 最终针对性测试：103 passed，覆盖 public_pool/shared_design/memos_adapter/multiuser/GateMem 前文与诊断/GroupMemBench。
- 全套测试中途检查：257 passed、5 failed、12 subtests passed。5 个失败均是旧 State/Sleep ScriptedModel 缺少修复调用响应；在隔离副本撤回此次 MemoryService 修改后，相关两组测试仍是同样 5 failed、55 passed、4 subtests passed。记录：`runtime/public-pool-baseline-check-result.txt`。没有为通过测试修改这些既有用例。
- 修改文件静态检查通过。
- 没有调用线上模型，也没有重跑正式数据集；测试验证结构与行为，不证明基准准确率提升。
