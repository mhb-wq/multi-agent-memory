# 2026-10-27 结构改动：一条消息一次调用直接产出卡片，卡片文本入索引

## 改了什么

之前每条消息走三层：

```
intake（切成 3–4 个主体片段）
  → 每个片段 memos_extract（MemOS 抽取成记忆条目）
  → 每个条目 cards（ENRICH 再写一张卡）
  → 卡片摘要入向量库（原文引用只存不索引）
```

现在 `card_mode="direct"` 走一层：

```
一条消息 → cards_direct（一次模型调用，返回 1–3 张卡）
  → 每张卡直接落库（主体、类别、引用、证据类型、时间、状态字段一次给全）
  → 卡片文本 = claim_text + "\n" + source_quote 入向量库
```

- 所有数据集共用同一实现（`src/ours_memory/multiuser.py::_cards_direct` +
  `src/ours_memory/service.py::_store_direct_card`），
  GroupMemBench 的 `groupmem.build_memory`、GateMem 的 `build_ours_memory`、
  PERMA（复用 `groupmem.build_memory`）都设置 `card_mode="direct"`，三处一致。
- 向量 id 使用 `uuid5(event_id|claim)`：MemOS 的 `TextualMemoryItem.id` 必须是合法 UUID，
  之前用 `cardmem_...` 会被 pydantic 拒绝、导致 0 条向量写入（已修）。
- 默认仍是 `card_mode="extract"`，旧路径与既有测试不变；`Settings.card_mode` 只有
  `extract|direct` 两个取值。

## 为什么

1. **成本**：一条消息从 8.3 次模型调用降到约 1 次（Sleep 另算）；实测同一场景
   （42 条消息）从 586 秒降到 138 秒，约 4 倍。
2. **检索保真**：索引文本变成“摘要 + 原话”，问题里的原词、数字、专有名词可以直接命中；
   之前只索引 MemOS 摘要。
3. **一致性**：存什么就索引什么，和 Mem0/MemOS/A-MEM 的做法同构。

## 代价与边界

- 卡片数量仍由模型决定（1–3 张/条），本轮实测约 2–3 张/条；如果后续要压缩，
  需要再做“同事件合并/最小信息量过滤”。
- 一张卡对应一个事件，事件文本就是这张卡的引用片段；删除仍以事件/卡片为单位，
  “值级删除”尚未实现。
- 默认 `extract` 保留给单元测试与旧配置；正式实验的构建函数显式改为 `direct`。

## 验证（同一批 10 场景、同一模型、同一判分）

| 指标 | extract（旧） | direct（新） |
| --- | ---: | ---: |
| 每条消息模型调用（样本） | 8.3 | 1.25 |
| GroupMemBench 平均耗时/场景 | 1,323 s | 334 s |
| 卡片总数 / 每消息 | 3,698 / 3.30 | 3,042 / 2.71 |
| GroupMemBench 正确 | 3/10 | 4/10 |
| 证据进候选 | 7/10 | 7/10 |
| 证据被选中并返回 | 6/10 | 5/10 |
| GateMem action / utility / over-refusal | .444 / 0 / 1.0 | **.556 / .25 / .75** |
| GateMem 平均 episode 耗时 | 351–899 s | 106–273 s |
| PERMA 正确 / 总耗时 | 8/10 / 3,522 s | 7/10 / 1,060 s |

结论：**成本与卡片膨胀按预期下降，正确率变化在 10 题样本的噪声内，证据链没有改善**。
原因不是索引文本，而是候选池仍为 20 张、卡片仍有 300+ 张，且筛选器会漏掉证据卡；
下一步应把 `candidate_limit` 提到 40，并把 `select_empty`/`select_fallback` 纳入指标。

- 单元测试：`tests/test_diag_fixes.py::test_direct_card_mode_one_call_and_indexes_quote`
  一条消息只发一次 `cards_direct`，不发 `intake`/`cards`；两张卡落库；
  向量库文本等于 `claim + "\n" + quote`。
- 核心测试 136 项通过。
- 真实场景：42 条消息 42/42 写入成功、检索返回 9 张卡（修复 UUID 之前为 0）。
