# 实验数据集调研：多用户、多 Agent 的 Card—State—Sleep

日期：2026-10-26。对应实现：`configs/multiuser.json`、`src/ours_memory/{multiuser,state,control,governance}.py`。

## 0. 先确认我们要验证什么

当前系统做的事（读代码后的结论）：

| 环节 | 实现事实 | 实验要问的问题 |
| --- | --- | --- |
| 写入 | `/messages` 由 LLM 从原文抽取 `subject/category/quote`；`/events` 由可信 runner 直接给定主体和类别 | Card 抽取的主体、类别、原文片段是否可靠？说话人≠主体时会不会记错人？ |
| Card | 带 `author_id/subject/category/item_id/source_message_id/evidence_kind/event_time/valid_from/valid_to`，同一事项可写多人的卡片，来源可级联 | 来源链、多用户副本、证据类型（转述 vs 确认）是否经得起对抗问法？ |
| State | 写卡片只排队；只有 `/sleep` 更新 State；操作 `INIT/CONFIRM/UPDATE/CONFLICT/INVALIDATE/IGNORE`；区间前闭后开、保留历史、冲突不强行合并 | 事实更新后，"当前值"是否正确、"历史值"是否还能查、冲突是否被保留而不是编一个答案？ |
| 读 | 权限（请求者∩Agent，主体×类别×任务×时间）→ 授权域检索 → 内部筛选 LLM 用 State 选卡片 → 证据包 | State 过滤是否提升答案且不引入泄漏？旧 State 会不会错误否决新证据？ |
| 交接/删除 | 交接生成接收者专用包；删除先阻断再清理，按来源级联 | 跨 Agent 交接后接收者是否只看到自己有权的内容？删除/撤权后旧包是否失效？ |

现有验证只有工程测试（140 passed、合成模型回复、哈希向量），README 明确写了"尚未取得真实模型或公开数据集分数"。所以下一步实验要补的是：**真实对话数据上的答案质量 + 状态正确性 + 权限正确性**，而不是再补工程测试。

**三条必须被数据支撑的主张**：

1. 多用户共享记忆时，按"主体×类别×Agent×时间"隔离/共享是可行的（既不漏也不过度保守）；
2. State 的"只在 Sleep 更新 + 保留历史 + 冲突显式化"能减少**旧状态污染新证据**和**把计划/转述当既成事实**两类错误；
3. 检索前的权限过滤 + 交接时的二次检查，能把**越权证据**挡在提示词之外，而不只是靠最后答案复核。

---

## 1. 选型判据

同时满足越多越好：

1. **多主体**：至少有 "A 说 B 的事"，能构造 `subject ≠ author`；
2. **时间演化**：同一事实在后续会话被修改、推翻、或产生冲突，能测 `UPDATE/INVALIDATE/CONFLICT`；
3. **有 gold 证据**：标到 session / dialog / message 级，才能算证据命中率、构造"故意不给某人的证据"来测泄漏；
4. **能构造权限**：主体、类别、Agent、任务、时限五个维度能映射到数据里的字段；
5. **规模可承受**：我们的写入链路每条消息要过 MemOS 抽取 + embedding，几万条消息的 LLM 成本要可控。

---

## 2. 推荐的数据集（三个，互补）

### 2.1 GroupMemBench —— 多用户、多频道（首选）

- 出处：UC Santa Barbara + Microsoft，2026-05。[论文 2605.14498](https://arxiv.org/abs/2605.14498)；[数据 kimperyang/GroupMemBench](https://huggingface.co/datasets/kimperyang/GroupMemBench)；[代码与题目 UCSB-NLP-Chang/GroupMemBench](https://github.com/UCSB-NLP-Chang/GroupMemBench)。
- 检测什么：群组对话里的记忆——多作者、多频道、多阶段，题目绑定了**提问人**（`asking_user_id`），六类问题各打一个失败模式：`multi_hop`、`knowledge_update`、`temporal`、`user_implicit`、`term_ambiguity`、`abstention`。
- 场景：模拟企业协作频道（类 Teams），四个域：Finance、Technology、**Healthcare**、Manufacturing。Healthcare 域有 10 个频道、3 万条消息、6 个用户，角色是 Nurse Leader / Health IT Specialist / Clinical Project Manager / Quality Improvement Coordinator，围绕 EHR 实施、用药核对、感染控制、远程医疗扩容等项目推进；消息带 `phase_name/topic/reply_to/is_noise/is_decision_point`，决策点还带 `decision_label/decision_change_metadata`。
- 规模：Healthcare 题目 145 道（multi_hop 48、abstention 34、temporal 37、knowledge_update 17、term_ambiguity 7、user_implicit 2）；每条消息一个 JSON 对象，频道名 → 按时间排序的消息数组。
- 论文结果：最强记忆系统平均只有 46.0%，**knowledge_update 仅 27.1%**、term_ambiguity 37.7%——正好是我们要打的点。

**真实场景（Healthcare 域原样摘录）**

1. 状态被推翻又改回（knowledge_update_1，提问人 User_3）：
   - 问：`How is the team currently handling credential type and expiration storage between HRIS and LMS?`
   - 答：`Capture credential type in HRIS now but store expiration in the LMS credential record (synced via nightly job) until the HRIS schema update is available.`
   - 证据链里团队先说"Enforce upstream in HRIS"（2025-07-13），两天后改口"LMS-only hosting creates cleaner audit trails; HRIS will fragment evidence … confirm LMS as system of record by Wednesday"（2025-07-15）。回答用旧决策就是错的。
2. 时间题（temporal_1，提问人 User_1）：`What is the exact datetime (YYYY-MM-DD HH:MM) by which staffing changes need to be confirmed?` → `2025-08-05 12:00`。
3. 跨消息推理（multi_hop_2，提问人 User_3）：`When is the 30-minute handoff involving ITBuild and ClinicalInformatics scheduled to take place?` → `2025-07-17`。
4. 依赖提问人身份（user_implicit_1，提问人 User_2）：`What is the open item I still need to resolve for the Telehealth Platform Infrastructure requirements?` → `whether after-hours coverage changes the remote support model`。同一个问题换个人问，答案不同——天然对应我们的"请求者身份 + 主体绑定"。
5. 弃答（abstention_1，提问人 User_1）：`Who is responsible for making the final decision on whether documentation shifts remain in scope or move to training-only?` → 语料里没有，正确答案是拒答。

**为什么最贴合我们**

- `channel` → 协作空间/Agent 职责；`author` → principal；`role` → 业务 Agent（clinical/logistics）；
- `reply_to`/`topic` → Card 的 `sources`/`item_id`；`is_decision_point` + 决策元数据 → State 的 `UPDATE/CONFLICT`；
- `asking_user_id` → 请求者身份，可直接做"同一问题不同请求者"的隔离实验；
- `is_noise` → 干扰消息，可测检索噪声下的 State 稳定性；
- Healthcare 域的角色分工与我们的 care 场景（临床/协调/物流）几乎一一对应。

**局限**：没有权限/可见性标签（要自己造 grant）；不是患者病历对话，而是医疗机构的项目协作；单域 3 万条消息，全量跑我们这条 LLM 抽取链路成本偏高（建议先取 1–2 个 channel 或 1–2 个 phase）。

---

### 2.2 LongMemEval —— 状态更新与弃答（State 生命周期）

- 出处：ICLR 2025，[论文 2410.10813](https://arxiv.org/abs/2410.10813)；[代码 xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)；数据 [longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)（MIT）。
- 检测什么：聊天助手的五项长期记忆能力——信息抽取、多会话推理、**知识更新**、**时间推理**、**弃答**。500 道题，每题带一段可扩展的用户—助手历史，历史里混入大量干扰会话（"干草堆"式设计）。
- 场景：单人长期使用助手的生活流：健身/跑步、买车保养、旅行、做饭、游戏目标、工作变动……会话按时间戳串起来，同一件事会在不同会话里被更新。
- 规模：`longmemeval_oracle.json`（只保留证据会话，约 15 MB，适合先跑通）；`longmemeval_s_cleaned.json`（约 115k tokens/实例，约 40 个会话，约 278 MB）；`longmemeval_m_cleaned.json`（约 500 个会话/实例，2.7 GB）。题型：temporal-reasoning 133、multi-session 133、knowledge-update 78、single-session-user 70、single-session-assistant 56、single-session-preference 30；另有 30 道弃答题（`question_id` 以 `_abs` 结尾）。
- 论文结论：商用助手/长上下文模型在跨会话记忆上平均掉约 30 个点。

**真实场景（oracle 文件原样摘录）**

1. 目标被更新（knowledge-update）：`What was my previous goal for my Apex Legends level before I updated my goal?` → `level 100`。证据是 2023-06-16 说 "determined to reach level 100"，2023-09-30 改口 "my current goal is to reach level 150"。这正好对应我们 State 的 `UPDATE` + 历史区间：当前值 150、历史值 100 都要能答。
2. 计数型状态更新：`How many Korean restaurants have I tried in my city?` → `four`（前一会话说 3 家，后一会话变 4 家）。
3. 时间推理：`What was the first issue I had with my new car after its first service?` → `GPS system not functioning correctly`（需要按会话时间排序取"第一次"）。
4. 弃答（`_abs`）：`When did I book the Airbnb in Sacramento?` → `The information provided is not enough. You only mentioned booking Airbnb in San Francisco.` 这正是我们 prompts 里"问题本身不是证据、不许顺着用户的猜测确认"的考点。
5. 偏好：`Can you recommend some resources where I can learn more about video editing?` → 期望答案要求"专门推荐 Adobe Premiere Pro 的资源"，因为历史里用户说过自己的偏好。

**为什么贴合**：`knowledge-update` 78 题 + `_abs` 30 题直接对应 State 的更新/失效/待确认与弃答；`answer_session_ids` 是 gold 证据，可精确算证据命中率，也可按 session 做权限掩码（"这段历史对当前 Agent 不可见"），测"证据被拿走后是拒答还是编"。

**局限**：单用户，没有天然的多人共享；要自己把 session 分配给不同 principal/agent 来造多用户场景（这也是最容易控制变量的做法）。

---

### 2.3 LoCoMo —— 长时程、时间与来源归因

- 出处：ACL 2024，[论文 2402.17753](https://arxiv.org/abs/2402.17753)；[代码与数据 snap-research/locomo](https://github.com/snap-research/locomo)（`data/locomo10.json`）。
- 检测什么：超长期对话记忆——问答（单跳/多跳/时间/开放域/**对抗**）、事件摘要、多模态对话生成。每段对话由两个 LLM 角色按 persona 和事件图生成，再人工校订。
- 场景：两个朋友（Caroline & Melanie、Jon & Gina、John & Maria……）跨 19–32 个会话、从 2023 年 5 月聊到 2024 年 1 月，内容涵盖家人、工作、旅行、健身、收养、公益跑等；每个会话有真实时间戳，每个说话人有独立的事件摘要。
- 规模（当前仓库 10 段对话，是原 50 段里挑的最长、标注最好的子集）：1,986 道 QA —— 单跳 841、对抗 446、时间 321、多跳 282、开放域 96；每题带 `evidence`（如 `D11:1`，即第 11 个会话第 1 轮）。
- 对抗题的设计：题干故意张冠李戴，正确答案是"这件事不是问的那个人做的"。

**真实场景（conv-26，Caroline & Melanie）**

1. 主体≠说话人：`When is Melanie's daughter's birthday?` → `13 August`；证据 D11:1 是 Melanie 自己说 "We celebrated my daughter's birthday with a concert…"。卡片的主体是"女儿"，作者是 Melanie——正是我们 `ingest` 里 `author_id` 与 `subject` 分离的考点。
2. 跨会话多跳：`What is Caroline's relationship status?` → `Single`；证据 D3:13 + D2:14（两个不同会话）。
3. 时间题：`When did Caroline go to the LGBTQ support group?` → `7 May 2023`，而所在会话时间是 `8 May 2023`——需要区分"事件时间"和"记录时间"，对应我们 Card 的 `event_time` 与 `recorded_at`。
4. 对抗（来源归因）：`What did Caroline realize after her charity race?`（证据 D2:3）——实际是 Melanie 跑完公益跑后意识到"self-care is important"，题干把事安到 Caroline 头上；标准答案要拒绝这个前提。我们的 `ANSWER`/`OUTPUT_CHECK` 提示里写了"问题不是证据、不许确认用户的猜测"，这题就是它的天然测试。
5. 第三方信息：`How long have Mel and her husband been married?` → 5 年（证据 D3:16），信息由 Melanie 提供，但问的是她丈夫。

**为什么贴合**：长时程 + 会话时间戳 + 每题 gold dialog id，可同时测时间有效性、来源归因、多跳；两人的共享事件可以用来构造"共同来源副本"和"仅一方可见"的权限变体；对抗题能直接量化"把 A 的事说成 B 的"这类来源错误。

**局限**：只有两个说话人，没有天然的多 Agent/权限结构；图像只给 URL 和 caption（不发布原图）；对抗题的判分需要按"拒绝前提"单独统计，不能并进普通准确率。

---

## 3. 三个数据集怎么分工

| 我们想证明的 | GroupMemBench | LongMemEval | LoCoMo |
| --- | --- | --- | --- |
| 多用户/多 Agent 共享与隔离 | ★★★ 频道×角色×提问人 | ★ 可按 session 造 | ★★ 两人+第三方 |
| State 更新/冲突/历史 | ★★★ 决策点+knowledge_update | ★★★ knowledge-update 78 题 | ★★ 目标/关系随时间变 |
| 弃答与不确定性 | ★★ abstention 34 题 | ★★★ `_abs` 30 题 | ★★ 对抗题 |
| 时间推理 | ★★ temporal 37 题 | ★★★ 133 题 | ★★★ 321 题 |
| 来源/说话人归因 | ★★ reply_to/role | —（助手语料） | ★★★ 主体≠说话人 |
| 权限泄漏可控实验 | ★★★ 频道+角色可切 | ★★★ session 可掩码 | ★★ 两人可分 |
| 单条消息成本 | 高（3 万条/域） | 中（每实例几十会话） | 低（10 段×约 600 轮） |

建议的主实验：**GroupMemBench(Healthcare) 做主场**（多用户+状态更新+弃答齐全），**LongMemEval 做 State 生命周期的受控实验**（更新/历史/弃答，变量最干净），**LoCoMo 做长时程与来源归因**（对抗题是现成的"问题不是证据"测试集）。

---

## 4. 建议的实验矩阵

| 实验 | 数据 | 对照组 | 指标 |
| --- | --- | --- | --- |
| E1 状态更新 | LongMemEval knowledge-update + `_abs` | 全上下文、BM25 RAG、向量 RAG、MemOS 原生、我们（无 Sleep / 无 State 过滤 / 完整） | 答案准确率（LLM judge）、**旧值错误率**、弃答正确率、证据 session P/R |
| E2 多用户隔离 | GroupMemBench Healthcare，按频道/角色造 grant | 无权限（全可见）、仅按 Agent 过滤、完整（主体×类别×任务×时间） | 答案准确率、**越权证据率**（进入提示词的无权卡片占比）、该拒答时的拒答率、token 成本 |
| E3 长时程与归因 | LoCoMo 10 段 | 同上 + 把 `author` 当 `subject` 的消融 | 分类准确率（单跳/多跳/时间/开放域/对抗）、证据 dialog 命中率、**错误归因率** |
| E4 交接与撤权 | 任一数据集子集 | 无二次检查、有二次检查 | 接收者包中的越权卡片数、撤权后旧包失效比例 |
| E5 删除 | GroupMemBench 多用户副本 | 仅删主副本、按来源级联 | 删除后残留可读副本数、重放是否被拒 |

统一口径：`current` / `history` 两种问法（我们已支持 `purpose`）；同一批题在"有权限/无权限"两种设置下各跑一遍，分别报 utility 和 leakage。

---

## 5. 接入要点（按现有代码）

**字段映射**

| 数据字段 | 我们的字段 | 说明 |
| --- | --- | --- |
| `author` / `speaker` | `author_id` + 事件 `principal_id` | 写入者身份，由 runner 令牌绑定 |
| 消息主体 | `subject_id` | 多数数据要推断；`/events` 可直接给，`/messages` 走 LLM 抽取 |
| 频道/角色/话题 | `category`（`clinical/logistics/contact`） | 需扩 `control.categories` |
| `topic` / `phase_name` / `item_id` | `item_id` | 同一事项跨消息聚合 |
| `reply_to` / 前序对话 | `sources` | 派生来源链、级联删除 |
| `timestamp` | `recorded_at` + `observed_at` | 数据集时间要显式传入，不能用导入时间 |
| 决策点/更新 | State 的 `valid_from/valid_to` | 决策变化的时间点即 `valid_from` |
| 可见性 | `grant`（`/control/grants`） | 五维：subject/category/task/agent/时间窗 |

**必须先补的两处**

1. `configs/multiuser.json` 的 `state_keys` 目前只有 `appointment`(text) 和 `budget`(number/CNY)。LongMemEval/GroupMemBench 的更新题需要 `goal`、`preference`、`deadline`、`owner`、`status` 之类的键，否则 knowledge-update 落不到 State 上，只能当普通 Card 检索。
2. `categories` 目前固定三类。GroupMemBench 的角色/话题更细，建议扩到 5–8 类，或按域建独立配置，避免一张卡跨类混装。

**成本控制**

- 全量 3 万条消息过 LLM 抽取不现实。先用 `/events`（runner 直接给 subject/category），只把 `source_quote` 校验留给程序；这样每组实验的成本主要是 embedding + 检索。
- LongMemEval 先跑 `oracle`（每题 1–3 个会话），跑通后再上 `_s_cleaned` 的子集（50–100 题）。
- 每题固定注入的干扰会话数、候选上限（`candidate_limit=20`）要写进实验配置，否则各组不可比。

**权限怎么造（关键）**

三份数据都没有权限标签，推荐统一用"证据掩码 + 授权矩阵"：

1. 从 gold 证据出发，把每条证据 session/message 分配给一个 `subject`；
2. 生成 2–4 个 principal 和 2 个业务 Agent（clinical/logistics），给不同的 `category` 和任务；
3. 正向组：请求者有覆盖该证据的 grant；负向组：故意不给（或给到期的 grant），正确答案应是"证据不足/无权"，而不是编造；
4. 泄漏率 = 进入模型上下文的无权卡片数 / 全部上下文卡片数；同时记录最终答案里是否出现无权信息（两道口径）。

---

## 6. 备选与"很贴合但暂时拿不到"

| 数据集 | 一句话 | 现状 |
| --- | --- | --- |
| **MUMBench**（AIM，[2609.12320](https://arxiv.org/abs/2609.12320)） | 672 条多用户交互，每条带 `CREATE/READ/UPDATE/DELETE/NOOP` 期望操作 + `PRIVATE/PUBLIC` 可见性，4 个域（Coding/Customer Support/Education/Travel） | **最贴合我们的 Card+权限+删除**，但截至 2026-10 未找到公开下载（HF/GitHub 均无）；论文附录有 schema 和生成提示，可按同构自建 |
| CoMemBench（[2609.32192](https://arxiv.org/abs/2609.32192)） | 800 条多 Agent 工作流，测"该共享的共享、该隔离的隔离"、artifact 交接 | 匿名 HF 仓库 [comembench/anonymous-submission](https://huggingface.co/datasets/comembench/anonymous-submission)，正式发布待定 |
| MEMTRACK（[2510.01353](https://arxiv.org/abs/2510.01353)） | Slack/Linear/Git 多平台、异步冲突、状态跟踪，带 Correctness/Efficiency/Redundancy 指标 | 数据在 Google Drive（论文脚注），可下 |
| PerLTQA（[2402.16288](https://arxiv.org/abs/2402.16288)） | 30 个人物、8,593 问，含 profile/社会关系/事件/对话；中英双语 | [GitHub](https://github.com/Elvin-Yiming-Du/PerLTQA) 可下；适合测"联系人类别 + 主体≠说话人" |
| H2HMem（[2606.09461](https://arxiv.org/abs/2606.09461)） | 人—人多方对话（含临床记录场景）、多模态、异步冲突 | 发布状态需再确认；与 care 场景最像 |
| AgentLeak（[2602.11510](https://arxiv.org/abs/2602.11510)） | 1,000 个多 Agent 隐私泄漏场景（医疗/金融/法律/企业），量内部通道泄漏 | 未见公开下载；可作为泄漏指标的参照设计 |
| MedAgentBench | 300 个 FHIR 临床任务，测 Agent 读写病历 | 偏工具执行，不测长期记忆；可作为"clinical 类工具观察"的补充 |

---

## 7. 参考链接

- GroupMemBench：[论文](https://arxiv.org/abs/2605.14498) · [数据](https://huggingface.co/datasets/kimperyang/GroupMemBench) · [代码/题目](https://github.com/UCSB-NLP-Chang/GroupMemBench)
- LongMemEval：[论文](https://arxiv.org/abs/2410.10813) · [代码](https://github.com/xiaowu0162/LongMemEval) · [数据](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
- LoCoMo：[论文](https://arxiv.org/abs/2402.17753) · [代码/数据](https://github.com/snap-research/locomo)
- 参考系统：MemOS（`vendor/MemOS`，固定提交见 `upstream.lock.json`）、Collaborative Memory（[2505.18279](https://arxiv.org/abs/2505.18279)）

> 许可提醒：LongMemEval 为 MIT；LoCoMo 仓库有 License 链接但未在 README 注明；GroupMemBench 为合成数据、仓库未写 License。投稿前需逐个确认再分发条款。
