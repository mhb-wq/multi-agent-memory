# Card–State–Sleep 多用户记忆系统

`src/ours_memory` 包含当前记忆实现。系统从多人消息抽取卡片及参与角色，检索候选历史卡片与用户状态，联合判断新旧关系与状态更新，并保存来源和状态历史。配置默认入口为 `configs/multiuser.json`。

状态维护的细分校验、短编号、一次格式修复、批量回滚以及待处理规则见[联合判断校验与恢复](docs/联合判断校验与恢复.md)。初始卡片的人物和角色不会因联合判断自动回写修正；系统能力及失败处理应以代码为准。

`experiments/gatemem` 保留 GateMem 接口和历史研究脚本，其中部分脚本依赖本地历史运行目录、特定服务或数据；不应将其当作开箱即用的正式实验入口。历史结果不随本仓库发布。`vendor/MemOS` 为 `upstream.lock.json` 所标记的 MemOS 源码快照，许可证随附。

在已安装测试依赖的环境中，可运行：

```powershell
python -m pytest tests/test_joint_repair.py tests/test_joint_state_update.py tests/test_immediate_person_matter.py tests/test_llm_failures.py -q
```

`.env.example` 只展示变量名；复制并配置时不要将真实密钥或访问令牌提交到 Git。模型权重及运行数据库未随源码提供。实验所用响应模型、向量模型、服务和所有开销需要单独配置并记录。
