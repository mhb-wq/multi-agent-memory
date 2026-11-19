# Card–State–Sleep 多用户记忆系统

`src/ours_memory` 包含当前记忆实现。系统从多人消息抽取卡片及参与角色，检索候选历史卡片与用户状态，联合判断新旧关系与状态更新，并保存来源和状态历史。配置默认入口为 `configs/multiuser.json`。

当前入口只接受共享记忆池、联合判断、即时状态维护和独立状态检索；旧的版本切换配置已移除。旧版命令行、HTTP 接口、演示入口以及旧状态维护流程已移除。部分共用底层服务、历史脚本和历史测试仍待进一步梳理，不能认为旧代码已全部清除。

状态维护的细分校验、短编号、一次格式修复、批量回滚以及待处理规则见[联合判断校验与恢复](docs/联合判断校验与恢复.md)。初始卡片的人物和角色不会因联合判断自动回写修正；系统能力及失败处理应以代码为准。

`experiments/gatemem` 保留 GateMem 接口和历史研究脚本，其中部分脚本依赖本地历史运行目录、特定服务或数据；不应将其当作正式全量实验入口。当前正式入口已接入 `baselines/GateMem/scripts/run_relay_experiment.py`，使用 `--agent ours`，可选 `--domain all` 覆盖四个领域。它通过同一个 GateMem 评测和 Judge 入口，按场景隔离存储，并记录待处理项。详见[GateMem 正式入口说明](docs/GateMem正式入口.md)。历史结果不随本仓库发布。`vendor/MemOS` 为 `upstream.lock.json` 所标记的 MemOS 源码快照，许可证随附。

在已安装测试依赖的环境中，可运行：

```powershell
python -m pytest tests/test_current_method.py tests/test_joint_repair.py tests/test_joint_state_update.py tests/test_immediate_person_matter.py tests/test_independent_state_read.py tests/test_source_message_storage.py tests/test_llm_failures.py tests/test_formal_embedding.py -q
```

这组检查只覆盖新版核心路径。完整历史测试集尚未迁移，部分测试仍依赖仓库中未包含的历史实验脚本或已删除的旧流程；不能把这组通过视为全量测试通过。

`.env.example` 只展示变量名；复制并配置时不要将真实密钥或访问令牌提交到 Git。模型权重及运行数据库未随源码提供。实验所用响应模型、向量模型、服务和所有开销需要单独配置并记录。
