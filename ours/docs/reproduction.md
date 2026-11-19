# 当前版本的复用边界

当前默认配置为 `configs/multiuser.json`。仅保留多用户版本，旧项目、旧配置和旧版使用说明已删除。

## 官方 MemOS

上游：https://github.com/MemTensor/MemOS

固定提交 `adcb9f270c982f2b9ff8db9714792829d3bd6ab8`，MemoryOS 2.0.33，Apache-2.0。源码和许可证在 `vendor/MemOS`，固定版本及关键文件哈希在 `upstream.lock.json`。未修改上游源码。

实际复用 GeneralMemCube、GeneralTextMemory 的提取/读写/检索/删除、OpenAILLM、UniversalAPIEmbedder、Qdrant，以及 SchedulerDispatcher。按主体、协作空间、资料类别划分物理记忆域；跨库候选采用排名合并。不声称复现 TreeTextMemory、全部路由调度策略或论文指标。

## Collaborative Memory

参考：https://arxiv.org/abs/2505.18279

权限及来源传播是受其思想启发的本地适配，不是该论文官方代码或实验分数复现。当前控制层区分请求者、Agent、记忆主体、资料类别、任务与时限。授权由可信配置和认证控制操作提供，普通文本不能生成权限。

## Card—State—Sleep

Card 为事实补上说话人、主体、来源、事项、证据类型、时间和范围。写入只排队，正式 State 在 Sleep 更新。State 只供内部筛选，业务 Agent 得到筛选后的卡片。尚未 Sleep 的新证据不会直接被旧状态否定。

SQLite 保存卡片、状态、权限与持久化任务；MemOS/Qdrant 保存原生事实与向量。删除先阻断访问，再清理来源相关记录，失败支持恢复。没有跨库分布式事务，采用稳定 ID 和持久化补偿。部署限定单进程。

源代码保留当前版本依赖的底层逻辑及其回归测试；测试中的历史兼容案例不是另一份可运行项目。本机依赖已归入 `.deps`，通过 `.venv` 启动，不再依赖旧目录。

## 验证与未完成部分

单元测试和合成模型集成演示只能说明工程约束可执行。真实模型、公开数据集、对抗评测及统计结果需另行实验。不能将固定模型回复、哈希测试向量得到的通过率写成任务准确率。

未开放私有计算后的独立披露授权，不靠答案复核扩大读取范围。业务领域的状态确认规则、任意字段级权限、多进程部署、完整 Agent 执行器尚未实现。内部筛选 LLM 可处理 State；外部模型日志、外部上下文和备份不在本地删除范围内。
