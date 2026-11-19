# Ours 的 GateMem 正式入口

Ours 注册为 GateMem 的 `ours` agent，通过 `baselines/GateMem/scripts/run_relay_experiment.py` 使用与基线相同的 `run_eval.py`、官方回答模板、规则评分和 Judge。每个 episode 创建独立数据库和向量目录，按 checkpoint 增量摄入消息；不将题目隐藏标注交给记忆或回答模型。

目前尚未实现对每个样本独立重复 Judge 三次及其汇总，正式实验前必须先补齐并验证这一要求。

正式运行前必须核实精确模型 ID、模型实际身份及服务能力，准备模型权重、各项依赖和余额，并确定中断后的处理方案。下例中的占位符需要替换；执行命令会调用付费模型，不是离线检查。

```powershell
cd baselines/GateMem
python scripts/run_relay_experiment.py `
  --agent ours `
  --model <已确认的中转站精确模型ID> `
  --domain all `
  --run-name <唯一运行名> `
  --ours-python <安装了Ours与MemOS依赖的Python可执行文件>
```

`--ours-python` 可省略：若 `ours/.venv` 的 Python 存在则自动选用，否则使用启动器当前 Python。`--domain` 也可指定单一领域；全量 `all` 与单领域结果不可重复计数。每次调用只运行一个模型组，切换中转站分组前应等待前一组全部退出。运行目录一旦存在，启动器不会覆盖。

入口将同一模型 ID 和中转地址用于记忆写入、最终回答及 Judge；通过同一个本地服务使用 `all-MiniLM-L6-v2`、384 维、最大长度 256 和归一化向量。方法使用共享池，开启联合状态更新、立即维护和独立状态读取；系统未显式强制供应商思考模式。默认取最多 20 张卡，用户状态可能额外进入回答，不能称作所有系统最终信息量严格相同。

输出目录的 `experiment_manifest.json` 记录参数、原始数据和提示词哈希；`predictions.jsonl`、`scores.jsonl`、`judge_scores.jsonl` 和 `summary.json` 使用 GateMem 标准输出。`model_calls.jsonl` 保存中转调用使用量和返回模型名；各场景的 `ours_episodes/` 包含协议、消息处理日志、调用记录、待处理与状态恢复记录。接口未返回 usage 时不应记为零。模型返回名可能是别名，不足以独立证明实际模型身份。

当前不支持安全自动续跑。发生中断时保留完整目录与额外调用成本，不得删除数据后冒充原过程继续；修复恢复机制前从受控的新运行按既定方案重跑。隐私/删除结果不能独立证明当前共享池实现了完整治理。

离线检查（不调用模型）：

```powershell
cd baselines/GateMem
python -m unittest discover -s tests -p 'test_ours_formal.py' -v
```

这些检查使用模拟模型和本地假数据，还会核对真实数据的场景/检查点数量；不能证明三家中转模型、真实服务、完整长对话或全部开销已经验证。
