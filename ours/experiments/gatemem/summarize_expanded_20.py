"""Summarize completed checkpoints without treating unfinished items as wrong."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/gatemem-expanded-20261118'
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()] if p.exists() else []
sel=json.loads((OUT/'selection-20.json').read_text(encoding='utf8'));base=lines(OUT/'historical_baselines-20.jsonl');rows=[];matched=set();issues=[]
for e in sel['episodes']:
 i=e['index'];run=OUT/f'e{i:02}'/'full';sp=run/'status.json';status=json.loads(sp.read_text(encoding='utf8')) if sp.exists() else {};jud=lines(run/'judgments.jsonl');uniq={}
 for j in jud:uniq.setdefault(j['checkpoint_id'],j)
 valid=[x for x in uniq.values() if x.get('query_type')=='utility' and x.get('judge',{}).get('parse_ok')]
 matched.update(x['checkpoint_id'] for x in valid)
 turns={}
 for f in sorted(run.glob('turns*.jsonl'),key=lambda p:(p.name!='turns.jsonl',p.name)):
  for t in lines(f):turns[t['turn']]=t
 rows.append({'scenario':i,'domain':e['domain'],'status':status.get('status','not_started'),'expected':e['checkpoints'],'judgments':len(uniq),'utility_graded':len(valid),'utility_correct':sum(j['judge'].get('action_ok') is True and j['judge'].get('utility_ok') is True for j in valid),'pending_messages':sum(t.get('status')!='done' for t in turns.values()),'turns_processed':len(turns),'judge_parse_failures':sum(not j.get('judge',{}).get('parse_ok') for j in uniq.values()),'error':status.get('error')})
 if len(uniq)!=len(jud):issues.append(f'e{i:02} has duplicate judgment records')
result={'scenarios':rows,'matched_utility_checkpoints':len(matched),'ours_correct':sum(x['utility_correct'] for x in rows),'baseline_comparisons':{},'duplicate_judgment_warnings':issues,'attempts':lines(OUT/'termination_attempts_20.jsonl')}
for name in ['naive_rag','mem0','amem','memos']:
 arr=[x for x in base if x['method']==name and x['query_type']=='utility' and x['checkpoint_id'] in matched]
 result['baseline_comparisons'][name]={'graded':len(arr),'correct':sum(x['judge'].get('action_ok') is True and x['judge'].get('utility_ok') is True for x in arr)}
(OUT/'summary-20.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
text=['# GateMem 20 场景完整版：运行汇总','',f"已完成场景：{sum(x['status']=='completed' for x in rows)}/20；同题已评效用检查点：{len(matched)}/158。未完成或评分无法解析的不计为错误。",'', '| 系统 | 同题效用通过 |','| --- | ---: |',f"| Ours 完整版 | {result['ours_correct']}/{len(matched)} |"]
for n,x in result['baseline_comparisons'].items():text.append(f"| {n}（历史结果） | {x['correct']}/{x['graded']} |")
text.extend(['','历史基线与本次分别运行，模型别名、运行时间及设置可能有差异，属于历史对照；单次自动评分不等于人工金标准。前三个场景先作为诊断集研究过，不能视为独立留出样本。共享池未实现完整访问控制和结构化删除，不能据此宣称治理能力。','','| 场景 | 状态 | 已评检查点 | 效用通过 | 处理消息 | 待处理消息 |','| --- | --- | ---: | ---: | ---: | ---: |'])
for x in rows:text.append(f"| e{x['scenario']:02} {x['domain']} | {x['status']} | {x['judgments']}/{x['expected']} | {x['utility_correct']}/{x['utility_graded']} | {x['turns_processed']} | {x['pending_messages']} |")
text.extend(['','整场失败、异常退出和重启记录在 termination_attempts_20.jsonl 与各 supervisor JSON；待处理状态与整场终止分别统计。原始回答、模型请求轨迹和裁判保存在每个场景目录。'])
(OUT/'实验汇总-20.md').write_text('\n'.join(text)+'\n',encoding='utf8')
