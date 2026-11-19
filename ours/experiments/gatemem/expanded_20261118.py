"""Exploratory GateMem pilot: full vs cards-only, official incremental runner."""
from __future__ import annotations
import json, os, sys, time, threading, argparse
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import requests
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runtime/gatemem-expanded-20261118'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments/gatemem/official/code')]
from ours_memory.groupmem import build_memory, close_memory, ReplayClock, Trace
from ours_memory.control import Actor
from ours_memory.models import stable_id
from bench.agents.base import BaseMemoryAgent
from bench.eval.runner import run_episode
from bench.eval.judge import run_llm_judge
from bench.llm import LLMConfig, LLMRouter

MODEL='deepseek-flash'
LIMIT=None if os.environ.get('GATEMEM_UNBOUNDED')=='1' else 6_000_000
PORT=int(os.environ.get('GATEMEM_PORT','18177'))
LEDGER_PATH=OUT/os.environ.get('GATEMEM_LEDGER','ledger.json')
MARGIN=50_000
ledger={'limit':LIMIT,'margin':MARGIN,'used':0,'calls':[],'status':'running'}
lock=threading.Lock()
def save(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(path)
def save_ledger():save(LEDGER_PATH,ledger)
class Proxy(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_POST(self):
  body=self.rfile.read(int(self.headers.get('Content-Length','0')))
  with lock:
   if ledger['status']!='running':return self.reply(503,{'error':{'message':'experiment guard stopped'}})
   if LIMIT is not None and ledger['used']+MARGIN>LIMIT:
    ledger['status']='budget_stopped';save_ledger();return self.reply(429,{'error':{'message':'budget stopped'}})
   try:
    obj=json.loads(body)
    if obj.get('model')!=MODEL or obj.get('stream'):return self.reply(400,{'error':{'message':'unsupported request'}})
    obj['thinking']={'type':'disabled'}
    body=json.dumps(obj,ensure_ascii=False).encode('utf8')
    for attempt in range(3):
     response=requests.post('https://api.ciyuan.fast/v1/chat/completions',data=body,headers={'Authorization':'Bearer '+os.environ['GATEMEM_SECRET'],'Content-Type':'application/json','User-Agent':'Mozilla/5.0'},timeout=120)
     if response.status_code==403 and '1010' in response.text and attempt<2:continue
     break
    try:data=response.json()
    except ValueError:data={'error':{'message':'gateway http '+str(response.status_code)}}
    usage=data.get('usage') or {}; tokens=usage.get('total_tokens')
    ledger['calls'].append({'http':response.status_code,'usage':usage,'model':data.get('model'),'phase':os.environ.get('GATEMEM_PHASE','unknown')})
    if isinstance(tokens,int):ledger['used']+=tokens
    elif response.ok:ledger['status']='missing_usage_stopped'
    save_ledger();self.reply(response.status_code,data)
   except Exception as e:
    ledger['status']='uncertain_request_stopped';ledger['calls'].append({'error':type(e).__name__});save_ledger();self.reply(503,{'error':{'message':'request failed; stop'}})
 def reply(self,status,data):
  body=json.dumps(data).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)

def readlines(path):return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines() if x.strip()]
def router(max_tokens=2048,temperature=0.2):
 return LLMRouter(LLMConfig(provider='deepseek',model=MODEL,api_base=f'http://127.0.0.1:{PORT}/v1',api_key_env='DEEPSEEK_API_KEY',temperature=temperature,max_output_tokens=max_tokens,timeout_s=150,max_retries=1))
class BudgetWindow(Exception):
 pass

class Agent(BaseMemoryAgent):
 def __init__(self,run,group,episode,resume=False):
  super().__init__(top_k=20,llm_router=router(4096));self.run=run;self.group=group;self.episode_config=episode;self.prev=[];self.count=0;self.resume=resume
  self.old_turns={}
  if resume:
   for file in sorted(run.glob('turns*.jsonl'), key=lambda p: (p.name != 'turns.jsonl', p.name)):
    for row in readlines(file):self.old_turns[row['turn']]=row
  self.cached_predictions={}
  if resume:
   for file in sorted(run.glob('predictions*.jsonl')):
    for row in readlines(file):self.cached_predictions[row['checkpoint_id']]=row['output']
  self.log_turns=run/('turns-resume.jsonl' if resume else 'turns.jsonl')
 def reset(self,episode):
  super().reset(episode);self.clock=ReplayClock('2026-01-01T00:00:00+00:00'); users=sorted(p['principal_id'] for p in episode['entities']['principals']);self.users=users;self.space='GateMem:'+episode['episode_id']
  trace=Trace(self.run/'trace');self.trace=trace;self.memory=build_memory(self.run,{'space':self.space,'users':users,'access_mode':'shared_pool'},trace,self.clock)
  if self.group=='cards_only':
   self.memory.settings.state_enabled=False;self.memory.settings.sleep_enabled=False;self.memory.settings.joint_state_update_enabled=False;self.memory.settings.independent_state_read_enabled=False
 def ingest(self,turn):
  if ledger['status']!='running':raise RuntimeError('budget or provider guard stopped')
  if LIMIT is not None and ledger['used']+120_000>LIMIT:raise BudgetWindow('reserve remaining budget for scoring completed checkpoints')
  raw=turn.timestamp; ts=raw
  if raw:
   parsed=datetime.fromisoformat(raw.replace('Z','+00:00'))
   if parsed.tzinfo is None:ts=raw+'+00:00'
  if ts:self.clock.value=ts
  metadata={'role':turn.speaker_role,'msg_node':turn.turn_id}
  if ts:metadata['timestamp']=ts
  else:metadata['source_time_unknown']='true'
  self.trace.context=turn.turn_id
  old=self.old_turns.get(turn.turn_id)
  # All earlier turns have already been committed. Only retry a pending operation
  # at the exact boundary, before any newer message exists.
  source_id=stable_id('message_v2',self.space,turn.speaker_principal_id,turn.turn_id)
  source=self.memory.store.get('source_message_v2',source_id) if old else None
  old_time=(source or {}).get('observed_at')
  old_time_naive=bool(old_time and datetime.fromisoformat(old_time.replace('Z','+00:00')).tzinfo is None)
  latest_sequence=max((x.get('arrival_sequence',0) for x in self.memory.store.all('source_message_v2') if x.get('space')==self.space),default=0) if old else 0
  retry_boundary = (old and source and not old_time_naive and source.get('arrival_sequence',0)==latest_sequence and turn.turn_id == max(self.old_turns, key=lambda x: int(x[1:])) and old.get('processing_status') in {'extraction_pending', 'state_maintenance_pending', 'card_storage_pending'})
  if old and not retry_boundary:
   if not self.memory.store.get('source_message_v2',source_id):raise RuntimeError('missing committed source during resume')
   self.prev.append({'id':source_id,'speaker':turn.speaker_principal_id,'role':turn.speaker_role,'text':turn.text,'timestamp':ts})
   while sum(len(x['text']) for x in self.prev[-10:])>6000:self.prev.pop(0)
   self.count+=1
   return
  r=self.memory.ingest_message(Actor(turn.speaker_principal_id,'ingestor',self.space,'pilot'),message_id=turn.turn_id,text=turn.text,observed_at=ts,metadata=metadata,fallback_category='conversation',prior_context=self.prev[-10:])
  self.prev.append({'id':r['message_id'],'speaker':turn.speaker_principal_id,'role':turn.speaker_role,'text':turn.text,'timestamp':ts})
  while sum(len(x['text']) for x in self.prev[-10:])>6000:self.prev.pop(0)
  self.count+=1
  with self.log_turns.open('a',encoding='utf8') as f:f.write(json.dumps({'turn':turn.turn_id,'status':r.get('status'),'processing_status':r.get('processing_status'),'cards':sum(len(x.get('card_ids',[])) for x in r.get('events',[]))},ensure_ascii=False)+'\n')
  if self.count%10==0:print(self.group,self.run.parent.name,'turn',self.count,'tokens',ledger['used'],flush=True)
 def query(self,ck):
  if ck.checkpoint_id in self.cached_predictions:return self.cached_predictions[ck.checkpoint_id]
  if ledger['status']!='running':raise RuntimeError('budget or provider guard stopped')
  if LIMIT is not None and ledger['used']+120_000>LIMIT:raise BudgetWindow('reserve remaining budget for scoring completed checkpoints')
  self.trace.context=ck.checkpoint_id; actor=Actor(ck.asker_principal_id,'answerer',self.space,'query')
  bundle=self.memory.read(actor,subject_ids=self.users,query=ck.query_text,purpose='current')
  if not self.memory.validate_bundle(actor,bundle['bundle_id'])['ready']:raise RuntimeError('bundle invalid')
  cards=bundle['cards'][:20]; states=bundle.get('user_states',[]) if self.group=='full' else []
  retrieved=[{'text':c.get('claim_text','')+'\nSource: '+str((c.get('source') or {}).get('quote','')),'record_id':c['id'],'principal_id':c.get('author_id','')} for c in cards]
  block=self._format_retrieved_memory_block(retrieved)
  if self.group=='full':block+='\n\nCurrent user states:\n'+json.dumps(states,ensure_ascii=False)
  output=self._run_llm(checkpoint=ck,retrieved_memory=retrieved,rendered_memory_block_override=block)
  output['incomplete_message_count']=bundle.get('incomplete_message_count',0);output['pending_state_count']=bundle.get('pending_state_count',0)
  return output

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,required=True,choices=range(1,21));parser.add_argument('--group',required=True,choices=['full','cards_only']);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 # Each independent process uses a standalone ledger; the orchestration tracks cumulative usage.
 prior=LEDGER_PATH
 if prior.exists():
  ledger.update(json.loads(prior.read_text(encoding='utf8')))
 ledger['limit']=LIMIT
 if ledger['status']=='budget_stopped' and (LIMIT is None or ledger['used']+MARGIN<LIMIT): ledger['status']='running'
 if ledger['status']!='running':raise RuntimeError('budget guard stopped')
 server=ThreadingHTTPServer(('127.0.0.1',PORT),Proxy);threading.Thread(target=server.serve_forever,daemon=True).start()
 os.environ['DEEPSEEK_API_KEY']='local-proxy';os.environ['OURS_LLM_BASE_URL']=f'http://127.0.0.1:{PORT}/v1';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['GATEMEM_PHASE']=f'e{args.index:02}_{args.group}'
 sel=json.loads((OUT/os.environ.get('GATEMEM_SELECTION','selection.json')).read_text(encoding='utf8'));item=next(x for x in sel['episodes'] if x['index']==args.index)
 directory=ROOT/'experiments/gatemem/official/data/gatemem/data'/item['domain'];ep=next(x for x in readlines(directory/'episodes.jsonl') if x['episode_id']==item['episode_id']);cks=[x for x in readlines(directory/'checkpoints.jsonl') if x['episode_id']==item['episode_id']]
 run=OUT/f'e{args.index:02}'/args.group
 if args.resume and not run.exists():raise RuntimeError('resume directory missing')
 run.mkdir(parents=True,exist_ok=args.resume)
 protocol={'memory_model':MODEL,'embedding':{'model':'sentence-transformers/all-MiniLM-L6-v2','dimensions':384},'reasoning_policy':'disabled','method':{'joint_state_update_enabled':args.group=='full','immediate_state_maintenance_enabled':args.group=='full','independent_state_read_enabled':args.group=='full'}}
 
 if args.resume and json.loads((run/'protocol.json').read_text(encoding='utf8'))!=protocol:raise RuntimeError('resume protocol differs')
 save(run/'protocol.json',protocol);agent=Agent(run,args.group,ep,args.resume);status={'status':'running','expected_checkpoints':len(cks),'episode_id':ep['episode_id'],'group':args.group};save(run/'status.json',status)
 try:
  prediction_file=run/('predictions-resume.jsonl' if args.resume else 'predictions.jsonl')
  partial=False
  try:preds=run_episode(agent=agent,episode=ep,checkpoints=cks,prediction_path=str(prediction_file))
  except BudgetWindow:
   partial=True
   preds=readlines(prediction_file) if prediction_file.exists() else []
  preds=list({p['checkpoint_id']:p for p in preds}.values())
  os.environ['GATEMEM_PHASE']=f'e{args.index:02}_{args.group}_judge'
  js,meta=run_llm_judge(episodes=[ep],checkpoints=cks,predictions=preds,judge_router=router(4096,0),prompt_path=str(ROOT/'experiments/gatemem/official/code/bench/prompts/judge_prompt.txt'),out_path=str(run/'judgments.jsonl'),resume=(run/'judgments.jsonl').exists(),concurrency=1)
  status.update(status='budget_partial_judged' if partial else 'completed',predictions=len(preds),judgments=meta['n_judged'],judge_meta=meta)
 except Exception as exc:status.update(status='failed',error=type(exc).__name__,detail=str(exc)[:300])
 finally:
  if hasattr(agent,'memory'):
   save(run/'calls.json',agent.memory.store.call_records());close_memory(agent.memory)
  status['total_tokens_so_far']=ledger['used'];save(run/'status.json',status);print(json.dumps(status,ensure_ascii=False),flush=True);server.shutdown()
if __name__=='__main__':main()
