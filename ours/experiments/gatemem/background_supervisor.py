"""Detached, resumable supervisor for one GateMem group."""
import argparse, ctypes, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runtime/gatemem-expanded-20261118'
parser=argparse.ArgumentParser()
parser.add_argument('--index',type=int,required=True)
parser.add_argument('--group',required=True)
parser.add_argument('--port',type=int,required=True)
parser.add_argument('--ledger',required=True)
parser.add_argument('--wait-pid',type=int)
a=parser.parse_args()
run=OUT/f'e{a.index:02}'/a.group
ledger_path=OUT/a.ledger
supervisor=OUT/f'supervisor-e{a.index:02}-{a.group}.json'
def save(obj):
 tmp=supervisor.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(supervisor)
def alive(pid):
 kernel=ctypes.windll.kernel32
 handle=kernel.OpenProcess(0x1000,False,pid)
 if not handle:return False
 try:
  code=ctypes.c_ulong();return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
 finally:kernel.CloseHandle(handle)
selection=os.environ.get('GATEMEM_SELECTION','selection.json')
state={'selection':selection,'status':'starting','index':a.index,'group':a.group,'port':a.port,'ledger':str(ledger_path),'restarts':0,'events':[]}
save(state)
if a.wait_pid:
 state['status']='waiting_for_existing_run';save(state)
 while alive(a.wait_pid):time.sleep(5)
for attempt in range(9):
 status_path=run/'status.json'
 if status_path.exists():
  current=json.loads(status_path.read_text(encoding='utf8'))
  if current.get('status')=='completed':state['status']='completed';save(state);break
 if ledger_path.exists():
  data=json.loads(ledger_path.read_text(encoding='utf8'))
  if data.get('status')=='uncertain_request_stopped':
   data.setdefault('calls',[]).append({'event':'automatic_resume_after_unknown_transport_outcome','unknown_charge':True})
   data['status']='running'
   ledger_path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
   state['events'].append({'event':'transport_retry','time':time.time()})
  elif data.get('status')=='missing_usage_stopped':
   state['status']='stopped_missing_usage';save(state);break
 env=os.environ.copy();env.update(GATEMEM_UNBOUNDED='1',GATEMEM_PORT=str(a.port),GATEMEM_LEDGER=a.ledger)
 cmd=[sys.executable,str(ROOT/'experiments/gatemem/expanded_20261118.py'),'--index',str(a.index),'--group',a.group]
 if run.exists():cmd.append('--resume')
 state['status']='running';state['restarts']=attempt;save(state)
 with (OUT/f'supervisor-e{a.index:02}-{a.group}.log').open('a',encoding='utf8') as log:
  result=subprocess.run(cmd,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 status=json.loads(status_path.read_text(encoding='utf8')) if status_path.exists() else {}
 with (OUT/'termination_attempts_20.jsonl').open('a',encoding='utf8') as audit:
  audit.write(json.dumps({'scenario':a.index,'group':a.group,'attempt':attempt+1,'status':status.get('status'),'error':status.get('error'),'detail':status.get('detail'),'exit_code':result.returncode,'time':time.time()},ensure_ascii=False)+'\n')
 if status.get('status')=='completed':state['status']='completed';save(state);break
 data=json.loads(ledger_path.read_text(encoding='utf8')) if ledger_path.exists() else {}
 if data.get('status') not in {'uncertain_request_stopped','budget_stopped'}:
  state.update(status='stopped_needs_review',last_run_status=status.get('status'),exit_code=result.returncode,error=status.get('error'));save(state);break
 state['events'].append({'event':'restart_scheduled','run_status':status.get('status'),'time':time.time()});save(state);time.sleep(min(10*(attempt+1),60))
else:state['status']='stopped_retry_limit';save(state)
