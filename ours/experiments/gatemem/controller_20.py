"""Run independent GateMem scenes with bounded, memory-aware concurrency."""
import ctypes,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/gatemem-expanded-20261118';DATA=json.loads((OUT/'selection-20.json').read_text(encoding='utf8'))
MAX_WORKERS=6;MIN_FREE_GB=1.8;indices=[x['index'] for x in DATA['episodes'] if x['index'] not in DATA['previous_three_reused']]
class MemoryStatus(ctypes.Structure):
 _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong),('ullTotalPhys',ctypes.c_ulonglong),('ullAvailPhys',ctypes.c_ulonglong),('ullTotalPageFile',ctypes.c_ulonglong),('ullAvailPageFile',ctypes.c_ulonglong),('ullTotalVirtual',ctypes.c_ulonglong),('ullAvailVirtual',ctypes.c_ulonglong),('ullAvailExtendedVirtual',ctypes.c_ulonglong)]
def free_gb():
 s=MemoryStatus();s.dwLength=ctypes.sizeof(s)
 if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):return 0
 return s.ullAvailPhys/(1024**3)
def save(obj):
 p=OUT/'controller-20.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');t.replace(p)
env=os.environ.copy();env.update(GATEMEM_SELECTION='selection-20.json',GATEMEM_UNBOUNDED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
active={};finished={};queue=indices[:];state={'status':'running','max_workers':MAX_WORKERS,'min_free_gb':MIN_FREE_GB,'precompleted':[3,16,18],'queue':queue,'active':[],'finished':{},'threads_per_worker':1}
save(state)
while queue or active:
 for idx,child in list(active.items()):
  code=child.poll()
  if code is not None:
   path=OUT/f'supervisor-e{idx:02}-full.json';details=json.loads(path.read_text(encoding='utf8')) if path.exists() else {};finished[str(idx)]={'exit_code':code,'status':details.get('status','missing_supervisor_status')};del active[idx]
 if queue and len(active)<MAX_WORKERS and free_gb()>=MIN_FREE_GB:
  idx=queue.pop(0);port=18200+idx
  cmd=[sys.executable,str(ROOT/'experiments/gatemem/background_supervisor.py'),'--index',str(idx),'--group','full','--port',str(port),'--ledger',f'ledger-e{idx:02}-full.json']
  active[idx]=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 state.update(queue=queue,active=sorted(active),finished=finished,free_memory_gb=round(free_gb(),2),updated_at=time.time());save(state)
 time.sleep(5)
state['status']='completed' if all(x['status']=='completed' for x in finished.values()) else 'completed_with_failures';state['updated_at']=time.time();save(state)
try:subprocess.run([sys.executable,str(ROOT/'experiments/gatemem/summarize_expanded_20.py')],cwd=str(ROOT),env=env,check=False,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
except Exception:pass
