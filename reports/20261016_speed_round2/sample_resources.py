"""Read-only NPU sampling for an explicitly named experiment; no training imports."""
import argparse, datetime, json, re, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('run',type=Path); p.add_argument('--max-seconds',type=int,default=900); a=p.parse_args()
stop=time.monotonic()+a.max_seconds
with (a.run/'resource_samples.jsonl').open('x') as f:
 while time.monotonic()<stop and not (a.run/'exit_code').exists():
  sample={'at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'last_step':0,'cards':{}}
  loss=a.run/'logs/loss.jsonl'
  if loss.exists():
   lines=loss.read_text().splitlines()
   if lines:
    try: sample['last_step']=json.loads(lines[-1])['step']
    except json.JSONDecodeError: pass
  for card in [0,4]:
   r=subprocess.run(['npu-smi','info','-t','usages','-i',str(card)],capture_output=True,text=True,timeout=10)
   fields={}
   for line in r.stdout.splitlines():
    if ':' in line:
     key,value=line.rsplit(':',1)
     try: fields[key.strip()]=float(value.strip())
     except ValueError: pass
   sample['cards'][str(card)]=fields
  f.write(json.dumps(sample)+'\n'); f.flush(); time.sleep(4)
