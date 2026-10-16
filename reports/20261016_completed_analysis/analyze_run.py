import json,re,math,csv,statistics
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
LOG=ROOT.parent.parent/'logs/robotwin_1ep_fast'
rows=[];malformed=[]
for n,line in enumerate((LOG/'loss.jsonl').read_text().splitlines(),1):
 try:rows.append(json.loads(line))
 except json.JSONDecodeError:malformed.append(n)
log=(LOG/'train.log').read_text();hits=re.findall(r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) - INFO - __main__ - Step (\d+)/',log)
def dt(s):return datetime.strptime(s,'%m/%d/%Y %H:%M:%S').replace(tzinfo=ZoneInfo('Asia/Shanghai'))
keys=['loss','vla_loss','depth_loss','future_depth_loss','future_video_loss','grad_norm','step_time','lr']
def stats(rs):
 return {k:{'mean':float(np.mean([r[k] for r in rs])),'median':float(np.median([r[k] for r in rs])),'min':min(r[k] for r in rs),'max':max(r[k] for r in rs),'p95':float(np.quantile([r[k] for r in rs],.95))} for k in keys}
windows={}
for label,lo,hi in [('first100',1,100),('first500',1,500),('postwarmup500',115,614),('middle500',2501,3000),('penultimate500',4718,5217),('last500',5218,5717),('last100',5618,5717)]:
 rs=[r for r in rows if lo<=r['step']<=hi];windows[label]={'start_step':lo,'end_step':hi,'count':len(rs),'metrics':stats(rs)}
launch=json.loads((LOG/'tmux_launch.json').read_text());start=datetime.fromisoformat(launch['started_at'])
loop=re.search(r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}).*rank0 Start training',log)
if not loop:
 loop=re.search(r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}).*Schedule:',log)
loop_time=dt(loop[1]);first,last=dt(hits[0][0]),dt(hits[-1][0])
events={}
for label,pat in [('dcp_saved','Saved checkpoint to'),('hf_start','\\[async_hf\\] saving HF checkpoint'),('hf_finished','\\[async_hf\\] HF checkpoint finished'),('obs_verified','OBS checkpoint verified'),('checkpoint_complete','Checkpoint complete'),('final_summary','\\[async_hf\\] summary:')]:
 matches=re.findall(r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})[^\n]*'+pat,log)
 if matches:events[label]=dt(matches[-1])
counts=Counter(r['step'] for r in rows);nonfinite=[{'step':r['step'],'metric':k,'value':str(r[k])} for r in rows for k in keys if not math.isfinite(r[k])]
base=json.loads((LOG/'previous_run_baseline.json').read_text());steady=(last-first).total_seconds()/(len(hits)-1)
summary={'analyzed_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'run':'robotwin_1ep_fast','exit_code':int((LOG/'exit_code').read_text()),'steps':len(rows),'epoch':rows[-1]['epoch'],'integrity':{'malformed_json_lines':malformed,'missing_steps':sorted(set(range(1,5718))-set(counts)),'duplicate_steps':[s for s,c in counts.items() if c>1],'nonfinite_values':nonfinite},'first':rows[0],'last':rows[-1],'all':stats(rows),'steady_excluding_first':stats(rows[1:]),'windows':windows,'timeline':{'launch':start.isoformat(),'loop_start':loop_time.isoformat(),'first_update':first.isoformat(),'last_update':last.isoformat(),**{k:v.isoformat() for k,v in events.items()}},'durations_seconds':{'launch_to_loop':(loop_time-start).total_seconds(),'loop_to_last_update':(last-loop_time).total_seconds(),'launch_to_last_update':(last-start).total_seconds(),'last_update_to_obs_verified':(events['obs_verified']-last).total_seconds(),'hf_export':(events['hf_finished']-events['hf_start']).total_seconds(),'hf_finished_to_obs_verified':(events['obs_verified']-events['hf_finished']).total_seconds(),'launch_to_completion':(events['final_summary']-start).total_seconds()},'throughput':{'steady_wall_seconds_per_update':steady,'steady_global_samples_per_second':96/steady,'planned_sample_slots':5717*96,'dataset_frames':548893,'old_seconds_per_update':base['last500_wall_seconds_per_step'],'time_reduction_pct':(1-steady/base['last500_wall_seconds_per_step'])*100,'throughput_multiplier':base['last500_wall_seconds_per_step']/steady,'device_hours_loop':(last-loop_time).total_seconds()/3600*8,'device_hours_launch_to_complete':(events['final_summary']-start).total_seconds()/3600*8},'outliers':{'step_time_over10_postfirst':[{'step':r['step'],'seconds':r['step_time']} for r in rows[1:] if r['step_time']>10],'top10_slowest':sorted([{'step':r['step'],'seconds':r['step_time']} for r in rows],key=lambda x:x['seconds'],reverse=True)[:10],'top10_vla_loss':sorted([{'step':r['step'],'vla_loss':r['vla_loss']} for r in rows],key=lambda x:x['vla_loss'],reverse=True)[:10]}}
(ROOT/'training_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
with (ROOT/'metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({k:summary[k] for k in ['steps','integrity','first','last','timeline','durations_seconds','throughput','outliers']},ensure_ascii=False,indent=2))
print('WINDOWS',json.dumps({k:{m:round(v['metrics'][m]['mean'],6) for m in keys} for k,v in windows.items()},indent=2))
print('STEP_TIME_STEADY',summary['steady_excluding_first']['step_time']);print('GRAD',summary['all']['grad_norm'])
