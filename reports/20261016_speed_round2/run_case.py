#!/usr/bin/env python3
"""Run one bounded, isolated speed experiment with no checkpoint or upload."""
import argparse, datetime, hashlib, json, os, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('case');p.add_argument('--source',type=Path,required=True)
p.add_argument('--monitor',choices=['0','1'],default='0')
p.add_argument('--attention',choices=['eager','sdpa_fp32'],default='eager')
p.add_argument('--adamw-foreach',choices=['0','1'],default='0')
p.add_argument('--checkpoint',choices=['true','false'],default='true')
p.add_argument('--steps',type=int,default=60)
a=p.parse_args()
if not a.case.replace('_','').isalnum() or a.steps<20: raise SystemExit('invalid case or step count')
source=a.source.resolve(); run=Path('/cache/lingbot/profiling/speed_round2_20261016')/a.case
if run.exists(): raise SystemExit('Refusing to reuse an existing experiment directory')
if not (source/'assets/norm_stats/robotwin.json').exists(): raise SystemExit('Missing normalization input')
run.mkdir(parents=True)
env=os.environ.copy()
env.update(REPO_ROOT=str(source),PYTHONPATH=str(source)+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else ''),
 PROFILE_ROOT=str(run),RUN_NAME='round2_'+a.case,STOP_AFTER_STEPS=str(a.steps),MASTER_PORT='62531',
 NPROC_PER_NODE='8',ASCEND_RT_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',MICRO_BATCH='12',GRAD_ACCUM='1',GLOBAL_BATCH='96',
 NUM_TRAIN_EPOCHS='1',LR='5e-5',LR_MIN='1e-5',LR_WARMUP_RATIO='0.02',TRAIN_EXPERT_ONLY='true',
 ENABLE_FP32='true',ENABLE_MIXED_PRECISION='true',ENABLE_GRAD_CKPT=a.checkpoint,USE_COMPILE='false',
 LINGBOT_NPU_MOE_IMPL='gmm',LINGBOT_MOE_MONITOR_BATCH_COUNTS=a.monitor,LINGBOT_JOINT_ATTN_BACKEND=a.attention,
 LINGBOT_ADAMW_FOREACH_NPU=a.adamw_foreach)
for k in ['DRY_RUN','MAX_STEPS','LOAD_CHECKPOINT_PATH','EXTRA_ARGS','ALIGN_MERGE','ALIGN_JSON','TRAIN_SCRIPT']:
 env.pop(k,None)
manifest={'started_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'source_snapshot':str(source),'steps':a.steps,'moe_backend':'gmm','monitor':a.monitor,'attention':a.attention,
 'adamw_foreach':a.adamw_foreach,
 'gradient_checkpointing':a.checkpoint,
 'profile_recording':False,'checkpoints':False,'uploads':False,'global_batch':96,'comparison_steps':[13,a.steps],
 'hashes':{str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*') if f.is_file() and f.suffix in ('.py','.sh','.yaml','.json')}}
(run/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
with (run/'resource_collector.log').open('w') as log:
 collector=subprocess.Popen([sys.executable,str(Path(__file__).with_name('sample_resources.py')),str(run),'--max-seconds','1200'],stdout=log,stderr=subprocess.STDOUT)
 try:
  with (run/'console.log').open('w') as output:
   result=subprocess.run(['bash',str(source/'tools/profile_robotwin_npu.sh'),'--train.profile_ranks',''],env=env,cwd=source,stdout=output,stderr=subprocess.STDOUT)
  status=result.returncode
 finally:
  status=locals().get('status',99)
  (run/'exit_code').write_text(str(status)+'\n')
  try: collector.wait(timeout=20)
  except subprocess.TimeoutExpired: collector.terminate();collector.wait(timeout=10)
print(json.dumps({'case':a.case,'exit_code':status,'root':str(run)}))
raise SystemExit(status)
