import os,logging,importlib.util,sys,json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ['MOX_OBS_CLIENT_LOG']='0';logging.disable(logging.CRITICAL)
R=Path('/home/ma-user/work/lingbot-vla-v2');P=R/'reports/20261016_completed_analysis';L=R/'logs/robotwin_1ep_fast'
spec=importlib.util.spec_from_file_location('obs_storage',R/'lingbotvla/utils/obs_storage.py');mod=importlib.util.module_from_spec(spec);sys.modules['obs_storage']=mod;spec.loader.exec_module(mod)
uri='obs://yw-2030-extern/Partner_Qin/robot_1014/robotwin_1ep_fast';storage=mod.ObsStorage(uri)
files={}
for n in ['train.log','loss.jsonl','exit_code','obs_checkpoints.jsonl','lingbotvla_cli.yaml']:
 files['logs/'+n]=L/n
for f in (L/'runs').glob('*'):
 if f.is_file():files['logs/runs/'+f.name]=f
for f in P.iterdir():
 if f.is_file() and f.name not in ['archive_receipt.json','archive_receipt_verification.json','archive_final.py']:
  files['analysis/'+f.name]=f
records=storage._upload_files(files,storage.root_uri,verification='sha256')
receipt={'completed_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'run_uri':uri,'purpose':'Final logs and completed training analysis; immutable checkpoint objects were not changed','files':records}
(P/'archive_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
verified=storage._upload_files({'analysis/archive_receipt.json':P/'archive_receipt.json'},storage.root_uri,verification='sha256')
(P/'archive_receipt_verification.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2))
print(json.dumps({'uploaded_count':len(records),'local_bytes':sum(p.stat().st_size for p in files.values()),'root':uri,'report':uri+'/analysis/report.html','verified':'remote full SHA256 for every small artifact','receipt':str(P/'archive_receipt.json')},ensure_ascii=False))
