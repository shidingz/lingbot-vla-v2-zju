#!/usr/bin/env python3
"""Copy the approved OBS HF checkpoint into the approved private ModelScope repo."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/20261016_modelscope_upload'
STAGE = Path('/cache/lingbot/modelscope_upload/robotwin_1ep_step5717')
REPO = 'shidingz/lingbot-vla-v2-robotwin-1ep-npu'
SOURCE = 'obs://yw-2030-extern/Partner_Qin/robot_1014/robotwin_1ep_fast/checkpoints/global_step_5717'
REVISION = 'master'
os.environ['MOX_OBS_CLIENT_LOG'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'
logging.disable(logging.CRITICAL)


def emit(event, **kwargs):
    print(json.dumps({'time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                      'event': event, **kwargs}, ensure_ascii=False), flush=True)


def save(name, data):
    REPORT.mkdir(parents=True, exist_ok=True)
    p = REPORT / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def api_client():
    from modelscope.hub.api import HubApi

    class ExistingPrivateRepoApi(HubApi):
        # SDK upload methods normally auto-create public repos. Never allow that.
        def create_repo(self, repo_id, **kwargs):
            if repo_id != REPO:
                raise RuntimeError('Unexpected repository')
            require_private(self)
            return f'https://modelscope.cn/models/{REPO}'

    api = ExistingPrivateRepoApi()
    api.get_cookies(cookies_required=True)
    require_private(api)
    return api


def require_private(api):
    info = api.get_model(REPO, revision=None)
    if info.get('Visibility') != 1:
        raise RuntimeError('Repository must already exist and be private')
    return info


def remote_files(api):
    return {f['Path']: f for f in api.get_model_files(
        REPO, revision=REVISION, recursive=True, use_cookies=True)
        if f.get('Type') != 'tree'}


def mismatches(records, remote):
    return [r['path'] for r in records if
            r['path'] not in remote or
            int(remote[r['path']].get('Size', -1)) != r['size'] or
            remote[r['path']].get('Sha256', '').lower() != r['sha256']]


def worker():
    api = api_client()
    plan = json.loads((REPORT / 'upload_plan.json').read_text())
    pending = mismatches(plan['files'], remote_files(api))
    if not pending:
        emit('already_uploaded')
        return
    emit('upload_start', file_count=len(pending))
    result = api.upload_folder(
        repo_id=REPO, folder_path=str(STAGE / 'hf_ckpt'),
        allow_patterns=pending, max_workers=2, revision=REVISION,
        commit_message='Upload RoboTwin 1-epoch checkpoint, global step 5717')
    results = result if isinstance(result, list) else [result]
    commits = [{key: getattr(c, key, None) for key in ('oid', 'commit_url')}
               for c in results]
    save('commits.json', commits)
    emit('commit_complete', commits=commits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upload-worker', action='store_true')
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    if args.upload_worker:
        worker()
        return
    started = time.monotonic()
    api = api_client()
    emit('private_repo_confirmed', repo=REPO)
    spec = importlib.util.spec_from_file_location('obs_transfer', ROOT / 'lingbotvla/utils/obs_storage.py')
    obs = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = obs
    spec.loader.exec_module(obs)
    storage = obs.ObsStorage(SOURCE)
    manifest, digest = storage._committed(SOURCE, 'checkpoint')
    if manifest.get('global_step') != 5717 or manifest.get('epoch') != 1:
        raise RuntimeError('Unexpected checkpoint identity')
    records = [r for r in manifest['files'] if r['path'].startswith('hf_ckpt/')]
    if len(records) != 17:
        raise RuntimeError('Unexpected HF file count')
    normalized = [{**r, 'path': r['path'].removeprefix('hf_ckpt/')} for r in records]
    save('upload_plan.json', {'repo': REPO, 'visibility': 'private',
         'source': SOURCE + '/hf_ckpt/', 'manifest_sha256': digest,
         'revision': REVISION, 'files': normalized})
    before = remote_files(api)
    conflicts = [r['path'] for r in normalized if r['path'] in before and
                 r['path'] in mismatches([r], before)]
    if conflicts:
        raise RuntimeError('Destination has differing model files; refusing overwrite')
    save('remote_before.json', [dict(Path=f['Path'], Size=f.get('Size'), Sha256=f.get('Sha256'))
                                for f in before.values()])
    # Initialize OBS before starting threads. Download only the requested HF tree.
    storage.file
    emit('obs_download_start', files=len(records), bytes=sum(r['size'] for r in records))
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(storage._download_records, [r], SOURCE, STAGE): r for r in records}
        for job in as_completed(jobs):
            job.result()
            r = jobs[job]
            emit('download_verified', path=r['path'], bytes=r['size'])
    index = json.loads((STAGE / 'hf_ckpt/model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    expected_shards = {r['path'] for r in normalized if r['path'].endswith('.safetensors')}
    if shards != expected_shards:
        raise RuntimeError('HF index does not match the uploaded shard set')
    emit('all_source_files_verified', seconds=round(time.monotonic() - started, 1))
    for attempt in range(1, 4):
        emit('upload_attempt', attempt=attempt)
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--upload-worker'],
                                    timeout=3600, check=False)
            if result.returncode == 0:
                break
            emit('upload_retry_needed', exit_code=result.returncode)
        except subprocess.TimeoutExpired:
            emit('upload_attempt_timeout')
        if attempt == 3:
            raise RuntimeError('Upload exhausted retries; verified cache retained')
        time.sleep(5)
    require_private(api)
    remote = remote_files(api)
    bad = mismatches(normalized, remote)
    if bad:
        emit('verification_mismatch', files=bad)
        raise RuntimeError('Remote file metadata verification failed')
    receipt = {'completed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               'repo': REPO, 'url': f'https://modelscope.cn/models/{REPO}',
               'visibility': 'private', 'source': SOURCE + '/hf_ckpt/',
               'source_manifest_sha256': digest, 'revision': REVISION,
               'file_count': len(normalized), 'total_bytes': sum(r['size'] for r in normalized),
               'elapsed_seconds': round(time.monotonic() - started, 1),
               'validation': 'OBS manifest SHA256 -> local full SHA256 -> ModelScope Size/Sha256 for every file; HF shard index complete',
               'files': [{**r, 'remote_size': int(remote[r['path']]['Size']),
                          'remote_sha256': remote[r['path']]['Sha256']} for r in normalized]}
    save('upload_receipt.json', receipt)
    emit('upload_verified', files=len(normalized), seconds=receipt['elapsed_seconds'])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Never print HTTP exception bodies or signed headers.
        emit('failed', error_type=type(exc).__name__)
        sys.exit(1)
