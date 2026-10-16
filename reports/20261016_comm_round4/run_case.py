"""Snapshot and run one isolated 8-card communication comparison."""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TZ = dt.timezone(dt.timedelta(hours=8))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--mesh", choices=("8", "2x4", "4x2"), default="8")
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--capture-start", type=int, default=0)
    a = p.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", a.case) or a.steps < 20:
        raise SystemExit("Invalid case or steps")
    root = Path("/cache/lingbot/profiling/comm_round4_20261016") / a.case
    root.mkdir(parents=True, exist_ok=False)
    run, snapshot = root / "run_case", root / "source"
    run.mkdir()
    snapshot.mkdir()
    before = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True)
    (run / "npu_before.txt").write_text(before.stdout)
    if before.stdout.count("No running processes found in NPU") != 8:
        raise SystemExit("Expected eight idle NPUs")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".pytest_cache", "kernel_meta")
    for directory in ("lingbotvla", "tasks", "configs", "tools", "assets/norm_stats"):
        shutil.copytree(REPO / directory, snapshot / directory, ignore=ignore)
    for name in ("train_robotwin_npu.sh", "train_robotwin_npu_optimized.sh", "pyproject.toml"):
        shutil.copy2(REPO / name, snapshot / name)
    if a.capture_start:
        if a.capture_start < 121 or a.capture_start + 4 >= a.steps:
            raise SystemExit("Capture must follow measurement and leave export/cleanup steps")
        spec = importlib.util.spec_from_file_location("instrument", REPO / "reports/20261016_round3_profile/instrument_snapshot.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        instrumentation = module.apply(snapshot)
        (run / "instrumentation.json").write_text(json.dumps(instrumentation, indent=2))
    hashes = {str(f.relative_to(snapshot)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(snapshot.rglob("*")) if f.is_file()}
    script = snapshot / "train_robotwin_npu.sh"
    keys = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)(?:[:+?-])", script.read_text()))
    env = os.environ.copy()
    for k in keys - {"HOME", "CODEX_HOME"}:
        env.pop(k, None)
    for k in list(env):
        if k.startswith("LINGBOT_") or k in ("GATE_GRAD_PROBE", "ASCEND_LAUNCH_BLOCKING", "TASK_QUEUE_ENABLE"):
            env.pop(k, None)
    overrides = dict(REPO_ROOT=str(snapshot), RUN_NAME="round4_" + a.case,
                     OUTPUT_DIR=str(run / "run"), LOG_DIR=str(run / "logs"),
                     OBS_OUTPUT_DIR="", OBS_DATASET_URI="", ENABLE_RESUME="false",
                     SAVE_HF_WEIGHTS="false", SAVE_STEPS="0", SAVE_EPOCHS="0",
                     STOP_AFTER_STEPS=str(a.steps), MASTER_PORT="62561", PYTHONUNBUFFERED="1",
                     TASK_QUEUE_ENABLE="1")
    if a.mesh != "8":
        replicate, shard = a.mesh.split("x")
        overrides["EXTRA_ARGS"] = (
            f"--train.data_parallel_replicate_size {replicate} "
            f"--train.data_parallel_shard_size {shard}"
        )
    for pair in a.env:
        k, v = pair.split("=", 1)
        if not k.startswith("LINGBOT_") and k != "TASK_QUEUE_ENABLE":
            raise SystemExit("Only performance switches are accepted")
        overrides[k] = v
    env.update(overrides)
    env["PYTHONPATH"] = str(snapshot)
    command = ["bash", str(script), "--train.enable_profiling", "true",
               "--train.profile_only", "true", "--train.profile_ranks", "all" if a.capture_start else "",
               "--train.profile_start_step", str(a.capture_start or 13), "--train.profile_end_step", str(a.capture_start + 4 if a.capture_start else 17),
               "--train.profile_record_shapes", "false", "--train.profile_profile_memory", "false", "--train.profile_with_stack", "false",
               "--train.profile_trace_dir", str(run / "trace")]
    dry = subprocess.run(command, cwd=snapshot, env={**env, "DRY_RUN": "1"}, capture_output=True, text=True, check=True)
    (run / "dry_run.log").write_text(dry.stdout + dry.stderr)
    manifest = dict(started_at=dt.datetime.now(TZ).isoformat(), source_snapshot=str(snapshot), mesh=a.mesh,
                    steps=a.steps, command=command, environment_overrides=overrides,
                    hashes=hashes, profile_recording=bool(a.capture_start), capture_start=a.capture_start, checkpoints=False, uploads=False,
                    dataset_list_sha256=hashlib.sha256(Path("/cache/lingbot/robotwin_data/train_robotwin_clean.txt").read_bytes()).hexdigest())
    (run / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    (HERE / (a.case + ".json")).write_text(json.dumps({"run": str(run), "source": str(snapshot)}, indent=2))
    print(json.dumps({"status": "starting", "run": str(run)}), flush=True)
    sampler = REPO / "reports/20261016_round3_profile/sample_resources.py"
    status = 99
    with (run / "resource_collector.log").open("w") as log:
        collector = subprocess.Popen([sys.executable, str(sampler), str(run), "--max-seconds", "2400"], stdout=log, stderr=subprocess.STDOUT)
        try:
            with (run / "console.log").open("w") as out:
                status = subprocess.run(command, cwd=snapshot, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
        finally:
            (run / "exit_code").write_text(str(status) + "\n")
            try:
                collector.wait(timeout=15)
            except subprocess.TimeoutExpired:
                collector.terminate()
                collector.wait(timeout=10)
    changes = [name for name, h in hashes.items() if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != h]
    (run / "snapshot_hash_check.json").write_text(json.dumps({"passed": not changes, "changed": changes, "checked_files": len(hashes)}, indent=2))
    print(json.dumps({"status": "finished", "exit_code": status, "run": str(run), "snapshot_unchanged": not changes}), flush=True)
    raise SystemExit(status or bool(changes))


if __name__ == "__main__":
    main()
