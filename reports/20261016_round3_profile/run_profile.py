"""Bounded eight-rank profiling of the current main launcher defaults.

Copies source before launching; never saves checkpoints or uploads artifacts.
Performance flags, batch, optimizer and LR are intentionally not overridden.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TZ = dt.timezone(dt.timedelta(hours=8))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=132)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.steps < 128:
        raise SystemExit("Profile run must include export and cleanup steps")
    root = args.root or Path("/cache/lingbot/profiling") / (
        "round3_profile_" + dt.datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    run = root / f"profile_{args.steps}"
    run.mkdir()
    snapshot = root / "source"
    snapshot.mkdir()
    before = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True)
    (run / "npu_before.txt").write_text(before.stdout)
    if before.stdout.count("No running processes found in NPU") != 8:
        raise SystemExit("Expected eight idle NPUs; no training launched")

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".pytest_cache", "kernel_meta")
    for directory in ("lingbotvla", "tasks", "configs", "tools"):
        shutil.copytree(REPO / directory, snapshot / directory, ignore=ignore)
    shutil.copytree(REPO / "assets/norm_stats", snapshot / "assets/norm_stats", ignore=ignore)
    for name in ("train_robotwin_npu.sh", "train_robotwin_npu_optimized.sh", "pyproject.toml"):
        shutil.copy2(REPO / name, snapshot / name)
    from instrument_snapshot import apply
    instrumentation = apply(snapshot)
    write_json(run / "instrumentation.json", instrumentation)
    hashes = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob("*")) if p.is_file()}

    script = snapshot / "train_robotwin_npu.sh"
    override_keys = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)(?:[:+?-])", script.read_text()))
    env = os.environ.copy()
    for key in override_keys - {"HOME", "CODEX_HOME"}:
        env.pop(key, None)
    env.pop("GATE_GRAD_PROBE", None)
    overrides = dict(
        REPO_ROOT=str(snapshot), RUN_NAME=f"round3_profile_{args.steps}",
        OUTPUT_DIR=str(run / "run"), LOG_DIR=str(run / "logs"),
        OBS_OUTPUT_DIR="", OBS_DATASET_URI="", ENABLE_RESUME="false",
        SAVE_HF_WEIGHTS="false", SAVE_STEPS="0", SAVE_EPOCHS="0",
        STOP_AFTER_STEPS=str(args.steps), MASTER_PORT="62549", PYTHONUNBUFFERED="1",
    )
    env.update(overrides)
    env["PYTHONPATH"] = str(snapshot) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = ["bash", str(script),
               "--train.enable_profiling", "true", "--train.profile_only", "true",
               "--train.profile_ranks", "all", "--train.profile_start_step", "121",
               "--train.profile_end_step", "125", "--train.profile_trace_dir", str(run / "trace"),
               "--train.profile_record_shapes", "false", "--train.profile_profile_memory", "false",
               "--train.profile_with_stack", "false"]
    dry = subprocess.run(command, cwd=snapshot, env={**env, "DRY_RUN": "1"},
                         capture_output=True, text=True, check=True)
    (run / "default_dry_run.log").write_text(dry.stdout + dry.stderr)
    for expected in ("LINGBOT_NPU_MOE_IMPL=gmm", "LINGBOT_JOINT_ATTN_BACKEND=sdpa_fp32",
                     "LINGBOT_MOE_MONITOR_BATCH_COUNTS=1", "LINGBOT_ADAMW_FOREACH_NPU=1",
                     "--train.enable_gradient_checkpointing false", "--train.global_batch_size 96"):
        if expected not in dry.stdout:
            raise RuntimeError(f"Unexpected main launcher defaults: missing {expected}")
    manifest = dict(
        started_at=dt.datetime.now(TZ).isoformat(), source_snapshot=str(snapshot),
        source_repository=str(REPO), steps=args.steps, command=command,
        environment_overrides=overrides, performance_flags_from_main_defaults=True,
        profile_recording=True, checkpoints=False, uploads=False,
        hashes=hashes, native_profile_steps=[121,124], native_warmup_step=120,
        planned_unprofiled_windows=[[13,42],[81,116]], instrumentation=instrumentation,
        dataset_list_sha256=hashlib.sha256(Path("/cache/lingbot/robotwin_data/train_robotwin_clean.txt").read_bytes()).hexdigest(),
        limits=["Bounded integration run, not full-epoch or task-success validation.",
                "Resources sampled on all 8 cards; allocator peak log covers rank 0.",
                "Model/data paths retained; their full contents are not rehashed."],
    )
    write_json(run / "experiment_manifest.json", manifest)
    write_json(HERE / "latest_run.json", {"root": str(root), "run": str(run), "steps": args.steps})
    print(json.dumps({"status": "starting", "run": str(run), "steps": args.steps}), flush=True)

    sampler = HERE / "sample_resources.py"
    shutil.copy2(sampler, run / "sample_resources.py")
    status = 99
    with (run / "resource_collector.log").open("w") as resource_log:
        collector = subprocess.Popen([sys.executable, str(run / "sample_resources.py"), str(run),
                                      "--max-seconds", "3600"], stdout=resource_log, stderr=subprocess.STDOUT)
        try:
            with (run / "console.log").open("w") as output:
                status = subprocess.run(command, cwd=snapshot, env=env,
                                        stdout=output, stderr=subprocess.STDOUT).returncode
        finally:
            (run / "exit_code").write_text(str(status) + "\n")
            try:
                collector.wait(timeout=20)
            except subprocess.TimeoutExpired:
                collector.terminate()
                collector.wait(timeout=10)
    changes = [name for name, digest in hashes.items()
               if not (snapshot / name).is_file()
               or hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != digest]
    write_json(run / "snapshot_hash_check.json", {"passed": not changes, "changed": changes,
                                                  "checked_files": len(hashes)})
    after = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
    (run / "npu_after.txt").write_text(after.stdout + after.stderr)
    print(json.dumps({"status": "finished", "exit_code": status, "run": str(run),
                      "snapshot_unchanged": not changes}), flush=True)
    raise SystemExit(status or bool(changes))


if __name__ == "__main__":
    main()
