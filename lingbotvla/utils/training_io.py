"""Small, CPU-only orchestration helpers for OBS training artifacts."""
import json
import os
import re
import shutil
from pathlib import Path


def rank_zero_action(distributed, rank, action, label):
    """Broadcast rank-zero failures as well as results before any rank continues."""
    payload = [None]
    if rank == 0:
        try:
            payload[0] = {"result": action(), "error": None}
        except Exception as exc:
            payload[0] = {"result": None, "error": f"{type(exc).__name__}: {exc}"}
    distributed.broadcast_object_list(payload, src=0)
    if payload[0]["error"]:
        raise RuntimeError(f"{label} failed; local checkpoint retained: {payload[0]['error']}")
    return payload[0]["result"]


def snapshot_run_assets(args, dataset_root):
    """Archive small, actual training inputs; never modify the dataset itself."""
    root = Path(args.train.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    robot_source = Path(args.data.robot_config_root) / "robotwin.yaml"
    if not robot_source.is_absolute():
        robot_source = Path.cwd() / robot_source
    import yaml
    robot_cfg = yaml.safe_load(robot_source.read_text())
    stats_source = Path(robot_cfg["norm_stats"])
    if not stats_source.is_absolute():
        stats_source = Path.cwd() / stats_source
    for source, relative in (
        (robot_source, "configs/robot_configs/robotwin.yaml"),
        (stats_source, "assets/norm_stats/robotwin.json"),
        (Path(args.data.train_path), "train_robotwin.txt"),
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    provenance = {
        "obs_dataset_uri": args.data.obs_dataset_uri,
        "local_dataset_root": dataset_root,
        "train_path": args.data.train_path,
        "source_status": "friend-provided LeRobot conversion; official clean provenance not independently verified"
            if args.data.obs_dataset_uri.endswith("/RoboTwin_lerobot_v30/") or args.data.obs_dataset_uri.endswith("/RoboTwin_lerobot_v30")
            else "user-configured dataset",
        "raw_dataset_modified": False,
        "video_boundary_policy": "same-episode last valid frame for at most two missing tail frames",
        "pyav_decode_threads": int(os.environ.get("LINGBOT_VIDEO_DECODE_THREADS", "1")),
        "pyav_frame_conversion": "nearest-timestamp selection before RGB conversion",
        "code_repository": str(repo),
    }
    (root / "data_source.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")


def cleanup_published_checkpoint(checkpoint_dir, run_dir):
    """Only a direct global_step_N child of this run may be removed."""
    checkpoint = Path(checkpoint_dir)
    expected_parent = Path(run_dir).resolve() / "checkpoints"
    if checkpoint.is_symlink() or checkpoint.resolve().parent != expected_parent:
        raise ValueError("Refusing checkpoint cleanup outside this run's checkpoints directory")
    if not re.fullmatch(r"global_step_\d+", checkpoint.name):
        raise ValueError("Refusing cleanup of a non-checkpoint directory")
    if checkpoint.exists():
        shutil.rmtree(checkpoint)


def publish_checkpoint(storage, checkpoint_dir, run_dir, logging_dir, step, epoch, remove_local=True):
    """Upload first, record confirmation, and only then reclaim the local files."""
    run = Path(run_dir)
    logs = Path(logging_dir)
    # Copy live logs to stable per-save snapshots before the upload starts logging.
    snapshot = run / "logs"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in ("loss.jsonl", "train.log"):
        source = logs / name
        if source.is_file() and source.resolve() != (snapshot / name).resolve():
            shutil.copy2(source, snapshot / name)
    result = storage.upload_checkpoint(checkpoint_dir, run_dir, step, epoch)
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "obs_checkpoints.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"step": step, "epoch": epoch, **result}, ensure_ascii=False) + "\n")
    if remove_local:
        cleanup_published_checkpoint(checkpoint_dir, run_dir)
    return result


def resume_signature(args):
    """Settings that must agree when continuing an existing optimizer trajectory."""
    train_fields = (
        "optimizer", "lr", "lr_min", "lr_start", "lr_decay_style", "lr_decay_ratio",
        "lr_warmup_ratio", "weight_decay", "global_batch_size", "train_expert_only",
        "chunk_size", "action_dim", "max_state_dim", "use_moe_expert_lr",
    )
    align = dict(args.train.align_params)
    align.pop("visual_dir", None)
    return {
        "train": {key: getattr(args.train, key) for key in train_fields},
        "align_params": align,
        "obs_dataset_uri": args.data.obs_dataset_uri.rstrip("/"),
    }
