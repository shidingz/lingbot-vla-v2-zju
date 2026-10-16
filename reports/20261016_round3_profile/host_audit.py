"""Read-only audit of profiling configuration against the completed 160-step run."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import yaml


HERE = Path(__file__).resolve().parent
BASELINE = Path("/cache/lingbot/profiling/round2_default_20261016_185821/default_160")


def flatten(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    out = {}
    for key, child in value.items():
        out.update(flatten(child, f"{prefix}.{key}" if prefix else str(key)))
    return out


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    before = flatten(yaml.safe_load((BASELINE / "logs/lingbotvla_cli.yaml").read_text()))
    after = flatten(yaml.safe_load((run / "logs/lingbotvla_cli.yaml").read_text()))
    differences = []
    for key in sorted(set(before) | set(after)):
        if before.get(key) == after.get(key):
            continue
        if key in ("train.output_dir", "train.logging_dir", "train.profile_trace_dir"):
            category = "isolated_output_destination"
        elif key.startswith("train.profile_") or key == "train.enable_profiling":
            category = "profiling_configuration"
        elif key == "train.stop_after_steps":
            category = "bounded_experiment_length"
        else:
            category = "unexpected_training_difference"
        differences.append(dict(field=key, baseline=before.get(key), current=after.get(key),
                                classification=category))
    old_manifest = load(BASELINE / "experiment_manifest.json")
    manifest = load(run / "experiment_manifest.json")
    old_snapshot = Path(old_manifest["source_snapshot"])
    snapshot = Path(manifest["source_snapshot"])
    paths = [p for d in ("lingbotvla", "tasks") for p in (snapshot / d).rglob("*.py")]
    source_changes = []
    for path in sorted(paths):
        relative = path.relative_to(snapshot)
        old = old_snapshot / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not old.exists() or digest != hashlib.sha256(old.read_bytes()).hexdigest():
            source_changes.append(str(relative))
    instrumentation = load(run / "instrumentation.json")
    expected_instrumented = {"lingbotvla/utils/training_profiler.py", "tasks/vla/train_lingbotvla.py"}
    instrumentation_hashes_match = True
    for change in instrumentation["changes"]:
        rel = change["path"]
        instrumentation_hashes_match &= (
            change["sha256_before"] == hashlib.sha256((old_snapshot / rel).read_bytes()).hexdigest()
            and change["sha256_after"] == hashlib.sha256((snapshot / rel).read_bytes()).hexdigest()
        )
    dry = (run / "default_dry_run.log").read_text()
    expected = dict(LINGBOT_NPU_MOE_IMPL="gmm", LINGBOT_JOINT_ATTN_BACKEND="sdpa_fp32",
                    LINGBOT_MOE_MONITOR_BATCH_COUNTS="1", LINGBOT_ADAMW_FOREACH_NPU="1")
    switches = {key: f"{key}={value}" in dry for key, value in expected.items()}
    checks = dict(
        no_training_config_difference=all(d["classification"] != "unexpected_training_difference" for d in differences),
        optimization_switches_match=all(switches.values()),
        checkpoint_disabled=after.get("train.enable_gradient_checkpointing") is False,
        dataset_list_hash_equal=manifest.get("dataset_list_sha256") == old_manifest.get("dataset_list_sha256"),
        source_changes_only_reviewed_instrumentation=set(source_changes) == expected_instrumented,
        instrumentation_hashes_match=instrumentation_hashes_match,
        no_added_synchronization=instrumentation["additional_sync_calls"] == 0,
    )
    result = dict(audited_at=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
                  baseline=str(BASELINE), candidate=str(run),
                  effective_yaml_differences=differences,
                  config_passed=all(checks.values()),
                  checks=checks, actual_launcher_switch_checks=switches,
                  optimization_flags=expected, source_files_compared=len(paths),
                  changed_training_python_files=source_changes,
                  source_review_note="Reviewed instrumentation.json diffs: training changes wrap existing sync/scalar-read blocks in record_function ranges; profiler changes wrap four existing functions and aliases with capture-window-gated ranges. No tensor mathematics, optimizer policy, data transformation, or explicit synchronization added. Timing annotations can still perturb performance and profiled steps must be excluded.",
                  effective_data_config={key: value for key, value in after.items() if any(x in key for x in ("num_workers", "prefetch", "pin_memory", "batch_size"))},
                  actual_profile_config={key: value for key, value in after.items() if "profil" in key},
                  limitations=["Full external data/model contents and imported dependencies are not rehashed.",
                               "Profiling and CPU stack sampled windows cannot be used as unprofiled throughput."])
    destination = HERE / "host_config_audit.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(destination), "config_passed": result["config_passed"],
                      "yaml_difference_count": len(differences), "changed_training_python_files": source_changes}))
    raise SystemExit(not result["config_passed"])


if __name__ == "__main__":
    main()
