"""Summarize completed profiling run, preserving measurement exclusions."""
import collections
import datetime as dt
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("round2", HERE.parent / "20261016_speed_round2/analyze_round2.py")
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)

def resources(samples, first, last):
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    count = 0
    for sample in samples:
        # Reject a query cycle crossing either boundary, rather than pretending
        # that all eight cards were measured simultaneously.
        if not first - 1 <= sample["last_step"] <= sample["last_step_at_end"] < last:
            continue
        count += 1
        for card, query in sample["cards"].items():
            if query.get("returncode") != 0:
                continue
            for metric, value in query.get("metrics", {}).items():
                if analyzer.finite(value):
                    grouped[card][metric].append(value)
    return {"query_cycles": count,
            "cards": {card: {metric: analyzer.stats(values) for metric, values in metrics.items()}
                      for card, metrics in sorted(grouped.items())},
            "limitation": "Eight cards sampled in groups of four; per-card timestamps retained. Samples are not simultaneous, continuous time averages, or MFU."}

def main():
    metadata = json.loads((HERE / "latest_run.json").read_text())
    run = Path(metadata["run"])
    case, data = analyzer.read_case(run, metadata["steps"])
    assert case["eligible"], case["exclusion_reasons"]
    assert case["updates"] == metadata["steps"]
    assert case["lr_horizons"] == [5717]
    assert case["global_batch_size"] == 96
    snapshot = json.loads((run / "snapshot_hash_check.json").read_text())
    assert snapshot["passed"], snapshot
    host = json.loads((run / "host_sampling.json").read_text())
    host_excluded = host.get("exclude_steps_inclusive")
    assert host_excluded and all(c == 0 for c in host["return_codes"].values()), host
    windows = {}
    for start, end in [(13,42),(81,116),(121,124)]:
        result = analyzer.window_summary(case, data, start, end)
        result["resources"] = resources(data["resources"], start, end)
        overlap = start <= host_excluded[1] and host_excluded[0] <= end
        result["host_sampling_overlap"] = overlap
        result["usable_for_speed_comparison"] &= not overlap
        if start < 120:
            assert result["usable_for_speed_comparison"]
        windows[f"{start}-{end}"] = result
    baseline = windows["81-116"]
    capture = windows["121-124"]
    overhead = 100 * (capture["logged_step_seconds"]["mean"] / baseline["logged_step_seconds"]["mean"] - 1)
    summary = {
        "completed_at": dt.datetime.now().astimezone().isoformat(),
        "status": "passed", "run": str(run), "updates": case["updates"],
        "exit_code": case["exit_code"], "all_numeric_values_finite": not case["nonfinite_numeric_records"],
        "lr_horizon": 5717, "global_batch_size": 96,
        "rank0_peak_allocated_gib": case["peak_allocated_gib"],
        "windows": windows, "host_sampling_excluded_steps": host_excluded,
        "native_warmup_step": 120, "native_active_steps": [121,124],
        "capture_logged_time_overhead_pct_vs_unprofiled": overhead,
        "snapshot_hash_check": snapshot,
        "limits": [
            "Profiler measurements include collection overhead; their absolute durations cannot be substituted for unprofiled step durations.",
            "The capture-window logged-time increase is a within-run comparison, not an independent A/B attribution of profiler cost; it excludes trace export after update 124 and other outside-timer work.",
            "CPU scopes and kernel totals are asynchronous/inclusive and must not be added into wall time.",
            "Device communication elapsed includes dependencies and cannot by itself prove a network bandwidth bottleneck.",
            "All steps from 120 onward are excluded from ordinary throughput; trace export after update 124 may delay later updates.",
            "Only four native steps per rank and two ranks of Python stacks are captured; not a full epoch.",
        ],
    }
    (HERE / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"status":summary["status"], "updates":case["updates"],
                      "baseline_wall_s":baseline["wall_seconds_per_update"],
                      "profiled_logged_s":capture["logged_step_seconds"]["mean"],
                      "overhead_pct":overhead,
                      "rank0_peak_allocated_gib":case["peak_allocated_gib"]},indent=2))

if __name__ == "__main__":
    main()
