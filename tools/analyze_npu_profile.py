#!/usr/bin/env python3
"""Summarize native Ascend profiler CSVs and phase ranges without summing nested CPU time."""
import argparse
import collections
import csv
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def number(value):
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return 0.0


def rows(path):
    with path.open(newline="") as stream:
        yield from csv.DictReader(stream)


def union_us(intervals):
    end = float("-inf")
    duration = 0.0
    for left, right in sorted(intervals):
        duration += max(0.0, right - max(left, end))
        end = max(end, right)
    return duration


def analyze(output):
    trace = json.loads((output / "trace_view.json").read_text())
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    phases = collections.defaultdict(list)
    for event in events:
        name = event.get("name", "")
        if event.get("ph") == "X" and name.startswith(("train.", "teacher.", "model.moe_", "ProfilerStep#")):
            phases[name].append(number(event["dur"]))
    steps = [v for k, vs in phases.items() if k.startswith("ProfilerStep#") for v in vs]
    count = len(steps)
    phase_summary = {k: {"calls": len(v), "total_s": sum(v) / 1e6,
                         "per_step_s": sum(v) / max(count, 1) / 1e6,
                         "mean_call_ms": statistics.mean(v) / 1e3}
                     for k, v in phases.items() if not k.startswith("ProfilerStep#")}
    operators = collections.defaultdict(collections.Counter)
    for row in rows(output / "operator_details.csv"):
        a = operators[row["Name"]]
        a["calls"] += 1
        for src, key in [("Host Self Duration(us)", "host_self_s"),
                         ("Host Total Duration(us)", "host_total_s"),
                         ("Device Self Duration(us)", "device_self_s"),
                         ("Device Total Duration(us)", "device_total_s")]:
            a[key] += number(row[src]) / 1e6
    kernel_types = collections.defaultdict(collections.Counter)
    compute_intervals, communication_intervals = [], []
    for row in rows(output / "kernel_details.csv"):
        start = number(row["Start Time(us)"])
        dur = number(row["Duration(us)"])
        a = kernel_types[row["Type"]]
        a["calls"] += 1
        a["duration_s"] += dur / 1e6
        core = row.get("Accelerator Core", "")
        if core == "HCCL" or "hccl" in row["Name"].lower():
            communication_intervals.append((start, start + dur))
        else:
            compute_intervals.append((start, start + dur))
    step_table = list(rows(output / "step_trace_time.csv"))
    step_metrics = {k: statistics.mean(number(r[k]) for r in step_table) / 1e6
                    for k in step_table[0] if k not in ("Device_id", "Step")} if step_table else {}
    def top(field):
        return [{"name": k, **dict(v)} for k, v in sorted(operators.items(), key=lambda kv: kv[1][field], reverse=True)[:25]]
    return {
        "output": str(output), "recorded_steps": count,
        "host_step_mean_s": statistics.mean(steps) / 1e6 if steps else None,
        "phases": phase_summary,
        "step_trace_mean_s": step_metrics,
        "compute_union_s_per_step": union_us(compute_intervals) / max(count, 1) / 1e6,
        "communication_union_s_per_step": union_us(communication_intervals) / max(count, 1) / 1e6,
        "top_host_self": top("host_self_s"), "top_device_self": top("device_self_s"),
        "top_device_total_inclusive": top("device_total_s"),
        "kernel_types": [{"type": k, **dict(v)} for k, v in sorted(kernel_types.items(), key=lambda kv: kv[1]["duration_s"], reverse=True)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    paths = sorted((args.profile_root / "trace").glob("rank*/*/ASCEND_PROFILER_OUTPUT"))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(analyze, paths))
    for path, summary in zip(paths, results):
        rank = path.parents[1].name
        summaries[rank] = summary
        print(rank, "steps", summaries[rank]["recorded_steps"], "host_step_s", summaries[rank]["host_step_mean_s"], flush=True)
    if not summaries:
        raise SystemExit("No native profiler outputs found")
    (args.output / "profile_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
