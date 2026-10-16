#!/usr/bin/env python3
"""Attribute CPU ranges/synchronization in the *new* round-3 Ascend traces.

All duration tables are host-side, inclusive unless explicitly marked exclusive.
Device overlap answers what was in flight while the CPU waited, not what caused
the wait. HCCL duration and runtime synchronization must not be added together.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
import statistics

import ijson


def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def duration(intervals):
    return sum(end - start for start, end in merge(intervals))


def intersection(left, right):
    left, right = merge(left), merge(right)
    i = j = 0
    result = []
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if b > a:
            result.append((a, b))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def events(path):
    with path.open("rb") as stream:
        first = stream.read(1024).lstrip()[:1]
        stream.seek(0)
        yield from ijson.items(stream, "item" if first == b"[" else "traceEvents.item", use_float=True)


def custom_scope(name):
    return name.startswith(("train.", "teacher.", "model.moe_", "optimizer.", "Optimizer.step#"))


def sync_kind(name):
    lower = name.lower()
    if "synchronize" in lower or "synchronizestream" in lower:
        return "runtime_synchronize"
    if "_local_scalar_dense" in lower:
        return "scalar_device_read"
    if name == "aten::item":
        return "scalar_item"
    if name == "c10d::wait_tensor" or "wait_work" in lower:
        return "distributed_wait"
    return None


def analyze(output):
    scopes, syncs, steps = [], [], []
    categories = collections.Counter()
    for event in events(output / "trace_view.json"):
        categories[event.get("cat", "")] += 1
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        cat = event.get("cat", "")
        if name.startswith("ProfilerStep#") or custom_scope(name) or (sync_kind(name) and cat not in ("kernel", "HCCL")):
            start = float(event["ts"])
            row = {"name": name, "cat": cat, "pid": event.get("pid"), "tid": event.get("tid"), "start": start, "end": start + float(event["dur"])}
            if name.startswith("ProfilerStep#"):
                steps.append(row)
            elif custom_scope(name):
                scopes.append(row)
            else:
                row["kind"] = sync_kind(name)
                syncs.append(row)
    if not steps:
        raise ValueError(f"No captured ProfilerStep in {output}")
    step_intervals = merge((s["start"], s["end"]) for s in steps)
    step_us = duration(step_intervals)
    nsteps = len(steps)
    by_thread = collections.defaultdict(list)
    by_os_tid = collections.defaultdict(list)
    by_name = collections.defaultdict(list)
    for scope in scopes:
        by_thread[(scope["pid"], scope["tid"])].append(scope)
        by_os_tid[scope["tid"]].append(scope)
        by_name[scope["name"]].append(scope)
    for values in by_thread.values():
        values.sort(key=lambda x: (x["start"], -x["end"]))
    sync_owners, sync_ops = collections.defaultdict(list), collections.defaultdict(list)
    for event in syncs:
        thread_scopes = by_thread.get((event["pid"], event["tid"]), [])
        # AscendCL uses a synthetic trace PID, but retains the actual OS TID.
        if not thread_scopes and event["name"].startswith("AscendCL@"):
            thread_scopes = by_os_tid.get(event["tid"], [])
        candidates = [s for s in thread_scopes if s["start"] <= event["start"] + 1 and s["end"] >= event["end"] - 1]
        owner = min(candidates, key=lambda s: s["end"] - s["start"])["name"] if candidates else "<outside_instrumented_scope>"
        interval = (event["start"], event["end"])
        sync_owners[owner].append(interval)
        sync_ops[event["name"]].append(interval)

    # Kernel tables use the same host-aligned microsecond clock as trace_view.
    compute, hccl, memcpy = [], [], []
    kernel_path = output / "kernel_details.csv"
    with kernel_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            start = float(row["Start Time(us)"])
            interval = (start, start + float(row["Duration(us)"]))
            if row.get("Accelerator Core", "") == "HCCL" or "hccl" in row.get("Name", "").lower():
                hccl.append(interval)
            else:
                compute.append(interval)
            if "memcpy" in (row.get("Type", "") + row.get("Name", "")).lower():
                memcpy.append(interval)
    compute, hccl, memcpy = merge(compute), merge(hccl), merge(memcpy)
    device = merge(compute + hccl)
    all_sync = intersection([(e["start"], e["end"]) for e in syncs], step_intervals)

    operator_scope_totals = collections.defaultdict(collections.Counter)
    with (output / "operator_details.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if custom_scope(row["Name"]):
                counter = operator_scope_totals[row["Name"]]
                counter["calls"] += 1
                for key in ("Host Self Duration(us)", "Host Total Duration(us)", "Device Self Duration(us)", "Device Total Duration(us)"):
                    counter[key] += float(row.get(key) or 0)

    phase_stats = {}
    for name, rows in by_name.items():
        intervals = intersection([(r["start"], r["end"]) for r in rows], step_intervals)
        nested = []
        contained_sync = []
        for row in rows:
            for child in by_thread[(row["pid"], row["tid"])]:
                if child is not row and child["start"] >= row["start"] and child["end"] <= row["end"]:
                    nested.append((child["start"], child["end"]))
            for event in syncs:
                same_thread = (event["pid"], event["tid"]) == (row["pid"], row["tid"])
                same_thread = same_thread or (event["name"].startswith("AscendCL@") and event["tid"] == row["tid"])
                if same_thread and event["start"] >= row["start"] - 1 and event["end"] <= row["end"] + 1:
                    contained_sync.append((event["start"], event["end"]))
        inclusive_us = duration(intervals)
        child_us = duration(intersection(nested, intervals))
        sync_us = duration(intersection(contained_sync, intervals))
        phase_stats[name] = {
            "calls": len(rows),
            "host_inclusive_s_per_step": inclusive_us / nsteps / 1e6,
            "host_excluding_nested_custom_scopes_s_per_step": (inclusive_us - child_us) / nsteps / 1e6,
            "contained_sync_union_s_per_step": sync_us / nsteps / 1e6,
            "host_step_fraction": inclusive_us / step_us,
            "operator_csv_host_self_s_per_step": operator_scope_totals[name]["Host Self Duration(us)"] / nsteps / 1e6,
            "operator_csv_device_total_inclusive_s_per_step": operator_scope_totals[name]["Device Total Duration(us)"] / nsteps / 1e6,
            "operator_csv_device_self_s_per_step": operator_scope_totals[name]["Device Self Duration(us)"] / nsteps / 1e6,
        }

    def summarize_sync(intervals):
        intervals = intersection(intervals, step_intervals)
        total = duration(intervals)
        active = duration(intersection(intervals, device))
        return {"host_union_s_per_step": total / nsteps / 1e6,
                "device_active_overlap_s_per_step": active / nsteps / 1e6,
                "device_uncovered_overlap_s_per_step": (total - active) / nsteps / 1e6,
                "hccl_overlap_s_per_step": duration(intersection(intervals, hccl)) / nsteps / 1e6}

    return {
        "output": str(output), "captured_steps": nsteps,
        "step_names": [s["name"] for s in steps],
        "host_step_s": step_us / nsteps / 1e6,
        "trace_categories": dict(categories),
        "phases": dict(sorted(phase_stats.items(), key=lambda x: x[1]["host_inclusive_s_per_step"], reverse=True)),
        "sync_union": summarize_sync(all_sync),
        "sync_by_innermost_scope": {name: summarize_sync(values) for name, values in sorted(sync_owners.items(), key=lambda x: duration(x[1]), reverse=True)},
        "sync_by_operator": {name: {"calls": len(values), **summarize_sync(values)} for name, values in sorted(sync_ops.items(), key=lambda x: duration(x[1]), reverse=True)},
        "kernel_clock_overlap_verified": bool(intersection(device, step_intervals)),
        "memcpy_device_union_s_per_step": duration(intersection(memcpy, step_intervals)) / nsteps / 1e6,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted((args.run / "trace").glob("rank*/*/ASCEND_PROFILER_OUTPUT"))
    if not paths:
        raise SystemExit("No new trace outputs found")
    result = {"run": str(args.run.resolve()), "ranks": {}, "interpretation": [
        "Host ranges are inclusive and overlap parent ranges; do not sum teachers+subranges or optimizer+NewtonSchulz.",
        "Host runtime wait is often time spent draining earlier device work, not independent CPU compute cost.",
        "Device-active overlap with a CPU range is contemporaneous, not causal attribution to that scope.",
        "Synchronization and HCCL wall time overlap and must not be added.",
        "Profiler durations are instrumented samples; use the same run pre-capture log window for throughput.",
        "Exclusive here excludes only nested custom ranges, not every aten/runtime child.",
        "operator_csv_device_total_inclusive is summed correlated device work, may overlap and is not wall time or an additive fraction.",
        "AscendCL runtime scopes use a synthetic PID; same OS thread ID maps them to Python scopes in each rank trace.",
    ]}
    for path in paths:
        rank = path.parents[1].name
        result["ranks"][rank] = analyze(path)
        print(rank, result["ranks"][rank]["host_step_s"], flush=True)
    names = set.intersection(*(set(r["phases"]) for r in result["ranks"].values()))
    result["common_phases_across_ranks"] = {
        name: {"mean_host_s_per_step": statistics.mean(vals), "min_host_s_per_step": min(vals), "max_host_s_per_step": max(vals)}
        for name in sorted(names)
        for vals in [[r["phases"][name]["host_inclusive_s_per_step"] for r in result["ranks"].values()]]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
