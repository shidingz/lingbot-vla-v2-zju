#!/usr/bin/env python3
"""CPU-only Ascend analysis; stream one rank at a time and never sum nested wall time.

Usage:
  python3 analyze_profile.py --trace-root /path/to/run/trace --output-dir ./analysis
  python3 analyze_profile.py --trace-root /path/to/run/trace --output-dir ./csv --csv-only

Standard library only. Input artifacts are read-only. Output directory is explicit.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import datetime
import json
import math
import re
import statistics
from decimal import Decimal
from pathlib import Path


def number(value):
    try:
        result = float(str(value).strip())
        return result if math.isfinite(result) else 0.0
    except (ValueError, TypeError):
        return 0.0


def ns(value):
    """Epoch microseconds -> integer nanoseconds without float cancellation."""
    return int(Decimal(str(value).strip()) * 1000)


def rows(path):
    with path.open(newline="") as stream:
        yield from csv.DictReader(stream)


def merge(intervals):
    result = []
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if result and left <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], right))
        else:
            result.append((left, right))
    return result


def length(intervals):
    return sum(right - left for left, right in intervals)


def intersect(left, right):
    """Intersect two sorted, disjoint interval sets."""
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if b > a:
            result.append((a, b))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def gaps(intervals):
    return [(a[1], b[0]) for a, b in zip(intervals, intervals[1:]) if b[0] > a[1]]


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def is_communication(name, kind, core):
    return (core.upper() in {"COMMUNICATION", "HCCL"}
            or "hccl" in name.lower() or name.lower().startswith("hcom")
            or kind.lower().startswith(("hcom", "hccl")))


def iter_events(path):
    """Stream a Chrome trace array, or an object containing traceEvents.

    Buffer use is bounded by chunk size plus the largest individual event.
    No third-party ijson dependency and no whole-file JSON materialization.
    """
    decoder, chunk_size = json.JSONDecoder(), 2 ** 20
    with path.open() as stream:
        buffer = stream.read(chunk_size)
        first = buffer.lstrip()
        if first.startswith("["):
            pos = buffer.index("[") + 1
        else:
            match = re.search(r'"traceEvents"\s*:\s*\[', buffer)
            if not match:
                raise ValueError(f"Cannot find traceEvents array near beginning of {path}")
            pos = match.end()
        while True:
            while pos < len(buffer) and buffer[pos] in " \r\n\t,":
                pos += 1
            if pos < len(buffer) and buffer[pos] == "]":
                return
            try:
                event, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError as exc:
                more = stream.read(chunk_size)
                if not more:
                    raise ValueError(f"Incomplete trace {path}") from exc
                buffer = buffer[pos:] + more
                pos = 0
                continue
            pos = end
            if not isinstance(event, dict):
                raise ValueError(f"Unexpected trace element: {type(event)}")
            yield event
            if pos >= chunk_size:
                buffer = buffer[pos:] + stream.read(chunk_size)
                pos = 0


def top_counts(groups, nsteps, limit=30):
    output = []
    for name, counts in sorted(groups.items(), key=lambda item: item[1]["duration_ns"], reverse=True)[:limit]:
        output.append({"name": name, "calls": counts["calls"],
                       "calls_per_step": counts["calls"] / nsteps,
                       "duration_sum_s_per_step": counts["duration_ns"] / nsteps / 1e9,
                       "mean_call_us": counts["duration_ns"] / max(counts["calls"], 1) / 1e3})
    return output


def analyze_csv(output):
    official_rows = list(rows(output / "step_trace_time.csv"))
    if not official_rows:
        raise ValueError(f"No official steps: {output}")
    step_ids = {str(r["Step"]) for r in official_rows}
    nsteps = len(official_rows)
    official = [{"step": str(r["Step"]), "device_id": r["Device_id"],
                 **{key + "_s": number(value) / 1e6 for key, value in r.items()
                    if key not in {"Step", "Device_id"}}} for r in official_rows]
    by_step = {s: {"compute": [], "communication": []} for s in step_ids}
    types, cores, collectives, shapes = (collections.defaultdict(collections.Counter) for _ in range(4))
    durations, excluded = [], collections.Counter()
    first_hcom, kernel_count, pipe_columns = None, 0, []
    collective_events = []
    pipes = collections.defaultdict(collections.Counter)
    for line, row in enumerate(rows(output / "kernel_details.csv"), 2):
        if line == 2:
            pipe_columns = [key for key in row if key.startswith(("aic", "aiv", "cube_"))]
        step = str(row.get("Step Id", ""))
        if step not in by_step:
            excluded[step] += 1
            continue
        name, kind, core = row["Name"], row["Type"], row.get("Accelerator Core", "")
        start, dur = ns(row["Start Time(us)"]), ns(row["Duration(us)"])
        if dur < 0:
            raise ValueError("Negative kernel duration")
        kernel_count += 1
        comm = is_communication(name, kind, core)
        category = "communication" if comm else "compute"
        by_step[step][category].append((start, start + dur))
        for group, key in ((types, kind), (cores, core)):
            group[key]["calls"] += 1
            group[key]["duration_ns"] += dur
        if comm:
            collectives[kind]["calls"] += 1
            collectives[kind]["duration_ns"] += dur
            if first_hcom is None:
                first_hcom = {"line": line, "name": name, "type": kind, "core": core}
            collective_events.append({"step": step, "name": name, "type": kind,
                                      "start_ns": start, "end_ns": start + dur,
                                      "duration_ms": dur / 1e6})
        else:
            durations.append(dur / 1e3)
            # Hardware pipe counters are kernel-local metrics, never whole-step MFU.
            aic_time, aiv_time = number(row.get("aicore_time(us)", row.get("aic_time(us)"))), number(row.get("aiv_time(us)"))
            pipe = pipes[kind + " / " + core]
            pipe["calls"] += 1
            if aic_time > 0:
                pipe["measured_aic_calls"] += 1
                pipe["measured_aic_kernel_duration_ns"] += dur
                pipe["aic_time_us"] += aic_time
                pipe["aic_mac_time_us"] += number(row.get("aic_mac_time(us)"))
                pipe["aic_mte1_time_us"] += number(row.get("aic_mte1_time(us)"))
                pipe["aic_mte2_time_us"] += number(row.get("aic_mte2_time(us)"))
                pipe["aic_fixpipe_time_us"] += number(row.get("aic_fixpipe_time(us)"))
                if row.get("cube_utilization(%)", "") not in {"", "N/A", "None"}:
                    pipe["cube_utilization_valid_calls"] += 1
                    pipe["cube_utilization_valid_duration_ns"] += dur
                    pipe["duration_weighted_cube_utilization_percent_numerator"] += dur * number(row["cube_utilization(%)"])
            if aiv_time > 0:
                pipe["measured_aiv_calls"] += 1
                pipe["aiv_time_us"] += aiv_time
                pipe["aiv_vec_time_us"] += number(row.get("aiv_vec_time(us)"))
            if any(term in kind.lower() for term in ("matmul", "gemm", "flashattention")):
                key = json.dumps([kind, row.get("Input Shapes", ""), row.get("Input Data Types", "")])
                shapes[key]["calls"] += 1
                shapes[key]["duration_ns"] += dur
    all_compute, all_communication, all_gaps, per_step = [], [], [], []
    for step in sorted(by_step, key=lambda value: int(value)):
        compute, comm = (merge(by_step[step][key]) for key in ("compute", "communication"))
        combined = merge(compute + comm)
        comp_ns, comm_ns, combined_ns = map(length, (compute, comm, combined))
        free = gaps(combined)
        raw = {"step": step,
               "compute_union_s": comp_ns / 1e9,
               "communication_union_s": comm_ns / 1e9,
               "overlap_s": (comp_ns + comm_ns - combined_ns) / 1e9,
               "communication_not_overlapped_s": (combined_ns - comp_ns) / 1e9,
               "compute_or_communication_union_s": combined_ns / 1e9,
               "kernel_envelope_s": (combined[-1][1] - combined[0][0]) / 1e9 if combined else 0,
               "gaps_inside_kernel_envelope_s": length(free) / 1e9,
               "device_start_ns": combined[0][0] if combined else None,
               "device_end_ns": combined[-1][1] if combined else None}
        off = next(r for r in official if r["step"] == step)
        raw["minus_official_s"] = {
            "computing": raw["compute_union_s"] - off["Computing_s"],
            "communication": raw["communication_union_s"] - off["Communication_s"],
            "communication_not_overlapped": raw["communication_not_overlapped_s"] - off["Communication(Not Overlapped)_s"]}
        per_step.append(raw)
        all_compute.extend(compute)
        all_communication.extend(comm)
        all_gaps.extend(free)
    op_groups = collections.defaultdict(collections.Counter)
    operator_path = output / "operator_details.csv"
    if operator_path.exists():
        for row in rows(operator_path):
            group = op_groups[row["Name"]]
            group["calls"] += 1
            for source, target in (("Host Self Duration(us)", "host_self_sum_s_per_step"),
                                   ("Host Total Duration(us)", "host_inclusive_sum_s_per_step"),
                                   ("Device Self Duration(us)", "device_self_sum_s_per_step"),
                                   ("Device Total Duration(us)", "device_inclusive_sum_s_per_step")):
                group[target] += number(row.get(source)) / nsteps / 1e6
    op_top = {}
    for field in ("host_self_sum_s_per_step", "host_inclusive_sum_s_per_step", "device_self_sum_s_per_step"):
        op_top[field] = [{"name": name, **dict(values), "calls_per_step": values["calls"] / nsteps}
                         for name, values in sorted(op_groups.items(), key=lambda item: item[1][field], reverse=True)[:30]]
    api_rows = []
    if (output / "api_statistic.csv").exists():
        for row in rows(output / "api_statistic.csv"):
            api_rows.append({"name": row["API Name"], "level": row.get("Level"),
                             "calls_per_step": number(row["Count"]) / nsteps,
                             "duration_sum_s_per_step": number(row["Time(us)"]) / nsteps / 1e6})
        api_rows.sort(key=lambda row: row["duration_sum_s_per_step"], reverse=True)
    gap_ms = [(b - a) / 1e6 for a, b in all_gaps]
    result = {
        "source_directory": str(output), "recorded_steps": nsteps,
        "official_steps": official,
        "official_mean_s_per_step": {key: statistics.mean(row[key] for row in official)
                                     for key in official[0] if key.endswith("_s")},
        "corrected_steps": per_step,
        "corrected_mean_s_per_step": {key: statistics.mean(row[key] for row in per_step)
                                      for key, value in per_step[0].items()
                                      if key.endswith("_s") and isinstance(value, (int, float))},
        "kernel_count": kernel_count, "kernel_calls_per_step": kernel_count / nsteps,
        "excluded_kernel_rows_by_step_id": dict(excluded), "communication_classification_evidence": first_hcom,
        "kernel_types": top_counts(types, nsteps, 60), "accelerator_cores": top_counts(cores, nsteps),
        "collectives": top_counts(collectives, nsteps), "collective_events": collective_events,
        "matmul_attention_shapes": top_counts(shapes, nsteps, 40),
        "pipe_metric_columns": pipe_columns,
        "pipe_metric_semantics": {
            "cube_utilization_percent": "Installed CANN computes aic_total_cycles / (aic_frequency * ai_core_num * task_duration) * 100 with unit conversion; mac_ratio is only checked for nonzero, not multiplied. It measures kernel-local core activity/occupancy, not MAC pipeline efficiency or model MFU.",
            "cube_definition_source": "/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/profiler_tool/analysis/viewer/runtime_report.py:253",
            "mac_fraction": "sum(aic_mac_time(us)) / sum(aicore_time(us)) for rows with positive aicore_time; a kernel-local pipe-time ratio, not whole-step FLOP utilization.",
            "aggregation": "Cube percentage is weighted by duration of measured kernels of the same type/core. Kernels may overlap; no per-type values are added as wall time. Missing exported columns are listed explicitly."},
        "kernel_pipe_counters": [{"name": name, **dict(value),
                                  "aic_mac_time_fraction": value["aic_mac_time_us"] / value["aic_time_us"] if value["aic_time_us"] else None,
                                  "aiv_vec_time_fraction": value["aiv_vec_time_us"] / value["aiv_time_us"] if value["aiv_time_us"] else None,
                                  "duration_weighted_cube_utilization_percent": value["duration_weighted_cube_utilization_percent_numerator"] / value["cube_utilization_valid_duration_ns"] if value["cube_utilization_valid_duration_ns"] else None}
                                 for name, value in sorted(pipes.items(), key=lambda item: item[1]["measured_aic_kernel_duration_ns"], reverse=True)],
        "noncommunication_kernel_duration_us": {
            "count": len(durations), "median": percentile(durations, 0.5),
            "p90": percentile(durations, 0.9), "p99": percentile(durations, 0.99),
            "under_10us_count": sum(d < 10 for d in durations),
            "under_50us_count": sum(d < 50 for d in durations),
            "under_100us_count": sum(d < 100 for d in durations)},
        "device_gaps": {"definition": "No recorded compute or communication inside each step's first-to-last kernel envelope; endpoints excluded.",
                        "count": len(gap_ms), "sum_s_per_step": sum(gap_ms) / 1000 / nsteps,
                        "median_ms": percentile(gap_ms, .5), "p90_ms": percentile(gap_ms, .9),
                        "max_ms": max(gap_ms, default=0),
                        "over_1ms_count": sum(t > 1 for t in gap_ms),
                        "over_1ms_sum_s_per_step": sum(t for t in gap_ms if t > 1) / 1000 / nsteps},
        "operators": op_top, "top_api": api_rows[:40],
        "sync_copy_api": [row for row in api_rows if re.search(r"synchron|memcpy|wait|event", row["name"], re.I)],
        "communication_detail": analyze_communication(output / "communication.json", nsteps),
    }
    return result, merge(all_compute), merge(all_communication), merge(all_gaps)


def analyze_communication(path, nsteps):
    if not path.exists():
        return {"available": False}
    content = json.loads(path.read_text())
    breakdown, links = collections.Counter(), collections.defaultdict(collections.Counter)
    count, zero = 0, 0
    for step in content.values():
        for operators in step.values():
            for name, values in operators.items():
                if name.lower().startswith("total"):
                    continue
                count += 1
                info = values.get("Communication Time Info", {})
                zero += all(number(info.get(k)) == 0 for k in ("Transit Time(ms)", "Wait Time(ms)", "Synchronization Time(ms)"))
                breakdown.update({key: number(value) for key, value in info.items() if key.endswith("(ms)")})
                for link, value in values.get("Communication Bandwidth Info", {}).items():
                    links[link]["MB"] += number(value.get("Transit Size(MB)"))
                    links[link]["ms"] += number(value.get("Transit Time(ms)"))
    return {"available": True, "operations": count, "zero_transit_wait_sync_operations": zero,
            "time_info_sum_s_per_step": {key: value / nsteps / 1000 for key, value in breakdown.items()},
            "bandwidth": {key: {"transit_MB_per_step": value["MB"] / nsteps,
                                "transit_subtask_sum_s_per_step": value["ms"] / nsteps / 1000,
                                "subtask_aggregate_GB_s": value["MB"] / value["ms"] if value["ms"] else None}
                          for key, value in links.items()},
            "limits": "Time-info zero values may mean incomplete plane coverage, not no transfer/wait. SDMA aggregates other links; do not add it to HCCS. Subtask summed time is not elapsed critical-path transfer time."}


def analyze_trace(path, nsteps, compute, communication, device_gaps):
    scopes, selected, profiler_steps = collections.defaultdict(list), collections.defaultdict(list), []
    event_types, process_names = collections.Counter(), {}
    event_count = 0
    selected_pattern = re.compile(r"synchron|memcpy|^aten::(?:item|_local_scalar_dense)$|launch|enqueue", re.I)
    for event in iter_events(path):
        event_count += 1
        name, category = event.get("name", ""), str(event.get("cat", ""))
        event_types[(event.get("ph", ""), category)] += 1
        if event.get("ph") == "M" and name == "process_name":
            process_names[str(event.get("pid"))] = event.get("args", {}).get("name")
        if event.get("ph") != "X" or "dur" not in event:
            continue
        # Only retain a small subset. Device task events have their own CSV above.
        scope = name.startswith(("train.", "teacher.", "model.", "optim.", "Optimizer.step", "FullySharded", "FSDP", "round3.")) or category == "user_annotation"
        is_step = name.startswith("ProfilerStep#")
        is_selected = bool(selected_pattern.search(name)) and category not in {"kernel", "Communication", "communication"}
        if not (scope or is_step or is_selected):
            continue
        start, dur = ns(event["ts"]), ns(event["dur"])
        interval = (start, start + dur)
        if is_step:
            profiler_steps.append({"name": name, "start_ns": start, "end_ns": start + dur, "duration_s": dur / 1e9})
        elif scope:
            scopes[name].append(interval)
        if is_selected:
            selected[(name, category)].append(interval)
    combined = merge(compute + communication)
    phase_summary = {}
    for name, intervals in sorted(scopes.items()):
        merged = merge(intervals)
        phase_summary[name] = {
            "calls": len(intervals), "calls_per_step": len(intervals) / nsteps,
            "host_inclusive_sum_s_per_step": length(intervals) / nsteps / 1e9,
            "host_union_s_per_step": length(merged) / nsteps / 1e9,
            "mean_call_ms": length(intervals) / max(len(intervals), 1) / 1e6,
            "temporal_overlap_compute_s_per_step": length(intersect(merged, compute)) / nsteps / 1e9,
            "temporal_overlap_communication_s_per_step": length(intersect(merged, communication)) / nsteps / 1e9,
            "temporal_overlap_device_gaps_s_per_step": length(intersect(merged, device_gaps)) / nsteps / 1e9,
            "calls_intervals_ns": intervals if len(intervals) <= 32 else None,
        }
    selected_summary = []
    for (name, category), intervals in selected.items():
        merged = merge(intervals)
        selected_summary.append({"name": name, "category": category, "calls": len(intervals),
                                 "calls_per_step": len(intervals) / nsteps,
                                 "inclusive_sum_s_per_step": length(intervals) / nsteps / 1e9,
                                 "union_s_per_step": length(merged) / nsteps / 1e9,
                                 "temporal_overlap_device_gaps_s_per_step": length(intersect(merged, device_gaps)) / nsteps / 1e9})
    selected_summary.sort(key=lambda value: value["union_s_per_step"], reverse=True)
    top_gaps = []
    for left, right in sorted(device_gaps, key=lambda pair: pair[1] - pair[0], reverse=True)[:20]:
        overlapping = []
        for name, intervals in scopes.items():
            overlap = length(intersect(merge(intervals), [(left, right)]))
            if overlap:
                overlapping.append({"scope": name, "overlap_ms": overlap / 1e6})
        top_gaps.append({"start_ns": left, "end_ns": right, "duration_ms": (right - left) / 1e6,
                         "overlapping_scopes": sorted(overlapping, key=lambda value: value["overlap_ms"], reverse=True)[:12]})
    return {"trace_events": event_count, "process_names": process_names,
            "event_categories": [{"ph": key[0], "category": key[1], "count": value} for key, value in event_types.most_common()],
            "profiler_steps": profiler_steps,
            "host_profiler_step_mean_s": statistics.mean(step["duration_s"] for step in profiler_steps) if profiler_steps else None,
            "phases": phase_summary, "sync_copy_launch_intervals": selected_summary[:100],
            "largest_device_gaps": top_gaps,
            "limits": "Scopes are host annotations and may be nested/asynchronous. Scope/device overlap is temporal coincidence, not dependency or causal attribution. Do not add phase sums, APIs, or kernels to infer elapsed step time. Device gaps alone do not prove CPU launch starvation; stream dependencies and missing task categories may contribute."}


def find_outputs(root):
    if root.name == "ASCEND_PROFILER_OUTPUT":
        return [root]
    return sorted(root.glob("**/ASCEND_PROFILER_OUTPUT"), key=lambda p: str(p))


def rank_name(output):
    for parent in output.parents:
        if re.fullmatch(r"rank\d+", parent.name):
            return parent.name
    return output.parent.name


def aggregate(results, csv_only):
    values = list(results.values())
    keys = sorted(set.intersection(*(set(v["official_mean_s_per_step"]) for v in values)))
    official = {key: {"mean": statistics.mean(v["official_mean_s_per_step"][key] for v in values),
                      "min_rank": min(v["official_mean_s_per_step"][key] for v in values),
                      "max_rank": max(v["official_mean_s_per_step"][key] for v in values)} for key in keys}
    phases = {}
    if not csv_only:
        phase_names = sorted(set().union(*(set(v.get("trace", {}).get("phases", {})) for v in values)))
        for name in phase_names:
            phase_values = [v["trace"]["phases"][name] for v in values if name in v.get("trace", {}).get("phases", {})]
            phases[name] = {"ranks": len(phase_values), **{
                key: statistics.mean(v[key] for v in phase_values)
                for key in ("host_union_s_per_step", "calls_per_step", "temporal_overlap_device_gaps_s_per_step")}}
    return {"rank_count": len(results), "ranks": list(results), "official_rank_statistics_s_per_step": official,
            "host_phase_rank_means": phases,
            "kernel_calls_per_rank_step_mean": statistics.mean(v["kernel_calls_per_step"] for v in values),
            "collective_cross_rank_timing": collective_cross_rank_timing(results)}


def collective_cross_rank_timing(results):
    groups = collections.defaultdict(dict)
    for rank, result in results.items():
        for event in result["collective_events"]:
            key = (event["step"], event["name"])
            if rank in groups[key]:
                raise ValueError(f"Duplicate collective name within rank/step: {rank} {key}")
            groups[key][rank] = event
    common, incomplete = [], 0
    for (step, name), events in groups.items():
        if len(events) != len(results):
            incomplete += 1
            continue
        starts, ends = [e["start_ns"] for e in events.values()], [e["end_ns"] for e in events.values()]
        common.append({"step": step, "name": name, "type": next(iter(events.values()))["type"],
                       "start_skew_ms": (max(starts) - min(starts)) / 1e6,
                       "end_skew_ms": (max(ends) - min(ends)) / 1e6,
                       "envelope_ms": (max(ends) - min(starts)) / 1e6,
                       "last_start_to_last_end_ms": (max(ends) - max(starts)) / 1e6,
                       "latest_start_rank": max(events, key=lambda rank: events[rank]["start_ns"]),
                       "earliest_start_rank": min(events, key=lambda rank: events[rank]["start_ns"]),
                       "rank_durations_ms": {rank: e["duration_ms"] for rank, e in events.items()}})
    skew = [e["start_skew_ms"] for e in common]
    return {"matching": "Exact (Step Id, kernel Name), only groups present on all captured ranks.",
            "matched_collectives": len(common), "incomplete_groups": incomplete,
            "start_skew_ms": {"median": percentile(skew, .5), "p90": percentile(skew, .9), "max": max(skew, default=0)},
            "largest_start_skew": sorted(common, key=lambda e: e["start_skew_ms"], reverse=True)[:40],
            "by_type": {kind: {"count": len(sub), "mean_start_skew_ms": statistics.mean(e["start_skew_ms"] for e in sub),
                               "mean_last_start_to_last_end_ms": statistics.mean(e["last_start_to_last_end_ms"] for e in sub)}
                        for kind in sorted({e["type"] for e in common})
                        for sub in [[e for e in common if e["type"] == kind]]},
            "limits": "Cross-device timestamps use profiler clock alignment. Arrival skew is evidence of rank timing imbalance, not exact blocked time; transfer may overlap arrivals. Last-start-to-last-end includes communication execution/synchronization and is not pure wire time. Never sum per-collective envelopes as elapsed step time."}


def self_test():
    assert ns("1792136148265875.540\t") - ns("1792136148265875.500") == 40
    assert merge([(1, 5), (2, 3), (5, 8), (10, 11)]) == [(1, 8), (10, 11)]
    assert intersect([(1, 5), (7, 10)], [(3, 8)]) == [(3, 5), (7, 8)]
    assert is_communication("hcom_allGather__2", "hcom_allGather", "COMMUNICATION")
    assert not is_communication("MatMul", "MatMul", "AI_CORE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv-only", action="store_true")
    parser.add_argument("--ranks", default="", help="Optional comma-separated ranks, e.g. 0,4")
    args = parser.parse_args()
    self_test()
    outputs = find_outputs(args.trace_root)
    wanted = {"rank" + value.removeprefix("rank") for value in args.ranks.split(",") if value}
    outputs = [p for p in outputs if not wanted or rank_name(p) in wanted]
    if not outputs:
        raise SystemExit("No ASCEND_PROFILER_OUTPUT directories found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for output in outputs:
        rank = rank_name(output)
        if rank in results:
            raise ValueError(f"Multiple captures for {rank}; choose one run directory")
        print(f"READ {rank}: CSV", flush=True)
        result, compute, communication, device_gaps = analyze_csv(output)
        result["rank"] = rank
        (args.output_dir / f"{rank}_csv.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        if not args.csv_only:
            print(f"READ {rank}: streaming trace", flush=True)
            result["trace"] = analyze_trace(output / "trace_view.json", result["recorded_steps"], compute, communication, device_gaps)
        (args.output_dir / f"{rank}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        results[rank] = result
        print(f"DONE {rank}: steps={result['recorded_steps']} official={result['official_mean_s_per_step']}", flush=True)
    summary = {"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "trace_root": str(args.trace_root.resolve()), "csv_only": args.csv_only,
               "algorithm": "Read ranks sequentially. Parse epoch microseconds exactly to integer ns; classify COMMUNICATION/HCCL or hcom/hccl names/types; merge per-step intervals, calculate overlap by set intersection. Official step CSV is reported independently. JSON trace is streamed.",
               "limitations": ["Inclusive host/operator durations are not additive elapsed wall time.",
                               "Rank means are not a sum across devices and do not measure arithmetic-unit MFU.",
                               "Official Free is neither recorded compute nor communication, not a causal classification of CPU/data/communication waits.",
                               "Raw kernel gap envelopes omit leading/trailing step gaps; official Stage/Free have different boundaries.",
                               "Kernel accelerator-core duration indicates task intervals, not fraction of arithmetic units executing instructions.",
                               "Profiler overhead must be compared against a no-trace run separately."],
               "aggregate": aggregate(results, args.csv_only), "per_rank_files": {rank: f"{rank}.json" for rank in results}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("WROTE", args.output_dir / "summary.json", flush=True)


if __name__ == "__main__":
    main()
