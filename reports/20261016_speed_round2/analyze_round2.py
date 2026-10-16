#!/usr/bin/env python3
"""Audit bounded round-2 runs using files only; never imports torch or uses NPU.

Example:
  python reports/20261016_speed_round2/analyze_round2.py
  python reports/20261016_speed_round2/analyze_round2.py --window 13:60 --window 31:60

Only successful, consecutive runs with >= min-steps updates enter comparisons.
The wall interval for steps A..B is timestamp[B] - timestamp[A - 1].
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import yaml

METRICS = ["HBM Usage Rate(%)", "Aicore Usage Rate(%)", "Aivector Usage Rate(%)",
           "Aicpu Usage Rate(%)", "HBM Bandwidth Usage Rate(%)"]
PAIRED = ["loss", "vla_loss", "depth_loss", "future_depth_loss", "future_video_loss",
          "grad_norm", "lr", "expert_lr"]
DESTINATION_CONFIG_FIELDS = {"train.logging_dir", "train.output_dir", "train.profile_trace_dir"}

TIME_RE = re.compile(r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) - INFO - __main__ - Step (\d+)/(\d+)")


def jsonl(path):
    if not path.exists():
        return [], []
    rows, errors = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except (ValueError, TypeError) as exc:
            errors.append({"line": i, "error": str(exc)})
    return rows, errors


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def stats(values):
    if not values:
        return None
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p95": percentile(values, .95),
            "min": min(values), "max": max(values),
            "stddev": statistics.pstdev(values)}


def config_diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for k in sorted(set(a) | set(b)):
            result += config_diff(a.get(k), b.get(k), f"{path}.{k}".strip("."))
        return result
    return [] if a == b else [{"field": path, "baseline": a, "candidate": b}]



def is_runtime_artifact(name):
    # Keep the raw hashes; exempt only identified compiler output locations.
    return name.startswith("kernel_meta/") or name == "fusion_result.json"


def source_provenance(path):
    manifest_path = path / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    snapshot = Path(manifest["source_snapshot"]) if manifest.get("source_snapshot") else None
    missing, changed, runtime_changed = [], [], []
    for name, expected in manifest.get("hashes", {}).items():
        if snapshot is None:
            break
        source_file = snapshot / name
        if not source_file.is_file():
            missing.append(name)
        elif hashlib.sha256(source_file.read_bytes()).hexdigest() != expected:
            (runtime_changed if is_runtime_artifact(name) else changed).append(name)
    data_file = path / "run/data_source.json"
    dataset_list = path / "run/train_robotwin.txt"
    data = json.loads(data_file.read_text()) if data_file.exists() else {}
    return {"manifest_present": bool(manifest), "manifest_path": str(manifest_path),
            "manifest_settings": {k: v for k, v in manifest.items() if k != "hashes"},
            "manifest_hash_count": len(manifest.get("hashes", {})),
            "current_snapshot_missing_files": missing,
            "current_snapshot_changed_nonruntime_files": changed,
            "current_snapshot_changed_runtime_files": runtime_changed,
            "data_source": data,
            "dataset_list_sha256": hashlib.sha256(dataset_list.read_bytes()).hexdigest() if dataset_list.exists() else None,
            "limits": ["Current snapshot hashes can detect edits since the recorded manifest, but cannot prove historical imports or byte identity at every update.",
                       "kernel_meta/ files and the root fusion_result.json are runtime compiler metadata and are not classified as training source edits; their original hashes remain recorded.",
                       "External model weights and dataset contents have paths but no full hashes or per-batch fingerprints; identical paths are evidence, not bitwise proof.",
                       "The launcher records selected settings, not a complete package-version or effective-environment snapshot."]}, manifest


def fairness_summary(baseline, candidate, raw_a, raw_b):
    differences = config_diff(raw_a["config"], raw_b["config"])
    non_destination = [d for d in differences if d["field"] not in DESTINATION_CONFIG_FIELDS]
    hashes_a = raw_a["manifest"].get("hashes", {})
    hashes_b = raw_b["manifest"].get("hashes", {})
    source_differences = [{"file": f, "baseline": hashes_a.get(f), "candidate": hashes_b.get(f),
                           "runtime_metadata": is_runtime_artifact(f)}
                          for f in sorted(set(hashes_a) | set(hashes_b))
                          if hashes_a.get(f) != hashes_b.get(f)]
    pa = baseline["provenance"]; pb = candidate["provenance"]
    da = {k: v for k, v in pa["data_source"].items() if k != "code_repository"}
    db = {k: v for k, v in pb["data_source"].items() if k != "code_repository"}
    return {"effective_configs_equal_except_destinations": not non_destination,
            "non_destination_config_differences": non_destination,
            "lr_horizons_match": baseline["lr_horizons"] == candidate["lr_horizons"],
            "dataset_list_hashes_match": (pa["dataset_list_sha256"] == pb["dataset_list_sha256"])
             if pa["dataset_list_sha256"] and pb["dataset_list_sha256"] else None,
            "data_source_metadata_equal_except_repository": da == db if da and db else None,
            "source_manifest_differences": source_differences,
            "interpretation": "Review settings and source differences before attributing speed changes to a single feature. Completion and finite metrics are prerequisite checks; numerical equivalence has no automatic tolerance-based verdict here."}


def resource_summary(samples, first, last):
    selected = [s for s in samples if isinstance(s.get("last_step"), (int, float))
                and first - 1 <= s["last_step"] < last]
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for sample in selected:
        for card, fields in sample.get("cards", {}).items():
            for metric in METRICS:
                if finite(fields.get(metric)):
                    grouped[str(card)][metric].append(fields[metric])
    return {"selection": f"{first - 1} <= last_step < {last}",
            "sample_records": len(selected),
            "first_sample_at": selected[0].get("at") if selected else None,
            "last_sample_at": selected[-1].get("at") if selected else None,
            "cards": {card: {k: stats(v) for k, v in metrics.items()}
                      for card, metrics in grouped.items()},
            "limitation": "last_step is read before sequential npu-smi queries for cards 0 and 4. "
                          "Readings may cross update boundaries and are not simultaneous. "
                          "The 4-second sleep plus query time limits sampling resolution; "
                          "no strict timing error bound is available. These short-window samples "
                          "are not MFU or full-run utilization averages."}


def read_case(path, min_steps):
    rows, errors = jsonl(path / "logs/loss.jsonl")
    samples, sample_errors = jsonl(path / "resource_samples.jsonl")
    exit_path = path / "exit_code"
    try:
        exit_code = int(exit_path.read_text().strip()) if exit_path.exists() else None
    except ValueError:
        exit_code = "invalid"
    log_path = path / "logs/train.log"
    if not log_path.exists():
        log_path = path / "console.log"
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    clocks = {}; clock_counts = collections.Counter(); horizons = set()
    for stamp, number, horizon in TIME_RE.findall(text):
        number = int(number)
        clocks[number] = dt.datetime.strptime(stamp, "%m/%d/%Y %H:%M:%S")
        clock_counts[number] += 1; horizons.add(int(horizon))
    cfg_path = path / "logs/lingbotvla_cli.yaml"
    config = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    config = config or {}
    train = config.get("train", {})
    steps = [r.get("step") for r in rows]
    contiguous = steps == list(range(1, len(rows) + 1))
    finite_required = all(finite(r.get(k)) for r in rows for k in ("loss", "grad_norm", "step_time"))
    bad_numeric = [{"step": r.get("step"), "field": k, "value": str(r.get(k))}
                   for r in rows for k in PAIRED + ["step_time"]
                   if k in r and not finite(r[k])]
    reasons = []
    if exit_code != 0: reasons.append("launcher_pending" if exit_code is None else "launcher_nonzero_or_invalid")
    if len(rows) < min_steps: reasons.append(f"fewer_than_{min_steps}_updates")
    if not contiguous: reasons.append("nonconsecutive_or_duplicate_updates")
    if errors: reasons.append("invalid_loss_jsonl")
    if not finite_required or bad_numeric: reasons.append("nonfinite_or_missing_metrics")
    peak = [float(x) for x in re.findall(r"VRAM usage after epoch \d+: cur [\d.]+GB, max ([\d.]+)GB", text)]
    provenance, manifest = source_provenance(path)
    case = {"path": str(path), "updates": len(rows), "last_step": steps[-1] if steps else None,
            "exit_code": exit_code, "eligible": not reasons, "exclusion_reasons": reasons,
            "contiguous_from_step_1": contiguous, "all_required_metrics_finite": finite_required,
            "nonfinite_numeric_records": bad_numeric, "jsonl_errors": errors,
            "resource_jsonl_errors": sample_errors,
            "peak_allocated_gib": max(peak) if peak else None,
            "peak_memory_scope": "Rank-0 max allocated memory reported at run end; GiB despite log label GB. Not all-rank reserved memory.",
            "duplicate_timestamp_steps": {str(k): v for k, v in clock_counts.items() if v > 1},
            "timestamp_log": str(log_path), "timestamp_resolution_s": 1,
            "lr_horizons": sorted(horizons),
            "global_batch_size": train.get("global_batch_size"),
            "seed": train.get("seed"), "model_path": config.get("model", {}).get("model_path"),
            "enable_resume": train.get("enable_resume"),
            "profile_ranks": train.get("profile_ranks"),
            "source_hashes": {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [path / "logs/loss.jsonl", cfg_path, log_path] if p.exists()},
            "provenance": provenance, "windows": {}}
    return case, {"rows": rows, "clocks": clocks, "config": config, "resources": samples, "manifest": manifest}


def window_summary(case, data, first, last):
    by_step = {r["step"]: r for r in data["rows"]}
    missing = [s for s in range(first, last + 1) if s not in by_step]
    if missing:
        return {"available": False, "missing_steps": missing}
    rows = [by_step[s] for s in range(first, last + 1)]
    values = [r["step_time"] for r in rows]
    if not all(finite(v) and v > 0 for v in values):
        return {"available": False, "reason": "nonfinite_or_nonpositive_step_time"}
    clocks = data["clocks"]; count = last - first + 1
    wall = ((clocks[last] - clocks[first - 1]).total_seconds()
            if last in clocks and first - 1 in clocks else None)
    train = data["config"].get("train", {})
    profiler_enabled = bool(train.get("profile_ranks"))
    profiler_start = train.get("profile_start_step")
    # Exclude all measurement steps after warmup starts: capture/export may also disturb peers.
    profile_risk = profiler_enabled and (profiler_start is None or last >= int(profiler_start) - 1)
    valid_wall = wall is not None and wall > 0 and not case["duplicate_timestamp_steps"]
    return {"available": True, "steps_inclusive": [first, last], "updates": count,
            "logged_step_seconds": stats(values),
            "wall_start_step": first - 1, "wall_end_step": last,
            "wall_start_at": clocks[first - 1].isoformat() if first - 1 in clocks else None,
            "wall_end_at": clocks[last].isoformat() if last in clocks else None,
            "wall_total_seconds": wall, "wall_seconds_per_update": wall / count if valid_wall else None,
            "wall_timestamp_quantization_uncertainty_seconds_per_update": 1 / count,
            "global_samples_per_second": case["global_batch_size"] * count / wall
             if valid_wall and case["global_batch_size"] else None,
            "profiler_overlap_risk": profile_risk,
            "usable_for_speed_comparison": not profile_risk,
            "loss": stats([r["loss"] for r in rows]),
            "grad_norm": stats([r["grad_norm"] for r in rows]),
            "resources": resource_summary(data["resources"], first, last)}


def pair_rows(a, b, first, last):
    aa = {r["step"]: r for r in a}; bb = {r["step"]: r for r in b}
    steps = sorted(set(aa) & set(bb) & set(range(first, last + 1)))
    result = {"steps_inclusive": [first, last], "paired_updates": len(steps), "metrics": {}}
    for key in PAIRED:
        rows = [{"step": s, "baseline": aa[s][key], "candidate": bb[s][key],
                 "abs_diff": abs(aa[s][key] - bb[s][key]),
                 "relative_diff": abs(aa[s][key] - bb[s][key]) / max(abs(aa[s][key]), 1e-12)}
                for s in steps if finite(aa[s].get(key)) and finite(bb[s].get(key))]
        if rows:
            result["metrics"][key] = {"paired_values": len(rows),
                "exactly_equal_values": sum(r["abs_diff"] == 0 for r in rows),
                "all_scalar_values_exactly_equal": all(r["abs_diff"] == 0 for r in rows),
                "mean_abs_diff": statistics.mean(r["abs_diff"] for r in rows),
                "max_abs_diff": max(rows, key=lambda r: r["abs_diff"]),
                "max_relative_diff": max(rows, key=lambda r: r["relative_diff"])}
    return result


def parse_window(value):
    try:
        first, last = map(int, value.split(":"))
        if first < 2 or last < first: raise ValueError
        return first, last
    except ValueError as exc:
        raise argparse.ArgumentTypeError("window must be FIRST:LAST with 2 <= FIRST <= LAST") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("/cache/lingbot/profiling/speed_round2_20261016"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().with_name("round2_analysis.json"))
    parser.add_argument("--baseline", default="gmm_baseline_v2")
    parser.add_argument("--min-steps", type=int, default=60)
    parser.add_argument("--window", type=parse_window, action="append")
    args = parser.parse_args()
    cases = {}; raw = {}
    for path in sorted(args.root.iterdir()):
        if path.is_dir() and ((path / "logs").exists() or (path / "console.log").exists()):
            cases[path.name], raw[path.name] = read_case(path, args.min_steps)
    eligible = [name for name, case in cases.items() if case["eligible"]]
    windows = list(args.window or [(13, 60), (31, 60)])
    if eligible:
        end = min(cases[name]["last_step"] for name in eligible)
        if end >= 13: windows.append((13, end))
    windows = sorted(set(windows))
    for name in eligible:
        for first, last in windows:
            cases[name]["windows"][f"{first}-{last}"] = window_summary(cases[name], raw[name], first, last)
    comparisons = {}
    if args.baseline in eligible:
        for name in eligible:
            if name == args.baseline: continue
            end = min(cases[name]["last_step"], cases[args.baseline]["last_step"])
            comparison = {"baseline": args.baseline, "candidate": name,
                "config_differences": config_diff(raw[args.baseline]["config"], raw[name]["config"]),
                "fairness": fairness_summary(cases[args.baseline], cases[name], raw[args.baseline], raw[name]),
                "numerical_verdict": "Descriptive paired scalar differences only; no automatic claim of full-gradient or training-quality equivalence.",
                "all_update_numerics": pair_rows(raw[args.baseline]["rows"], raw[name]["rows"], 1, end),
                "windows": {}}
            for first, last in windows:
                key = f"{first}-{last}"; a = cases[args.baseline]["windows"][key]; b = cases[name]["windows"][key]
                if not all(x.get("available") and x.get("usable_for_speed_comparison") for x in [a, b]): continue
                ratio = a["logged_step_seconds"]["mean"] / b["logged_step_seconds"]["mean"]
                wall_ratio = a["wall_seconds_per_update"] / b["wall_seconds_per_update"] if a["wall_seconds_per_update"] and b["wall_seconds_per_update"] else None
                comparison["windows"][key] = {"logged_mean_speedup": ratio,
                    "logged_mean_time_reduction_pct": 100 * (1 - 1 / ratio),
                    "wall_speedup": wall_ratio,
                    "wall_time_reduction_pct": 100 * (1 - 1 / wall_ratio) if wall_ratio else None,
                    "paired_numerics": pair_rows(raw[args.baseline]["rows"], raw[name]["rows"], first, last)}
            comparisons[name] = comparison
    result = {"generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "root": str(args.root), "baseline": args.baseline, "eligible_cases": eligible,
        "comparison_windows": [list(w) for w in windows], "cases": cases, "comparisons": comparisons,
        "limits": ["Only exit_code=0 with consecutive, finite updates and at least min-steps updates enters comparisons. Failed and unfinished cases remain visible but excluded.",
                   "Warmup/initialization steps 1-12 are excluded from default speed windows; profile-affected windows are excluded.",
                   "Wall time for A..B uses timestamp[B] minus timestamp[A-1], divided by B-A+1. Whole-second logging gives under 1 second total quantization error.",
                   "Resource last_step selection has query-boundary and sampling-cadence error; resource readings are descriptive, not device MFU.",
                   "Sequential short runs cannot prove full-epoch convergence or robot-task quality. Similar scalar loss/gradient norms do not prove full gradient equivalence."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "eligible_cases": eligible,
                      "excluded": {n: c["exclusion_reasons"] for n, c in cases.items() if not c["eligible"]},
                      "speedups": {n: {w: v["logged_mean_speedup"] for w, v in c["windows"].items()} for n, c in comparisons.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
