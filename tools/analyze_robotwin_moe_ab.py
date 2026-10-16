#!/usr/bin/env python3
"""Compare the unprofiled intervals of controlled full-model MoE runs."""
import argparse
import datetime
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import yaml


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def case_summary(root, first, last):
    launcher_exit_code = int((root / "exit_code").read_text().strip())
    rows = [json.loads(s) for s in (root / "logs/loss.jsonl").read_text().splitlines()]
    assert [x["step"] for x in rows] == list(range(1, 41)), "expected 40 complete updates"
    assert all(math.isfinite(x[k]) for x in rows for k in ["loss", "grad_norm"])
    text = (root / "console.log").read_text(errors="replace")
    post_run_note = None
    if launcher_exit_code != 0:
        # Keep the known post-training shell failure visible; never hide a
        # training failure or rewrite the original launcher's exit status.
        audit = json.loads((root / "training_completion.json").read_text())
        assert root.name == "torch" and launcher_exit_code == 2
        assert audit["training_completed"] and audit["optimizer_updates"] == 40
        assert "unexpected EOF while looking for matching" in text
        assert "Profiling run: skip checkpoint at step 40." in text
        assert "[async_hf] summary:" in text
        post_run_note = audit["post_training_launcher_error"]
    clocks = {int(step): datetime.datetime.strptime(t, "%m/%d/%Y %H:%M:%S")
              for t, step in re.findall(r"(\d\d/\d\d/\d{4} \d\d:\d\d:\d\d) - INFO - __main__ - Step (\d+)/5717", text)}
    values = [r["step_time"] for r in rows if first <= r["step"] <= last]
    wall = (clocks[last] - clocks[first - 1]).total_seconds() / len(values)
    peak = re.findall(r"VRAM usage after epoch 1: cur [\d.]+GB, max ([\d.]+)GB", text)
    return {
        "root": str(root), "steps": len(rows), "steady_range_inclusive": [first, last],
        "launcher_exit_code": launcher_exit_code, "post_training_launcher_note": post_run_note,
        "steady_step_time_mean_s": statistics.mean(values),
        "steady_step_time_median_s": statistics.median(values),
        "steady_step_time_p95_s": percentile(values, .95),
        "steady_wall_seconds_per_update": wall,
        "global_samples_per_second": 96 / wall,
        "peak_allocated_gib": float(peak[-1]) if peak else None,
        "finite_loss_and_gradients": True,
        "first_loss": rows[0]["loss"], "last_loss": rows[-1]["loss"],
    }, rows


def resource_summary(root, first, last):
    grouped = {name: {} for name in ["torch", "gmm"]}
    for line in (root / "resources.jsonl").read_text().splitlines():
        sample = json.loads(line)
        if not first - 1 <= sample["last_logged_step"] < last:
            continue
        for card in ["0", "4"]:
            values = grouped[sample["case"]].setdefault(card, {})
            for key, value in re.findall(r"^\s*([^:\n]+?)\s*:\s*([\d.]+)\s*$", sample[card], re.MULTILINE):
                if key in ["Aicore Usage Rate(%)", "Aivector Usage Rate(%)", "HBM Bandwidth Usage Rate(%)", "HBM Usage Rate(%)"]:
                    values.setdefault(key, []).append(float(value))
    return {name: {card: {key: {"mean": statistics.mean(v), "min": min(v), "max": max(v), "samples": len(v)}
                         for key, v in metrics.items()} for card, metrics in cards.items()}
            for name, cards in grouped.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases, records, configs = {}, {}, {}
    for name in ["torch", "gmm"]:
        cases[name], records[name] = case_summary(args.root / name, 13, 29)
        configs[name] = yaml.safe_load((args.root / name / "logs/lingbotvla_cli.yaml").read_text())
    differences = []
    for section, fields in configs["torch"].items():
        for key, value in fields.items():
            other = configs["gmm"][section][key]
            if value != other:
                differences.append({"section": section, "field": key, "torch": value, "gmm": other})
                assert section == "train" and key in ["output_dir", "logging_dir", "profile_ranks", "profile_start_step", "profile_end_step", "profile_trace_dir"], f"unexpected config difference: {section}.{key}"
    paired = [{"step": a["step"], "loss_abs_diff": abs(a["loss"] - b["loss"]),
               "grad_norm_abs_diff": abs(a["grad_norm"] - b["grad_norm"])}
              for a, b in zip(records["torch"], records["gmm"])]
    a, b = cases["torch"], cases["gmm"]
    summary = {
        "cases": cases, "steady_comparison_steps_inclusive": [13, 29],
        "speedup_wall": a["steady_wall_seconds_per_update"] / b["steady_wall_seconds_per_update"],
        "step_time_reduction_pct": 100 * (1 - b["steady_wall_seconds_per_update"] / a["steady_wall_seconds_per_update"]),
        "speedup_logged_time": a["steady_step_time_mean_s"] / b["steady_step_time_mean_s"],
        "max_loss_abs_diff": max(r["loss_abs_diff"] for r in paired),
        "max_grad_norm_abs_diff": max(r["grad_norm_abs_diff"] for r in paired),
        "paired_updates": paired,
        "resources": resource_summary(args.root, 13, 29),
        "effective_config_differences": differences,
        "limits": ["Short 40-step runs, fixed initialization/data/LR horizon, sequential backend order.",
                   "Speed excludes initialization, profiler warmup/capture/export and the following collective wait.",
                   "Resource metrics are short-window npu-smi samples, not MFU or epoch averages.",
                   "No full-epoch training or RoboTwin success-rate evaluation."],
    }
    (args.output / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ["cases", "speedup_wall", "step_time_reduction_pct", "max_loss_abs_diff", "max_grad_norm_abs_diff"]}, indent=2))


if __name__ == "__main__":
    main()
