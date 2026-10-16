"""Summarize main-thread py-spy samples without treating them as device time."""
import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def stage_ranges(path):
    result = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            expr = item.context_expr
            if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Attribute):
                continue
            if expr.func.attr != "scope" or not expr.args or not isinstance(expr.args[0], ast.Constant):
                continue
            name = expr.args[0].value
            if isinstance(name, str) and name.startswith("train."):
                result.append((node.lineno, node.end_lineno, name))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    metadata = json.loads((run / "host_sampling.json").read_text())
    snapshot = Path(metadata["source_snapshot"])
    train_path = snapshot / "tasks/vla/train_lingbotvla.py"
    ranges = stage_ranges(train_path)
    results = {}
    for rank in (0, 4):
        data = json.loads((run / f"host_rank{rank}.speedscope.json").read_text())
        frames = data["shared"]["frames"]
        profiles = [p for p in data["profiles"] if '"MainThread"' in p["name"]]
        assert len(profiles) == 1, profiles
        profile = profiles[0]
        assert profile["type"] == "sampled" and profile["unit"] == "seconds"
        stages, leaves, inclusive, stacks, selected = [defaultdict(float) for _ in range(5)]
        selected_names = {"recover_focal_shift", "solve_optimal_focal_shift", "batched_newton_schulz",
                          "_step_megabatch", "_step_megabatch_chunk", "get_depth_target",
                          "get_video_target", "_rgb_to_unit_float", "_engine_run_backward"}
        total = sum(profile["weights"])
        for ids, weight in zip(profile["samples"], profile["weights"]):
            stack = [frames[i] for i in ids]
            def label(f):
                return f"{f.get('file', '?')}:{f.get('line', '?')}:{f['name']}"
            if not stack:
                stages["unknown_empty_stack"] += weight
                continue
            leaves[label(stack[-1])] += weight
            stacks[tuple(label(f) for f in stack)] += weight
            for frame in set(label(f) for f in stack):
                inclusive[frame] += weight
            for name in {f["name"] for f in stack} & selected_names:
                selected[name] += weight
            active = "outside_named_train_stage"
            for f in stack:
                if f.get("file", "").endswith("tasks/vla/train_lingbotvla.py"):
                    line = f.get("line")
                    matches = [(end-start, name) for start, end, name in ranges
                               if isinstance(line, int) and start <= line <= end]
                    if matches:
                        active = min(matches)[1]
            stages[active] += weight
        def top(counts, limit=30):
            return [dict(location=k, sampled_seconds=v, sample_fraction_pct=100*v/total)
                    for k, v in sorted(counts.items(), key=lambda item: -item[1])[:limit]]
        results[str(rank)] = dict(
            main_thread_name=profile["name"], total_threads=len(data["profiles"]),
            main_thread_samples=len(profile["samples"]), main_thread_sampled_seconds=total,
            stage_samples=top(stages), top_leaf_samples=top(leaves),
            top_inclusive_frame_samples=top(inclusive),
            selected_function_inclusive_samples=top(selected),
            top_stacks=[dict(stack=list(stack), sampled_seconds=seconds, sample_fraction_pct=100*seconds/total)
                        for stack, seconds in sorted(stacks.items(), key=lambda item: -item[1])[:20]],
            raw_profile=str(run / f"host_rank{rank}.speedscope.json"),
        )
    output = dict(run=str(run), metadata=metadata, ranks=results,
                  interpretation=[
                      "Percentages are main-thread Python stack sample fractions, including blocking/idle time.",
                      "They are neither CPU busy percentages nor NPU kernel-time percentages.",
                      "Inclusive frame percentages overlap and must not be added.",
                      "Phase labels are disjoint and derived from actual snapshot training scope source ranges.",
                      "Native extensions are not unwound: a Python frame may represent waiting on NPU, communication, or C/C++ work.",
                      "Nonblocking sampling may miss or produce inconsistent stacks; cross-check the native trace.",
                  ])
    (HERE / "host_sampling_analysis.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    lines = ["Low-rate Python host stack sampling (20 Hz, 30 seconds, ranks 0/4)",
             "Percentages below include idle/blocking time and are not NPU or CPU utilization.",
             f"Excluded training steps: {metadata['exclude_steps_inclusive']}", ""]
    for rank, data in results.items():
        lines.append(f"Rank {rank}: {data['main_thread_samples']} main-thread samples ({data['main_thread_sampled_seconds']:.3f}s weighted)")
        lines.extend(f"  {row['location']}: {row['sample_fraction_pct']:.2f}%" for row in data["stage_samples"])
        lines.append("Top leaves:")
        lines.extend(f"  {row['sample_fraction_pct']:.2f}% {row['location']}" for row in data["top_leaf_samples"][:12])
        lines.append("")
    (HERE / "host_analysis.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
