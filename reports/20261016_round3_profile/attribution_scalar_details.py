#!/usr/bin/env python3
"""Map rank-0 scalar reads to nearest custom scope and enclosing aten operator."""
import argparse
import collections
import json
from pathlib import Path

from attribution_scope_analysis import events, custom_scope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    by_thread = collections.defaultdict(list)
    steps = 0
    for event in events(args.trace):
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        is_runtime_sync = name.startswith("AscendCL@") and "synchronize" in name.lower()
        if event.get("cat") != "cpu_op" and not is_runtime_sync:
            continue
        if name.startswith("ProfilerStep#"):
            steps += 1
        start = float(event["ts"])
        # The AscendCL PID is synthetic; its TID is the real originating thread.
        by_thread[event.get("tid")].append((start, start + float(event["dur"]), name))
    stats = collections.defaultdict(collections.Counter)
    direct = collections.defaultdict(collections.Counter)
    runtime = collections.defaultdict(collections.Counter)
    for values in by_thread.values():
        stack = []
        for start, end, name in sorted(values, key=lambda x: (x[0], -x[1])):
            while stack and not (stack[-1][0] <= start and stack[-1][1] >= end - 0.5):
                stack.pop()
            if name == "aten::_local_scalar_dense":
                scope = next((r[2] for r in reversed(stack) if custom_scope(r[2])), "<outside_instrumented_scope>")
                parent = next((r[2] for r in reversed(stack) if r[2] not in ("aten::item", "aten::_local_scalar_dense") and not custom_scope(r[2]) and not r[2].startswith("ProfilerStep#")), "<direct_in_scope>")
                for key, table in ((scope, stats), (scope + " | " + parent, direct)):
                    table[key]["calls"] += 1
                    table[key]["host_inclusive_us"] += end - start
            elif name.startswith("AscendCL@") and "synchronize" in name.lower():
                scope = next((r[2] for r in reversed(stack) if custom_scope(r[2])), "<outside_instrumented_scope>")
                parent = next((r[2] for r in reversed(stack) if not custom_scope(r[2]) and not r[2].startswith(("ProfilerStep#", "AscendCL@"))), "<direct_in_scope>")
                key = scope + " | " + parent + " | " + name
                runtime[key]["calls"] += 1
                runtime[key]["host_inclusive_us"] += end - start
            stack.append((start, end, name))
    result = {"trace": str(args.trace), "captured_steps": steps,
              "by_scope": dict(stats),
              "by_scope_and_enclosing_operator": dict(sorted(direct.items(), key=lambda x: x[1]["calls"], reverse=True)),
              "runtime_synchronize_by_scope_and_parent": dict(sorted(runtime.items(), key=lambda x: x[1]["host_inclusive_us"], reverse=True)),
              "interpretation": "Counts/host ranges of aten::_local_scalar_dense only; Python source locations require native stack support or targeted follow-up ranges. Enclosing aten::Int/ScalarImplicit/is_nonzero indicates scalar-shape/control conversion, not Python loss logging."}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"captured_steps": steps, "by_scope": result["by_scope"], "top_parent_operators": dict(list(result["by_scope_and_enclosing_operator"].items())[:20]), "top_runtime_sync_parents": dict(list(result["runtime_synchronize_by_scope_and_parent"].items())[:20])}, indent=2))


if __name__ == "__main__":
    main()
