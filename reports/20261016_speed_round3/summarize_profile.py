"""Compare like-for-like four-update, all-card native profiler captures."""
import importlib.util
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "20261016_round3_profile"
spec = importlib.util.spec_from_file_location("profile_parser", OLD / "analyze_profile.py")
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)


def complete_types(data):
    """The general parser retains only the top 60 types; scan all for counts."""
    kinds = defaultdict(lambda: [0, 0])
    steps = {row["step"] for row in data["official_steps"]}
    with (Path(data["source_directory"]) / "kernel_details.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["Step Id"] in steps:
                kinds[row["Type"]][0] += 1
                kinds[row["Type"]][1] += parser.ns(row["Duration(us)"])
    n = data["recorded_steps"]
    data["kernel_types"] = [{"name": name, "calls_per_step": count/n, "duration_sum_s_per_step": duration/1e9/n}
                            for name, (count, duration) in kinds.items()]
    total = sum(count for count, _ in kinds.values())
    if "kernel_count" in data:
        assert total == data["kernel_count"]
    data["kernel_count"] = total
    data["kernel_calls_per_step"] = total/n
    return data


def read_counts_and_official(output):
    """Read exactly the fields used here; full counters remain in raw CSVs."""
    with (output / "step_trace_time.csv").open(newline="") as f:
        official = [{"step": r["Step"], "device_id": r["Device_id"],
                     **{key + "_s": parser.number(value)/1e6 for key, value in r.items()
                        if key not in ("Step", "Device_id")}} for r in csv.DictReader(f)]
    assert len(official) == 4 and {r["step"] for r in official} == {"120", "121", "122", "123"}
    return {"source_directory": str(output), "recorded_steps": len(official), "official_steps": official,
            "official_mean_s_per_step": {key: statistics.mean(r[key] for r in official)
                                         for key in official[0] if key.endswith("_s")}}


def summarize(ranks):
    result = {"ranks": len(ranks), "steps_per_rank": [r["recorded_steps"] for r in ranks],
              "kernels_per_update": statistics.mean(r["kernel_calls_per_step"] for r in ranks)}
    result["official_mean_s_per_step"] = {key: statistics.mean(r["official_mean_s_per_step"][key] for r in ranks)
                                            for key in ranks[0]["official_mean_s_per_step"]}
    ops = sorted(set(item["name"] for r in ranks for item in r["kernel_types"]))
    result["kernel_types"] = {op: {key: statistics.mean(next((item[key] for item in r["kernel_types"] if item["name"] == op), 0) for r in ranks)
                                         for key in ("calls_per_step", "duration_sum_s_per_step")} for op in ops}
    return result


def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--case", choices=("default_profile", "no_prefetch"), default="default_profile")
    arguments.add_argument("--output-dir", type=Path, default=HERE)
    args = arguments.parse_args()
    run = Path("/cache/lingbot/profiling/speed_round3_20261016") / args.case / "run_case"
    target = args.output_dir / "profile"
    target.mkdir(exist_ok=True, parents=True)
    outputs = parser.find_outputs(run / "trace")
    if len(outputs) != 8:
        raise SystemExit(f"Expected eight rank outputs, got {len(outputs)}")
    new, old = [], []
    for path in outputs:
        rank = parser.rank_name(path)
        cached = target / (rank + "_csv.json")
        if cached.exists():
            data = json.loads(cached.read_text())
            assert data["source_directory"] == str(path)
        else:
            data = read_counts_and_official(path)
        complete_types(data)
        (target / (rank + "_csv.json")).write_text(json.dumps(data, indent=2))
        new.append(data)
        old.append(complete_types(json.loads((OLD / "analysis" / (rank + ".json")).read_text())))
    result = {"case": args.case, "baseline": summarize(old), "optimized": summarize(new),
              "limitations": ["Profiler timings include instrumentation overhead; wall throughput is measured outside capture.",
                              "Kernel duration sums are not critical-path elapsed time.",
                              "The model follows the same 121–124 update window, but floating-point rounding may change routing decisions."]}
    result["kernel_count_reduction_percent"] = (1 - result["optimized"]["kernels_per_update"] / result["baseline"]["kernels_per_update"]) * 100
    (args.output_dir / "profile_comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key not in ("baseline", "optimized")}, indent=2))
    for key in ("baseline", "optimized"):
        print(key, result[key]["kernels_per_update"], result[key]["official_mean_s_per_step"])


if __name__ == "__main__":
    main()
