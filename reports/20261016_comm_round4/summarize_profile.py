"""Compare all-card native 121–124 update captures for communication trials."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/cache/lingbot/profiling/comm_round4_20261016")
PREVIOUS = HERE.parent / "20261016_speed_round3/summarize_profile.py"
spec = importlib.util.spec_from_file_location("round3_summarizer", PREVIOUS)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def read_trace(trace, case):
    outputs = helper.parser.find_outputs(trace)
    if len(outputs) != 8:
        raise RuntimeError(f"{case}: expected eight profiler outputs, got {len(outputs)}")
    ranks = []
    for output in outputs:
        record = helper.complete_types(helper.read_counts_and_official(output))
        ranks.append(record)
    return helper.summarize(ranks)


def main():
    prior = Path("/cache/lingbot/profiling/speed_round3_20261016/no_prefetch/run_case/trace")
    result = {"round3_baseline": read_trace(prior, "round3_baseline")}
    for root in sorted(ROOT.iterdir()):
        trace = root / "run_case/trace"
        if trace.is_dir() and (root / "run_case/exit_code").exists():
            result[root.name] = read_trace(trace, root.name)
    (HERE / "profile_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    for case, data in result.items():
        print(case, "kernels/step", data["kernels_per_update"])
        print("official", data["official_mean_s_per_step"])
        for name in ("hcom_allGather_", "hcom_reduceScatter_", "hcom_allReduce_"):
            print(name, data["kernel_types"].get(name))


if __name__ == "__main__":
    main()
