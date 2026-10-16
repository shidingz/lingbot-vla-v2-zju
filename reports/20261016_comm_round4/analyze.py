"""Read-only comparison of completed runs; log wall time includes data waits."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("round2_analysis", HERE.parent / "20261016_speed_round2/analyze_round2.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
ROOT = Path("/cache/lingbot/profiling/comm_round4_20261016")


def resources(rows, first, last):
    by_card = {str(i): {} for i in range(8)}
    selected = [r for r in rows if first <= r.get("last_step", -1) and r.get("last_step_at_end", last) < last]
    for row in selected:
        for card, query in row["cards"].items():
            for metric, value in query.get("metrics", {}).items():
                by_card[card].setdefault(metric, []).append(value)
    return {"query_cycles": len(selected), "cards": {c: {m: helper.stats(v) for m, v in values.items()} for c, values in by_card.items()},
            "limitation": "5-second polling plus query time; asynchronous hardware utilization samples, not MFU."}


def main():
    cases, raw = {}, {}
    for root in sorted(ROOT.iterdir()):
        run = root / "run_case"
        if not (run / "exit_code").exists():
            continue
        case, data = helper.read_case(run, 120)
        cases[root.name], raw[root.name] = case, data
        for lo, hi in ((13, 60), (31, 110), (81, 110), (115, 120)):
            window = helper.window_summary(case, data, lo, hi)
            window["resources"] = resources(data["resources"], lo, hi)
            case["windows"][f"{lo}:{hi}"] = window
    result = {"cases": cases, "comparisons": {}}
    if "baseline_8" in cases:
        for name, case in cases.items():
            if name == "baseline_8" or not case["eligible"] or not cases["baseline_8"]["eligible"]:
                continue
            result["comparisons"][name] = comparison = {
                "paired_metrics": helper.pair_rows(raw["baseline_8"]["rows"], raw[name]["rows"], 1, 120),
                "config_differences": helper.config_diff(raw["baseline_8"]["config"], raw[name]["config"]),
                "dataset_list_hash_matches": raw["baseline_8"]["manifest"]["dataset_list_sha256"] == raw[name]["manifest"]["dataset_list_sha256"],
                "performance": {},
            }
            for key in case["windows"]:
                before = cases["baseline_8"]["windows"][key]["wall_seconds_per_update"]
                after = case["windows"][key]["wall_seconds_per_update"]
                if before and after and case["windows"][key]["usable_for_speed_comparison"]:
                    comparison["performance"][key] = {"baseline_s": before, "candidate_s": after,
                        "throughput_gain_percent": (before / after - 1) * 100,
                        "update_time_reduction_percent": (1 - after / before) * 100}
    (HERE / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({"cases": {k: {"steps": v["updates"], "exit": v["exit_code"], "eligible": v["eligible"], "peak_gib": v["peak_allocated_gib"]} for k, v in cases.items()},
                      "comparisons": {k: v["performance"] for k, v in result["comparisons"].items()}}, indent=2))


if __name__ == "__main__":
    main()
