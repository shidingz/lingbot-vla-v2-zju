"""Validate the completed default-entry run and summarize its measured windows."""
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
latest = json.loads((HERE / "latest_run.json").read_text())
run = Path(latest["run"])
steps = latest["steps"]
spec = importlib.util.spec_from_file_location(
    "round2_analysis", REPO / "reports/20261016_speed_round2/analyze_round2.py"
)
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)
case, raw = analyzer.read_case(run, steps)
assert case["eligible"], case["exclusion_reasons"]
rows = raw["rows"]
assert [row["step"] for row in rows] == list(range(1, steps + 1))
assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v, (int, float)))
train = raw["config"]["train"]
expected = {
    "micro_batch_size": 12, "global_batch_size": 96, "gradient_accumulation_steps": 1,
    "optimizer": "muon", "enable_fp32": True, "enable_mixed_precision": True,
    "enable_gradient_checkpointing": False, "train_expert_only": True,
    "enable_resume": False, "profile_only": True, "profile_ranks": "", "obs_output_dir": "",
}
for key, value in expected.items():
    assert train[key] == value, (key, train[key], value)
assert case["lr_horizons"] == [5717]
text = (run / "console.log").read_text(errors="replace")
assert "moe=gmm joint_attention=sdpa_fp32 monitor_batch=1 adamw_foreach=1 gradient_checkpointing=false" in text
assert f"Profiling run: skip checkpoint at step {steps}." in text
assert not list((run / "run").glob("**/global_step_*"))
snapshot_check = json.loads((run / "snapshot_hash_check.json").read_text())
assert snapshot_check["passed"], snapshot_check
snapshot = Path(raw["manifest"]["source_snapshot"])
production_files = [
    "train_robotwin_npu.sh", "train_robotwin_npu_optimized.sh",
    "configs/vla/robotwin/robotwin_npu.yaml",
    "lingbotvla/ops/npu_moe.py", "lingbotvla/ops/joint_attention.py",
    "lingbotvla/optim/optimizer.py",
    "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py",
]
source_hashes = {}
for name in production_files:
    current = hashlib.sha256((REPO / name).read_bytes()).hexdigest()
    frozen = hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
    assert current == frozen, name
    source_hashes[name] = current
launch = json.loads((HERE / "launcher_validation.json").read_text())
assert launch["passed"]
for name, digest in launch["source_sha256"].items():
    assert source_hashes[name] == digest, name
windows = {f"{first}-{last}": analyzer.window_summary(case, raw, first, last)
           for first, last in [(13, 60), (13, steps), (115, steps)]}
assert all(w["available"] and w["usable_for_speed_comparison"] for w in windows.values())
warmup = int(5717 * train["lr_warmup_ratio"])
assert warmup == 114 and rows[113]["lr"] > rows[112]["lr"] and rows[114]["lr"] < rows[113]["lr"]
historical = json.loads((REPO / "reports/20261016_speed_round2/final_summary.json").read_text())
history = {item["case"]: item for item in historical["results"]}
summary = {
    "completed_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
    "status": "passed", "run": str(run), "updates": steps, "launcher_exit_code": 0,
    "performance_flags_from_main_defaults": True, "launcher_checks_passed": launch["check_count"],
    "all_numeric_log_values_finite": True, "lr_horizon": 5717, "lr_warmup_steps": warmup,
    "peak_allocated_rank0_gib": case["peak_allocated_gib"], "windows": windows,
    "source_sha256": source_hashes, "snapshot_hash_check": snapshot_check,
    "new_default_run_name": "robotwin_1ep_optimized",
    "historical_controlled_ab": {
        "source": "../20261016_speed_round2/final_summary.json",
        "baseline": history["baseline_repeat"], "optimized": history["combined_no_checkpoint"],
        "throughput_gain_pct": historical["throughput_gain_vs_repeat_pct"],
    },
    "limits": [
        "This is a default-entry integration run; historical controlled A/B results are reported separately.",
        "160 updates, not full-epoch convergence or RoboTwin task-success validation.",
        "No full-size DCP save/resume in this run; existing optimizer state-dict tests were retained.",
        "Peak allocation is rank 0 only; resource sampling covers cards 0/4 and is not MFU.",
    ],
}
(HERE / "completion.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
lines = [
    "第二轮优化已接入默认训练入口", "",
    "主入口：bash train_robotwin_npu.sh",
    "兼容入口：bash train_robotwin_npu_optimized.sh（委托主入口，默认值统一维护）",
    "默认：GMM + FP32 SDPA + MoE监控合批 + AdamW foreach + 关闭动作层梯度重计算。",
    "默认实验名：robotwin_1ep_optimized；旧 robotwin_1ep_fast 的产物保留。",
    "单独使用 YAML 时仍保留 eager + 重计算；主脚本在 SDPA 组合下覆盖为关闭重计算。",
    "", f"集成验收：8卡，global batch96，连续{steps}步，exit0，所有日志数值有限。",
    f"学习率总步数5717；跨过第{warmup}步预热；{launch['check_count']}项启动检查通过。",
    f"rank0峰值已分配显存：{case['peak_allocated_gib']:.2f} GiB。", "",
]
for name, window in windows.items():
    lines.append(f"第{name}步：实际间隔 {window['wall_seconds_per_update']:.4f} s/update；"
                 f"吞吐 {window['global_samples_per_second']:.4f} samples/s；"
                 f"日志均值 {window['logged_step_seconds']['mean']:.4f} s/update。")
lines += [
    "", "已有同条件60步A/B：3.5417 → 2.6042秒/步，耗时降低26.47%，吞吐提高36%。",
    "本次是默认入口的更长集成验证，没有把历史对照直接当成新的同场A/B。", "",
    "低显存模式：ENABLE_GRAD_CKPT=true bash train_robotwin_npu.sh",
    "回退第一轮：LINGBOT_JOINT_ATTN_BACKEND=eager LINGBOT_MOE_MONITOR_BATCH_COUNTS=0 "
    "LINGBOT_ADAMW_FOREACH_NPU=0 ENABLE_GRAD_CKPT=true bash train_robotwin_npu.sh",
    "恢复旧实验：RUN_NAME=robotwin_1ep_fast bash train_robotwin_npu.sh（默认enable_resume=true）。",
    "", "限制：尚未用新默认跑完整epoch或闭环评测；本次没有执行完整大小的DCP保存/恢复。",
    f"原始日志：{run}", "数值/硬件正确性证据：../20261016_speed_round2/",
]
(HERE / "analysis.txt").write_text("\n".join(lines) + "\n")
print(json.dumps({"status": "passed", "steps": steps, "peak_allocated_rank0_gib": case["peak_allocated_gib"],
                  "windows": {k: {f: v[f] for f in ["wall_seconds_per_update", "global_samples_per_second"]}
                              for k, v in windows.items()}}, indent=2))
