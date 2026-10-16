#!/usr/bin/env python3
"""Read-only launcher validation. DRY_RUN never starts NPU work or storage I/O."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MAIN = ROOT / "train_robotwin_npu.sh"
WRAPPER = ROOT / "train_robotwin_npu_optimized.sh"
CONFIG = ROOT / "configs/vla/robotwin/robotwin_npu.yaml"
checks = []
config_source = CONFIG.read_text()
assert re.search(r"^  enable_gradient_checkpointing: true\b", config_source, re.M)
assert re.search(r"^  attention_implementation: eager\b", config_source, re.M)

# Clear only launcher configuration inherited by this test process, retaining
# HOME, CODEX_HOME, PATH and the rest of the established execution environment.
configured_names = set(re.findall(r'^([A-Z][A-Z0-9_]*)="\$\{', MAIN.read_text(), re.M))
configured_names.update({
    "ASCEND_RT_VISIBLE_DEVICES", "LINGBOT_JOINT_ATTN_BACKEND",
    "LINGBOT_MOE_MONITOR_BATCH_COUNTS", "LINGBOT_ADAMW_FOREACH_NPU",
    "LINGBOT_NPU_MOE_IMPL", "LINGBOT_VIDEO_DECODE_THREADS",
})
configured_names.difference_update({"HOME", "CODEX_HOME"})
base_environment = os.environ.copy()
for name in configured_names:
    base_environment.pop(name, None)
base_environment["DRY_RUN"] = "1"


def validate(script, case, updates=None, cli=None, expected=None, error=None):
    env = base_environment.copy()
    env.update(updates or {})
    result = subprocess.run(
        ["bash", str(script), *(cli or [])], env=env, cwd=ROOT,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45,
    )
    logfile = OUT / f"{script.stem}_{case}.log"
    logfile.write_text(result.stdout)
    failures = []
    actual = {}
    if error:
        if result.returncode == 0:
            failures.append("invalid input was accepted")
        if error not in result.stdout:
            failures.append(f"missing rejection message for {error}")
        if "torchrun " in result.stdout:
            failures.append("invalid settings reached command output")
    else:
        if result.returncode != 0:
            failures.append(f"unexpected exit {result.returncode}")
        lines = result.stdout.splitlines()
        actual["exports"] = {}
        for line in lines:
            if line.startswith("export "):
                name, value = shlex.split(line)[1].split("=", 1)
                actual["exports"][name] = value
        commands = [line for line in lines if line.startswith("torchrun ")]
        if len(commands) != 1:
            failures.append(f"expected one command; found {len(commands)}")
        else:
            tokens = shlex.split(commands[0])
            options = {}
            for index, token in enumerate(tokens):
                if token.startswith(("--train.", "--data.")):
                    if "=" in token:
                        name, value = token.split("=", 1)
                    else:
                        name, value = token, tokens[index + 1]
                    options[name[2:]] = value
            actual["options"] = options
            actual["config"] = tokens[tokens.index("tasks/vla/train_lingbotvla.py") + 1]
            actual["nproc"] = tokens[tokens.index("--nproc-per-node") + 1]
        for section, values in (expected or {}).items():
            if isinstance(values, dict):
                for name, value in values.items():
                    found = actual.get(section, {}).get(name)
                    if found != value:
                        failures.append(f"{section}.{name}: expected {value!r}, got {found!r}")
            elif actual.get(section) != values:
                failures.append(f"{section}: expected {values!r}, got {actual.get(section)!r}")
        if "[DRY_RUN]" not in result.stdout:
            failures.append("missing DRY_RUN guard")
        if expected and expected.get("options", {}).get("train.enable_gradient_checkpointing"):
            value = expected["options"]["train.enable_gradient_checkpointing"]
            if not re.search(rf"重计算\s*: {value}（", result.stdout):
                failures.append("summary does not reflect final checkpoint override")
    checks.append({
        "case": f"{script.name}:{case}", "passed": not failures,
        "exit_code": result.returncode, "failures": failures,
        "log": logfile.name, "actual": actual,
    })


default_exports = {
    "LINGBOT_NPU_MOE_IMPL": "gmm",
    "LINGBOT_JOINT_ATTN_BACKEND": "sdpa_fp32",
    "LINGBOT_MOE_MONITOR_BATCH_COUNTS": "1",
    "LINGBOT_ADAMW_FOREACH_NPU": "1",
}
default_options = {
    "train.output_dir": "/cache/lingbot/robotwin_train/robotwin_1ep_optimized",
    "train.logging_dir": str(ROOT / "logs/robotwin_1ep_optimized"),
    "train.obs_output_dir": "obs://yw-2030-extern/Partner_Qin/robot_1014/robotwin_1ep_optimized",
    "train.enable_gradient_checkpointing": "false",
    "train.micro_batch_size": "12", "train.gradient_accumulation_steps": "1",
    "train.global_batch_size": "96", "train.num_train_epochs": "1",
    "train.enable_resume": "true", "train.lr": "5e-5", "train.lr_min": "1e-5",
    "train.train_expert_only": "true",
}
for script in (MAIN, WRAPPER):
    syntax = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    checks.append({"case": f"{script.name}:syntax", "passed": syntax.returncode == 0,
                   "exit_code": syntax.returncode, "failures": [syntax.stderr] if syntax.returncode else []})
    validate(script, "defaults", expected={"exports": default_exports,
             "options": default_options, "nproc": "8", "config": str(CONFIG)})
    run_name = "round2_launcher_validation"
    validate(script, "new_run_name", {"RUN_NAME": run_name}, expected={"options": {
        "train.output_dir": f"/cache/lingbot/robotwin_train/{run_name}",
        "train.logging_dir": str(ROOT / "logs" / run_name),
        "train.obs_output_dir": f"obs://yw-2030-extern/Partner_Qin/robot_1014/{run_name}",
    }})
    validate(script, "rollback", {
        "LINGBOT_JOINT_ATTN_BACKEND": "eager", "LINGBOT_MOE_MONITOR_BATCH_COUNTS": "0",
        "LINGBOT_ADAMW_FOREACH_NPU": "0", "ENABLE_GRAD_CKPT": "true",
    }, expected={"exports": {**default_exports, "LINGBOT_JOINT_ATTN_BACKEND": "eager",
        "LINGBOT_MOE_MONITOR_BATCH_COUNTS": "0", "LINGBOT_ADAMW_FOREACH_NPU": "0"},
        "options": {"train.enable_gradient_checkpointing": "true"}})
    validate(script, "empty_attention", {"LINGBOT_JOINT_ATTN_BACKEND": ""}, expected={
        "exports": {**default_exports, "LINGBOT_JOINT_ATTN_BACKEND": ""},
        "options": {"train.enable_gradient_checkpointing": None}})
    validate(script, "eager_auto_checkpoint", {"LINGBOT_JOINT_ATTN_BACKEND": "eager"}, expected={
        "exports": {"LINGBOT_JOINT_ATTN_BACKEND": "eager"},
        "options": {"train.enable_gradient_checkpointing": None}})
    validate(script, "low_memory", {"ENABLE_GRAD_CKPT": "true"}, expected={
        "exports": default_exports, "options": {"train.enable_gradient_checkpointing": "true"}})
    validate(script, "cli_override", cli=["--train.enable_gradient_checkpointing", "true"],
             expected={"options": {"train.enable_gradient_checkpointing": "true"}})
    validate(script, "extra_args_override", {"EXTRA_ARGS": "--train.enable_gradient_checkpointing=true"},
             expected={"options": {"train.enable_gradient_checkpointing": "true"}})
    validate(script, "cli_wins_over_extra_and_env", {
        "ENABLE_GRAD_CKPT": "true", "EXTRA_ARGS": "--train.enable_gradient_checkpointing false",
    }, cli=["--train.enable_gradient_checkpointing=true"],
             expected={"options": {"train.enable_gradient_checkpointing": "true"}})
    validate(script, "absolute_config", {"CONFIG": str(CONFIG)}, expected={"config": str(CONFIG)})
    for name, bad in (("LINGBOT_JOINT_ATTN_BACKEND", "invalid"),
                      ("LINGBOT_MOE_MONITOR_BATCH_COUNTS", "2"),
                      ("LINGBOT_ADAMW_FOREACH_NPU", "true"),
                      ("LINGBOT_NPU_MOE_IMPL", "invalid"), ("RUN_NAME", "../bad")):
        validate(script, f"invalid_{name.lower()}", {name: bad}, error=name)

report = {
    "generated_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    "scope": "Read-only bash syntax and DRY_RUN argument/environment checks; no training, NPU operation or OBS I/O.",
    "checks": checks, "passed": all(item["passed"] for item in checks),
    "check_count": len(checks), "source_sha256": {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (MAIN, WRAPPER, CONFIG)
    },
    "configuration_note": "Standalone YAML retains eager + checkpoint=true. Main launcher defaults to sdpa_fp32 and conditionally overrides checkpoint=false; explicit empty/eager backend inherits YAML checkpoint policy.",
}
(OUT / "launcher_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"passed": report["passed"], "checks": len(checks),
                  "failed": [item for item in checks if not item["passed"]]}, ensure_ascii=False, indent=2))
raise SystemExit(0 if report["passed"] else 1)
