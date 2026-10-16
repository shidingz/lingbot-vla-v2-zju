"""Run hardware correctness checks after the isolated baseline releases NPUs."""
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
baseline_exit = Path("/cache/lingbot/profiling/speed_round3_20261016/baseline/run_case/exit_code")
deadline = time.monotonic() + 900
while not baseline_exit.exists():
    if time.monotonic() > deadline:
        raise SystemExit("Baseline did not finish in time")
    time.sleep(5)
if baseline_exit.read_text().strip() != "0":
    raise SystemExit("Baseline failed")
before = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True)
if before.stdout.count("No running processes found in NPU") != 8:
    raise SystemExit("Expected eight idle NPUs")
env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": str(ROOT), "ROUND3_TEST_DEVICE": "npu:0"}
commands = [
    ("npu_tests.txt", [sys.executable, "-m", "pytest", "tests/test_round3_optimizations.py", "-q"]),
    ("distributed_test.log", [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", str(HERE / "check_distributed.py")]),
]
for name, command in commands:
    print("Starting " + name, flush=True)
    with (HERE / name).open("w") as output:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    print(f"{name}: exit {result.returncode}", flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
print("All hardware checks passed", flush=True)
