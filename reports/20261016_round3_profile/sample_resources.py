"""Read-only all-card resource sampling; timestamps retain query skew."""
import argparse
import concurrent.futures
import datetime
import json
from pathlib import Path
import subprocess
import time

TZ = datetime.timezone(datetime.timedelta(hours=8))

def now():
    return datetime.datetime.now(TZ).isoformat()

def step(run):
    p = run / "logs/loss.jsonl"
    if not p.exists():
        return 0
    for line in reversed(p.read_text().splitlines()):
        try:
            return json.loads(line)["step"]
        except (json.JSONDecodeError, KeyError):
            pass
    return 0

def query(card):
    start = now()
    try:
        r = subprocess.run(["npu-smi", "info", "-t", "usages", "-i", str(card)],
                           capture_output=True, text=True, timeout=10)
        fields = {}
        for line in r.stdout.splitlines():
            if ":" in line:
                key, value = line.rsplit(":", 1)
                try:
                    fields[key.strip()] = float(value.strip())
                except ValueError:
                    pass
        return str(card), {"start_at": start, "end_at": now(), "returncode": r.returncode, "metrics": fields}
    except subprocess.TimeoutExpired:
        return str(card), {"start_at": start, "end_at": now(), "error": "timeout"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--max-seconds", type=int, default=3600)
    a = p.parse_args()
    deadline = time.monotonic() + a.max_seconds
    with (a.run / "resource_samples.jsonl").open("x") as f, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while time.monotonic() < deadline and not (a.run / "exit_code").exists():
            sample = {"at": now(), "last_step": step(a.run)}
            sample["cards"] = dict(pool.map(query, range(8)))
            sample["end_at"] = now()
            sample["last_step_at_end"] = step(a.run)
            f.write(json.dumps(sample) + "\n")
            f.flush()
            time.sleep(5)

if __name__ == "__main__":
    main()
