"""Attach low-rate Python stack sampling only to this experiment's train ranks.

No native extension unwinding, no subprocess tracing, and no process arguments
or environments are printed. The recorded step window must be excluded from
unprofiled throughput. Idle stacks describe waiting locations, not CPU use.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


TZ = dt.timezone(dt.timedelta(hours=8))


def stamp():
    return dt.datetime.now(TZ).isoformat()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def step(run):
    path = run / "logs/loss.jsonl"
    if not path.exists():
        return 0
    records = path.read_text().splitlines()
    for line in reversed(records):
        try:
            return int(json.loads(line)["step"])
        except (ValueError, KeyError):
            continue
    return 0


def rank_pids(snapshot):
    result = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if (proc / "cwd").resolve() != snapshot:
                continue
            args = (proc / "cmdline").read_bytes().split(b"\0")
            if b"tasks/vla/train_lingbotvla.py" not in args:
                continue
            env = dict(item.split(b"=", 1) for item in
                       (proc / "environ").read_bytes().split(b"\0") if b"=" in item)
            rank = int(env.get(b"RANK", b"-1"))
            if rank not in (0, 4):
                continue
            status = (proc / "status").read_text().splitlines()
            parent = int(next(line.split()[1] for line in status if line.startswith("PPid:")))
            parent_args = (Path("/proc") / str(parent) / "cmdline").read_bytes().split(b"\0")
            if not any(Path(os.fsdecode(arg)).name == "torchrun" for arg in parent_args if arg):
                continue
            if rank in result:
                raise RuntimeError(f"Multiple direct torchrun children for rank {rank}")
            result[rank] = int(proc.name)
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            continue
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--start-step", type=int, default=50)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--rate", type=int, default=20)
    args = parser.parse_args()
    run, snapshot = args.run.resolve(), args.snapshot.resolve()
    if not run.is_dir() or not snapshot.is_dir():
        raise SystemExit("Run and snapshot directories must exist")
    pyspy = shutil.which("py-spy")
    if not pyspy:
        raise SystemExit("py-spy not available")
    meta = dict(run=str(run), source_snapshot=str(snapshot), created_at=stamp(),
                rate_hz=args.rate, duration_seconds=args.duration, start_after_step=args.start_step,
                idle_stacks=True, nonblocking=True, native_stacks=False, subprocesses=False,
                limitations=["Idle stacks show waits and cannot measure CPU utilization.",
                             "Nonblocking Python stacks may be incomplete or inconsistent.",
                             "Sampled steps must be excluded from unprofiled throughput.",
                             "Two ranks are sampled; this is not a full eight-rank wall-time attribution."])
    write(run / "host_sampling.json", meta)
    wait_started = time.monotonic()
    while step(run) < args.start_step:
        if (run / "exit_code").exists():
            raise SystemExit("Training finished before CPU sampling window")
        if time.monotonic() - wait_started > 1500:
            raise SystemExit("Timed out waiting for CPU sampling window")
        time.sleep(2)
    begin_step = step(run)
    if begin_step > args.start_step + 10:
        raise SystemExit("Sampling window has passed; do not interfere with later native profiling")
    pids = rank_pids(snapshot)
    if set(pids) != {0, 4}:
        raise SystemExit(f"Expected direct train rank 0/4 processes; found ranks {sorted(pids)}")
    meta.update(started_at=stamp(), completed_step_at_start=begin_step,
                process_ids={str(k): v for k, v in pids.items()}, commands={})
    children, logs = {}, []
    try:
        for rank, pid in sorted(pids.items()):
            output = run / f"host_rank{rank}.speedscope.json"
            cmd = [pyspy, "record", "--pid", str(pid), "--rate", str(args.rate),
                   "--duration", str(args.duration), "--format", "speedscope",
                   "--output", str(output), "--idle", "--nonblocking", "--full-filenames"]
            meta["commands"][str(rank)] = cmd
            log = (run / f"host_rank{rank}.log").open("w")
            logs.append(log)
            children[rank] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        write(run / "host_sampling.json", meta)
        codes = {str(rank): child.wait(timeout=args.duration + 30)
                 for rank, child in children.items()}
    finally:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
        for log in logs:
            log.close()
        meta.update(finished_at=stamp(), completed_step_at_end=step(run),
                    return_codes={str(rank): child.returncode for rank, child in children.items()})
        meta["exclude_steps_inclusive"] = [begin_step, meta["completed_step_at_end"] + 2]
        meta["exclusion_note"] = "Conservative guard: include start boundary and two updates after last completed step."
        write(run / "host_sampling.json", meta)
    print(json.dumps({k: meta[k] for k in ("started_at", "finished_at", "completed_step_at_start",
                                           "completed_step_at_end", "exclude_steps_inclusive", "return_codes")}))
    raise SystemExit(any(code != 0 for code in codes.values()))


if __name__ == "__main__":
    main()
