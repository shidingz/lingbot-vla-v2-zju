#!/usr/bin/env python
"""Small CPU-only decoder benchmark with the pre-optimization reader as reference.

This loads files directly, avoiding lingbotvla's NPU initialization. It never
edits dataset files or launches training. Thread counts are deliberately bounded
by default; torchvision's PyAV backend ignores VideoReader(num_threads=...).
"""

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time
import warnings

import torch
import torchvision


REPO = Path(__file__).resolve().parents[1]
CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
THREAD_ENV = "LINGBOT_VIDEO_DECODE_THREADS"


def load_file(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, REPO / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def optimized_decoder():
    return load_file(
        "standalone_pyav_decoder", "lingbotvla/data/vla_data/pyav_decoder.py"
    ).decode_video_frames_pyav


def reference_decode(video_path, timestamps, tolerance_s, threads=1):
    """Old video_utils PyAV path, with thread control and reliable cleanup only.

    Every traversed frame is converted to RGB by VideoReader. Selection uses the
    original torch default dtype, cdist and strict tolerance, including tie order.
    """
    torchvision.set_video_backend("pyav")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        reader = torchvision.io.VideoReader(str(video_path), "video")
    if threads is not None:
        reader.container.streams.video[0].thread_count = threads
    loaded_frames, loaded_ts = [], []
    try:
        reader.seek(min(timestamps), keyframes_only=True)
        last_ts = max(timestamps)
        for frame in reader:
            loaded_frames.append(frame["data"])
            loaded_ts.append(frame["pts"])
            if frame["pts"] >= last_ts:
                break
    finally:
        reader.container.close()
    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)
    distance = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    nearest_distance, nearest_index = distance.min(1)
    assert (nearest_distance < tolerance_s).all(), (
        f"Timestamp tolerance violated: {nearest_distance.tolist()}, tolerance={tolerance_s}"
    )
    result = torch.stack([loaded_frames[index] for index in nearest_index])
    return result.type(torch.uint8)


def episode_metadata(root, episode_index):
    import pyarrow.parquet as pq

    fields = ["episode_index", "length"] + [
        f"videos/{camera}/{field}"
        for camera in CAMERAS
        for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
    ]
    for path in sorted((Path(root) / "meta/episodes").rglob("*.parquet")):
        rows = pq.read_table(
            path, columns=fields, filters=[("episode_index", "=", episode_index)]
        ).to_pylist()
        if rows:
            return rows[0]
    raise ValueError(f"Episode {episode_index} not found under {root}")


def episode_video(root, episode, camera):
    prefix = f"videos/{camera}"
    return Path(root) / (
        f"{prefix}/chunk-{episode[prefix + '/chunk_index']:03d}/"
        f"file-{episode[prefix + '/file_index']:03d}.mp4"
    )


def bounded_queries(episode, camera, timestamps, fps=15):
    bounds = load_file(
        "standalone_video_bounds", "lingbotvla/data/vla_data/video_episode_bounds.py"
    )
    prefix = f"videos/{camera}"
    return bounds.bound_episode_video_timestamps(
        timestamps,
        from_timestamp=episode[prefix + "/from_timestamp"],
        to_timestamp=episode[prefix + "/to_timestamp"],
        fps=fps,
        episode_length=episode["length"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=1273)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(value < 1 for value in args.threads):
        parser.error("repeats and threads must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    decode = optimized_decoder()
    episode = episode_metadata(args.root, args.episode)
    first = min(episode["length"] // 2, max(0, episode["length"] - 50))
    requests = []
    for camera in CAMERAS:
        timestamps, _ = bounded_queries(episode, camera, [first / 15, (first + 49) / 15])
        requests.append((episode_video(args.root, episode, camera), timestamps))
    results = []
    for threads in args.threads:
        os.environ[THREAD_ENV] = str(threads)
        elapsed = {"reference": [], "optimized": []}
        for repeat in range(args.repeats):
            for path, timestamps in requests:
                order = ("reference", "optimized") if repeat % 2 == 0 else ("optimized", "reference")
                outputs = {}
                for implementation in order:
                    start = time.perf_counter()
                    outputs[implementation] = (
                        reference_decode(path, timestamps, 1e-4, threads)
                        if implementation == "reference" else decode(path, timestamps, 1e-4)
                    )
                    elapsed[implementation].append(time.perf_counter() - start)
                if not torch.equal(outputs["reference"], outputs["optimized"]):
                    raise AssertionError(f"Pixel mismatch at {path}: {timestamps}")
        record = {
            "threads": threads,
            "requests_per_implementation": len(elapsed["reference"]),
            "reference_seconds": sum(elapsed["reference"]),
            "optimized_seconds": sum(elapsed["optimized"]),
            "reference_median_seconds": statistics.median(elapsed["reference"]),
            "optimized_median_seconds": statistics.median(elapsed["optimized"]),
            "pixels_identical": True,
        }
        record["measured_speedup"] = record["reference_seconds"] / record["optimized_seconds"]
        results.append(record)
        print(json.dumps(record), flush=True)
    report = {
        "scope": "single CPU worker; three real cameras; warm-cache/order effects and host contention apply",
        "reference": "old torchvision PyAV RGB-every-frame path, explicitly constrained to the same decoder threads",
        "root": str(args.root), "episode": args.episode,
        "repeats": args.repeats, "results": results,
        "torch_threads": torch.get_num_threads(),
        "torch_npu_imported": "torch_npu" in __import__("sys").modules,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
