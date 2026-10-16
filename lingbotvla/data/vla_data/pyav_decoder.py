"""CPU PyAV decoding with bounded threads and RGB conversion of selected frames.

Keep the keyframe seek and float32 nearest-PTS selection used by the original
torchvision VideoReader path. Intermediate codec frames still have to be decoded,
but do not need RGB conversion or a torch tensor.
"""

import logging
import math
import os
from pathlib import Path

import av
import torch


def _decode_thread_count() -> int:
    value = os.environ.get("LINGBOT_VIDEO_DECODE_THREADS", "1")
    try:
        threads = int(value)
    except ValueError as error:
        raise ValueError("LINGBOT_VIDEO_DECODE_THREADS must be a positive integer") from error
    if threads < 1:
        raise ValueError("LINGBOT_VIDEO_DECODE_THREADS must be a positive integer; 0 enables unbounded automatic threads")
    return threads


def decode_video_frames_pyav(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Return uint8 TCHW frames, preserving query order and repeated queries.

    LINGBOT_VIDEO_DECODE_THREADS defaults to one codec thread per reader. The
    decoder is opened per call, never shared between DataLoader workers or cached.
    Missing frame timestamps are an error rather than an inferred frame position.
    """
    if not timestamps:
        raise ValueError("At least one video timestamp is required")
    if not all(math.isfinite(timestamp) for timestamp in timestamps):
        raise ValueError("Video timestamps must be finite")

    threads = _decode_thread_count()
    video_path = str(video_path)
    first_ts, last_ts = min(timestamps), max(timestamps)
    container = av.open(video_path, metadata_errors="ignore")
    try:
        streams = container.streams.get(video=0)
        if not streams:
            raise ValueError(f"No video stream found in {video_path}")
        stream = streams[0]
        # VideoReader's num_threads argument is ignored by its PyAV backend.
        # Set the codec context before seek/decode so dav1d cannot auto-create
        # one thread per host CPU in every DataLoader worker.
        stream.codec_context.thread_count = threads
        if stream.time_base is None or stream.time_base <= 0:
            raise ValueError(f"Invalid video stream time base in {video_path}")

        # Match torchvision.io.VideoReader.seek(..., keyframes_only=True),
        # including its rounding and handling of a negative seek timestamp.
        offset = int(round(max(first_ts, 0) / stream.time_base))
        container.seek(offset, backward=True, any_frame=False, stream=stream)

        loaded_frames = []
        loaded_timestamps = []
        decoder = container.decode(video=0)
        while True:
            try:
                frame = next(decoder)
            except (StopIteration, av.error.EOFError):
                break
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Decoded video frame has no timestamp in {video_path}")
            current_ts = float(frame.pts * frame.time_base)
            if not math.isfinite(current_ts):
                raise ValueError(f"Decoded video frame has a non-finite timestamp in {video_path}")
            if log_loaded_timestamps:
                logging.info(f"frame loaded at timestamp={current_ts:.4f}")
            loaded_frames.append(frame)
            loaded_timestamps.append(current_ts)
            if current_ts >= last_ts:
                break

        if not loaded_frames:
            raise ValueError(f"No video frames decoded for timestamps {timestamps} in {video_path}")

        # Keep float32 cdist and first-minimum tie breaking, including for long
        # merged v3 videos where float32 timestamp rounding matters.
        query_ts = torch.tensor(timestamps, dtype=torch.float32)
        loaded_ts = torch.tensor(loaded_timestamps, dtype=torch.float32)
        dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
        min_, argmin_ = dist.min(1)
        is_within_tol = min_ < tolerance_s
        assert is_within_tol.all(), (
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
            "It means that the closest frame that can be loaded from the video is too far away in time."
            "This might be due to synchronization issues with timestamps during data collection."
            "To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts}"
            f"\nvideo: {video_path}"
            "\nbackend: pyav"
        )

        # RGB conversion is substantially more expensive than retaining the
        # native codec frames. Convert each selected frame once, preserving the
        # exact VideoReader color conversion chain and duplicate query ordering.
        selected = argmin_.tolist()
        converted = {
            index: torch.as_tensor(loaded_frames[index].to_rgb().to_ndarray()).permute(2, 0, 1)
            for index in dict.fromkeys(selected)
        }
        closest_frames = torch.stack([converted[index] for index in selected]).type(torch.uint8)
        if log_loaded_timestamps:
            closest_ts = loaded_ts[argmin_]
            logging.info(f"{closest_ts=}")
        return closest_frames
    finally:
        container.close()
