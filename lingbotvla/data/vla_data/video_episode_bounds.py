"""Read-only protection for the exclusive video ranges in LeRobot v3 metadata."""

import math


class VideoEpisodeBoundaryError(ValueError):
    """Invalid episode/video alignment that must not be hidden by resampling."""


def bound_episode_video_timestamps(
    timestamps,
    *,
    from_timestamp,
    to_timestamp,
    fps,
    episode_length,
    max_missing_frames=2,
    tolerance_s=1e-4,
):
    """Shift episode timestamps, repeating its last video frame for a short tail.

    ``to_timestamp`` is exclusive. A video may be one or two frames shorter
    than the episode's table; only those missing tail queries are repaired.
    In-range timestamps are returned unchanged apart from the existing offset.
    The returned positions identify repairs, for worker-local diagnostics.
    This does not establish that the remaining video frames are aligned.
    """
    try:
        start, end, rate = float(from_timestamp), float(to_timestamp), float(fps)
        length = int(episode_length)
    except (TypeError, ValueError, OverflowError) as error:
        raise VideoEpisodeBoundaryError("Non-numeric video metadata") from error
    if not all(math.isfinite(x) for x in (start, end, rate)) or rate <= 0 or start < 0 or end <= start:
        raise VideoEpisodeBoundaryError(f"Invalid video range/rate: [{start}, {end}), fps={rate}")
    if length != episode_length or length <= 0:
        raise VideoEpisodeBoundaryError(f"Invalid episode length: {episode_length}")
    frame_span = (end - start) * rate
    video_frames = round(frame_span)
    if video_frames <= 0 or abs(frame_span - video_frames) > tolerance_s * rate:
        raise VideoEpisodeBoundaryError(f"Video duration is not on the frame grid: {frame_span} frames")
    shortfall = length - video_frames
    if shortfall > max_missing_frames:
        raise VideoEpisodeBoundaryError(
            f"Video is {shortfall} frames shorter than episode {length}; allowed={max_missing_frames}"
        )

    last_video_ts = (video_frames - 1) / rate
    last_data_ts = (length - 1) / rate
    shifted, repaired_positions = [], []
    for position, value in enumerate(timestamps):
        try:
            timestamp = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise VideoEpisodeBoundaryError(f"Non-numeric video query: {value!r}") from error
        if not math.isfinite(timestamp) or timestamp < -tolerance_s or timestamp > last_data_ts + tolerance_s:
            raise VideoEpisodeBoundaryError(
                f"Query {timestamp} lies outside episode data [0, {last_data_ts}]"
            )
        if timestamp > last_video_ts + tolerance_s:
            frame = round(timestamp * rate)
            if (
                shortfall <= 0
                or abs(timestamp * rate - frame) > tolerance_s * rate
                or not 1 <= frame - (video_frames - 1) <= max_missing_frames
            ):
                raise VideoEpisodeBoundaryError(f"Unrepairable video tail query {timestamp}, last={last_video_ts}")
            timestamp = last_video_ts
            repaired_positions.append(position)
        shifted.append(start + timestamp)
    return shifted, repaired_positions
