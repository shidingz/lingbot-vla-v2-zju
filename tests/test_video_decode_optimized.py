"""Pixel-identical CPU regressions against the old torchvision/PyAV path.

Set LINGBOT_TEST_VIDEO_ROOT to exercise three real RoboTwin camera streams as
well. Modules are loaded by filename, avoiding package-level NPU initialization.
"""

import importlib.util
from fractions import Fraction
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("video_decode_benchmark", REPO / "tools/bench_video_decode.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)

import av
import numpy as np
import torch


class OptimizedVideoDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.decoder_module = bench.load_file(
            "test_pyav_decoder", "lingbotvla/data/vla_data/pyav_decoder.py"
        )
        cls.decode = staticmethod(cls.decoder_module.decode_video_frames_pyav)
        cls.temp = tempfile.TemporaryDirectory(prefix="lingbot-video-regression-")
        cls.video = Path(cls.temp.name) / "frames.mkv"
        with av.open(str(cls.video), "w") as output:
            stream = output.add_stream("ffv1", rate=16)
            stream.width, stream.height = 32, 24
            stream.pix_fmt = "yuv420p"
            stream.thread_count = 1
            for index in range(24):
                pixels = np.empty((24, 32, 3), dtype=np.uint8)
                pixels[..., 0] = index * 9
                pixels[..., 1] = np.arange(32, dtype=np.uint8)[None, :] * 7
                pixels[..., 2] = np.arange(24, dtype=np.uint8)[:, None] * 10
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compare(self, video, timestamps, tolerance=1e-4, threads=1):
        expected = bench.reference_decode(video, timestamps, tolerance, threads)
        with patch.dict(os.environ, {bench.THREAD_ENV: str(threads)}):
            actual = self.decode(video, timestamps, tolerance)
        self.assertEqual(actual.dtype, torch.uint8)
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertTrue(torch.equal(actual, expected), f"Pixel mismatch: {video}, {timestamps}")
        return actual

    def test_first_last_frames_and_single_frame(self):
        self.compare(self.video, [0.0, 23 / 16], tolerance=0.001)
        self.compare(self.video, [11 / 16], tolerance=0.001)

    def test_duplicate_unsorted_queries(self):
        actual = self.compare(self.video, [12 / 16, 0, 12 / 16, 5 / 16], tolerance=0.001)
        self.assertTrue(torch.equal(actual[0], actual[2]))

    def test_halfway_tie_and_epsilon(self):
        # Matroska uses millisecond timestamps: frames 10 and 12 are exact here.
        self.compare(self.video, [0, 10 / 16 - 1e-7, 10 / 16 + 1e-7], tolerance=0.001)
        self.compare(self.video, [0, 10.5 / 16], tolerance=0.034)

    def test_eof_chooses_last_frame_within_tolerance(self):
        self.compare(self.video, [23 / 16 + 1e-4], tolerance=0.001)

    def test_tolerance_violation_remains_an_error(self):
        for timestamps, tolerance in [([0.03125], 0.001), ([0.0], 0.0), ([1.6], 0.01)]:
            with self.subTest(timestamps=timestamps), self.assertRaises(AssertionError):
                bench.reference_decode(self.video, timestamps, tolerance)
            with self.subTest(timestamps=timestamps), patch.dict(os.environ, {bench.THREAD_ENV: "1"}):
                with self.assertRaises(AssertionError):
                    self.decode(self.video, timestamps, tolerance)

    def test_one_two_four_decoder_threads(self):
        for threads in (1, 2, 4):
            with self.subTest(threads=threads):
                self.compare(self.video, [0, 9 / 16, 23 / 16], tolerance=0.001, threads=threads)

    def mock_container(self, *, missing_pts=False, rgb_error=False, decode_error=False):
        stream = SimpleNamespace(time_base=Fraction(1, 1), codec_context=SimpleNamespace(thread_count=0))
        frames = []
        for index in range(3):
            frame = SimpleNamespace(pts=index, time_base=Fraction(1, 1))
            frame.to_ndarray = Mock(return_value=np.full((2, 2, 3), index, dtype=np.uint8))
            frame.to_rgb = Mock(return_value=frame)
            frames.append(frame)
        if missing_pts:
            frames[0].pts = None
        if rgb_error:
            frames[0].to_rgb.side_effect = RuntimeError("RGB conversion failed")
        container = SimpleNamespace(
            streams=SimpleNamespace(get=Mock(return_value=[stream])),
            seek=Mock(), close=Mock(), decode=Mock(return_value=iter(frames)),
        )
        if decode_error:
            container.decode.side_effect = RuntimeError("decode failed")
        return container, frames, stream

    def test_convert_only_unique_requested_frames_and_close(self):
        container, frames, stream = self.mock_container()
        with patch.object(self.decoder_module.av, "open", return_value=container), patch.dict(
            os.environ, {bench.THREAD_ENV: "1"}
        ):
            actual = self.decode("mock.mp4", [2.0, 0.5, 0.0, 2.0], 0.6)
        self.assertEqual(actual[:, 0, 0, 0].tolist(), [2, 0, 0, 2])
        self.assertEqual([frame.to_rgb.call_count for frame in frames], [1, 0, 1])
        self.assertEqual(stream.codec_context.thread_count, 1)
        container.seek.assert_called_once_with(0, backward=True, any_frame=False, stream=stream)
        container.close.assert_called_once()

    def test_strict_halfway_tolerance_rejects_before_rgb_and_closes(self):
        container, frames, _ = self.mock_container()
        with patch.object(self.decoder_module.av, "open", return_value=container), patch.dict(
            os.environ, {bench.THREAD_ENV: "1"}
        ), self.assertRaises(AssertionError):
            self.decode("mock.mp4", [0.5], 0.5)
        self.assertEqual(sum(frame.to_rgb.call_count for frame in frames), 0)
        container.close.assert_called_once()

    def test_container_closes_on_decode_timestamp_and_rgb_errors(self):
        for option, error_type in (("decode_error", RuntimeError), ("missing_pts", ValueError), ("rgb_error", RuntimeError)):
            container, _, _ = self.mock_container(**{option: True})
            with self.subTest(option=option), patch.object(
                self.decoder_module.av, "open", return_value=container
            ), patch.dict(os.environ, {bench.THREAD_ENV: "1"}), self.assertRaises(error_type):
                self.decode("mock.mp4", [0.0], 1e-4)
            container.close.assert_called_once()

    def test_invalid_thread_settings_fail_before_opening_video(self):
        for value in ("0", "-1", "abc", "1.5"):
            with self.subTest(value=value), patch.dict(os.environ, {bench.THREAD_ENV: value}), patch.object(
                self.decoder_module.av, "open"
            ) as opened, self.assertRaises(ValueError):
                self.decode("mock.mp4", [0.0], 1e-4)
            opened.assert_not_called()

    @unittest.skipUnless(os.environ.get("LINGBOT_TEST_VIDEO_ROOT"), "real dataset path not provided")
    def test_real_random_episodes_three_cameras(self):
        root = Path(os.environ["LINGBOT_TEST_VIDEO_ROOT"])
        rng = random.Random(1016)
        for episode_id in (rng.randrange(0, 1000), rng.randrange(1000, 2000)):
            episode = bench.episode_metadata(root, episode_id)
            last = episode["length"] - 1
            mid = rng.randrange(1, max(2, last - 49))
            # First/last, duplicate, unordered and actual 50-step action span.
            queries = [last / 15, 0, mid / 15, (mid + 49) / 15, mid / 15]
            for camera in bench.CAMERAS:
                with self.subTest(episode=episode_id, camera=camera):
                    timestamps, _ = bench.bounded_queries(episode, camera, queries)
                    self.compare(bench.episode_video(root, episode, camera), timestamps)

    @unittest.skipUnless(os.environ.get("LINGBOT_TEST_VIDEO_ROOT"), "real dataset path not provided")
    def test_real_episode_2354_short_tail_preserved(self):
        root = Path(os.environ["LINGBOT_TEST_VIDEO_ROOT"])
        episode = bench.episode_metadata(root, 2354)
        queries = [15.933333396911621, 17.933332443237305]
        for camera in bench.CAMERAS:
            with self.subTest(camera=camera):
                timestamps, repaired = bench.bounded_queries(episode, camera, queries)
                if camera == "observation.images.cam_high":
                    self.assertEqual(repaired, [1])
                    self.assertAlmostEqual(timestamps[-1], 3749.133333333333, places=8)
                else:
                    self.assertEqual(repaired, [])
                self.compare(bench.episode_video(root, episode, camera), timestamps)

    @unittest.skipUnless(os.environ.get("LINGBOT_TEST_VIDEO_ROOT"), "real dataset path not provided")
    def test_real_episode_148_two_missing_tail_frames(self):
        root = Path(os.environ["LINGBOT_TEST_VIDEO_ROOT"])
        episode = bench.episode_metadata(root, 148)
        last = episode["length"] - 1
        for camera in bench.CAMERAS:
            with self.subTest(camera=camera):
                timestamps, repaired = bench.bounded_queries(episode, camera, [(last - 1) / 15, last / 15])
                if camera == "observation.images.cam_high":
                    self.assertEqual(repaired, [0, 1])
                    self.assertEqual(timestamps[0], timestamps[1])
                self.compare(bench.episode_video(root, episode, camera), timestamps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
