"""Boundary regressions; the pure helper tests require only the standard library."""

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "lingbotvla/data/vla_data/video_episode_bounds.py"
SPEC = importlib.util.spec_from_file_location("video_episode_bounds", MODULE_PATH)
bounds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bounds)


class VideoEpisodeBoundsTest(unittest.TestCase):
    def bound(self, timestamps, *, length=100, video_frames=100, start=10.0, fps=15, **kwargs):
        return bounds.bound_episode_video_timestamps(
            timestamps, from_timestamp=start, to_timestamp=start + video_frames / fps,
            fps=fps, episode_length=length, **kwargs,
        )

    def test_healthy_queries_preserve_existing_timestamps(self):
        requested = [0.0, 1.234567, 99 / 15]
        result, repaired = self.bound(requested)
        self.assertEqual(result, [10.0 + ts for ts in requested])
        self.assertEqual(repaired, [])

    def test_one_frame_short_tail_never_crosses_episode(self):
        result, repaired = self.bound([50 / 15, 99 / 15], video_frames=99)
        self.assertEqual(result, [10 + 50 / 15, 10 + 98 / 15])
        self.assertEqual(repaired, [1])
        self.assertLess(result[-1], 10 + 99 / 15)

    def test_two_frame_tail_repeats_last_frame_for_both_queries(self):
        result, repaired = self.bound([98 / 15, 99 / 15], video_frames=98)
        self.assertEqual(result, [10 + 97 / 15] * 2)
        self.assertEqual(repaired, [0, 1])

    def test_short_episode_in_range_queries_unchanged(self):
        self.assertEqual(self.bound([0.0, 1.0], video_frames=98), ([10.0, 11.0], []))

    def test_real_episode_2354_float32_timestamp(self):
        result, repaired = bounds.bound_episode_video_timestamps(
            [15.933333396911621, 17.933332443237305],
            from_timestamp=3731.266666666667, to_timestamp=3749.2,
            fps=15, episode_length=270,
        )
        self.assertEqual(repaired, [1])
        self.assertAlmostEqual(result[-1], 3749.133333333333, places=8)

    def test_float32_jitter_at_valid_last_frame_is_not_a_repair(self):
        requested = 99 / 15 - 1e-6
        self.assertEqual(self.bound([requested]), ([10 + requested], []))

    def test_shortfall_larger_than_two_raises_even_for_early_query(self):
        with self.assertRaises(bounds.VideoEpisodeBoundaryError):
            self.bound([0.0], video_frames=97)

    def test_query_beyond_data_episode_raises(self):
        with self.assertRaises(bounds.VideoEpisodeBoundaryError):
            self.bound([100 / 15])

    def test_off_grid_tail_query_raises(self):
        with self.assertRaises(bounds.VideoEpisodeBoundaryError):
            self.bound([98.5 / 15], video_frames=98)

    def test_invalid_metadata_and_queries_raise(self):
        for fps in [0, -15, float("nan"), float("inf")]:
            with self.subTest(fps=fps), self.assertRaises(bounds.VideoEpisodeBoundaryError):
                bounds.bound_episode_video_timestamps(
                    [0], from_timestamp=0, to_timestamp=1, fps=fps, episode_length=15,
                )
        for query in [-1, float("nan"), float("inf")]:
            with self.subTest(query=query), self.assertRaises(bounds.VideoEpisodeBoundaryError):
                self.bound([query])
        with self.assertRaises(bounds.VideoEpisodeBoundaryError):
            bounds.bound_episode_video_timestamps(
                [0], from_timestamp=0, to_timestamp=1.01, fps=15, episode_length=15,
            )
        for length in [None, float("nan"), float("inf"), 1.5]:
            with self.subTest(length=length), self.assertRaises(bounds.VideoEpisodeBoundaryError):
                self.bound([0], length=length)
        with self.assertRaises(bounds.VideoEpisodeBoundaryError):
            self.bound([None])


if __name__ == "__main__":
    unittest.main()
