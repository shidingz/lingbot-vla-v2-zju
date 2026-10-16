"""Exercise the real loader boundary hook with decoding mocked out."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from lingbotvla.data.vla_data import base_dataset
from lingbotvla.data.vla_data.multi_vla_dataset import MultiVLADataset
from lingbotvla.data.vla_data.video_episode_bounds import VideoEpisodeBoundaryError


class VideoEpisodeLoaderTest(unittest.TestCase):
    camera = "observation.images.cam_high"

    def make_loader(self, video_frames=99):
        loader = object.__new__(base_dataset.LeRobotDataset)
        loader.root = Path("/unused")
        loader.tolerance_s = 1e-4
        loader.video_backend = "pyav"
        loader.video_boundary_stats = {}
        episode = {
            "length": 100,
            f"videos/{self.camera}/from_timestamp": 10.0,
            f"videos/{self.camera}/to_timestamp": 10.0 + video_frames / 15,
        }
        loader.meta = SimpleNamespace(
            episodes=[episode], fps=15,
            get_video_file_path=lambda *_: "video.mp4",
        )
        return loader

    def test_v3_passes_protected_timestamp_to_decoder_and_records_clamp(self):
        loader = self.make_loader()
        decoded = torch.zeros((2, 3, 2, 2), dtype=torch.uint8)
        with patch.object(base_dataset, "LEROBOT_DATASET_API", "v3"), patch.object(
            base_dataset, "decode_video_frames", return_value=decoded
        ) as decoder, patch.object(base_dataset.logger, "warning") as warning:
            result = loader._query_videos({self.camera: [50 / 15, 99 / 15]}, 0)
            decoder.assert_called_once_with(
                Path("/unused/video.mp4"), [10 + 50 / 15, 10 + 98 / 15], 1e-4, "pyav",
            )
            self.assertTrue(torch.equal(result[self.camera], decoded))
            self.assertEqual(loader.video_boundary_stats[self.camera]["future_clamps"], 1)
            self.assertEqual(loader.video_boundary_stats[self.camera]["current_clamps"], 0)
            loader._query_videos({self.camera: [50 / 15, 99 / 15]}, 0)
            warning.assert_called_once()  # No log line per repaired sample.

    def test_v2_timestamps_and_diagnostics_are_unchanged(self):
        loader = self.make_loader()
        with patch.object(base_dataset, "LEROBOT_DATASET_API", "v2"), patch.object(
            base_dataset, "decode_video_frames", return_value=torch.zeros((2, 3, 2, 2))
        ) as decoder:
            loader._query_videos({self.camera: [0.0, 99 / 15]}, 0)
            self.assertEqual(decoder.call_args.args[1], [0.0, 99 / 15])
            self.assertEqual(loader.video_boundary_stats, {})

    def test_serious_alignment_error_is_not_randomly_resampled(self):
        dataset = object.__new__(MultiVLADataset)
        dataset.getdata = Mock(side_effect=VideoEpisodeBoundaryError("three frames missing"))
        with patch.object(MultiVLADataset, "__len__", return_value=1):
            with self.assertRaises(VideoEpisodeBoundaryError):
                dataset[0]
        dataset.getdata.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
