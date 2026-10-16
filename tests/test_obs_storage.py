"""Offline storage lifecycle tests: no torch, moxing, network or training."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zlib


MODULE_PATH = Path(__file__).resolve().parents[1] / "lingbotvla/utils/obs_storage.py"
SPEC = importlib.util.spec_from_file_location("obs_storage_under_test", MODULE_PATH)
obs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(obs)


class FakeCRC:
    def __init__(self):
        self.crc = 0

    def update(self, data):
        self.crc = zlib.crc32(data, self.crc)


class FakeFile:
    def __init__(self):
        self.objects = {}
        self.copies = []
        self.fail_destinations = set()
        self.bad_crc = set()
        self.missing_crc = set()

    def exists(self, uri):
        return uri in self.objects or any(key.startswith(uri.rstrip("/") + "/") for key in self.objects)

    def copy(self, source, destination, attach_crc64=False):
        self.copies.append((source, destination))
        if destination in self.fail_destinations:
            raise IOError("secret-auth-token-should-never-be-in-error-output")
        data = self.objects[source] if source.startswith("obs://") else Path(source).read_bytes()
        if destination.startswith("obs://"):
            self.objects[destination] = data
        else:
            Path(destination).write_bytes(data)

    def File(self, uri, mode):
        if mode != "rb":
            raise AssertionError(mode)
        return io.BytesIO(self.objects[uri])

    def read_metadata(self, uri):
        content = self.objects[uri]
        checksum = zlib.crc32(content)
        if uri in self.bad_crc:
            checksum += 1
        return {"contentLength": len(content),
                "crc64": None if uri in self.missing_crc else str(checksum)}


class ObsStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.local = Path(self.temp.name)
        self.backend = FakeFile()
        self.store = obs.ObsStorage("obs://bucket/runs/example", backend=self.backend,
                                    crc64_factory=FakeCRC, retries=2, workers=2, retry_wait=0)

    def write(self, path, data=b"sample"):
        path = self.local / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def dataset(self):
        for name in ("data/chunk-000/file-000.parquet", "videos/cam/chunk-000/file-000.mp4",
                     "meta/info.json", "meta/stats.json", "meta/tasks.parquet",
                     "meta/episodes/chunk-000/file-000.parquet"):
            self.write("dataset/" + name, name.encode())
        self.write("dataset/backup_provenance.json", b"not part of the data tree")
        return self.local / "dataset"

    def checkpoint(self, step=1):
        root = f"run/checkpoints/global_step_{step}"
        for name in ("model/.metadata", "model/rank0.distcp", "optimizer/.metadata",
                     "optimizer/rank0.distcp", "extra_state/rank0.pt", "hf_ckpt/model.safetensors"):
            self.write(root + "/" + name, (name + str(step)).encode())
        for name in ("lingbotvla_cli.yaml", "data_source.json", "assets/norm_stats/robotwin.json",
                     "configs/robot_configs/robotwin.yaml", "model_assets/config.json",
                     "logs/loss.jsonl", "logs/train.log", "train_robotwin.txt", "images/example.png"):
            self.write("run/" + name, name.encode())
        return self.local / root, self.local / "run"

    def upload_checkpoint(self, step=1):
        checkpoint, run = self.checkpoint(step)
        return self.store.upload_checkpoint(checkpoint, run, step, step)

    def recommit_manifest(self, manifest, checkpoint_root):
        raw = obs._json_bytes(manifest)
        digest = hashlib.sha256(raw).hexdigest()
        self.backend.objects[checkpoint_root + "/manifest.json"] = raw
        self.backend.objects[checkpoint_root + "/_SUCCESS"] = obs._json_bytes({"manifest_sha256": digest})
        latest_uri = self.store.root_uri + "/latest.json"
        if latest_uri in self.backend.objects:
            latest = json.loads(self.backend.objects[latest_uri])
            latest["manifest_sha256"] = digest
            self.backend.objects[latest_uri] = obs._json_bytes(latest)

    def test_constructor_does_not_initialize_backend(self):
        store = obs.ObsStorage("obs://bucket/run")
        self.assertIsNone(store._backend)

    def test_dataset_round_trip_keeps_metadata_outside_data_tree(self):
        uri = "obs://bucket/datasets/RoboTwin_lerobot_v30"
        result = self.store.upload_dataset(self.dataset(), uri)
        self.assertEqual(result["file_count"], 6)
        self.assertEqual(result["records_uri"], "obs://bucket/datasets/_records/RoboTwin_lerobot_v30")
        self.assertNotIn(uri + "/backup_provenance.json", self.backend.objects)
        self.assertNotIn(uri + "/manifest.json", self.backend.objects)
        destination = self.local / "cache"
        self.assertEqual(self.store.ensure_dataset(uri, destination), str(destination))
        copies = len(self.backend.copies)
        self.store.ensure_dataset(uri, destination)
        self.assertEqual(len(self.backend.copies), copies, "verified cache should avoid repeat downloads")
        (destination / "meta/info.json").write_bytes(b"damaged")
        with self.assertRaisesRegex(obs.ObsStorageError, "another empty OBS_CACHE_DIR"):
            self.store.ensure_dataset(uri, destination)
        self.assertEqual((destination / "meta/info.json").read_bytes(), b"damaged")
        self.assertEqual(len(self.backend.copies), copies)

    def test_checkpoint_round_trip_preserves_resume_and_inference_assets(self):
        result = self.upload_checkpoint()
        remote = self.store.root_uri + "/checkpoints/global_step_1"
        self.assertEqual(result["remote_checkpoint_uri"], remote)
        self.assertIn(remote + "/model/.metadata", self.backend.objects)
        self.assertIn(self.store.root_uri + "/lingbotvla_cli.yaml", self.backend.objects)
        restored = Path(self.store.download_latest(self.local / "restored"))
        self.assertTrue((restored / "optimizer/rank0.distcp").is_file())
        self.assertTrue((restored / "hf_ckpt/model.safetensors").is_file())
        self.assertTrue((restored.parents[1] / "assets/norm_stats/robotwin.json").is_file())
        self.assertTrue((restored.parents[1] / "logs/loss.jsonl").is_file())
        self.assertTrue((restored.parents[1] / "images/example.png").is_file())
        self.assertTrue((self.local / "run/checkpoints/global_step_1/model/rank0.distcp").is_file())
        with self.assertRaises(obs.ObsStorageError):
            self.store.assert_new_run()

    def test_file_failure_propagates_without_marker_or_cleanup(self):
        checkpoint, run = self.checkpoint()
        remote = self.store.root_uri + "/checkpoints/global_step_1"
        self.backend.fail_destinations.add(remote + "/model/rank0.distcp")
        with self.assertRaises(obs.ObsStorageError) as error:
            self.store.upload_checkpoint(checkpoint, run, 1, 1)
        self.assertNotIn("secret-auth-token", str(error.exception))
        self.assertNotIn(remote + "/_SUCCESS", self.backend.objects)
        self.assertNotIn(self.store.root_uri + "/latest.json", self.backend.objects)
        self.assertTrue((checkpoint / "model/rank0.distcp").is_file())

    def test_crc_mismatch_is_not_marked_successful(self):
        checkpoint, run = self.checkpoint()
        remote = self.store.root_uri + "/checkpoints/global_step_1"
        self.backend.bad_crc.add(remote + "/hf_ckpt/model.safetensors")
        with self.assertRaises(obs.ObsStorageError):
            self.store.upload_checkpoint(checkpoint, run, 1, 1)
        self.assertNotIn(remote + "/_SUCCESS", self.backend.objects)

    def test_missing_remote_crc_uses_full_sha_readback(self):
        remote = self.store.root_uri + "/checkpoints/global_step_1"
        self.backend.missing_crc.add(remote + "/hf_ckpt/model.safetensors")
        result = self.upload_checkpoint()
        self.assertEqual(result["remote_checkpoint_uri"], remote)

    def test_failed_latest_can_be_retried_without_reuploading_weights(self):
        checkpoint, run = self.checkpoint()
        latest = self.store.root_uri + "/latest.json"
        self.backend.fail_destinations.add(latest)
        with self.assertRaises(obs.ObsStorageError):
            self.store.upload_checkpoint(checkpoint, run, 1, 1)
        self.backend.fail_destinations.clear()
        remote_model = self.store.root_uri + "/checkpoints/global_step_1/model/rank0.distcp"
        count = sum(dst == remote_model for _, dst in self.backend.copies)
        self.store.upload_checkpoint(checkpoint, run, 1, 1)
        self.assertEqual(sum(dst == remote_model for _, dst in self.backend.copies), count)
        (checkpoint / "model/rank0.distcp").write_bytes(b"different run")
        with self.assertRaises(obs.ObsStorageError):
            self.store.upload_checkpoint(checkpoint, run, 1, 1)

    def test_missing_latest_is_distinct_from_invalid_or_corrupt_latest(self):
        self.assertIsNone(self.store.download_latest(self.local / "restored"))
        self.backend.objects[self.store.root_uri + "/latest.json"] = b"{bad json"
        with self.assertRaises(obs.ObsStorageError):
            self.store.download_latest(self.local / "restored")

    def test_missing_latest_with_existing_checkpoint_is_an_error(self):
        self.backend.objects[self.store.root_uri + "/checkpoints/global_step_1/model/data"] = b"partial"
        with self.assertRaisesRegex(obs.ObsStorageError, "no latest pointer"):
            self.store.download_latest(self.local / "restored")

    def test_matching_partial_dataset_cache_can_finish_without_replacing_files(self):
        uri = "obs://bucket/datasets/example"
        self.store.upload_dataset(self.dataset(), uri)
        path = self.write("cache/meta/info.json", b"meta/info.json")
        before = path.stat().st_mtime_ns
        self.store.ensure_dataset(uri, self.local / "cache")
        self.assertEqual(path.stat().st_mtime_ns, before)

    def test_manifest_parent_traversal_is_rejected_before_download(self):
        result = self.upload_checkpoint()
        manifest = result["manifest"]
        manifest["files"][0]["path"] = "../escape"
        self.recommit_manifest(manifest, result["remote_checkpoint_uri"])
        copies = len(self.backend.copies)
        with self.assertRaises(obs.ObsStorageError):
            self.store.download_latest(self.local / "restored")
        self.assertEqual(len(self.backend.copies), copies)

    def test_manifest_cannot_restore_unexpected_run_assets(self):
        result = self.upload_checkpoint()
        manifest = result["manifest"]
        manifest["run_files"][0]["path"] = "checkpoints/other/overwrite"
        self.recommit_manifest(manifest, result["remote_checkpoint_uri"])
        with self.assertRaises(obs.ObsStorageError):
            self.store.download_latest(self.local / "restored")

    def test_corrupt_download_never_returns_a_restorable_checkpoint(self):
        result = self.upload_checkpoint()
        self.backend.objects[result["remote_checkpoint_uri"] + "/model/rank0.distcp"] = b"truncated"
        with self.assertRaises(obs.ObsStorageError):
            self.store.download_latest(self.local / "restored")
        self.assertFalse((self.local / "restored/checkpoints/global_step_1/model/rank0.distcp").exists())

    def test_dataset_source_link_cannot_escape_requested_tree(self):
        root = self.dataset()
        outside = self.write("outside.txt", b"private")
        (root / "data/escape").symlink_to(outside)
        with self.assertRaises(obs.ObsStorageError):
            self.store.upload_dataset(root, "obs://bucket/datasets/example")
        self.assertEqual(self.backend.copies, [])


if __name__ == "__main__":
    unittest.main()
