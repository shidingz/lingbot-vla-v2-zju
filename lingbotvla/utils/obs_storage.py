"""Verified OBS transfers, independent of torch and imported only when enabled.

Checkpoints are immutable after their _SUCCESS object is written. The caller
owns training synchronization and local checkpoint cleanup. Dataset records live
beside the dataset, under ``_records/<dataset_name>``; the dataset itself contains
only the original data/, meta/ and videos/ trees.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import threading
import time
from urllib.parse import unquote, urlsplit


LOG = logging.getLogger(__name__)
FORMAT_VERSION = 1
MAX_JSON_BYTES = 16 * 1024 * 1024
RUN_FILES = ("lingbotvla_cli.yaml", "data_source.json")
OPTIONAL_RUN_FILES = ("train_robotwin.txt", "logs/loss.jsonl", "logs/train.log")
RUN_TREES = ("assets/norm_stats", "configs/robot_configs", "model_assets", "images")
DATA_TREES = ("data", "meta", "videos")


class ObsStorageError(RuntimeError):
    """A transfer or verification failed; local source files remain intact."""


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ObsStorageError("Invalid relative object path")
    parts = value.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ObsStorageError("Object path must not contain empty, dot or parent components")
    decoded = unquote(value)
    if decoded != value and ("/" in decoded or "\\" in decoded or decoded in (".", "..")):
        raise ObsStorageError("Encoded path separators are not allowed")
    return value


def _uri(value: str) -> str:
    parsed = urlsplit(str(value))
    if parsed.scheme != "obs" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ObsStorageError("Expected obs://bucket/path without a query or fragment")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", parsed.netloc):
        raise ObsStorageError("Invalid OBS bucket name")
    path = parsed.path.strip("/")
    if path:
        _relative(path)
    return "obs://" + parsed.netloc + ("/" + path if path else "")


def _join(root: str, relative: str) -> str:
    return root.rstrip("/") + "/" + _relative(relative)


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_name(name: str) -> bool:
    return (name.startswith((".obs-", ".hf_ckpt.tmp.", ".ema_hf_ckpt.tmp."))
            or name.endswith((".upload_record", ".download_record", ".tmp")))


def _files(root: Path) -> dict[str, Path]:
    """Inventory regular files, rejecting links outside the requested source."""
    root = root.resolve()
    if not root.is_dir():
        raise ObsStorageError(f"Missing source directory: {root}")
    found = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not _temporary_name(d))
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise ObsStorageError("Symlinked subdirectories are not supported")
        for name in sorted(names):
            if _temporary_name(name):
                continue
            path = Path(directory) / name
            if not path.is_file():
                raise ObsStorageError("Only regular files can be uploaded")
            if not path.resolve().is_relative_to(root):
                raise ObsStorageError("Source symlink escapes its directory")
            found[_relative(path.relative_to(root).as_posix())] = path
    return found


def _target(root: Path, relative: str) -> Path:
    relative = _relative(relative)
    root = root.absolute()
    result = root.joinpath(*PurePosixPath(relative).parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ObsStorageError("Download target escapes its directory")
    if result.is_symlink():
        raise ObsStorageError("Refusing to overwrite a symlink")
    return result


class ObsStorage:
    """OBS run storage with bounded workers and verified, resumable transfers.

    ``backend`` is a mox.file-compatible object for tests. ``crc64_factory`` may
    also be injected for tests; production uses the installed OBS SDK's Crc64.
    Public methods never delete a local checkpoint or start model training.
    """

    def __init__(self, root_uri: str, retries: int = 3, workers: int = 2,
                 backend=None, crc64_factory=None, retry_wait: float = 1.0):
        self.root_uri = _uri(root_uri)
        self.retries = max(1, int(retries))
        self.workers = max(1, int(workers))
        self.retry_wait = max(0.0, float(retry_wait))
        self._backend = backend
        self._crc64_factory = crc64_factory
        self._init_lock = threading.Lock()

    @property
    def file(self):
        if self._backend is None:
            with self._init_lock:
                if self._backend is None:
                    # SDK DEBUG logs can contain signed request headers.
                    os.environ["MOX_OBS_CLIENT_LOG"] = "0"
                    for key, value in {
                        "MOX_FILE_RETRY_COUNT": "5", "MOX_FILE_RETRY_WAIT": "1",
                        "MOX_FILE_RETRY_MAX_WAIT": "15",
                        "MOX_FILE_PARTIAL_MAXIMUM_SIZE": str(1024 ** 3),
                        "MOX_FILE_LARGE_FILE_METHOD": "2",
                        "MOX_FILE_LARGE_FILE_TASK_NUM": "4",
                        "MOX_FILE_LARGE_FILE_PART_SIZE": str(64 * 1024 ** 2),
                    }.items():
                        os.environ.setdefault(key, value)
                    try:
                        import moxing as mox
                        mox.file.set_auth(obs_client_log=False, client_timeout=120, large_file_method=2)
                    except Exception as exc:
                        raise ObsStorageError(f"Cannot initialize moxing ({type(exc).__name__})") from None
                    self._backend = mox.file
        return self._backend

    def _retry(self, operation, callback):
        for attempt in range(self.retries):
            try:
                return callback()
            except Exception as exc:
                # Do not interpolate SDK exception strings or response objects.
                if attempt + 1 == self.retries:
                    raise ObsStorageError(
                        f"{operation} failed after {self.retries} attempt(s) ({type(exc).__name__})"
                    ) from None
                LOG.warning("OBS %s retry %d/%d (%s)", operation, attempt + 1,
                            self.retries, type(exc).__name__)
                time.sleep(min(self.retry_wait * 2 ** attempt, 15.0))

    def _exists(self, uri):
        return bool(self._retry("object lookup", lambda: self.file.exists(uri)))

    def _read_bytes(self, uri):
        def read():
            with self.file.File(uri, "rb") as stream:
                data = stream.read(MAX_JSON_BYTES + 1)
            if not isinstance(data, bytes) or len(data) > MAX_JSON_BYTES:
                raise ObsStorageError("Invalid or oversized metadata object")
            return data
        return self._retry("read metadata", read)

    def _read_json(self, uri):
        raw = self._read_bytes(uri)
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ObsStorageError("Invalid JSON metadata") from None
        if not isinstance(value, dict):
            raise ObsStorageError("Expected a JSON metadata object")
        return value, raw

    def _write_json(self, uri, value):
        raw = _json_bytes(value)
        with tempfile.TemporaryDirectory(prefix="lingbot_obs_metadata_") as directory:
            source = Path(directory) / "metadata.json"
            source.write_bytes(raw)
            def write():
                self.file.copy(str(source), uri)
                if self._read_bytes(uri) != raw:
                    raise ObsStorageError("Metadata readback mismatch")
            self._retry("commit metadata", write)
        return hashlib.sha256(raw).hexdigest()

    def _crc(self):
        if self._crc64_factory is None:
            try:
                from obs.util import Crc64
            except ImportError:
                raise ObsStorageError("Installed OBS SDK lacks CRC64 support") from None
            self._crc64_factory = Crc64
        return self._crc64_factory()

    def _verify_sha(self, uri, record):
        digest = hashlib.sha256()
        count = 0
        with self.file.File(uri, "rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                count += len(chunk)
        if count != record["size"] or digest.hexdigest() != record["sha256"]:
            raise ObsStorageError("Remote SHA256 mismatch")

    def _upload_file(self, source: Path, uri: str, relative: str, verification: str):
        before = source.stat()
        digest = hashlib.sha256()
        crc = self._crc() if verification == "crc64" else None
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                if crc is not None:
                    crc.update(chunk)
        record = {"path": relative, "size": before.st_size, "sha256": digest.hexdigest()}
        if crc is not None:
            record["crc64"] = str(crc.crc)
        # Isolate the SDK's <source>.upload_record from checkpoint inventories.
        with tempfile.TemporaryDirectory(prefix="lingbot_obs_upload_") as directory:
            link = Path(directory) / "source"
            link.symlink_to(source.absolute())
            def upload():
                self.file.copy(str(link), uri, attach_crc64=crc is not None)
                metadata = self.file.read_metadata(uri)
                field = metadata.get if isinstance(metadata, dict) else lambda k: getattr(metadata, k, None)
                if int(field("contentLength")) != record["size"]:
                    raise ObsStorageError("Remote file size mismatch")
                remote_crc = field("crc64")
                if crc is not None and remote_crc is not None:
                    if int(remote_crc) != int(record["crc64"]):
                        raise ObsStorageError("Remote CRC64 mismatch")
                else:
                    self._verify_sha(uri, record)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                        after.st_size, after.st_mtime_ns, after.st_ino):
                    raise ObsStorageError("Source changed during upload")
            self._retry("upload and verify file", upload)
        return record

    def _upload_files(self, files, root, verification="crc64"):
        self.file  # Initialize before starting worker threads.
        records = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self._upload_file, source, _join(root, relative),
                                       relative, verification): relative
                       for relative, source in sorted(files.items())}
            errors = []
            for future in as_completed(futures):
                try:
                    records.append(future.result())
                except Exception as exc:
                    errors.append(f"{futures[future]} ({type(exc).__name__})")
            if errors:
                raise ObsStorageError("File transfers failed; no completion marker written: "
                                      + ", ".join(errors)) from None
        return sorted(records, key=lambda item: item["path"])

    @staticmethod
    def _validate_records(records):
        if not isinstance(records, list) or not records:
            raise ObsStorageError("Manifest must contain a nonempty file list")
        seen = set()
        for record in records:
            if not isinstance(record, dict):
                raise ObsStorageError("Invalid file record")
            relative = _relative(record.get("path"))
            if relative in seen:
                raise ObsStorageError("Duplicate path in manifest")
            seen.add(relative)
            if type(record.get("size")) is not int or record["size"] < 0:
                raise ObsStorageError("Invalid manifest file size")
            if not re.fullmatch(r"[a-f0-9]{64}", str(record.get("sha256", ""))):
                raise ObsStorageError("Invalid manifest SHA256")
        return records

    def _committed(self, records_root, kind):
        marker, _ = self._read_json(_join(records_root, "_SUCCESS"))
        manifest, raw = self._read_json(_join(records_root, "manifest.json"))
        digest = hashlib.sha256(raw).hexdigest()
        if marker.get("manifest_sha256") != digest:
            raise ObsStorageError("Manifest does not match its completion marker")
        if manifest.get("format_version") != FORMAT_VERSION or manifest.get("kind") != kind:
            raise ObsStorageError("Unsupported manifest format")
        self._validate_records(manifest.get("files"))
        if kind == "checkpoint":
            self._validate_records(manifest.get("run_files"))
            if any(not self._allowed_run_file(r["path"]) for r in manifest["run_files"]):
                raise ObsStorageError("Manifest includes an unexpected run asset path")
        return manifest, digest

    @staticmethod
    def _matches(files, records):
        return (set(files) == {r["path"] for r in records} and all(
            files[r["path"]].stat().st_size == r["size"]
            and _sha_file(files[r["path"]]) == r["sha256"] for r in records))

    @staticmethod
    def _allowed_run_file(relative):
        return (relative in RUN_FILES + OPTIONAL_RUN_FILES
                or any(relative.startswith(tree + "/") for tree in RUN_TREES))

    @staticmethod
    def _run_files(run_dir):
        run_dir = Path(run_dir).resolve()
        files = {}
        for name in RUN_FILES:
            path = _target(run_dir, name)
            if not path.is_file():
                raise ObsStorageError(f"Missing required run asset: {name}")
            files[name] = path
        for name in OPTIONAL_RUN_FILES:
            path = _target(run_dir, name)
            if path.is_file():
                files[name] = path
        for tree in RUN_TREES:
            directory = _target(run_dir, tree)
            if directory.is_dir():
                files.update({tree + "/" + rel: path for rel, path in _files(directory).items()})
        if not any(p.startswith("assets/norm_stats/") for p in files):
            raise ObsStorageError("Run assets must include normalization statistics")
        if not any(p.startswith("configs/robot_configs/") for p in files):
            raise ObsStorageError("Run assets must include robot configuration")
        return files

    def assert_new_run(self):
        """Refuse a new run if this prefix already contains committed training."""
        if self._exists(_join(self.root_uri, "latest.json")) or self._exists(_join(self.root_uri, "_SUCCESS")):
            raise ObsStorageError("OBS run already exists; enable resume or choose a different run name")
        checkpoint_root = _join(self.root_uri, "checkpoints")
        if self._exists(checkpoint_root):
            # Even a partial previous run must not be silently overwritten.
            raise ObsStorageError("OBS run has checkpoint objects; resume or choose a different run name")

    def _latest(self, checkpoint_relative, step, epoch, digest):
        latest_uri = _join(self.root_uri, "latest.json")
        if self._exists(latest_uri):
            previous, _ = self._read_json(latest_uri)
            if int(previous.get("global_step", -1)) > step:
                raise ObsStorageError("Refusing to move latest checkpoint backwards")
        value = {"format_version": FORMAT_VERSION, "checkpoint_path": checkpoint_relative,
                 "global_step": step, "epoch": epoch, "manifest_sha256": digest}
        self._write_json(latest_uri, value)
        return value

    def upload_checkpoint(self, local_checkpoint_dir, run_dir, global_step, epoch):
        """Commit one complete local checkpoint; does not delete local files."""
        if type(global_step) is not int or global_step < 0 or type(epoch) is not int or epoch < 0:
            raise ObsStorageError("Step and epoch must be nonnegative integers")
        checkpoint_relative = f"checkpoints/global_step_{global_step}"
        remote = _join(self.root_uri, checkpoint_relative)
        files = _files(Path(local_checkpoint_dir))
        if not files or not all(any(p.startswith(tree + "/") for p in files)
                                for tree in ("model", "optimizer", "extra_state")):
            raise ObsStorageError("A resumable checkpoint needs model, optimizer and extra_state files")
        if any(p.startswith("_run_assets/") or p in ("manifest.json", "_SUCCESS") for p in files):
            raise ObsStorageError("Reserved checkpoint metadata path in local source")
        assets = self._run_files(run_dir)
        if self._exists(_join(remote, "_SUCCESS")):
            manifest, digest = self._committed(remote, "checkpoint")
            if (manifest.get("global_step") != global_step or manifest.get("epoch") != epoch
                    or not self._matches(files, manifest["files"])
                    or not self._matches(assets, manifest["run_files"])):
                raise ObsStorageError("Committed checkpoint differs from local source; refusing overwrite")
        else:
            file_records = self._upload_files(files, remote)
            asset_records = self._upload_files(assets, _join(remote, "_run_assets"))
            # The run-root copy preserves the existing inference directory contract.
            self._upload_files(assets, self.root_uri)
            manifest = {"format_version": FORMAT_VERSION, "kind": "checkpoint",
                        "global_step": global_step, "epoch": epoch,
                        "files": file_records, "run_files": asset_records}
            digest = self._write_json(_join(remote, "manifest.json"), manifest)
            self._write_json(_join(remote, "_SUCCESS"), {"manifest_sha256": digest})
        latest = self._latest(checkpoint_relative, global_step, epoch, digest)
        return {"remote_checkpoint_uri": remote, "manifest": manifest, "latest": latest}

    def _download_records(self, records, remote, local_root, overwrite_existing=True):
        records = self._validate_records(records)
        local_root = Path(local_root).absolute()
        local_root.mkdir(parents=True, exist_ok=True)
        for record in records:
            target = _target(local_root, record["path"])
            if target.is_file() and target.stat().st_size == record["size"] and _sha_file(target) == record["sha256"]:
                continue
            if target.exists() and not overwrite_existing:
                raise ObsStorageError("Existing dataset differs from OBS; choose another empty OBS_CACHE_DIR")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".obs-download-", dir=target.parent) as directory:
                temporary = Path(directory) / "payload"
                def download():
                    self.file.copy(_join(remote, record["path"]), str(temporary))
                    if temporary.stat().st_size != record["size"] or _sha_file(temporary) != record["sha256"]:
                        raise ObsStorageError("Downloaded file SHA256 mismatch")
                self._retry("download and verify file", download)
                os.replace(temporary, target)

    def download_latest(self, local_run_dir):
        """Return a verified local checkpoint, or None only if latest is absent."""
        uri = _join(self.root_uri, "latest.json")
        if not self._exists(uri):
            if self._exists(_join(self.root_uri, "checkpoints")):
                raise ObsStorageError("OBS has checkpoint objects but no latest pointer; repair the pointer before resuming")
            return None
        latest, _ = self._read_json(uri)
        relative = latest.get("checkpoint_path")
        if not isinstance(relative, str) or not re.fullmatch(r"checkpoints/global_step_[0-9]+", relative):
            raise ObsStorageError("Invalid latest checkpoint path")
        remote = _join(self.root_uri, relative)
        manifest, digest = self._committed(remote, "checkpoint")
        if (digest != latest.get("manifest_sha256")
                or manifest.get("global_step") != latest.get("global_step")
                or manifest.get("epoch") != latest.get("epoch")
                or relative != f"checkpoints/global_step_{manifest.get('global_step')}"):
            raise ObsStorageError("Latest pointer and checkpoint manifest disagree")
        run_dir = Path(local_run_dir).absolute()
        checkpoint = _target(run_dir, relative)
        self._download_records(manifest["run_files"], _join(remote, "_run_assets"), run_dir)
        self._download_records(manifest["files"], remote, checkpoint)
        return str(checkpoint)

    @staticmethod
    def dataset_records_uri(destination_uri):
        destination_uri = _uri(destination_uri)
        parsed = urlsplit(destination_uri)
        path = parsed.path.strip("/")
        if not path:
            raise ObsStorageError("Dataset must use a named directory, not a bucket root")
        parent, _, name = path.rpartition("/")
        root = "obs://" + parsed.netloc + ("/" + parent if parent else "")
        return _join(root, "_records/" + name)

    @staticmethod
    def _dataset_files(local_dir):
        root = Path(local_dir).resolve()
        files = {}
        for tree in DATA_TREES:
            files.update({tree + "/" + rel: path for rel, path in _files(root / tree).items()})
        required = {"meta/info.json", "meta/stats.json", "meta/tasks.parquet"}
        if not required.issubset(files) or not any(p.startswith("meta/episodes/") for p in files):
            raise ObsStorageError("Dataset is missing LeRobot v3 metadata")
        if not all(any(p.startswith(tree + "/") for p in files) for tree in DATA_TREES):
            raise ObsStorageError("Dataset must contain data, metadata and videos")
        return files

    def upload_dataset(self, local_dir, destination_uri):
        """Upload only a LeRobot v3 data tree and verify full SHA256 by streaming."""
        destination_uri = _uri(destination_uri)
        records_root = self.dataset_records_uri(destination_uri)
        files = self._dataset_files(local_dir)
        if self._exists(_join(records_root, "_SUCCESS")):
            manifest, digest = self._committed(records_root, "dataset")
            if manifest.get("dataset_uri") != destination_uri or not self._matches(files, manifest["files"]):
                raise ObsStorageError("Committed dataset differs; refusing to overwrite it")
        else:
            records = self._upload_files(files, destination_uri, verification="sha256")
            manifest = {"format_version": FORMAT_VERSION, "kind": "dataset",
                        "dataset_uri": destination_uri, "files": records}
            digest = self._write_json(_join(records_root, "manifest.json"), manifest)
            self._write_json(_join(records_root, "_SUCCESS"), {"manifest_sha256": digest})
        return {"dataset_uri": destination_uri, "records_uri": records_root,
                "manifest_sha256": digest, "file_count": len(manifest["files"]),
                "total_bytes": sum(item["size"] for item in manifest["files"])}

    def ensure_dataset(self, uri, cache_dir):
        """Verify/reuse a local cache or fetch files from a committed dataset."""
        uri = _uri(uri)
        manifest, _ = self._committed(self.dataset_records_uri(uri), "dataset")
        if manifest.get("dataset_uri") != uri:
            raise ObsStorageError("Dataset manifest points to a different dataset")
        if any(item["path"].split("/", 1)[0] not in DATA_TREES for item in manifest["files"]):
            raise ObsStorageError("Dataset manifest includes a non-dataset path")
        # Validate every existing file before writing anything. In particular,
        # never repair the user's original dataset in place if it differs.
        existing = {}
        for tree in DATA_TREES:
            directory = _target(Path(cache_dir), tree)
            if directory.is_dir():
                existing.update({tree + "/" + rel: path for rel, path in _files(directory).items()})
            elif directory.exists():
                raise ObsStorageError("Dataset cache tree is not a directory")
        expected = {r["path"]: r for r in manifest["files"]}
        if not set(existing).issubset(expected) or any(
                path.stat().st_size != expected[name]["size"]
                or _sha_file(path) != expected[name]["sha256"] for name, path in existing.items()):
            raise ObsStorageError("Existing dataset differs from OBS; choose another empty OBS_CACHE_DIR")
        if set(existing) == set(expected):
            self._dataset_files(cache_dir)  # Also check the required LeRobot metadata.
            return str(Path(cache_dir).absolute())
        self._download_records(manifest["files"], uri, cache_dir, overwrite_existing=False)
        actual = self._dataset_files(cache_dir)
        if set(actual) != {r["path"] for r in manifest["files"]}:
            raise ObsStorageError("Local dataset contains files outside the committed inventory")
        return str(Path(cache_dir).absolute())
