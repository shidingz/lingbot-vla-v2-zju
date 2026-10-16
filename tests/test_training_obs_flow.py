"""CPU-only regressions for publication and the actual trainer's save control flow.

Import helpers by file path and extract trainer definitions with AST: importing
the trainer/package would activate the NPU compatibility shim in this workspace.
No model, device, process group, network call, or training entry point is started.
"""

import ast
import copy
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List
import unittest
from unittest.mock import Mock, patch


REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "tasks/vla/train_lingbotvla.py"
ARGUMENTS = REPO / "lingbotvla/utils/arguments.py"
CONFIG = REPO / "configs/vla/robotwin/robotwin_npu.yaml"
SPEC = importlib.util.spec_from_file_location(
    "_training_io_test_target", REPO / "lingbotvla/utils/training_io.py"
)
training_io = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training_io)


class BroadcastBus:
    def __init__(self):
        self.payload = None

    def endpoint(self, rank):
        def broadcast(payload, src):
            if rank == src:
                self.payload = copy.deepcopy(payload[0])
            else:
                payload[0] = copy.deepcopy(self.payload)

        return SimpleNamespace(broadcast_object_list=broadcast, barrier=lambda: None)


class PublicationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.logs = self.root / "small_logs"
        self.logs.mkdir()
        (self.logs / "loss.jsonl").write_text('{"step": 3, "loss": 0.3}\n')
        self.checkpoint = self.run / "checkpoints/global_step_3"
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / "weights.bin").write_bytes(b"checkpoint")

    def test_upload_failure_retains_checkpoint_and_does_not_record_success(self):
        storage = SimpleNamespace(upload_checkpoint=Mock(side_effect=OSError("connection lost")))
        with self.assertRaisesRegex(OSError, "connection lost"):
            training_io.publish_checkpoint(storage, self.checkpoint, self.run, self.logs, 3, 1)
        self.assertEqual((self.checkpoint / "weights.bin").read_bytes(), b"checkpoint")
        self.assertFalse((self.logs / "obs_checkpoints.jsonl").exists())
        self.assertEqual((self.run / "logs/loss.jsonl").read_text(), (self.logs / "loss.jsonl").read_text())

    def test_verified_publication_records_uri_before_removing_local_checkpoint(self):
        def upload(checkpoint, run, step, epoch):
            self.assertTrue((Path(checkpoint) / "weights.bin").is_file())
            self.assertEqual((step, epoch), (3, 1))
            return {"remote_checkpoint_uri": "obs://test/run/checkpoints/global_step_3"}

        result = training_io.publish_checkpoint(
            SimpleNamespace(upload_checkpoint=upload), self.checkpoint, self.run, self.logs, 3, 1
        )
        self.assertFalse(self.checkpoint.exists())
        record = json.loads((self.logs / "obs_checkpoints.jsonl").read_text())
        self.assertEqual(record["remote_checkpoint_uri"], result["remote_checkpoint_uri"])
        self.assertEqual((record["step"], record["epoch"]), (3, 1))
        self.assertTrue((self.logs / "loss.jsonl").exists())

    def test_keep_local_option(self):
        storage = SimpleNamespace(upload_checkpoint=lambda *args: {"remote_checkpoint_uri": "obs://test/step3"})
        training_io.publish_checkpoint(storage, self.checkpoint, self.run, self.logs, 3, 1, remove_local=False)
        self.assertTrue(self.checkpoint.is_dir())

    def test_cleanup_rejects_outside_nested_noncheckpoint_and_symlink_paths(self):
        outside = self.root / "outside/global_step_3"
        nested = self.run / "checkpoints/nested/global_step_3"
        noncheckpoint = self.run / "checkpoints/model_assets"
        for candidate in (outside, nested, noncheckpoint):
            candidate.mkdir(parents=True)
            with self.subTest(path=candidate), self.assertRaises(ValueError):
                training_io.cleanup_published_checkpoint(candidate, self.run)
            self.assertTrue(candidate.is_dir())
        link = self.run / "checkpoints/global_step_99"
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            training_io.cleanup_published_checkpoint(link, self.run)
        self.assertTrue(outside.is_dir())

    def test_cleanup_rejects_symlinked_checkpoints_parent(self):
        other_run = self.root / "linked_run"
        other_run.mkdir()
        outside = self.root / "external_checkpoints"
        outside.mkdir()
        (outside / "global_step_9").mkdir()
        (other_run / "checkpoints").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            training_io.cleanup_published_checkpoint(other_run / "checkpoints/global_step_9", other_run)
        self.assertTrue((outside / "global_step_9").is_dir())

    def test_rank_zero_failure_reaches_other_ranks_without_running_their_action(self):
        bus = BroadcastBus()
        root_action = Mock(side_effect=OSError("OBS unavailable"))
        peer_action = Mock()
        for rank, action in ((0, root_action), (1, peer_action)):
            with self.subTest(rank=rank), self.assertRaisesRegex(RuntimeError, "OBS unavailable"):
                training_io.rank_zero_action(bus.endpoint(rank), rank, action, "upload")
        root_action.assert_called_once()
        peer_action.assert_not_called()

    def test_rank_zero_result_reaches_other_ranks(self):
        bus = BroadcastBus()
        expected = {"resume_path": "/cache/run/checkpoints/global_step_3"}
        self.assertEqual(training_io.rank_zero_action(bus.endpoint(0), 0, lambda: expected, "restore"), expected)
        self.assertEqual(training_io.rank_zero_action(bus.endpoint(1), 1, lambda: self.fail("peer action ran"), "restore"), expected)


def load_actual_argument_definitions():
    """Use real dataclasses/parser without importing lingbotvla or the trainer."""
    tree = ast.parse(ARGUMENTS.read_text(), filename=str(ARGUMENTS))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level:
            continue
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "logger" for t in node.targets):
            continue
        nodes.append(node)
    module = ModuleType("_training_argument_test_target")
    module.__dict__["logger"] = SimpleNamespace(info_rank0=lambda *args, **kwargs: None)
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(ARGUMENTS), "exec"), module.__dict__)
    trainer = ast.parse(TRAINER.read_text(), filename=str(TRAINER))
    classes = [node for node in trainer.body if isinstance(node, ast.ClassDef) and node.name in {"MyTrainingArguments", "MyDataArguments", "Arguments"}]
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])), str(TRAINER), "exec"), module.__dict__)
    return module


class EffectiveArgumentsTest(unittest.TestCase):
    @staticmethod
    def dry_run_environment(**overrides):
        environment = {key: value for key, value in os.environ.items() if key in {"HOME", "PATH", "LANG"}}
        environment.update(DRY_RUN="1", CONFIG=str(CONFIG), REPO_ROOT=str(REPO))
        environment.update(overrides)
        return environment

    def test_launcher_command_parses_to_one_epoch_and_verified_obs_defaults(self):
        # Exercise the real dry-run command so shell overrides are tested too.
        environment = self.dry_run_environment()
        result = subprocess.run(
            ["bash", str(REPO / "train_robotwin_npu.sh")], cwd=REPO,
            env=environment, text=True, capture_output=True, check=True,
        )
        command = shlex.split(next(line for line in result.stdout.splitlines() if line.startswith("torchrun ")))
        entry = command.index("tasks/vla/train_lingbotvla.py")
        module = load_actual_argument_definitions()
        with patch.object(sys, "argv", command[entry:]), patch.dict(os.environ, {"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "8"}):
            args = module.parse_args(module.Arguments)
        self.assertEqual(args.train.num_train_epochs, 1)
        self.assertIsNone(args.train.max_steps)
        self.assertEqual(args.train.stop_after_steps, 0)
        self.assertEqual((args.train.save_epochs, args.train.save_steps), (1, 0))
        self.assertEqual((args.train.micro_batch_size, args.train.gradient_accumulation_steps, args.train.global_batch_size), (12, 1, 96))
        self.assertEqual((args.train.lr, args.train.lr_min, args.train.lr_warmup_ratio), (5e-5, 1e-5, 0.02))
        self.assertTrue(args.train.train_expert_only)
        self.assertTrue(args.train.enable_resume)
        self.assertTrue(args.train.obs_remove_local_checkpoint)
        self.assertEqual(args.train.align_params["depth_loss_weight"], 0.004)
        self.assertEqual(args.train.align_params["video"]["future_video_loss_weight"], 0.004)
        self.assertEqual(args.data.obs_dataset_uri, "obs://yw-2030-extern/Partner_Qin/robot_1014/datasets/RoboTwin_lerobot_v30/")
        self.assertEqual(args.train.obs_output_dir, "obs://yw-2030-extern/Partner_Qin/robot_1014/robotwin_1ep_optimized")
        self.assertEqual(args.train.logging_dir, str(REPO / "logs/robotwin_1ep_optimized"))
        args.train.compute_train_steps(dataset_length=548893)
        self.assertEqual(args.train.train_steps, 5717)
        self.assertEqual(args.train.train_steps * args.train.num_train_epochs, 5717)

    def test_run_name_moves_all_default_destinations_and_explicit_overrides_win(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for overrides in ({}, {"OUTPUT_DIR": str(root / "custom_output"), "LOG_DIR": str(root / "custom_logs"), "OBS_OUTPUT_DIR": "obs://test/custom_run"}):
                with self.subTest(overrides=overrides):
                    environment = self.dry_run_environment(RUN_NAME="trial.lr3e5-v2", REPO_ROOT=str(root), **overrides)
                    result = subprocess.run(["bash", str(REPO / "train_robotwin_npu.sh")], cwd=REPO, env=environment, text=True, capture_output=True, check=True)
                    command = shlex.split(next(line for line in result.stdout.splitlines() if line.startswith("torchrun ")))
                    for field, expected in (
                        ("output_dir", overrides.get("OUTPUT_DIR", "/cache/lingbot/robotwin_train/trial.lr3e5-v2")),
                        ("logging_dir", overrides.get("LOG_DIR", str(root / "logs/trial.lr3e5-v2"))),
                        ("obs_output_dir", overrides.get("OBS_OUTPUT_DIR", "obs://yw-2030-extern/Partner_Qin/robot_1014/trial.lr3e5-v2")),
                    ):
                        self.assertEqual(command[command.index(f"--train.{field}") + 1], expected)
                    self.assertEqual(list(root.iterdir()), [], "dry run must not create output/log directories")

    def test_run_name_rejects_path_traversal_and_shell_metacharacters(self):
        for name in ("../other", "/absolute", "has space", "a;touch file", ".hidden"):
            with self.subTest(name=name):
                result = subprocess.run(["bash", str(REPO / "train_robotwin_npu.sh")], cwd=REPO, env=self.dry_run_environment(RUN_NAME=name), text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("RUN_NAME", result.stderr)


class RestoreControlTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        tree = ast.parse(TRAINER.read_text(), filename=str(TRAINER))
        self.main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

    def test_restore_rewrites_old_machine_dataset_list_before_snapshot(self):
        run = self.root / "run"
        dataset = self.root / "current_cache"
        dataset.mkdir()
        args = SimpleNamespace(
            train=SimpleNamespace(output_dir=str(run), logging_dir=str(self.root / "logs"), load_checkpoint_path=None, enable_resume=True),
            data=SimpleNamespace(obs_dataset_uri="obs://test/RoboTwin_lerobot_v30/", obs_cache_dir=str(dataset), train_path="/old/list.txt", robot_config_root=str(REPO / "configs/robot_configs")),
        )

        def download(run_dir):
            (Path(run_dir) / "train_robotwin.txt").write_text("robotwin /old_machine/cache\n")
            return str(Path(run_dir) / "checkpoints/global_step_3")

        storage = SimpleNamespace(ensure_dataset=lambda *a: str(dataset), download_latest=download)
        prepare = copy.deepcopy(next(node for node in self.main.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_storage"))
        namespace = {"args": args, "obs_store": storage, "os": os}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[prepare], type_ignores=[])), str(TRAINER), "exec"), namespace)
        prepared = namespace["prepare_storage"]()
        args.data.train_path = prepared["train_path"]
        self.assertTrue(prepared["downloaded_resume"])
        self.assertEqual(Path(args.data.train_path).read_text(), f"robotwin {dataset}\n")
        previous_cwd = Path.cwd()
        try:
            os.chdir(REPO)
            training_io.snapshot_run_assets(args, prepared["dataset_root"])
        finally:
            os.chdir(previous_cwd)
        self.assertEqual((run / "train_robotwin.txt").read_text(), f"robotwin {dataset}\n")
        source_record = json.loads((run / "data_source.json").read_text())
        self.assertEqual(source_record["local_dataset_root"], str(dataset))
        self.assertFalse(source_record["raw_dataset_modified"])
        self.assertEqual((run / "assets/norm_stats/robotwin.json").read_bytes(), (REPO / "assets/norm_stats/robotwin.json").read_bytes())
        self.assertEqual(list(dataset.iterdir()), [])

    def test_actual_restore_branch_refuses_from_scratch_after_obs_load_failure(self):
        restore = copy.deepcopy(next(node for node in self.main.body if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "candidates"))
        logger = SimpleNamespace(info_rank0=Mock())
        namespace = {
            "candidates": ["/cache/run/checkpoints/global_step_3"],
            "Checkpointer": SimpleNamespace(load=Mock(side_effect=OSError("corrupt distributed shard"))),
            "model": object(), "optimizer": object(), "obs_store": object(),
            "args": SimpleNamespace(train=SimpleNamespace(load_checkpoint_path=None, allow_partial_checkpoint=False)),
            "logger": logger,
        }
        with self.assertRaisesRegex(RuntimeError, "refusing to restart from scratch"):
            exec(compile(ast.fix_missing_locations(ast.Module(body=[restore], type_ignores=[])), str(TRAINER), "exec"), namespace)
        logger.info_rank0.assert_not_called()


def calls(node, function_name):
    return any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == function_name for child in ast.walk(node))


def assigns(node, name):
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AugAssign) else []
    return any(isinstance(target, ast.Name) and target.id == name for target in targets)


def make_save_control_fixture(run_dir, logging_dir, *, save_steps=0, stop_after_steps=0, start_step=0):
    """Execute real scheduler/save/epoch branches, replacing only model work.

    A three-item fake dataloader makes epoch boundaries small and deterministic.
    The real save closure publishes through the real helper with fake storage.
    """
    tree = ast.parse(TRAINER.read_text(), filename=str(TRAINER))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    save_node = copy.deepcopy(next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "save_training_checkpoint"))
    factory = ast.parse("def make_saver():\n    last_saved_step = None\n    save_checkpoint_path = None\n    return save_training_checkpoint\n").body[0]
    factory.body.insert(-1, save_node)
    saved, exported, published = [], [], []
    train = SimpleNamespace(
        global_rank=0, local_rank=0, output_dir=str(run_dir), logging_dir=str(logging_dir),
        save_checkpoint_path=str(run_dir / "checkpoints"), obs_remove_local_checkpoint=True,
        train_steps=3, num_train_epochs=5, max_steps=None, stop_after_steps=stop_after_steps,
        save_steps=save_steps, save_epochs=1, profile_only=False,
        optimizer="muon", lr=5e-5, lr_min=1e-5, lr_start=0,
        lr_decay_style="cosine", lr_decay_ratio=1, lr_warmup_ratio=0.02,
        weight_decay=0.001, global_batch_size=96, train_expert_only=True,
        chunk_size=50, action_dim=55, max_state_dim=55, use_moe_expert_lr=True,
        align_params={"depth_loss_weight": 0.004},
    )

    def save(path, state, global_steps):
        saved.append((global_steps, state["extra_state"]["total_train_steps"]))
        checkpoint = Path(path) / f"global_step_{global_steps}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "weights.bin").write_text(str(global_steps))

    def upload(checkpoint, run, step, epoch):
        published.append((step, epoch))
        return {"remote_checkpoint_uri": f"obs://test/run/checkpoints/global_step_{step}"}

    class Loader:
        def __iter__(self):
            return iter([{}, {}, {}])
        def set_epoch(self, epoch):
            self.epoch = epoch
        def state_dict(self):
            return {}

    bus = BroadcastBus()
    namespace = {
        "args": SimpleNamespace(train=train, data=SimpleNamespace(drop_last=True, obs_dataset_uri="obs://test/data")),
        "List": List, "Dict": Dict, "Any": Any,
        "os": os, "rank_zero_action": training_io.rank_zero_action,
        "publish_checkpoint": training_io.publish_checkpoint,
        "resume_signature": training_io.resume_signature,
        "dist": bus.endpoint(0), "helper": SimpleNamespace(empty_cache=lambda: None, print_device_mem_info=lambda *args: None),
        "np": SimpleNamespace(random=SimpleNamespace(get_state=lambda: ("MT", SimpleNamespace(tolist=lambda: [1, 2]), 0, 0, 0.0))),
        "random": SimpleNamespace(getstate=lambda: ()),
        "torch": SimpleNamespace(get_rng_state=lambda: "cpu_rng", cuda=SimpleNamespace(get_rng_state=lambda: "device_rng")),
        "is_torch_npu_available": lambda: False,
        "model": object(), "optimizer": object(),
        "lr_scheduler": SimpleNamespace(state_dict=lambda: {"last_epoch": 0}),
        "train_dataloader": Loader(), "environ_meter": SimpleNamespace(state_dict=lambda: {}),
        # Match disabled profiling without importing device-backed trainer helpers.
        "training_profiler": SimpleNamespace(scope=lambda name: nullcontext()),
        "writer": SimpleNamespace(flush=lambda: None), "loss_file": SimpleNamespace(flush=lambda: None),
        "Checkpointer": SimpleNamespace(save=save), "obs_store": SimpleNamespace(upload_checkpoint=upload),
        "logger": SimpleNamespace(info_rank0=lambda *args: None),
        "save_hf_checkpoint_best_effort": lambda path, state, step, epoch, epoch_step: exported.append(step),
        "data_loader_tqdm": SimpleNamespace(close=lambda: None),
        "global_step": start_step, "start_epoch": start_step // 3, "start_step": start_step % 3,
        "reached_max_steps": False, "max_steps_driven": False,
    }
    horizon_index = next(i for i, node in enumerate(main.body) if assigns(node, "total_train_steps"))
    schedule_nodes = copy.deepcopy(main.body[horizon_index:horizon_index + 2])
    schedule_nodes.append(copy.deepcopy(next(node for node in main.body if assigns(node, "stop_step"))))
    exec(compile(ast.fix_missing_locations(ast.Module(body=schedule_nodes + [factory], type_ignores=[])), str(TRAINER), "exec"), namespace)
    namespace["save_training_checkpoint"] = namespace["make_saver"]()

    loop = copy.deepcopy(next(node for node in main.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "epoch"))
    inner = next(node for node in loop.body if isinstance(node, ast.For))
    inner.body = [
        node for node in inner.body
        if assigns(node, "current_epoch_step_for_eval") or assigns(node, "global_step")
        or isinstance(node, ast.Try)  # actual StopIteration/early-epoch error handling
        or (isinstance(node, ast.If) and (calls(node, "save_training_checkpoint") or any(assigns(child, "reached_max_steps") for child in ast.walk(node))))
    ]
    # The progress-bar creation is presentation only; preserve every save/stop branch.
    loop.body = [node for node in loop.body if not (isinstance(node, ast.If) and calls(node, "trange"))]
    code = compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), str(TRAINER), "exec")
    return namespace, code, saved, exported, published


class SaveBoundaryTest(unittest.TestCase):
    def run_fixture(self, **kwargs):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        namespace, code, saved, exported, published = make_save_control_fixture(root / "run", root / "logs", **kwargs)
        exec(code, namespace)
        return namespace, saved, exported, published

    def test_five_epochs_publish_exactly_once_at_each_boundary(self):
        namespace, saved, exported, published = self.run_fixture()
        self.assertEqual(saved, [(step, 15) for step in (3, 6, 9, 12, 15)])
        self.assertEqual(exported, [3, 6, 9, 12, 15])
        self.assertEqual(published, [(3, 1), (6, 2), (9, 3), (12, 4), (15, 5)])
        namespace["save_training_checkpoint"](15, 5, 3)
        self.assertEqual(len(saved), 5, "duplicate final save must not recreate a cleaned checkpoint")

    def test_step_epoch_and_final_save_collision_is_deduplicated(self):
        _, saved, exported, published = self.run_fixture(save_steps=3)
        self.assertEqual([step for step, _ in saved], [3, 6, 9, 12, 15])
        self.assertEqual(len(exported), len(published))
        self.assertEqual(len(published), 5)

    def test_pilot_stops_mid_epoch_without_shortening_schedule(self):
        namespace, saved, _, published = self.run_fixture(stop_after_steps=7)
        self.assertEqual(namespace["global_step"], 7)
        self.assertEqual(namespace["total_train_steps"], 15)
        self.assertEqual(saved, [(3, 15), (6, 15), (7, 15)])
        self.assertEqual(published[-1], (7, 3))
        resumed, saved_after_resume, _, _ = self.run_fixture(start_step=7)
        self.assertEqual(resumed["global_step"], 15)
        self.assertEqual(saved_after_resume, [(9, 15), (12, 15), (15, 15)])


if __name__ == "__main__":
    unittest.main()
