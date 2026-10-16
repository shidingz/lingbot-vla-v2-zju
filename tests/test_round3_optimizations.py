"""Compare scheduling changes with frozen pre-change code and independent math.

CPU by default; ROUND3_TEST_DEVICE=npu:0 enables the hardware checks. The
baseline snapshot is the original round-2 default used in the speed comparison.
"""
import ast
import copy
from contextlib import nullcontext
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(os.environ.get("ROUND3_BASELINE_SOURCE", "/cache/lingbot/profiling/speed_round3_20261016/baseline/source"))
DEVICE = os.environ.get("ROUND3_TEST_DEVICE", "cpu")
torch.set_num_threads(1)


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


muon = load_file(ROOT / "lingbotvla/optim/muon.py", "candidate_muon")
old_muon = load_file(BASELINE / "lingbotvla/optim/muon.py", "baseline_muon") if BASELINE.exists() else muon
norm = load_file(ROOT / "lingbotvla/ops/rms_norm.py", "candidate_norm")


def function(path, name, scope):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("adjust", [None, "original", "match_rms_adamw"])
def test_muon_trajectory_and_checkpoint(monkeypatch, nesterov, adjust):
    torch.manual_seed(92)
    shapes = [(3, 32, 48)] * 7 + [(2, 48, 32)] * 3 + [(32, 48)]
    ref = [torch.nn.Parameter(torch.randn(s, device=DEVICE) * .02) for s in shapes]
    got = [torch.nn.Parameter(v.detach().clone()) for v in ref]

    def make(cls, params, batch_size):
        monkeypatch.setenv("LINGBOT_MUON_LOCAL_BATCH", str(batch_size))
        return cls([{"params": params[:7]}, {"params": params[7:], "lr": .0006, "weight_decay": .03}],
                   lr=.0002, weight_decay=.001, nesterov=nesterov, adjust_lr_fn=adjust)

    a, b = make(old_muon.DistributedMuon, ref, 0), make(muon.DistributedMuon, got, 4)
    for step in range(6):
        if step == 2:
            b = make(muon.DistributedMuon, got, 4)
            b.load_state_dict(copy.deepcopy(a.state_dict()))
            assert b.local_batch_size == 4
        if step == 4:
            # The new optimizer's state must also remain readable by the old one.
            a = make(old_muon.DistributedMuon, ref, 0)
            a.load_state_dict(copy.deepcopy(b.state_dict()))
        for i, (p, q) in enumerate(zip(ref, got)):
            grad = torch.randn_like(p)
            p.grad = None if i == step else grad
            q.grad = None if i == step else grad.clone()
        a.step()
        b.step()
        for p, q in zip(ref, got):
            torch.testing.assert_close(p, q, rtol=0, atol=1e-7)
            assert set(a.state[p]) == set(b.state[q])
            if "momentum_buffer" in a.state[p]:
                torch.testing.assert_close(a.state[p]["momentum_buffer"], b.state[q]["momentum_buffer"], rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(4, 768, 512), (4, 512, 768)])
def test_ns_batch_preserves_independent_matrix_norms(shape):
    if DEVICE == "cpu":
        shape = (4, 48, 32) if shape[-2] > shape[-1] else (4, 32, 48)
    torch.manual_seed(123)
    values = [torch.randn(shape, device=DEVICE) for _ in range(3)]
    expected = torch.cat([muon.batched_newton_schulz(x) for x in values])
    actual = muon.batched_newton_schulz(torch.cat(values))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(12, 51, 768), (2, 37, 2048)])
def test_rms_norm_output_and_gradients(monkeypatch, dtype, shape):
    torch.manual_seed(42)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    w = torch.randn(shape[-1], device=DEVICE)
    grad = torch.randn(shape, device=DEVICE)
    outputs = []
    for enabled in ("0", "1"):
        monkeypatch.setenv("LINGBOT_NPU_FUSED_RMSNORM", enabled)
        xx, ww = x.clone().requires_grad_(), w.clone().requires_grad_()
        y = norm.qwen_rms_norm(xx, ww, 1e-6)
        (y * grad).sum().backward()
        outputs.append((y.detach(), xx.grad, ww.grad))
    for ref, got in zip(*outputs):
        torch.testing.assert_close(got, ref, rtol=2e-5, atol=3e-5)


@pytest.mark.skipif(DEVICE == "cpu", reason="NPU training autocast contract")
def test_rms_norm_keeps_fp32_under_autocast(monkeypatch):
    x = torch.randn(12, 51, 768, device=DEVICE)
    w = torch.randn(768, device=DEVICE)
    outputs = []
    for flag in ("0", "1"):
        monkeypatch.setenv("LINGBOT_NPU_FUSED_RMSNORM", flag)
        xx, ww = x.clone().requires_grad_(), w.clone().requires_grad_()
        with torch.autocast("npu", dtype=torch.bfloat16):
            y = norm.qwen_rms_norm(xx, ww, 1e-6)
        assert y.dtype == torch.float32
        y.square().mean().backward()
        outputs.append((y.detach(), xx.grad, ww.grad))
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=3e-5)


@pytest.mark.parametrize("class_name", ["AdaRMSNorm", "FixAdaRMSNorm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_adanorm_state_and_gradients(monkeypatch, class_name, dtype):
    if not BASELINE.exists():
        pytest.skip("Requires original snapshot")
    modules = []
    for root in (BASELINE, ROOT):
        src = root / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py"
        node = next(n for n in ast.walk(ast.parse(src.read_text())) if isinstance(n, ast.ClassDef) and n.name == class_name)
        scope = {"torch": torch, "nn": torch.nn, "qwen_rms_norm": norm.qwen_rms_norm}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(src), "exec"), scope)
        modules.append(scope[class_name](768, 768).to(DEVICE))
    torch.manual_seed(455)
    with torch.no_grad():
        # Exercise trained FiLM weights, not only the zero-initialized branch.
        for name, parameter in modules[0].named_parameters():
            if name != "weight":
                parameter.normal_(std=.01)
    modules[1].load_state_dict(modules[0].state_dict(), strict=True)
    assert list(modules[0].state_dict()) == list(modules[1].state_dict())
    torch.manual_seed(456)
    x, c = torch.randn(3, 51, 768, device=DEVICE, dtype=dtype), torch.randn(3, 768, device=DEVICE)
    results = []
    monkeypatch.setenv("LINGBOT_NPU_FUSED_RMSNORM", "1")
    for module in modules:
        xx, cc = x.clone().requires_grad_(), c.clone().requires_grad_()
        with torch.autocast("npu", dtype=torch.bfloat16) if DEVICE != "cpu" else nullcontext():
            y = module(xx, cc)
        y.float().square().mean().backward()
        results.append((y.detach(), xx.grad, cc.grad, *(p.grad for p in module.parameters())))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=3e-5)


def test_prefix_position_ids(monkeypatch):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    src = ROOT / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    build = function(src, "build_prefix_position_ids", {"os": os})
    cfg = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2), image_token_id=3, video_token_id=4, vision_start_token_id=5)
    hf = SimpleNamespace(config=cfg)
    hf.get_rope_index = lambda **kw: Qwen3VLModel.get_rope_index(hf, **kw)
    wrapper = SimpleNamespace(qwenvl=SimpleNamespace(model=hf))
    ids = torch.tensor([[5, 3, 3, 3, 3, 6, 7, 8, 0], [5, 3, 3, 3, 3, 6, 7, 0, 0]], device=DEVICE)
    mask = ids.ne(0).long()
    for grid in (torch.tensor([[1, 4, 4], [1, 4, 4]], device=DEVICE), None):
        outputs = []
        for value in ("0", "1"):
            monkeypatch.setenv("LINGBOT_HOST_METADATA", value)
            outputs.append(build(wrapper, ids, mask, grid))
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        assert outputs[1].device == ids.device


def test_teacher_shape_constants():
    src = ROOT / "lingbotvla/models/vla/vision_models/dino_video/lumos_dinov3/models/navit_video_vision_transformer.py"
    cached = function(src, "_cached_packed_cu_seqlens", {"torch": torch, "lru_cache": lru_cache})
    for lengths in ((512, 512, 512), (257, 514), (512, 512, 512)):
        got = cached(lengths, torch.device(DEVICE))
        ref = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=DEVICE)
        torch.cumsum(torch.tensor(lengths, dtype=torch.int32, device=DEVICE), 0, out=ref[1:])
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
        assert got.data_ptr() == cached(lengths, torch.device(DEVICE)).data_ptr()
    rgb = function(ROOT / "lingbotvla/models/vla/vision_models/module_utils.py", "_video_rgb_constants", {"torch": torch, "lru_cache": lru_cache})
    mean, std = rgb(torch.device(DEVICE), torch.float32)
    torch.testing.assert_close(mean.flatten(), torch.tensor([.485, .456, .406], device=DEVICE), rtol=0, atol=0)
    torch.testing.assert_close(std.flatten(), torch.tensor([.229, .224, .225], device=DEVICE), rtol=0, atol=0)


def test_vision_metadata_and_pos_embed_gradient(monkeypatch):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from typing import Optional
    if not BASELINE.exists():
        pytest.skip("Requires original snapshot")
    config = Qwen3VLVisionConfig(hidden_size=64, out_hidden_size=64, intermediate_size=128,
                               num_heads=4, depth=1, deepstack_visual_indexes=[], num_position_embeddings=64)
    model = Qwen3VLVisionModel(config).to(DEVICE)
    grid = torch.tensor([[1, 4, 4], [2, 8, 4]], device=DEVICE)
    outputs = []
    for root, flag in ((BASELINE, "0"), (ROOT, "1")):
        monkeypatch.setenv("LINGBOT_HOST_METADATA", flag)
        src = root / "lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py"
        scope = {"os": os, "torch": torch, "F": torch.nn.functional, "Optional": Optional}
        prepare = function(src, "preprcess_grid_thw", scope)
        metadata = prepare(model, grid)
        pos = model.fast_pos_embed_interpolate(grid.cpu() if flag == "1" else grid)
        model.zero_grad(set_to_none=True)
        pos.float().square().mean().backward()
        outputs.append((metadata, pos.detach(), model.pos_embed.weight.grad.detach().clone()))
    for index in (1, 2):
        torch.testing.assert_close(outputs[0][index], outputs[1][index], rtol=0, atol=0)
    a, b = outputs[0][0], outputs[1][0]
    for i in range(2):
        torch.testing.assert_close(a[1][i], b[1][i], rtol=0, atol=0)
    torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)
    assert a[3:] == b[3:]


def test_visual_grid_cache_invalidates_on_shape_values():
    src = ROOT / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    scope = {"torch": torch}
    get_features = function(src, "get_image_features", scope)

    class Visual:
        spatial_merge_size = 2
        prepares = 0

        def preprcess_grid_thw(self, grid_thw):
            self.prepares += 1
            sizes = (grid_thw.prod(-1) // 4).tolist()
            return None, object(), object(), sizes, max(sizes)

        def __call__(self, pixels, **kw):
            size = int(kw["grid_thw"].prod(-1).sum()) // 4
            return torch.zeros(size, 8), []

    visual = Visual()
    obj = SimpleNamespace(config=SimpleNamespace(precompute_grid_thw=True), qwenvl=SimpleNamespace(visual=visual),
                          pos_embeds=None, position_embeddings=None, cu_seqlens=None, visual_split_sizes=None, visual_max_seqlen=None)
    for grid, expected_tokens, preparations in (([[1, 4, 4]], 4, 1), ([[1, 4, 4]], 4, 1),
                                                ([[1, 6, 4]], 6, 2), ([[1, 4, 4], [1, 4, 4]], 4, 3)):
        output, _ = get_features(obj, None, torch.tensor(grid))
        assert output.shape == (len(grid), expected_tokens, 8)
        assert visual.prepares == preparations
