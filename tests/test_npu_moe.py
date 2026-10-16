"""Hardware numerical checks for the Ascend training backend."""
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from lingbotvla.ops import fused_moe
from lingbotvla.ops.npu_moe import grouped_moe_npu_forward


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is None or not torch.npu.is_available(),
    reason="requires an Ascend NPU",
)


def inputs(experts=8, tokens=97, dim=64, intermediate=32, topk=4, sparse=False):
    torch.npu.set_device(0)
    torch.manual_seed(20261016)
    x = torch.randn(tokens, dim, device="npu", dtype=torch.float32)
    route = torch.rand(tokens, topk, device="npu") * 0.2 + 0.2
    weights = [torch.randn(shape, device="npu") * 0.02 for shape in (
        (experts, intermediate, dim), (experts, intermediate, dim), (experts, dim, intermediate)
    )]
    if sparse:
        selected = torch.tensor([1, 3, experts - 2, experts - 1][:topk], device="npu").expand(tokens, topk).contiguous()
    else:
        selected = torch.rand(tokens, experts, device="npu").topk(topk, dim=-1).indices
    grad = torch.randn(tokens, dim, device="npu", dtype=torch.bfloat16)
    return experts, selected, [x, route, *weights], grad


def evaluate(fn, case, checkpointed=False, hidden_grad=True, weight_grad=True):
    experts, selected, values, grad = case
    leaves = [v.detach().clone().requires_grad_(hidden_grad if i == 0 else weight_grad if i > 1 else True)
              for i, v in enumerate(values)]

    def forward(*args):
        x, route, w1, w2, w3 = [v.to(torch.bfloat16) for v in args]
        return fn(experts, route, selected, x, w1, w2, w3)

    y = checkpoint(forward, *leaves, use_reentrant=False) if checkpointed else forward(*leaves)
    y.backward(grad)
    return [y.detach(), *(v.grad for v in leaves)]


def compare(actual, reference):
    for a, b in zip(actual, reference):
        if b is None:
            assert a is None
            continue
        assert a is not None and bool(a.isfinite().all())
        a, b = a.float(), b.float()
        relative = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
        assert relative < 2e-3, f"relative L2 error {relative}"


@pytest.mark.parametrize("sparse,topk", [(False, 4), (True, 4), (True, 1)])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_forward_and_all_gradients(sparse, topk, checkpointed):
    case = inputs(sparse=sparse, topk=topk)
    compare(evaluate(grouped_moe_npu_forward, case, checkpointed),
            evaluate(fused_moe.grouped_moe_torch_forward, case, checkpointed))


@pytest.mark.parametrize("hidden_grad,weight_grad", [(False, True), (True, False)])
def test_frozen_operands(hidden_grad, weight_grad):
    case = inputs(sparse=True)
    compare(evaluate(grouped_moe_npu_forward, case, hidden_grad=hidden_grad, weight_grad=weight_grad),
            evaluate(fused_moe.grouped_moe_torch_forward, case, hidden_grad=hidden_grad, weight_grad=weight_grad))


def test_training_size_with_checkpoint():
    case = inputs(experts=32, tokens=612, dim=768, intermediate=512)
    compare(evaluate(grouped_moe_npu_forward, case, checkpointed=True),
            evaluate(fused_moe.grouped_moe_torch_forward, case, checkpointed=True))


def test_backend_selection(monkeypatch):
    monkeypatch.setattr(fused_moe, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=False))
    case = inputs()

    def run(implementation):
        monkeypatch.setenv("LINGBOT_NPU_MOE_IMPL", implementation)
        def forward(experts, route, selected, x, w1, w2, w3):
            return fused_moe.fused_moe_forward(None, experts, route, selected, x, w1, w2, w3)
        return evaluate(forward, case, checkpointed=True)

    compare(run("gmm"), run("torch"))
    with pytest.raises(ValueError, match="LINGBOT_NPU_MOE_IMPL"):
        run("invalid")
