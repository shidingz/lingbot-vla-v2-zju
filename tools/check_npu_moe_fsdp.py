"""Run with torchrun to compare FSDP2/checkpoint gradients to a global reference."""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from lingbotvla.ops.fused_moe import grouped_moe_torch_forward
from lingbotvla.ops.npu_moe import grouped_moe_npu_forward


def local(t):
    return t.to_local() if isinstance(t, DTensor) else t


class Block(torch.nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native
        self.gate = torch.nn.Parameter(torch.randn(32, 64) * .02)
        self.gate_proj = torch.nn.Parameter(torch.randn(32, 32, 64) * .02)
        self.up_proj = torch.nn.Parameter(torch.randn(32, 32, 64) * .02)
        self.down_proj = torch.nn.Parameter(torch.randn(32, 64, 32) * .02)

    def forward(self, x):
        scores = torch.nn.functional.linear(x.float(), local(self.gate.float())).sigmoid()
        selected = scores.topk(4, dim=-1).indices
        route = scores.gather(1, selected).to(torch.bfloat16) * 4
        weights = [local(w.to(torch.bfloat16)) for w in [self.gate_proj, self.up_proj, self.down_proj]]
        fn = grouped_moe_npu_forward if self.native else grouped_moe_torch_forward
        return fn(32, route, selected, x.to(torch.bfloat16), *weights)


class Model(torch.nn.Module):
    def __init__(self, native):
        super().__init__()
        self.block = Block(native)

    def forward(self, x):
        return checkpoint(self.block, x, use_reentrant=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import os
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    torch.manual_seed(20261016)
    reference, native = Model(False).npu(), Model(True).npu()
    native.load_state_dict(reference.state_dict(), strict=True)
    mesh = init_device_mesh("npu", (dist.get_world_size(),))
    kwargs = dict(mesh=mesh, reshard_after_forward=False,
                  mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32))
    fully_shard(native.block, **kwargs)
    fully_shard(native, **kwargs)
    torch.manual_seed(100 + rank)
    x = torch.randn(31, 64, device="npu")
    target = torch.randn(31, 64, device="npu")
    ref_loss = (reference(x).float() - target).square().mean()
    ref_loss.backward()
    actual_loss = (native(x).float() - target).square().mean()
    actual_loss.backward()
    results = {}
    for (name, expected), (other_name, actual) in zip(reference.named_parameters(), native.named_parameters()):
        assert name == other_name and actual.grad is not None
        expected_grad = expected.grad
        dist.all_reduce(expected_grad)
        expected_grad /= dist.get_world_size()
        actual_grad = actual.grad.full_tensor() if isinstance(actual.grad, DTensor) else actual.grad
        relative = ((actual_grad - expected_grad).norm() / expected_grad.norm().clamp_min(1e-12)).item()
        norm = actual_grad.norm().item()
        assert bool(actual_grad.isfinite().all()) and norm > 0
        assert relative < .003, (name, relative)
        results[name] = {"relative_l2_error": relative, "gradient_norm": norm}
    # Both expert weights and the upstream FP32 router must receive gradients.
    optimizer = torch.optim.AdamW(native.parameters(), lr=1e-3)
    optimizer.step()
    assert len(optimizer.state) == len(list(native.parameters()))
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = {"world_size": dist.get_world_size(), "non_reentrant_checkpoint": True,
                  "strict_load_of_reference_weights": True, "parameters": results,
                  "optimizer_state_entries": len(optimizer.state)}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("FSDP2_MOE_GRADIENTS_OK", json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
