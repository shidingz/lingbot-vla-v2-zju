"""Compare 8-way FSDP2 with 2x4 HSDP for the current Muon optimizer."""
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

from lingbotvla.optim.muon import DistributedMuon


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(48, 64, bias=False)
        self.experts = torch.nn.Parameter(torch.randn(8, 64, 48) * 0.02)

    def forward(self, x):
        dense = self.linear(x)
        expert = torch.einsum("bi,eoi->beo", x, self.experts).mean(dim=1)
        return (dense + expert).square().mean()


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    assert dist.get_world_size() == 8
    mesh_8 = init_device_mesh("npu", (8,), mesh_dim_names=("shard",))
    mesh_2x4 = init_device_mesh("npu", (2, 4), mesh_dim_names=("replicate", "shard"))
    torch.manual_seed(216)
    fsdp = Tiny().to(f"npu:{rank}")
    hsdp = Tiny().to(f"npu:{rank}")
    hsdp.load_state_dict(fsdp.state_dict())
    fully_shard(fsdp, mesh=mesh_8, reshard_after_forward=False)
    fully_shard(hsdp, mesh=mesh_2x4, reshard_after_forward=False)
    os.environ["LINGBOT_MUON_LOCAL_BATCH"] = "8"
    opts = [DistributedMuon(model.parameters(), lr=2e-4, adjust_lr_fn="match_rms_adamw")
            for model in (fsdp, hsdp)]
    errors = {"loss": 0.0, "gradient": 0.0, "parameter": 0.0, "norm": 0.0}
    for step in range(3):
        torch.manual_seed(4000 + rank * 13 + step)
        x = torch.randn(12, 48, device=f"npu:{rank}")
        for model in (fsdp, hsdp):
            model.zero_grad(set_to_none=True)
        losses = [model(x) for model in (fsdp, hsdp)]
        errors["loss"] = max(errors["loss"], abs((losses[0] - losses[1]).item()))
        for loss in losses:
            loss.backward()
        for a, b in zip(fsdp.parameters(), hsdp.parameters()):
            ga, gb = a.grad.full_tensor(), b.grad.full_tensor()
            errors["gradient"] = max(errors["gradient"], (ga - gb).abs().max().item())
            torch.testing.assert_close(ga, gb, atol=1e-6, rtol=1e-5)
        norms = [torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
                 for model in (fsdp, hsdp)]
        norm_values = [n.full_tensor().item() if hasattr(n, "full_tensor") else n.item() for n in norms]
        errors["norm"] = max(errors["norm"], abs(norm_values[0] - norm_values[1]))
        for opt in opts:
            opt.step()
        for a, b in zip(fsdp.parameters(), hsdp.parameters()):
            pa, pb = a.full_tensor(), b.full_tensor()
            errors["parameter"] = max(errors["parameter"], (pa - pb).abs().max().item())
            torch.testing.assert_close(pa, pb, atol=1e-6, rtol=1e-5)
    dist.barrier()
    if rank == 0:
        result = {"passed": True, "ranks": 8, "steps": 3, "max_abs_error_rank0": errors}
        out = Path(__file__).with_name("hsdp_toy_result.json")
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
