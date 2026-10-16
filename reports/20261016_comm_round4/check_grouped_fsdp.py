"""Check that grouping adjacent twice-called FSDP2 blocks preserves gradients."""
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)

    def forward(self, x):
        return self.linear(x).tanh()


class Joint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block() for _ in range(4)])
        self.other = torch.nn.ModuleList([Block() for _ in range(4)])

    def forward(self, x):
        for a, b in zip(self.layers, self.other):
            x = a(x) + b(x)
            x = a(x) + b(x)
        return x.square().mean()


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    mesh = init_device_mesh("npu", (dist.get_world_size(),))
    torch.manual_seed(121)
    separate = Joint().to(f"npu:{rank}")
    grouped = Joint().to(f"npu:{rank}")
    grouped.load_state_dict(separate.state_dict())
    for block in (*separate.layers, *separate.other):
        fully_shard(block, mesh=mesh, reshard_after_forward=False)
    for i in range(0, len(grouped.layers), 2):
        fully_shard([grouped.layers[i], grouped.layers[i + 1]], mesh=mesh, reshard_after_forward=False)
    for block in grouped.other:
        fully_shard(block, mesh=mesh, reshard_after_forward=False)
    fully_shard(separate, mesh=mesh, reshard_after_forward=False)
    fully_shard(grouped, mesh=mesh, reshard_after_forward=False)
    opts = [torch.optim.SGD(model.parameters(), lr=.01) for model in (separate, grouped)]
    max_error = 0.0
    for step in range(3):
        torch.manual_seed(3000 + rank * 19 + step)
        x = torch.randn(8, 32, device=f"npu:{rank}")
        for model in (separate, grouped):
            model.zero_grad(set_to_none=True)
        losses = [model(x) for model in (separate, grouped)]
        for loss in losses:
            loss.backward()
        for a, b in zip(separate.parameters(), grouped.parameters()):
            aa, bb = a.grad.full_tensor(), b.grad.full_tensor()
            max_error = max(max_error, (aa - bb).abs().max().item())
            torch.testing.assert_close(aa, bb, rtol=1e-5, atol=1e-6)
        for opt in opts:
            opt.step()
        for a, b in zip(separate.parameters(), grouped.parameters()):
            aa, bb = a.full_tensor(), b.full_tensor()
            max_error = max(max_error, (aa - bb).abs().max().item())
            torch.testing.assert_close(aa, bb, rtol=1e-5, atol=1e-6)
    if rank == 0:
        result = {"passed": True, "ranks": dist.get_world_size(), "steps": 3,
                  "max_abs_error_rank0": max_error}
        Path(__file__).with_name("grouped_fsdp_toy_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
