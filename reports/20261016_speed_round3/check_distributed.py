"""Two-rank DTensor Muon equivalence including state restoration."""
import copy
import ast
import importlib.util
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/cache/lingbot/profiling/speed_round3_20261016/baseline/source")


def load(root, name):
    spec = importlib.util.spec_from_file_location(name, root / "lingbotvla/optim/muon.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(rank)
dist.init_process_group("hccl")
mesh = init_device_mesh("npu", (dist.get_world_size(),))
torch.manual_seed(135)
shapes = [(8, 64, 48)] * 7 + [(8, 48, 64)] * 3 + [(64, 48)] * 4
values = [torch.randn(shape, device=f"npu:{rank}") * .02 for shape in shapes]
ps = [[torch.nn.Parameter(distribute_tensor(v.clone(), mesh, [Shard(0)])) for v in values] for _ in range(2)]
os.environ["LINGBOT_MUON_LOCAL_BATCH"] = "4"
modules = [load(BASE, "old_muon"), load(ROOT, "new_muon")]
opts = [m.DistributedMuon(p, lr=.0002, adjust_lr_fn="match_rms_adamw") for m, p in zip(modules, ps)]
max_error = 0.0
for step in range(4):
    if step == 2:
        opts[1].load_state_dict(copy.deepcopy(opts[0].state_dict()))
    for i, shape in enumerate(shapes):
        grad = torch.randn(shape, device=f"npu:{rank}")
        for params in ps:
            params[i].grad = None if i == step else distribute_tensor(grad.clone(), mesh, [Shard(0)])
    for opt in opts:
        opt.step()
    for p, q in zip(*ps):
        torch.testing.assert_close(p.to_local(), q.to_local(), rtol=0, atol=1e-7)
        max_error = max(max_error, (p.to_local() - q.to_local()).abs().max().item())
        if "momentum_buffer" in opts[0].state[p]:
            torch.testing.assert_close(opts[0].state[p]["momentum_buffer"].to_local(), opts[1].state[q]["momentum_buffer"].to_local(), rtol=0, atol=0)
dist.barrier()

# Exercise the actual joint decoder's prefetch helper with modules called twice
# per layer, matching QKV and post-attention stages, including backward.
from torch.distributed._composable.fsdp import fully_shard
src = ROOT / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
node = next(n for n in ast.walk(ast.parse(src.read_text())) if isinstance(n, ast.FunctionDef) and n.name == "prefetch_after")


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.proj(x).tanh()


class Joint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
        for branch in self.branches:
            branch.layers = torch.nn.ModuleList([Block() for _ in range(3)])
        scope = {"models": self.branches, "active_models": [0, 1], "num_layers": 3}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(src), "exec"), scope)
        self.prefetch_after = scope["prefetch_after"]

    def forward(self, x, prefetch):
        for layer in range(3):
            for output_stage in (False, True):
                parts = []
                for i in range(2):
                    if prefetch:
                        self.prefetch_after(i, layer, output_stage)
                    parts.append(self.branches[i].layers[layer](x))
                x = parts[0] + parts[1]
        return x


torch.manual_seed(24)
models = [Joint().to(f"npu:{rank}"), Joint().to(f"npu:{rank}")]
models[1].load_state_dict(models[0].state_dict())
for model in models:
    for branch in model.branches:
        for layer in branch.layers:
            fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
for step in range(3):
    data = torch.randn(8, 64, device=f"npu:{rank}")
    outputs = []
    for enabled, model in zip((False, True), models):
        model.zero_grad(set_to_none=True)
        output = model(data, enabled)
        output.square().mean().backward()
        outputs.append(output.detach())
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for p, q in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(p.grad.to_local(), q.grad.to_local(), rtol=0, atol=0)
dist.barrier()
if rank == 0:
    result = {"passed": True, "ranks": dist.get_world_size(), "steps": 4, "max_parameter_error_rank0": max_error,
              "covers": ["Shard(0) 3D experts", "unchanged 2D all-gather", "missing gradients", "old checkpoint load", "FSDP2 two-stage prefetch forward/backward"]}
    (ROOT / "reports/20261016_speed_round3/distributed_test.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
dist.destroy_process_group()
