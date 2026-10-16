"""Compare the two HCCL all-gather APIs at representative Muon batch sizes."""
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
import torch_npu
from torch_npu.contrib import transfer_to_npu


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    world = dist.get_world_size()
    assert world == 8
    results = []
    for n, m, k in ((8, 64, 768), (4, 96, 4096), (16, 96, 4096)):
        x = torch.randn(n, m, k, device=f"npu:{rank}")

        def list_gather():
            parts = [torch.empty_like(x) for _ in range(world)]
            dist.all_gather(parts, x)
            return torch.cat(parts, dim=1)

        def tensor_gather():
            full = torch.empty((world * n, m, k), dtype=x.dtype, device=x.device)
            dist.all_gather_into_tensor(full, x)
            return full.view(world, n, m, k).permute(1, 0, 2, 3).reshape(n, world * m, k)

        expected = list_gather()
        actual = tensor_gather()
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for label, fn in (("list", list_gather), ("tensor", tensor_gather)):
            for _ in range(3):
                result = fn()
            torch.npu.synchronize()
            elapsed = []
            for _ in range(12):
                t0 = time.perf_counter()
                result = fn()
                torch.npu.synchronize()
                elapsed.append(time.perf_counter() - t0)
            del result
            median = statistics.median(elapsed)
            max_median = torch.tensor(median, device=x.device)
            dist.all_reduce(max_median, op=dist.ReduceOp.MAX)
            if rank == 0:
                results.append({"shape": [n, m, k], "api": label,
                                "median_wall_s_rank0": median,
                                "max_median_wall_s_across_ranks": max_median.item()})
    if rank == 0:
        output = {"ranks": world, "dtype": "float32", "correctness": "bitwise equal",
                  "warmups": 3, "measured_iterations": 12, "results": results}
        Path(__file__).with_name("muon_allgather_benchmark.json").write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
