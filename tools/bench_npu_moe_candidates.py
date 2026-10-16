#!/usr/bin/env python3
"""Isolated MoE experiments; does not change the training implementation."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from lingbotvla.ops.fused_moe import grouped_moe_torch_forward
from lingbotvla.ops.npu_moe import grouped_moe_npu_forward as grouped_native


def unbind_loop(num_experts, routing_weights, selected_experts, hidden_states, w1, w2, w3, *, concatenate):
    tokens, topk = selected_experts.shape
    dim = hidden_states.shape[-1]
    indices = selected_experts.reshape(-1)
    order = indices.argsort(stable=True)
    counts = torch.bincount(indices, minlength=num_experts).tolist()
    x = hidden_states[order // topk]
    a, b, c = w1.unbind(0), w2.unbind(0), w3.unbind(0)
    pieces = []
    y = None if concatenate else torch.empty_like(x)
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        end = start + count
        xe = x[start:end]
        value = (F.silu(xe @ a[expert].t()) * (xe @ b[expert].t())) @ c[expert].t()
        if concatenate:
            pieces.append(value)
        else:
            y[start:end] = value
        start = end
    if concatenate:
        y = torch.cat(pieces, dim=0)
    unsorted = torch.empty_like(y)
    unsorted[order] = y
    return (unsorted.view(tokens, topk, dim).float() * routing_weights.unsqueeze(-1).float()).sum(1).to(hidden_states.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--native-gmm", action="store_true")
    parser.add_argument("--routing", choices=["random", "sparse"], default="random")
    parser.add_argument("--dynamic-routing", action="store_true")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(20261016)
    experts, tokens, dim, hidden, topk = 32, 12 * 51, 768, 512, 4
    device = f"npu:{args.device}"
    x = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16).requires_grad_()
    route = (torch.rand(tokens, topk, device=device) * 0.2 + 0.2).to(torch.bfloat16).requires_grad_()
    def make_routing():
        if args.routing == "sparse":
            # Empty groups at both ends and in the middle of the expert stack.
            return torch.tensor([2, 9, 17, 29], device=device).expand(tokens, topk).contiguous()
        return torch.rand(tokens, experts, device=device).topk(topk, dim=-1).indices
    selected = make_routing()
    routing_schedule = [make_routing() for _ in range(args.iterations + 2)] if args.dynamic_routing else [selected] * (args.iterations + 2)
    weights = [torch.randn(shape, device=device).mul_(0.02).requires_grad_()
               for shape in [(experts, hidden, dim), (experts, hidden, dim), (experts, dim, hidden)]]
    gradient = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16)
    leaves = [x, route, *weights]
    candidates = {
        "baseline": grouped_moe_torch_forward,
        "unbind_weights": lambda *a: unbind_loop(*a, concatenate=False),
        "unbind_weights_and_cat": lambda *a: unbind_loop(*a, concatenate=True),
    }
    if args.native_gmm:
        candidates["native_gmm_explicit_backward"] = grouped_native
    results, reference_y, reference_grads = {}, None, None
    for name, fn in candidates.items():
        timings = []
        for iteration in range(args.iterations + 2):
            selected = routing_schedule[iteration]
            for leaf in leaves:
                leaf.grad = None
            torch.npu.synchronize()
            started = time.perf_counter()
            y = fn(experts, route, selected, x, *(w.to(torch.bfloat16) for w in weights))
            torch.npu.synchronize()
            forward_done = time.perf_counter()
            y.backward(gradient)
            torch.npu.synchronize()
            ended = time.perf_counter()
            if iteration >= 2:
                timings.append({"forward_ms": (forward_done - started) * 1000,
                                "backward_ms": (ended - forward_done) * 1000,
                                "total_ms": (ended - started) * 1000})
        if reference_y is None:
            reference_y = y.detach().clone()
            reference_grads = [leaf.grad.detach().clone() for leaf in leaves]
        errors = []
        for label, actual, expected in zip(["output", "hidden_grad", "route_grad", "w1_grad", "w2_grad", "w3_grad"],
                                           [y.detach(), *(leaf.grad for leaf in leaves)], [reference_y, *reference_grads]):
            a, b = actual.float(), expected.float()
            errors.append({"tensor": label, "relative_l2": ((a - b).norm() / b.norm().clamp_min(1e-12)).item(),
                           "max_abs": (a - b).abs().max().item(), "finite": bool(a.isfinite().all())})
        results[name] = {"median": {k: statistics.median(t[k] for t in timings) for k in timings[0]},
                         "iterations": timings, "errors_vs_baseline": errors}
        print(name, json.dumps(results[name]["median"]), flush=True)
    result = {"shape": {"experts": experts, "tokens": tokens, "dim": dim, "hidden": hidden, "topk": topk},
              "routing": args.routing, "dynamic_routing": args.dynamic_routing,
              "master_weights": "float32; cast to bfloat16 in forward", "results": results,
              "limits": "One MoE layer, synthetic routing, no FSDP/checkpoint/teacher/optimizer; not an end-to-end training speedup."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
