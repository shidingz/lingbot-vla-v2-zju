#!/usr/bin/env python3
"""Check candidate against the actual eager function without importing the model.

Defaults to CPU and does not initialize any NPU. NPU runs require --device npu:0.
Example after reserving a card:
  python tools/check_joint_attention.py --device npu:0 --benchmark --output result.json
All benchmarks are single attention-call microbenchmarks, not training throughput.
"""

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import statistics
import time

import einops
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_candidate(path):
    spec = importlib.util.spec_from_file_location("attention_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_real_eager():
    path = ROOT / "lingbotvla/models/vla/lingbot_vla/utils.py"
    tree = ast.parse(path.read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "our_eager_attention_forward")
    namespace = {"torch": torch, "nn": nn, "einops": einops}
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[func.name]


def errors(actual, expected):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    difference = actual - expected
    reference_is_zero = bool((expected == 0).all())
    reference_l2 = expected.norm().item()
    return {
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        "max_abs": difference.abs().max().item(),
        "reference_is_zero": reference_is_zero,
        "reference_l2": reference_l2,
        # Relative error is undefined for a zero reference, not a huge number.
        "relative_l2": None if reference_is_zero else (difference.norm() / expected.norm()).item(),
    }


def within_screening_budget(metrics, abs_limit, rel_limit):
    # Preserve both original budgets for nonzero references. For an exactly
    # zero reference (e.g. one-key softmax Q/K gradients), only absolute error
    # is meaningful. The numerical tolerance itself is unchanged.
    return (metrics["finite"] and metrics["max_abs"] <= abs_limit
            and (metrics["reference_is_zero"] or metrics["relative_l2"] <= rel_limit))


def check_error_rule():
    cases = [
        ("zero_reference_roundoff", [0.0, 1.33e-6, -5.4e-7], [0.0, 0.0, 0.0], True),
        ("zero_reference_real_error", [0.0, 2e-4], [0.0, 0.0], False),
        ("zero_reference_nan", [float("nan")], [0.0], False),
        ("nonzero_relative_error", [1.01e-3], [1e-3], False),
        ("nonzero_absolute_error", [1000.0005], [1000.0], False),
        ("nonzero_roundoff", [1.000001], [1.0], True),
    ]
    rows = []
    for name, actual, expected, should_pass in cases:
        metrics = errors(torch.tensor(actual), torch.tensor(expected))
        accepted = within_screening_budget(metrics, 1e-4, 1e-4)
        rows.append({"case": name, "expected_acceptance": should_pass,
                     "actual_acceptance": accepted, "passed": accepted == should_pass})
    if not all(row["passed"] for row in rows):
        raise AssertionError(f"Error-rule self-check failed: {rows}")
    return rows


def build_case(name, device, shape=(2, 17, 23, 8, 2, 16)):
    b, lq, lk, hq, hkv, d = shape
    generator = torch.Generator().manual_seed(412)
    q = torch.randn(b, lq, hq, d, generator=generator).to(device)
    k = torch.randn(b, lk, hkv, d, generator=generator).to(device)
    v = torch.randn(b, lk, hkv, d, generator=generator).to(device)
    # Build masks on CPU; production masks are already constructed by the model.
    mask = torch.ones(b, lq, lk, dtype=torch.bool)
    if name == "mixed":
        mask = torch.rand(b, lq, lk, generator=generator) > 0.35
        mask[:, :, 0] = True
    elif name == "prefix_action":
        prefix = min(7, lq, lk)
        mask[:, :prefix, prefix:] = False
        mask[0, :, -3:] = False
        mask[0, -3:, :] = False  # Padded query rows.
    elif name == "all_masked":
        mask.zero_()
    elif name == "one_key":
        mask.zero_()
        mask[:, :, 0] = True
    elif name == "cache":
        # Query length can be shorter than key length during cached inference.
        mask[0, :, -2:] = False
    elif name == "scaled_logits":
        q *= 4
        k *= 4
        mask[:, -1, :] = False
    elif name != "dense":
        raise ValueError(name)
    upstream = torch.randn(b, lq, hq * d, generator=generator).to(device)
    return (q, k, v), mask.to(device), upstream


def run_grad(func, tensors, mask, upstream):
    inputs = [x.detach().clone().requires_grad_(True) for x in tensors]
    out = func(*inputs, mask)
    gradients = torch.autograd.grad((out * upstream).sum(), inputs)
    return out, gradients


def check_case(name, funcs, device, bf16, shape):
    tensors, mask, upstream = build_case(name, device, shape)
    expected, expected_gradients = run_grad(funcs["eager"], tensors, mask, upstream)
    rows = []
    for backend in (["sdpa", "sdpa_bf16"] if bf16 else ["sdpa"]):
        actual, gradients = run_grad(funcs[backend], tensors, mask, upstream)
        metrics = {"output": errors(actual, expected)}
        metrics.update({label: errors(got, ref) for label, got, ref in
                        zip(("dq", "dk", "dv"), gradients, expected_gradients)})
        # Conservative screening budgets, not a claim about task quality.
        abs_limit, rel_limit = ((1e-4, 1e-4) if backend == "sdpa" else (0.10, 0.03))
        passed = all(within_screening_budget(m, abs_limit, rel_limit) for m in metrics.values())
        rows.append({"case": name, "shape_B_Lq_Lk_Hq_Hkv_D": shape, "backend": backend,
                     "metrics": metrics, "passed": passed,
                     "screening_limits": {"max_abs": abs_limit, "relative_l2": rel_limit,
                                          "zero_reference": "absolute_error_only"}})
    return rows


def benchmark(func, device, iterations, warmup, shape):
    tensors, mask, upstream = build_case("prefix_action", device, shape)
    inputs = [x.requires_grad_(True) for x in tensors]
    sync = torch.npu.synchronize if device.startswith("npu") else lambda: None
    def iteration():
        out = func(*inputs, mask)
        torch.autograd.grad((out * upstream).sum(), inputs)
    for _ in range(warmup):
        iteration()
    sync()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        iteration()
        sync()
        times.append((time.perf_counter() - start) * 1000)
    return {"median_forward_backward_ms": statistics.median(times),
            "min_ms": min(times), "max_ms": max(times), "samples_ms": times,
            "shape_B_Lq_Lk_Hq_Hkv_D": shape}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate", type=Path, default=ROOT / "lingbotvla/ops/joint_attention.py")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--actual-shape", action="store_true",
                        help="Also compare output/gradients at the profiled training shape")
    parser.add_argument("--shape", type=int, nargs=6, default=[12, 337, 337, 32, 8, 128],
                        metavar=("B", "Lq", "Lk", "Hq", "Hkv", "D"))
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-error-rule-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    error_rule_checks = check_error_rule()
    if args.check_error_rule_only:
        payload = {"device": "cpu", "error_rule_checks": error_rule_checks, "passed": True}
        output = json.dumps(payload, indent=2)
        if args.output:
            args.output.write_text(output + "\n")
        print(output)
        return
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
    candidate = load_candidate(args.candidate)
    if args.bf16 and not hasattr(candidate, "our_sdpa_bf16_attention_forward"):
        parser.error("--bf16 requires a separate --candidate with a BF16 implementation; the project candidate keeps FP32 inputs")
    funcs = {"eager": load_real_eager(), "sdpa": candidate.our_sdpa_attention_forward}
    if args.bf16:
        funcs["sdpa_bf16"] = candidate.our_sdpa_bf16_attention_forward
    shapes = {"dense": (2, 17, 23, 8, 2, 16), "mixed": (2, 17, 23, 8, 2, 16),
              "prefix_action": (2, 17, 23, 8, 2, 16), "all_masked": (2, 17, 23, 8, 2, 16),
              "one_key": (2, 17, 23, 8, 2, 16), "cache": (2, 3, 23, 8, 2, 16),
              "scaled_logits": (2, 17, 23, 8, 2, 16)}
    checks = []
    for name, shape in shapes.items():
        checks += check_case(name, funcs, args.device, args.bf16, shape)
    # Also cover matching query/KV head counts and NPU's production head dim.
    checks += check_case("dense", funcs, args.device, args.bf16, (1, 32, 32, 2, 2, 128))
    if args.actual_shape:
        checks += check_case("prefix_action", funcs, args.device, args.bf16, args.shape)
    payload = {"device": args.device, "torch_version": torch.__version__,
               "checks": checks, "passed": all(row["passed"] for row in checks),
               "error_rule_checks": error_rule_checks,
               "error_rule_version": "2-zero-reference-absolute-only",
               "scope": "Isolated attention output/input gradients; full training not validated"}
    if args.device.startswith("npu"):
        payload["torch_npu_version"] = torch_npu.__version__
    if args.benchmark:
        payload["benchmarks"] = {name: benchmark(func, args.device, args.iterations,
                                                   args.warmup, args.shape)
                                 for name, func in funcs.items()
                                 if args.bf16 or name != "sdpa_bf16"}
    output = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    print(output)
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
