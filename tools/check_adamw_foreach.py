"""Compare existing AdamW foreach=False/True on CPU or NPU, local or DTensor.

NPU is touched only when --device npu is explicitly provided. NPU mode enables
transfer_to_npu, matching the project. Run via torchrun for multi-rank DTensor;
plain python automatically creates a one-rank process group for DTensor cases.
"""
import argparse
import ast
import copy
import json
import os
import re
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor


def local(t):
    return t.to_local() if isinstance(t, DTensor) else t


def check_switch(source):
    tree = ast.parse(Path(source).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_muon_optimizer')
    expr = next(n.value for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'foreach' for t in n.targets))
    code = compile(ast.Expression(expr), str(source), 'eval')
    for npu in (False, True):
        for enabled in ('0', '1'):
            with patch.dict(os.environ, {'LINGBOT_ADAMW_FOREACH_NPU': enabled}):
                value = eval(code, {'os': os, 'is_torch_npu_available': lambda: npu})
            assert value == ((not npu) or enabled == '1'), (npu, enabled, value)
    return 'actual builder expression passed CPU/NPU x off/on truth table'


def check_builder_guard(source, device, mesh):
    tree = ast.parse(Path(source).read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                and n.name in {'CombinedOptimizer', '_split_param_groups_by_scaled_lr', 'build_muon_optimizer'}]
    class DummyMuon:
        def __init__(self, groups, **kwargs):
            self.param_groups = groups
            self.kwargs = kwargs
    scope = dict(torch=torch, nn=torch.nn, os=os, re=re, logging=logging,
                 DTensor=DTensor, AdamW=torch.optim.AdamW, Optimizer=torch.optim.Optimizer,
                 DistributedMuon=DummyMuon, is_torch_npu_available=lambda: True,
                 Any=Any, Dict=Dict, List=List, Optional=Optional, Sequence=Sequence, Tuple=Tuple)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), scope)
    local_param = torch.nn.Parameter(torch.ones(8, device=device))
    cases = [('local', [local_param], True)]
    if mesh is not None:
        dist_param = torch.nn.Parameter(distribute_tensor(torch.ones(8, device=device), mesh, [Shard(0)]))
        cases += [('dtensor', [dist_param], True), ('hybrid', [local_param, dist_param], False)]
    records = []
    for name, params, enabled_expected in cases:
        matrix = torch.nn.Parameter(torch.ones(2, 2))
        scope['split_muon_adamw_params'] = lambda *a, **kw: ([matrix], params, ['matrix.weight'], [f'norm{i}.weight' for i in range(len(params))])
        for enabled in ('0', '1'):
            with patch.dict(os.environ, {'LINGBOT_ADAMW_FOREACH_NPU': enabled}):
                opt = scope['build_muon_optimizer'](None, SimpleNamespace(), lr=5e-5, weight_decay=0.001)
            adamw = opt.optimizers[1]
            expected = enabled_expected if enabled == '1' else False
            assert adamw.param_groups[0]['foreach'] is expected
            assert all(x is y for x, y in zip(adamw.param_groups[0]['params'], params))
            assert adamw.param_groups[0]['fused'] is False
            assert adamw.param_groups[0]['weight_decay'] == 0.001
            assert adamw.param_groups[0]['lr'] == 5e-5
            records.append({'group_kind': name, 'switch': enabled, 'foreach': expected})
    return records


def grad_for(shape, step, index):
    gen = torch.Generator(device='cpu').manual_seed(20261016 + step * 1009 + index * 37)
    if step % 6 == 0:
        return torch.zeros(shape)
    return torch.randn(shape, generator=gen) * (0.3 if step % 3 else 0.001)


def run_case(args, device, mesh, layout, mixed):
    shapes = [(32,), (17,), (8, 16), (16, 9), (1,), (64,)]
    specs = [(shapes[i % len(shapes)], torch.bfloat16 if mixed and i % 2 else torch.float32)
             for i in range(args.parameter_count)]
    def place(t, index):
        t = t.to(device)
        if layout == 'dtensor' or (layout == 'hybrid' and index % 3 != 0):
            # Alternate replicated/sharded tensors, including uneven and empty
            # shards at world_size=8. This checks placement-aware list handling.
            placement = Replicate() if index % 4 == 0 else Shard(0)
            t = distribute_tensor(t, mesh, [placement])
        return t
    refs, candidates = [], []
    for i, (shape, dtype) in enumerate(specs):
        gen = torch.Generator(device='cpu').manual_seed(10000 + i)
        tensor = torch.randn(shape, generator=gen).to(dtype)
        refs.append(torch.nn.Parameter(place(tensor.clone(), i)))
        candidates.append(torch.nn.Parameter(place(tensor.clone(), i)))
    def make_opt(params, foreach):
        # Distinct groups cover independent LR/decay/betas and skipped gradients.
        split = max(1, len(params) // 2)
        groups = [dict(params=params[:split], lr=5e-4, weight_decay=0.001),
                  dict(params=params[split:], lr=1.4e-3, weight_decay=0.03)]
        return torch.optim.AdamW(groups, lr=5e-4, betas=(0.9, 0.95), eps=1e-8,
                                 foreach=foreach, fused=False, capturable=False)
    ref_opt = make_opt(refs, False)
    test_opt = make_opt(candidates, True)
    maxima = {'float32': {}, 'bfloat16': {}}
    failures = []
    checked = 0
    checkpoint_roundtrip = False
    for step in range(1, args.steps + 1):
        for i, ((shape, dtype), a, b) in enumerate(zip(specs, refs, candidates)):
            # Keep some parameters absent initially, and skip other steps later.
            skip = (i % 7 == 0 and step <= 2) or (i % 5 == 0 and step % 4 == 0)
            grad = None if skip else place(grad_for(shape, step, i).to(dtype), i)
            a.grad = None if grad is None else grad.clone()
            b.grad = None if grad is None else grad.clone()
        ref_opt.step()
        test_opt.step()
        if device.type == 'npu':
            torch.npu.synchronize()
        for i, (a, b) in enumerate(zip(refs, candidates)):
            state_a, state_b = ref_opt.state[a], test_opt.state[b]
            if set(state_a) != set(state_b):
                raise AssertionError(f'state keys differ at step={step}, param={i}')
            pairs = [('parameter', a, b)] + [(key, state_a[key], state_b[key]) for key in state_a]
            for field, x, y in pairs:
                # Preserve DTensor state layout and scalar-step CPU residency.
                assert type(x) == type(y), (field, type(x), type(y))
                if isinstance(x, DTensor):
                    assert x.placements == y.placements and x.shape == y.shape
                assert x.dtype == y.dtype and x.device == y.device
                xx, yy = local(x).detach().float().cpu(), local(y).detach().float().cpu()
                if xx.numel() == 0:
                    continue
                assert torch.isfinite(xx).all() and torch.isfinite(yy).all()
                err = (xx - yy).abs()
                key = 'bfloat16' if x.dtype == torch.bfloat16 else 'float32'
                stat = maxima[key].setdefault(field, {'max_abs': 0.0, 'max_relative_to_global_max': 0.0, 'unequal_elements': 0, 'elements_checked': 0})
                stat['max_abs'] = max(stat['max_abs'], float(err.max()))
                stat['max_relative_to_global_max'] = max(stat['max_relative_to_global_max'], float(err.max() / xx.abs().max().clamp(min=1e-20)))
                stat['unequal_elements'] += int((xx != yy).sum())
                stat['elements_checked'] += xx.numel()
                if field == 'step':
                    rtol, atol = 0, 0
                elif x.dtype == torch.bfloat16:
                    rtol, atol = args.bf16_rtol, args.bf16_atol
                else:
                    rtol, atol = args.fp32_rtol, args.fp32_atol
                close = err <= atol + rtol * xx.abs()
                if not bool(close.all()) and len(failures) < 20:
                    failures.append({'step': step, 'parameter': i, 'field': field,
                                     'dtype': str(x.dtype), 'max_abs': float(err.max()),
                                     'mismatches': int((~close).sum()), 'rtol': rtol, 'atol': atol})
                checked += 1
        if step == max(1, args.steps // 2):
            # Reload each path's own state; check repeated-step/skip counters and
            # placement survive. Native AdamW retains its standard state schema.
            ref_opt.load_state_dict(copy.deepcopy(ref_opt.state_dict()))
            test_opt.load_state_dict(copy.deepcopy(test_opt.state_dict()))
            checkpoint_roundtrip = True
    return {'layout': layout, 'mixed_fp32_bf16': mixed, 'steps': args.steps,
            'parameters': len(specs), 'checked_tensors': checked,
            'checkpoint_state_dict_roundtrip': checkpoint_roundtrip,
            'maxima': maxima, 'passed': not failures, 'failures': failures}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=['cpu', 'npu'], default='cpu')
    p.add_argument('--layouts', default='local,dtensor')
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--parameter-count', type=int, default=24)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--optimizer-source', type=Path)
    p.add_argument('--fp32-rtol', type=float, default=2e-5)
    p.add_argument('--fp32-atol', type=float, default=2e-7)
    p.add_argument('--bf16-rtol', type=float, default=0.02)
    p.add_argument('--bf16-atol', type=float, default=0.002)
    args = p.parse_args()
    torch.set_num_threads(1)
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if args.device == 'npu':
        import torch_npu
        import torch_npu.contrib.transfer_to_npu
        torch.npu.set_device(local_rank)
        device = torch.device('npu', local_rank)
    else:
        device = torch.device('cpu')
    layouts = args.layouts.split(',')
    assert set(layouts) <= {'local', 'dtensor', 'hybrid'}
    group_created = False
    temp = None
    if any(x != 'local' for x in layouts):
        if 'RANK' in os.environ:
            dist.init_process_group('hccl' if args.device == 'npu' else 'gloo')
        else:
            temp = tempfile.TemporaryDirectory(prefix='adamw_foreach_')
            dist.init_process_group('hccl' if args.device == 'npu' else 'gloo',
                                    init_method='file://' + str(Path(temp.name) / 'init'), rank=0, world_size=1)
        group_created = True
        mesh = init_device_mesh(args.device, (dist.get_world_size(),))
    else:
        mesh = None
    rank = dist.get_rank() if dist.is_initialized() else 0
    report = {'device': str(device), 'rank': rank,
              'world_size': dist.get_world_size() if dist.is_initialized() else 1,
              'torch': torch.__version__, 'cases': [], 'tolerances': {
                  'fp32_rtol': args.fp32_rtol, 'fp32_atol': args.fp32_atol,
                  'bf16_rtol': args.bf16_rtol, 'bf16_atol': args.bf16_atol},
              'timing_scope': 'Correctness test with explicit synchronizations; not a performance benchmark.'}
    try:
        if args.optimizer_source:
            report['builder_switch'] = check_switch(args.optimizer_source)
            report['builder_guard'] = check_builder_guard(args.optimizer_source, device, mesh)
        for layout in layouts:
            for mixed in (False, True):
                print(f'rank={rank}: checking {layout} mixed={mixed}', flush=True)
                report['cases'].append(run_case(args, device, mesh, layout, mixed))
        report['passed'] = all(c['passed'] for c in report['cases'])
    except Exception as exc:
        report['passed'] = False
        report['error'] = repr(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        target = args.output.with_name(args.output.stem + f'.rank{rank}' + args.output.suffix)
        target.write_text(json.dumps(report, indent=2) + '\n')
        print(f'rank={rank}: saved {target}; passed={report.get("passed")}', flush=True)
        if group_created:
            dist.destroy_process_group()
        if temp:
            temp.cleanup()
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
