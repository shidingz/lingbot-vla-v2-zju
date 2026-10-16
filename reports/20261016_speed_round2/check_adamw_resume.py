"""CPU-only checkpoint execution-policy tests against actual builder source."""
import argparse
import ast
import copy
import json
import logging
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    tree = ast.parse(args.source.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                and n.name in {'CombinedOptimizer', '_split_param_groups_by_scaled_lr', 'build_muon_optimizer'}]
    class StubMuon(torch.optim.SGD):
        def __init__(self, groups, **kwargs):
            super().__init__(groups, lr=kwargs['lr'])
    scope = dict(torch=torch, nn=torch.nn, os=os, re=re, logging=logging,
                 DTensor=DTensor, AdamW=torch.optim.AdamW, Optimizer=torch.optim.Optimizer,
                 DistributedMuon=StubMuon, is_torch_npu_available=lambda: True,
                 Any=Any, Dict=Dict, List=List, Optional=Optional, Sequence=Sequence, Tuple=Tuple)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(args.source), 'exec'), scope)
    rows = []
    with tempfile.TemporaryDirectory(prefix='adamw_resume_') as tmp:
        dist.init_process_group('gloo', init_method='file://' + str(Path(tmp) / 'init'), rank=0, world_size=1)
        mesh = init_device_mesh('cpu', (1,))
        def params_for(kind):
            local = torch.nn.Parameter(torch.linspace(-0.7, 1.0, 8))
            sharded = torch.nn.Parameter(distribute_tensor(torch.linspace(-0.9, 0.4, 8), mesh, [Shard(0)]))
            return [local] if kind == 'local' else [sharded] if kind == 'dtensor' else [local, sharded]
        def build(kind, enabled):
            params = params_for(kind)
            matrix = torch.nn.Parameter(torch.ones(2, 2))
            scope['split_muon_adamw_params'] = lambda *a, **kw: ([matrix], params, ['matrix.weight'], [f'norm{i}.weight' for i in range(len(params))])
            with patch.dict(os.environ, {'LINGBOT_ADAMW_FOREACH_NPU': '1' if enabled else '0'}):
                opt = scope['build_muon_optimizer'](None, SimpleNamespace(), lr=5e-5, weight_decay=0.001)
            return opt, params
        try:
            for kind in ('local', 'dtensor', 'hybrid'):
                for source_flag, target_flag in ((False, True), (True, False)):
                    for route in ('inner_direct_dcp_path', 'combined'):
                        src, src_params = build(kind, source_flag)
                        adam = src.optimizers[1]
                        for _ in range(3):
                            for p in src_params:
                                p.grad = torch.ones_like(p) * 0.2
                            adam.step()
                        # Distinct restored settings expose accidental resets.
                        adam.param_groups[0].update(lr=0.00071, weight_decay=0.025,
                                                   betas=(0.8, 0.88), eps=3e-7)
                        saved = copy.deepcopy(src.state_dict())
                        saved_adam = saved['optimizers'][1]
                        # Also emulate a foreign checkpoint with foreach=True
                        # for a mixed group; runtime guard must override it.
                        saved_adam['param_groups'][0]['foreach'] = source_flag
                        dst, dst_params = build(kind, target_flag)
                        with patch.dict(os.environ, {'LINGBOT_ADAMW_FOREACH_NPU': '0' if target_flag else '1'}):
                            # Load uses the captured construction policy, not a
                            # later environment mutation or checkpoint value.
                            if route == 'combined':
                                dst.load_state_dict(copy.deepcopy(saved))
                            else:
                                dst.optimizers[1].load_state_dict(copy.deepcopy(saved_adam))
                        dst_adam = dst.optimizers[1]
                        expected = target_flag and kind != 'hybrid'
                        actual = dst_adam.param_groups[0]['foreach']
                        assert actual is expected, (kind, source_flag, target_flag, route, actual)
                        got = dst_adam.state_dict()
                        for key, value in saved_adam['param_groups'][0].items():
                            if key not in ('foreach', 'params'):
                                assert got['param_groups'][0][key] == value, key
                        for param_id, state in saved_adam['state'].items():
                            assert set(got['state'][param_id]) == set(state)
                            for key, value in state.items():
                                other = got['state'][param_id][key]
                                assert type(value) == type(other) and value.dtype == other.dtype
                                if isinstance(value, DTensor):
                                    assert value.placements == other.placements
                                    value, other = value.to_local(), other.to_local()
                                torch.testing.assert_close(value, other, rtol=0, atol=0)
                        assert saved_adam['param_groups'][0]['foreach'] is source_flag
                        rows.append({'kind':kind, 'checkpoint_foreach':source_flag,
                                     'runtime_flag':target_flag, 'route':route,
                                     'loaded_foreach':actual, 'hyperparameters_and_moments_exact':True})
        finally:
            dist.destroy_process_group()
    report={'device':'cpu', 'world_size':1, 'source':str(args.source), 'checks':rows,
            'passed':len(rows)==12, 'scope':'Only foreach restored from captured runtime policy; LR/betas/decay/eps/state preserved.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
