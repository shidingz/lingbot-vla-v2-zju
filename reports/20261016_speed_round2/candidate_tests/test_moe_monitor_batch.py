"""CPU-only equivalence checks on actual project monitor source.

Copy this file to <project>/tests/test_moe_monitor_batch.py. By default it checks
that the environment switch preserves the original (disabled) source branch.
For independent baseline validation, set LINGBOT_MONITOR_BASELINE_SOURCE to the
unmodified model file in the baseline snapshot. LINGBOT_MONITOR_SOURCE may also
override the candidate source, useful before installing the patch.

No model imports or accelerators are initialized. The actual sequence-wise loss
function declarations are loaded from the repository with Triton disabled.
"""
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / 'lingbotvla/ops/triton_moe_loss.py').is_file())
ACTUAL_SOURCE = Path(os.environ.get('LINGBOT_MONITOR_SOURCE', str(ROOT / 'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py')))
BASELINE_SOURCE = Path(os.environ.get('LINGBOT_MONITOR_BASELINE_SOURCE', str(ACTUAL_SOURCE)))
namespace = {
    'torch': torch, 'F': F, 'os': os,
    'List': List, 'Optional': Optional, 'Tuple': Tuple, 'Union': Union,
    '_HAS_TRITON': False,
}
loss_source = ROOT / 'lingbotvla/ops/triton_moe_loss.py'
loss_ast = ast.parse(loss_source.read_text())
loss_functions = [n for n in loss_ast.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
exec(compile(ast.Module(body=loss_functions, type_ignores=[]), str(loss_source), 'exec'), namespace)


def load_method(name):
    source = Path(name)
    module = ast.parse(source.read_text())
    method = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == '_moe_losses_and_metrics')
    scope = dict(namespace)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    return scope['_moe_losses_and_metrics']


original = load_method(BASELINE_SOURCE)
candidate = load_method(ACTUAL_SOURCE)


def nested_equal(left, right, label='result'):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=label)
        assert left.requires_grad == right.requires_grad, label
        assert left.dtype == right.dtype, label
    elif isinstance(left, dict):
        assert list(left) == list(right), f'{label}: changed metric keys/order'
        for key in left:
            nested_equal(left[key], right[key], label + '.' + key)
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), label
        for i, (a, b) in enumerate(zip(left, right)):
            nested_equal(a, b, f'{label}[{i}]')
    else:
        assert left == right, label


def make_inputs(kind, n=4, aux=True, remap=False):
    torch.manual_seed(20261016)
    indices = [2 * i + 1 for i in range(n)] if remap else list(range(n))
    blocks = [SimpleNamespace() for _ in range((max(indices) + 1) if indices else 0)]
    logits = [torch.randn(12, 8, requires_grad=True) for _ in range(n)]
    for i, layer_id in enumerate(indices):
        block = blocks[layer_id]
        block.avg_topk_sigmoid_score = torch.rand(1)
        block.e_score_correction_bias = torch.randn(8)
        if kind != 'no_buffers' and not (kind == 'mixed_missing' and i % 3 == 0):
            counts = torch.randint(1, 30, (8,)).float()
            if kind == 'all_zero' or (kind in ('mixed', 'mixed_missing') and i % 2 == 0):
                counts.zero_()
            elif kind == 'dead_expert':
                counts[0] = 0
            block.last_tokens_per_expert = counts
    self = SimpleNamespace(
        config=SimpleNamespace(
            token_moe_layers=indices,
            router_z_loss_coeff=0.003 if aux else 0,
            sequence_wise_loss_coeff=0.0001 if aux else 0,
            sequence_wise_mode='per_sequence', router_activation='sigmoid', token_top_k=4,
        ),
        qwenvl_with_expert=SimpleNamespace(qwen_expert=SimpleNamespace(model=SimpleNamespace(
            layers=[SimpleNamespace(mlp=block) for block in blocks]
        ))),
    )
    return self, logits, torch.rand(3, 4)


def run_equivalence(kind, n=4, aux=True, remap=False):
    ref_inputs = make_inputs(kind, n=n, aux=aux, remap=remap)
    test_inputs = copy.deepcopy(ref_inputs)
    before = copy.deepcopy(test_inputs[0])
    with patch.dict(os.environ, {'LINGBOT_MOE_MONITOR_BATCH_COUNTS': '0'}):
        ref_output = original(*ref_inputs)
    with patch.dict(os.environ, {'LINGBOT_MOE_MONITOR_BATCH_COUNTS': '1'}):
        test_output = candidate(*test_inputs)
    nested_equal(ref_output, test_output)
    if aux and n:
        (ref_output[0] + ref_output[1]).backward()
        (test_output[0] + test_output[1]).backward()
        for ref_logits, test_logits in zip(ref_inputs[1], test_inputs[1]):
            torch.testing.assert_close(ref_logits.grad, test_logits.grad, rtol=0, atol=0)
    for old_layer, new_layer in zip(before.qwenvl_with_expert.qwen_expert.model.layers,
                                    test_inputs[0].qwenvl_with_expert.qwen_expert.model.layers):
        nested_equal(vars(old_layer.mlp), vars(new_layer.mlp), label='unchanged buffers')
    # Counts exported to the caller must remain snapshots after optimizer updates.
    exported = test_output[2].get('_token_moe_expert_counts', [])
    frozen = [(i, c.clone()) for i, c in exported]
    for layer in test_inputs[0].qwenvl_with_expert.qwen_expert.model.layers:
        if hasattr(layer.mlp, 'last_tokens_per_expert'):
            layer.mlp.last_tokens_per_expert.fill_(999)
    nested_equal(exported, frozen, label='snapshot stability')
    # Disabled switch must retain the baseline path/results.
    with patch.dict(os.environ, {'LINGBOT_MOE_MONITOR_BATCH_COUNTS': '0'}):
        disabled = candidate(*copy.deepcopy(ref_inputs))
    nested_equal(ref_output, disabled, label='disabled path')
    return {'case': kind, 'layers': n, 'auxiliary_losses': aux, 'remapped_layers': remap,
            'metric_keys': len(test_output[2]), 'exact_outputs_and_gradients': True}



@pytest.mark.parametrize('kind,n,aux,remap', [
    ('populated', 4, True, False),
    ('all_zero', 4, True, False),
    ('mixed', 4, True, False),
    ('mixed_missing', 4, True, False),
    ('no_buffers', 4, True, False),
    ('dead_expert', 4, True, False),
    ('mixed', 36, True, True),
    ('populated', 36, False, False),
    ('populated', 0, False, False),
])
def test_monitor_outputs_buffers_and_auxiliary_gradients(kind, n, aux, remap):
    run_equivalence(kind, n=n, aux=aux, remap=remap)


@pytest.mark.parametrize('kind,expected_scalar_reads,expected_transfer_shapes', [
    ('populated', 36, [(36,)]),
    ('all_zero', 36, [(36,)]),
    ('no_buffers', None, []),
])
def test_monitor_batches_device_reads(kind, expected_scalar_reads, expected_transfer_shapes):
    inputs = make_inputs(kind, n=36, aux=False)
    audit = {}
    real_cpu = torch.Tensor.cpu
    for name, function, enabled in [('original', original, '0'), ('candidate', candidate, '1')]:
        cpu_calls = []
        def tracked_cpu(tensor, *args, **kwargs):
            cpu_calls.append(tuple(tensor.shape))
            return real_cpu(tensor, *args, **kwargs)
        with patch.dict(os.environ, {'LINGBOT_MOE_MONITOR_BATCH_COUNTS': enabled}), patch.object(torch.Tensor, 'cpu', tracked_cpu):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                function(*copy.deepcopy(inputs))
        scalar_reads = sum(event.count for event in prof.key_averages() if event.key == 'aten::_local_scalar_dense')
        audit[name] = {'local_scalar_dense_calls': scalar_reads, 'explicit_cpu_transfer_shapes': cpu_calls}
    if expected_scalar_reads is None:
        # one_hot fallback may internally read scalars, depending on PyTorch.
        # No load buffers exist here, so this patch must preserve that path.
        assert audit['candidate']['local_scalar_dense_calls'] == audit['original']['local_scalar_dense_calls'], audit
    else:
        assert audit['original']['local_scalar_dense_calls'] == expected_scalar_reads, audit
        assert audit['candidate']['local_scalar_dense_calls'] == 0, audit
    assert audit['candidate']['explicit_cpu_transfer_shapes'] == expected_transfer_shapes, audit
