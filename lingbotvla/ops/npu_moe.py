"""Ascend grouped MoE with explicit first-order gradients and stacked weights.

torch_npu 2.6's grouped matmul does not register training gradients. Keep its
forward, input gradient and FP32 weight-gradient accumulation in one autograd
function so FSDP2's autograd-aware DTensor.to_local() remains connected.
"""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


class _NpuGroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, groups):
        import torch_npu

        ctx.save_for_backward(x, weight, groups)
        return torch_npu.npu_grouped_matmul(
            [x], [weight.transpose(-1, -2)],
            group_list=groups, split_item=3, group_type=0,
        )[0]

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        import torch_npu

        x, weight, groups = ctx.saved_tensors
        grad = grad.contiguous()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = torch_npu.npu_grouped_matmul(
                [grad], [weight], group_list=groups, split_item=3, group_type=0,
            )[0]
        if ctx.needs_input_grad[1]:
            # A single accumulator also keeps empty experts' gradients at zero.
            accumulator = torch.zeros(
                weight.shape[0], weight.shape[2], weight.shape[1],
                device=weight.device, dtype=torch.float32,
            )
            torch_npu.npu_grouped_matmul_add_(
                accumulator, x, grad, groups,
                transpose_x=True, transpose_weight=False, group_type=2,
            )
            dw = accumulator.transpose(-1, -2).to(weight.dtype).contiguous()
        return dx, dw, None


def grouped_moe_npu_forward(
    num_experts, routing_weights, selected_experts, hidden_states,
    fc1_1_weight, fc1_2_weight, fc2_weight,
):
    """Same routed SwiGLU math as the torch fallback, without host token counts.

    Operands are local BF16 tensors after the caller's FSDP2 conversion. No
    parameters or buffers are added, preserving existing checkpoint keys/layout.
    """
    tokens, topk = selected_experts.shape
    dim = hidden_states.shape[-1]
    if tokens == 0:
        return hidden_states * routing_weights.sum()
    if num_experts > 2**24:
        raise ValueError("NPU MoE FP32 sorting requires at most 2**24 experts")

    indices = selected_experts.reshape(-1)
    # Integer ArgSort falls back to AICPU on 910B. Expert IDs are represented
    # exactly in FP32, and stable sorting preserves the reference token order.
    order = indices.float().argsort(stable=True)
    counts = F.one_hot(indices, num_classes=num_experts).sum(0)
    groups = counts.cumsum(0)
    x = hidden_states[order // topk]
    gate = _NpuGroupedLinear.apply(x, fc1_1_weight, groups)
    up = _NpuGroupedLinear.apply(x, fc1_2_weight, groups)
    y = _NpuGroupedLinear.apply(F.silu(gate) * up, fc2_weight, groups)

    unsorted = torch.empty_like(y)
    unsorted[order] = y
    out = (unsorted.view(tokens, topk, dim).float()
           * routing_weights.unsqueeze(-1).float()).sum(1)
    return out.to(hidden_states.dtype)
