"""Qwen RMSNorm with an opt-in Ascend FP32 fused path.

Low-precision inputs retain Qwen's original cast-before-weight arithmetic.
"""
import os

import torch


def qwen_rms_norm(hidden_states, weight, eps):
    if (
        os.environ.get("LINGBOT_NPU_FUSED_RMSNORM", "0") == "1"
        and hidden_states.device.type == "npu"
        and hidden_states.dtype == weight.dtype == torch.float32
    ):
        import torch_npu

        return torch_npu.npu_rms_norm(hidden_states, weight, epsilon=eps)[0]
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)
