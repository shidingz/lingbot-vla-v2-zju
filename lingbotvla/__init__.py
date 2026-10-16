# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


__version__ = "0.0.1"

# On Ascend NPU environments, activate Huawei's CUDA->NPU compatibility shim as
# early as possible so the torch.cuda.* / device="cuda" idioms used across this
# codebase resolve to torch_npu. No-op on CUDA/CPU machines (torch_npu absent).
import importlib.util as _importlib_util

if _importlib_util.find_spec("torch_npu") is not None:
    import torch_npu.contrib.transfer_to_npu  # noqa: F401
