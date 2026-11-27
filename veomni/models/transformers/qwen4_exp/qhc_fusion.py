# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Module compatibility checks for explicit Qwen4-Exp QHC fusion."""

from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor


def validate_qhc_modules(qhc: nn.Module) -> None:
    """Reject child modules whose forward semantics would be bypassed by QHC."""
    children = [qhc.hc_norm, qhc.input_mix_weight_down, qhc.input_mix_weight_up]
    if qhc.block_inject_weight is not None:
        children.append(qhc.block_inject_weight)
    for projection in children[1:]:
        if type(projection) is not nn.Linear or projection.bias is not None:
            raise ValueError(
                "qhc_implementation='triton' requires plain bias-free nn.Linear projections; "
                "use qhc_implementation='eager' with adapters or wrapped projections."
            )
    for child in children:
        if isinstance(child, FSDPModule) or any(
            (child._forward_pre_hooks, child._forward_hooks, child._backward_pre_hooks, child._backward_hooks)
        ):
            raise ValueError(
                "qhc_implementation='triton' cannot bypass child-module hooks or separately sharded children. "
                "Wrap the decoder layer with FSDP and use qhc_implementation='eager' for child-module hooks."
            )
        if isinstance(child.weight, DTensor):
            raise ValueError("qhc_implementation='triton' requires materialized local child weights.")
