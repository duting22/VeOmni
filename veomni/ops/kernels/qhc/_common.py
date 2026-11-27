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

"""Tensor-contract validation shared by QHC's eager and Ascend implementations."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def validate_inputs(
    x: Tensor,
    norm_weight: Tensor,
    down_weight: Tensor,
    up_weight: Tensor,
    injection_weight: Tensor | None,
    *,
    hc_count: int,
    eps: float,
    fused: bool = False,
) -> int:
    """Validate flat features and bias-free linear weights, returning group width.

    QHC accepts ``x[..., hc_count * hidden]``, ``norm_weight[hc_count * hidden]``,
    ``down_weight[rank, hc_count * hidden]`` and the transposed projection shape
    ``up_weight[hc_count * hidden, rank]``. Injection adds a
    ``[hc_count, hc_count * hidden]`` weight. Strided noncontiguous inputs are
    supported; all tensors must share a device and projection dtype.
    """
    if isinstance(hc_count, bool) or not isinstance(hc_count, int) or hc_count <= 0:
        raise ValueError("hc_count must be a positive integer")
    if x.ndim < 1 or x.shape[-1] <= 0 or x.shape[-1] % hc_count:
        raise ValueError("QHC input's last dimension must be positive and divisible by hc_count")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("QHC eps must be finite and nonnegative")
    total_size = x.shape[-1]
    if norm_weight.shape != (total_size,):
        raise ValueError("QHC norm_weight must be flat and match the input's last dimension")
    if down_weight.ndim != 2 or down_weight.shape[0] <= 0 or down_weight.shape[1] != total_size:
        raise ValueError("QHC down_weight must have shape [rank, input_features] with positive rank")
    if up_weight.shape != (total_size, down_weight.shape[0]):
        raise ValueError("QHC up_weight must have shape [input_features, rank]")
    if injection_weight is not None and injection_weight.shape != (hc_count, total_size):
        raise ValueError("QHC injection_weight must have shape [hc_count, input_features]")
    weights = (down_weight, up_weight) if injection_weight is None else (down_weight, up_weight, injection_weight)
    for tensor in (x, norm_weight, *weights):
        if tensor.layout != torch.strided or tensor.device != x.device:
            raise ValueError("QHC requires strided tensors on the same device")
    if any(weight.dtype != x.dtype for weight in weights):
        raise TypeError("QHC projection weights must have the input dtype")
    if not x.is_floating_point() or not norm_weight.is_floating_point():
        raise TypeError("QHC requires floating-point input and norm weight")
    if fused and (x.dtype != torch.bfloat16 or norm_weight.dtype not in (torch.bfloat16, torch.float32)):
        raise TypeError("Ascend 950 Triton QHC requires BF16 input/projections and BF16/FP32 norm weight")
    return total_size // hc_count
