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

"""Portable QHC references preserving native projection and gating autograd."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ._common import validate_inputs


def _qhc(x, norm_weight, down_weight, up_weight, injection_weight, hc_count, eps):
    hidden = validate_inputs(x, norm_weight, down_weight, up_weight, injection_weight, hc_count=hc_count, eps=eps)
    grouped = x.float().unflatten(-1, (hc_count, hidden))
    normalized = (grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)).flatten(-2)
    normalized = (normalized * (1.0 + norm_weight.float())).to(x.dtype)
    down = F.linear(normalized, down_weight)
    up = F.linear(F.silu(down / hc_count), up_weight)
    mixed = (torch.sigmoid(up) * normalized).unflatten(-1, (hc_count, hidden)).mean(-2)
    if injection_weight is None:
        return mixed
    injection = 2.0 * torch.sigmoid(F.linear(normalized, injection_weight) / hc_count)
    return mixed, injection


def gated(
    x: Tensor,
    norm_weight: Tensor,
    down_weight: Tensor,
    up_weight: Tensor,
    injection_weight: Tensor,
    *,
    hc_count: int,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Return QHC ``(mixed[..., hidden], injection[..., hc_count])``.

    Grouped offset RMSNorm reduces in FP32 and casts before bias-free native
    GEMMs. Eager autograd retains ``None`` gradients for an unused output's
    projection branch, including empty batches.
    """
    return _qhc(x, norm_weight, down_weight, up_weight, injection_weight, hc_count, eps)


def mixer(
    x: Tensor,
    norm_weight: Tensor,
    down_weight: Tensor,
    up_weight: Tensor,
    *,
    hc_count: int,
    eps: float,
) -> Tensor:
    """Return QHC ``mixed[..., hidden]`` without an injection projection."""
    return _qhc(x, norm_weight, down_weight, up_weight, None, hc_count, eps)
