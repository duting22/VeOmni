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

"""Opt-in Ascend 950 grouped offset RMSNorm raw forward/backward pair.

Inputs are BF16 with a flat BF16 or FP32 offset weight. Normalization and
``normalized * (1 + weight.float())`` run in FP32 before the BF16 output cast.
Noncontiguous strided tensors are copied; unsupported devices, dtypes and
layouts raise. Empty batches launch no kernel and retain the same raw-pair
contract. Neither importing this module nor registering it imports Triton.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ....registry import SavedState
from ..._ascend950 import ascend950_device, launch_plan, require_ascend950
from ._triton_kernels import _backward_kernel, _forward_kernel


@dataclass(frozen=True)
class _Meta:
    """Normalization group width and flattened row count for the saved tensors."""

    group_size: int
    num_rows: int


def _validate(x: Tensor, weight: Tensor, eps: float, group_size: int | None) -> int:
    require_ascend950(x.device)
    if x.layout != torch.strided or weight.layout != torch.strided:
        raise ValueError("Ascend 950 offset RMSNorm supports only strided tensors")
    if x.dtype != torch.bfloat16 or weight.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("Ascend 950 offset RMSNorm requires BF16 input and BF16/FP32 weight")
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("Ascend 950 offset RMSNorm requires a nonempty feature dimension")
    if weight.device != x.device or weight.shape != (x.shape[-1],):
        raise ValueError("offset RMSNorm weight must be flat, match x's last dimension, and share its device")
    width = x.shape[-1] if group_size is None else group_size
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0 or x.shape[-1] % width:
        raise ValueError("group_size must be a positive integer dividing x's last dimension")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    return width


def forward(
    x: Tensor,
    weight: Tensor,
    *,
    eps: float,
    group_size: int | None = None,
) -> tuple[Tensor, SavedState]:
    """Normalize independent last-dim groups and return ``(output, SavedState)``.

    ``weight`` is one offset per feature, including every group. ``group_size``
    defaults to the complete last dimension. This pair is Ascend 950 only and
    never delegates grouped or empty inputs to an eager autograd wrapper.
    """
    width = _validate(x, weight, eps, group_size)
    x = x.contiguous()
    weight = weight.contiguous()
    total_size = x.shape[-1]
    rows = x.numel() // total_size
    groups = total_size // width
    output = torch.empty_like(x)
    rstd = torch.empty((groups, rows), dtype=torch.float32, device=x.device)
    with ascend950_device(x.device):
        # Plan even empty batches so an explicitly selected unavailable backend
        # or unsupported feature layout still raises consistently.
        block_rows, block_cols, nprog, even_cols = launch_plan(rows, width, x.element_size(), x.device, backward=False)
        if rows:
            _forward_kernel()[(nprog,)](
                output,
                x,
                weight,
                rstd,
                rows,
                total_size,
                eps,
                n_groups=groups,
                group_size=width,
                BLOCK_ROWS=block_rows,
                BLOCK_COLS=block_cols,
                EVEN_COLS=even_cols,
            )
    return output, SavedState((x, weight, rstd), _Meta(width, rows))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, Tensor]:
    """Return BF16 ``dx`` and a weight gradient with the original weight dtype.

    Accumulate partial weight gradients in FP32 and reduce with native PyTorch.
    Noncontiguous incoming gradients are copied without mutating the caller.
    """
    meta = saved.metadata
    if not isinstance(meta, _Meta):
        raise TypeError("Ascend 950 offset RMSNorm backward requires its own SavedState")
    x, weight, rstd = saved.tensors
    if grad_output.device != x.device or grad_output.shape != x.shape or grad_output.dtype != x.dtype:
        raise ValueError("offset RMSNorm output gradient must match the saved input's shape, device and dtype")
    if grad_output.layout != torch.strided:
        raise ValueError("Ascend 950 offset RMSNorm supports only strided output gradients")
    dx = torch.empty_like(x)
    with ascend950_device(x.device):
        if not meta.num_rows:
            return dx, torch.zeros_like(weight)
        block_rows, block_cols, nprog, even_cols = launch_plan(
            meta.num_rows, meta.group_size, x.element_size(), x.device, backward=True
        )
        total_size = x.shape[-1]
        partial_dw = torch.empty((nprog, total_size), dtype=torch.float32, device=x.device)
        _backward_kernel()[(nprog,)](
            dx,
            grad_output.contiguous(),
            x,
            weight,
            rstd,
            partial_dw,
            meta.num_rows,
            total_size,
            n_groups=total_size // meta.group_size,
            group_size=meta.group_size,
            total_size=total_size,
            BLOCK_ROWS=block_rows,
            BLOCK_COLS=block_cols,
            EVEN_COLS=even_cols,
        )
        dw = partial_dw.sum(dim=0).to(weight.dtype)
    return dx, dw
