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

"""Complete opt-in Ascend 950 QHC with native GEMMs and Triton vector kernels.

One custom autograd function owns grouped offset RMSNorm, down/up projections,
scaled SiLU, optional injection projection and fused sigmoid/group mixing.
The nested RMSNorm is a raw pair, avoiding nested autograd. Only BF16 input and
projection weights with BF16/FP32 norm weights are supported. A3 and unknown
devices fail explicitly; imports remain safe without NPU or Triton packages.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import once_differentiable

from ...registry import SavedState
from .._ascend950 import ascend950_device, launch_plan, require_ascend950
from ..rms_norm.offset import triton as rms_norm
from ._common import validate_inputs
from ._triton_kernels import _backward_kernel, _forward_kernel


def _linear_backward(grad_output, input_tensor, weight):
    grad_input = grad_output @ weight
    grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).transpose(0, 1) @ input_tensor.reshape(
        -1, input_tensor.shape[-1]
    )
    return grad_input, grad_weight


def _scaled_silu_backward(grad_output, preactivation, hc_count):
    scaled = preactivation / hc_count
    # Match F.silu's native backward and its BF16 output before the division.
    return torch.ops.aten.silu_backward(grad_output, scaled) / hc_count


class _QHCFunction(torch.autograd.Function):
    """Autograd owner for the complete QHC graph with optional injection output."""

    @staticmethod
    def forward(ctx, x, norm_weight, down_weight, up_weight, injection_weight, hc_count, eps):
        """Run native projections around raw grouped RMSNorm and fused postprocessing."""
        hidden = x.shape[-1] // hc_count
        rows = x.numel() // x.shape[-1]
        with ascend950_device(x.device):
            normed, norm_saved = rms_norm.forward(x, norm_weight, eps=eps, group_size=hidden)
            down = F.linear(normed, down_weight)
            activated = F.silu(down / hc_count)
            up = F.linear(activated, up_weight)
            logits = F.linear(normed, injection_weight) if injection_weight is not None else None
            mixed = up.new_empty((*x.shape[:-1], hidden))
            injection = up.new_empty((*x.shape[:-1], hc_count)) if logits is not None else None
            block_rows, block_cols, nprog, even_cols = launch_plan(
                rows, hidden, x.element_size(), x.device, backward=False, post=True
            )
            if rows:
                _forward_kernel()[(nprog,)](
                    mixed,
                    injection if injection is not None else mixed,
                    up,
                    normed,
                    logits if logits is not None else normed,
                    rows,
                    x.shape[-1],
                    n_groups=hc_count,
                    hidden_size=hidden,
                    BLOCK_ROWS=block_rows,
                    BLOCK_COLS=block_cols,
                    EVEN_COLS=even_cols,
                    HAS_INJECTION=logits is not None,
                )

        tensors = (*norm_saved.tensors, normed, down, activated, up, down_weight, up_weight)
        if logits is not None:
            tensors = (*tensors, logits, injection_weight)
        ctx.save_for_backward(*tensors)
        ctx.norm_count = len(norm_saved.tensors)
        ctx.norm_metadata = norm_saved.metadata
        ctx.hc_count = hc_count
        ctx.hidden = hidden
        ctx.num_rows = rows
        ctx.has_injection = logits is not None
        ctx.set_materialize_grads(False)
        return mixed if injection is None else (mixed, injection)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_mixed, grad_injection=None):
        """Differentiate only used outputs, retaining ``None`` for unused projections.

        Native GEMMs compute projection gradients. Scaled SiLU preserves the
        eager implementation's BF16 operator boundaries. RMSNorm runs its raw
        backward after adding the active projection branches' norm gradients.
        """
        if grad_mixed is None and grad_injection is None:
            return (None,) * 7
        saved = ctx.saved_tensors
        norm_saved = SavedState(saved[: ctx.norm_count], ctx.norm_metadata)
        normed, down, activated, up, down_weight, up_weight = saved[ctx.norm_count : ctx.norm_count + 6]
        if ctx.has_injection:
            logits, injection_weight = saved[ctx.norm_count + 6 :]
        else:
            logits = injection_weight = None

        has_mixed_grad = grad_mixed is not None
        has_injection_grad = grad_injection is not None
        grad_up = torch.empty_like(up) if has_mixed_grad else None
        grad_normed = torch.empty_like(normed) if has_mixed_grad else None
        grad_logits = torch.empty_like(logits) if has_injection_grad else None
        with ascend950_device(normed.device):
            if ctx.num_rows:
                block_rows, block_cols, nprog, even_cols = launch_plan(
                    ctx.num_rows,
                    ctx.hidden,
                    normed.element_size(),
                    normed.device,
                    backward=True,
                    post=True,
                )
                _backward_kernel()[(nprog,)](
                    grad_up if grad_up is not None else normed,
                    grad_normed if grad_normed is not None else normed,
                    grad_logits if grad_logits is not None else normed,
                    grad_mixed.contiguous() if has_mixed_grad else normed,
                    grad_injection.contiguous() if has_injection_grad else normed,
                    up,
                    normed,
                    logits if logits is not None else normed,
                    ctx.num_rows,
                    normed.shape[-1],
                    n_groups=ctx.hc_count,
                    hidden_size=ctx.hidden,
                    BLOCK_ROWS=block_rows,
                    BLOCK_COLS=block_cols,
                    EVEN_COLS=even_cols,
                    DO_MIXED=has_mixed_grad,
                    DO_INJECTION=has_injection_grad,
                )

            grad_down_weight = grad_up_weight = grad_injection_weight = None
            if has_mixed_grad:
                grad_activated, grad_up_weight = _linear_backward(grad_up, activated, up_weight)
                grad_down = _scaled_silu_backward(grad_activated, down, ctx.hc_count)
                grad_normed_down, grad_down_weight = _linear_backward(grad_down, normed, down_weight)
            if has_injection_grad:
                grad_normed_injection, grad_injection_weight = _linear_backward(grad_logits, normed, injection_weight)
                grad_normed = grad_normed_injection if grad_normed is None else grad_normed + grad_normed_injection
            if has_mixed_grad:
                # Eager adds direct mixing and injection before the down branch.
                grad_normed = grad_normed + grad_normed_down
            grad_x, grad_norm_weight = rms_norm.backward(grad_normed, norm_saved)

        return grad_x, grad_norm_weight, grad_down_weight, grad_up_weight, grad_injection_weight, None, None


def _qhc(x, norm_weight, down_weight, up_weight, injection_weight, hc_count, eps):
    require_ascend950(x.device)
    validate_inputs(
        x,
        norm_weight,
        down_weight,
        up_weight,
        injection_weight,
        hc_count=hc_count,
        eps=eps,
        fused=True,
    )
    return _QHCFunction.apply(x, norm_weight, down_weight, up_weight, injection_weight, hc_count, eps)


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
    """Return ``(mixed[..., hidden], injection[..., hc_count])`` on Ascend 950.

    Noncontiguous strided inputs are supported, including empty batches. A
    missing output gradient leaves that output's private projection parameters
    unused (``None``), matching eager autograd. This implementation never falls
    back to eager and supports first-order gradients only.
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
    """Return ``mixed[..., hidden]`` using complete Ascend 950 QHC without injection."""
    return _qhc(x, norm_weight, down_weight, up_weight, None, hc_count, eps)
