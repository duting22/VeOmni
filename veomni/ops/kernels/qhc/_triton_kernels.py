# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Lazy Ascend 950 Triton QHC gating, mixing and injection kernels.

Adapted from MindSpeed-MM ``mindspeed_mm/fsdp/ops/qhc/qhc_triton.py`` at
f742843bbcda9417730c5801beea4361ebc5f102, through mcore-bridge
``src/mcore_bridge/model/modules/kernels/qhc_npu.py`` at d89b06f.
Local changes defer imports, specialize backward on used output branches, and
preserve the eager graph's BF16 sigmoid, product and gradient boundaries.
Factories bind ``tl`` in module globals on first use for Ascend JIT lookup.
"""

from __future__ import annotations

from ...triton_cache import cached_triton_kernel


@cached_triton_kernel
def _forward_kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["num_rows"])
    def kernel(
        mixed_ptr,
        injection_ptr,
        up_ptr,
        normed_ptr,
        logits_ptr,
        num_rows,
        row_stride,
        n_groups: tl.constexpr,
        hidden_size: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        EVEN_COLS: tl.constexpr,
        HAS_INJECTION: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_step = tl.num_programs(0) * BLOCK_ROWS
        columns = tl.arange(0, BLOCK_COLS)
        col_mask = columns < hidden_size
        for row0 in range(pid * BLOCK_ROWS, num_rows, row_step):
            rows = (row0 + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
            row_mask = rows < num_rows
            mask = row_mask[:, None] if EVEN_COLS else row_mask[:, None] & col_mask[None, :]
            mixed = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)
            for group in tl.static_range(n_groups):
                offsets = rows[:, None] * row_stride + (group * hidden_size + columns)[None, :]
                up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                normed = tl.load(normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                probability = tl.sigmoid(up).to(up_ptr.dtype.element_ty).to(tl.float32)
                product = (probability * normed).to(normed_ptr.dtype.element_ty).to(tl.float32)
                mixed += product
                if HAS_INJECTION:
                    logit = tl.load(logits_ptr + rows * n_groups + group, mask=row_mask, other=0.0).to(tl.float32)
                    scaled = (logit / n_groups).to(logits_ptr.dtype.element_ty).to(tl.float32)
                    probability = tl.sigmoid(scaled).to(logits_ptr.dtype.element_ty).to(tl.float32)
                    injection = 2.0 * probability
                    tl.store(injection_ptr + rows * n_groups + group, injection, mask=row_mask)
            offsets = rows[:, None] * hidden_size + columns[None, :]
            tl.store(mixed_ptr + offsets, mixed / n_groups, mask=mask)

    return kernel


@cached_triton_kernel
def _backward_kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["num_rows"])
    def kernel(
        dup_ptr,
        dnorm_ptr,
        dlogits_ptr,
        dmixed_ptr,
        dinjection_ptr,
        up_ptr,
        normed_ptr,
        logits_ptr,
        num_rows,
        row_stride,
        n_groups: tl.constexpr,
        hidden_size: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        EVEN_COLS: tl.constexpr,
        DO_MIXED: tl.constexpr,
        DO_INJECTION: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_step = tl.num_programs(0) * BLOCK_ROWS
        columns = tl.arange(0, BLOCK_COLS)
        col_mask = columns < hidden_size
        for row0 in range(pid * BLOCK_ROWS, num_rows, row_step):
            rows = (row0 + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
            row_mask = rows < num_rows
            mask = row_mask[:, None] if EVEN_COLS else row_mask[:, None] & col_mask[None, :]
            if DO_MIXED:
                offsets = rows[:, None] * hidden_size + columns[None, :]
                dmixed = tl.load(dmixed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                dmean = (dmixed / n_groups).to(dmixed_ptr.dtype.element_ty).to(tl.float32)
            for group in tl.static_range(n_groups):
                if DO_MIXED:
                    offsets = rows[:, None] * row_stride + (group * hidden_size + columns)[None, :]
                    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                    normed = tl.load(normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                    probability = tl.sigmoid(up).to(up_ptr.dtype.element_ty).to(tl.float32)
                    dprobability = (dmean * normed).to(up_ptr.dtype.element_ty).to(tl.float32)
                    dup = dprobability * (1.0 - probability) * probability
                    dnorm = dmean * probability
                    tl.store(dup_ptr + offsets, dup, mask=mask)
                    tl.store(dnorm_ptr + offsets, dnorm, mask=mask)
                if DO_INJECTION:
                    offsets = rows * n_groups + group
                    logit = tl.load(logits_ptr + offsets, mask=row_mask, other=0.0).to(tl.float32)
                    dinjection = tl.load(dinjection_ptr + offsets, mask=row_mask, other=0.0).to(tl.float32)
                    scaled = (logit / n_groups).to(logits_ptr.dtype.element_ty).to(tl.float32)
                    probability = tl.sigmoid(scaled).to(logits_ptr.dtype.element_ty).to(tl.float32)
                    dprobability = (dinjection * 2.0).to(logits_ptr.dtype.element_ty).to(tl.float32)
                    dscaled = (dprobability * (1.0 - probability) * probability).to(logits_ptr.dtype.element_ty)
                    dlogit = dscaled.to(tl.float32) / n_groups
                    tl.store(dlogits_ptr + offsets, dlogit, mask=row_mask)

    return kernel
