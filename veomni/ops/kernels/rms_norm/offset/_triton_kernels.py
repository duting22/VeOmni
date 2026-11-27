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

"""Lazy Triton-Ascend grouped offset RMSNorm kernels for Ascend 950.

Adapted from MindSpeed-MM ``mindspeed_mm/fsdp/ops/qhc/qhc_triton.py`` at
f742843bbcda9417730c5801beea4361ebc5f102, through mcore-bridge
``src/mcore_bridge/model/modules/kernels/qhc_npu.py`` at d89b06f.
Local changes isolate the raw RMSNorm pair and defer Triton imports.
Factories bind ``tl`` in module globals on first use for Ascend JIT lookup.
"""

from __future__ import annotations

from ....triton_cache import cached_triton_kernel


@cached_triton_kernel
def _forward_kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["num_rows"])
    def kernel(
        y_ptr,
        x_ptr,
        weight_ptr,
        rstd_ptr,
        num_rows,
        row_stride,
        eps,
        n_groups: tl.constexpr,
        group_size: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        EVEN_COLS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_step = tl.num_programs(0) * BLOCK_ROWS
        columns = tl.arange(0, BLOCK_COLS)
        col_mask = columns < group_size
        for group in tl.static_range(n_groups):
            col0 = group * group_size
            gamma = 1.0 + tl.load(weight_ptr + col0 + columns, mask=col_mask, other=0.0).to(tl.float32)
            for row0 in range(pid * BLOCK_ROWS, num_rows, row_step):
                rows = (row0 + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
                row_mask = rows < num_rows
                mask = row_mask[:, None] if EVEN_COLS else row_mask[:, None] & col_mask[None, :]
                offsets = rows[:, None] * row_stride + (col0 + columns)[None, :]
                x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                variance = tl.sum(x * x, axis=1) / group_size
                rstd = 1.0 / tl.sqrt(variance + eps)
                normalized = x * rstd[:, None]
                y = normalized * gamma[None, :]
                tl.store(y_ptr + offsets, y, mask=mask)
                tl.store(rstd_ptr + group * num_rows + rows, rstd, mask=row_mask)

    return kernel


@cached_triton_kernel
def _backward_kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["num_rows"])
    def kernel(
        dx_ptr,
        dy_ptr,
        x_ptr,
        weight_ptr,
        rstd_ptr,
        partial_dw_ptr,
        num_rows,
        row_stride,
        n_groups: tl.constexpr,
        group_size: tl.constexpr,
        total_size: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        EVEN_COLS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_step = tl.num_programs(0) * BLOCK_ROWS
        columns = tl.arange(0, BLOCK_COLS)
        col_mask = columns < group_size
        for group in tl.static_range(n_groups):
            col0 = group * group_size
            gamma = 1.0 + tl.load(weight_ptr + col0 + columns, mask=col_mask, other=0.0).to(tl.float32)
            dw = tl.zeros((BLOCK_COLS,), dtype=tl.float32)
            for row0 in range(pid * BLOCK_ROWS, num_rows, row_step):
                rows = (row0 + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
                row_mask = rows < num_rows
                mask = row_mask[:, None] if EVEN_COLS else row_mask[:, None] & col_mask[None, :]
                offsets = rows[:, None] * row_stride + (col0 + columns)[None, :]
                x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                rstd = tl.load(rstd_ptr + group * num_rows + rows, mask=row_mask, other=0.0)
                dnorm = dy * gamma[None, :]
                dot = tl.sum(dnorm * x, axis=1)
                dx = rstd[:, None] * dnorm - x * dot[:, None] * (
                    rstd[:, None] * rstd[:, None] * rstd[:, None] / group_size
                )
                dw += tl.sum(dy * (x * rstd[:, None]), axis=0)
                tl.store(dx_ptr + offsets, dx, mask=mask)
            tl.store(partial_dw_ptr + pid * total_size + col0 + columns, dw, mask=col_mask)

    return kernel
