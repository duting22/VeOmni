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

"""Strict Ascend 950 (A5 / arch35) guards for opt-in Triton kernels.

Importing this module never imports torch_npu or Triton. Device names must
start with the documented arch35 names; unknown devices fail closed.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

import torch

from ...utils.device import IS_NPU_AVAILABLE


_DEVICE_PREFIXES = ("Ascend950", "Ascend910_95")


def _device_name(device: torch.device | None = None) -> str:
    if not IS_NPU_AVAILABLE or not hasattr(torch, "npu"):
        raise RuntimeError("Ascend 950 Triton kernels require an available torch_npu runtime")
    if device is not None and device.type not in ("npu", "privateuseone"):
        raise RuntimeError(f"Ascend 950 Triton kernels require NPU tensors, got {device}")
    index = torch.npu.current_device() if device is None or device.index is None else device.index
    return str(torch.npu.get_device_name(index))


def require_ascend950(device: torch.device | None = None) -> None:
    """Require an Ascend950 / Ascend910_95 device, rejecting A3 and unknown names."""
    name = _device_name(device)
    if not name.startswith(_DEVICE_PREFIXES):
        raise RuntimeError(f"Ascend 950 Triton kernels require A5 / arch35 (Ascend950 or Ascend910_95), got {name!r}")


@dataclass(frozen=True)
class Ascend950KernelRequirement:
    """Registry requirement for the explicit Ascend 950 Triton implementations."""

    device: ClassVar[str] = "npu"

    def matches(self) -> bool:
        """Return whether the active device is explicitly identified as arch35."""
        try:
            self.check()
        except (RuntimeError, AttributeError):
            return False
        return True

    def check(self) -> None:
        """Raise if the active device is not an Ascend 950 NPU."""
        require_ascend950()


@contextmanager
def ascend950_device(device: torch.device):
    """Validate the tensor device and select it for native and Triton launches."""
    require_ascend950(device)
    index = torch.npu.current_device() if device.index is None else device.index
    with torch.npu.device(index):
        yield


@lru_cache(maxsize=1024)
def launch_plan(
    num_rows: int,
    width: int,
    element_size: int,
    device: torch.device,
    *,
    backward: bool,
    post: bool = False,
) -> tuple[int, int, int, bool]:
    """Cache bounded row tiles using the selected 950 device's vector cores and UB.

    This is called inside ``ascend950_device``. Missing property fields use the
    source bridge's conservative scheduling defaults, not a backend fallback.
    Tiles exceeding the reported or default UB budget still fail explicitly.
    """
    import triton
    from triton.runtime import driver

    target = driver.active.get_current_target()
    if target.backend not in ("npu", "ascend"):
        raise RuntimeError(f"Ascend 950 kernels require Triton-Ascend, got backend {target.backend!r}")
    index = torch.npu.current_device() if device.index is None else device.index
    properties = driver.active.utils.get_device_properties(index)

    def value(names, default):
        for name in names:
            item = properties.get(name) if isinstance(properties, dict) else getattr(properties, name, None)
            if item is not None and int(item) > 0:
                return int(item)
        return default

    # Match mcore-bridge's qhc_npu.py when Triton-Ascend omits these fields.
    # The UB default is a scheduling budget, not a claim about physical capacity.
    cores = value(("num_vectorcore", "num_aivectorcore", "num_aicore", "num_cube_core"), 40)
    ub_size = value(("ubsize", "ub_size", "UB_SIZE"), 192 * 1024)
    block_cols = triton.next_power_of_2(width)
    multiplier = 16 if backward or post else 7
    rows_by_ub = (ub_size - 6144) // (block_cols * element_size * multiplier)
    if rows_by_ub < 1:
        raise ValueError(f"group width {width} exceeds the Ascend 950 Triton kernel's supported UB layout")
    max_rows = 8 if backward or post else 16
    rows = min(rows_by_ub, max_rows, triton.next_power_of_2(max(num_rows, 1)))
    block_rows = 1 << (rows.bit_length() - 1)
    nprog = max(1, min(triton.cdiv(num_rows, block_rows), cores))
    return block_rows, block_cols, nprog, block_cols == width
