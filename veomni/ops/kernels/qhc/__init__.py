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

"""QHC kernel family: gated mixing with injection, and mixing without injection.

Eager is the portable reference. The complete Triton implementation is an
explicit opt-in restricted to Ascend 950; it is never a host default.
"""

from ...registry import register_op
from .._ascend950 import Ascend950KernelRequirement
from . import eager, triton


register_op(
    "qhc",
    "gated",
    "eager",
    wrapper=eager.gated,
    description="PyTorch grouped offset normalization, native QHC projections, gated mixing and injection",
)

register_op(
    "qhc",
    "mixer",
    "eager",
    wrapper=eager.mixer,
    description="PyTorch grouped offset normalization, native QHC projections and gated mixing",
)

register_op(
    "qhc",
    "gated",
    "triton",
    wrapper=triton.gated,
    description="Ascend 950 complete Triton QHC with native GEMMs, gated mixing and injection",
    requirement=Ascend950KernelRequirement(),
    requires=("triton", "torch_npu"),
)

register_op(
    "qhc",
    "mixer",
    "triton",
    wrapper=triton.mixer,
    description="Ascend 950 complete Triton QHC with native GEMMs and gated mixing",
    requirement=Ascend950KernelRequirement(),
    requires=("triton", "torch_npu"),
)
