# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.triton_utils.importing import (
    HAS_TRITON,
    TRITON_IS_GFX906_VARIANT,
    TritonLanguagePlaceholder,
    TritonPlaceholder,
)

if TYPE_CHECKING or HAS_TRITON:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()
    tldevice = TritonLanguagePlaceholder()

__all__ = ["HAS_TRITON", "TRITON_IS_GFX906_VARIANT", "triton", "tl", "tldevice"]
