# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys

import pytest

from vllm.platforms import current_platform


def test_rocm_missing_flash_attn_does_not_fail_module_import(monkeypatch):
    """ROCm modules should import even when flash_attn is unavailable."""

    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "flash_attn":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    for module_name in (
        "vllm.attention.utils.fa_utils",
        "vllm.v1.attention.backends.mla.common",
    ):
        sys.modules.pop(module_name, None)

    fa_utils = importlib.import_module("vllm.attention.utils.fa_utils")
    assert fa_utils.is_flash_attn_varlen_func_available() is False
    with pytest.raises(ImportError, match="flash_attn is not installed"):
        fa_utils.flash_attn_varlen_func()

    mla_common = importlib.import_module("vllm.v1.attention.backends.mla.common")
    with pytest.raises(ImportError, match="flash_attn is not installed"):
        mla_common.flash_attn_varlen_func()
