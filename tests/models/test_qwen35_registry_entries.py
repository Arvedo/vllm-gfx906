# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types

from vllm.model_executor.models.registry import (
    _MULTIMODAL_MODELS,
    _SPECULATIVE_DECODING_MODELS,
)


def test_qwen35_multimodal_registry_entries():
    assert _MULTIMODAL_MODELS["Qwen3_5ForConditionalGeneration"] == (
        "qwen3_5",
        "Qwen3_5ForConditionalGeneration",
    )
    assert _MULTIMODAL_MODELS["Qwen3_5MoeForConditionalGeneration"] == (
        "qwen3_5",
        "Qwen3_5MoeForConditionalGeneration",
    )


def test_qwen35_spec_decode_registry_entries():
    assert _SPECULATIVE_DECODING_MODELS["Qwen3_5MTP"] == (
        "qwen3_5_mtp",
        "Qwen3_5MTP",
    )
    assert _SPECULATIVE_DECODING_MODELS["Qwen3_5MoeMTP"] == (
        "qwen3_5_mtp",
        "Qwen3_5MoeMTP",
    )


def test_qwen35_import_with_missing_mamba_copy_symbols(monkeypatch):
    import vllm.model_executor.layers.mamba.mamba_utils as real_mamba_utils

    stub_mamba_utils = types.ModuleType(real_mamba_utils.__name__)
    stub_mamba_utils.MambaStateDtypeCalculator = (
        real_mamba_utils.MambaStateDtypeCalculator
    )
    stub_mamba_utils.MambaStateShapeCalculator = (
        real_mamba_utils.MambaStateShapeCalculator
    )

    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.mamba.mamba_utils",
        stub_mamba_utils,
    )
    sys.modules.pop("vllm.model_executor.models.qwen3_5", None)

    qwen35_module = importlib.import_module("vllm.model_executor.models.qwen3_5")

    copy_funcs = qwen35_module.Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()
    assert len(copy_funcs) == 2
    assert all(callable(fn) for fn in copy_funcs)
