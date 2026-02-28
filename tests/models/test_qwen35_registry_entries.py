# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types

import pytest

import torch

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

    # Import must succeed even if the copy symbols are absent in mamba_utils.
    assert qwen35_module is not None

    copy_funcs = qwen35_module.Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()
    assert len(copy_funcs) == 2
    assert all(callable(fn) for fn in copy_funcs)

    # Confirm fallback path is used from qwen3_5 local compatibility shim.
    assert qwen35_module.MambaStateCopyFuncCalculator.__module__ == qwen35_module.__name__

    # Fallback copy funcs must be no-op callables, even with arbitrary args.
    assert copy_funcs[0](object(), object()) is None
    assert copy_funcs[1](state=None, cache=None) is None


def test_qwen35_import_with_missing_require_is_multimodal(monkeypatch):
    import vllm.model_executor.models.interfaces as real_interfaces

    monkeypatch.delattr(real_interfaces, "_require_is_multimodal", raising=False)
    sys.modules.pop("vllm.model_executor.models.qwen3_5", None)

    qwen35_module = importlib.import_module("vllm.model_executor.models.qwen3_5")

    # Import must succeed even if private helper is absent in interfaces.
    assert qwen35_module is not None

    required = qwen35_module._require_is_multimodal

    # Fallback helper keeps the same contract.
    with pytest.raises(ValueError, match="requires `is_multimodal`"):
        required(None)

    mask = torch.tensor([True, False])
    assert torch.equal(required(mask), mask)


def test_qwen35_mamba_state_dtype_compat_with_two_arg_calculator(monkeypatch):
    import vllm.model_executor.models.qwen3_5 as qwen35_module

    captured: dict[str, object] = {}

    def _two_arg_dtype_calculator(cls, model_dtype, mamba_cache_dtype):
        captured["model_dtype"] = model_dtype
        captured["mamba_cache_dtype"] = mamba_cache_dtype
        return (torch.float16, torch.float16)

    monkeypatch.setattr(
        qwen35_module.MambaStateDtypeCalculator,
        "gated_delta_net_state_dtype",
        classmethod(_two_arg_dtype_calculator),
    )

    vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(dtype=torch.bfloat16),
        cache_config=types.SimpleNamespace(
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="float32",
        ),
    )

    out = qwen35_module.Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(
        vllm_config
    )

    assert out == (torch.float16, torch.float16)
    assert captured == {
        "model_dtype": torch.bfloat16,
        "mamba_cache_dtype": "auto",
    }


def test_qwen35_mamba_state_dtype_compat_with_three_arg_calculator(monkeypatch):
    import vllm.model_executor.models.qwen3_5 as qwen35_module

    captured: dict[str, object] = {}

    def _three_arg_dtype_calculator(
        cls,
        model_dtype,
        mamba_cache_dtype,
        mamba_ssm_cache_dtype,
    ):
        captured["model_dtype"] = model_dtype
        captured["mamba_cache_dtype"] = mamba_cache_dtype
        captured["mamba_ssm_cache_dtype"] = mamba_ssm_cache_dtype
        return (torch.bfloat16, torch.bfloat16)

    monkeypatch.setattr(
        qwen35_module.MambaStateDtypeCalculator,
        "gated_delta_net_state_dtype",
        classmethod(_three_arg_dtype_calculator),
    )

    vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(dtype=torch.float16),
        cache_config=types.SimpleNamespace(
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="float32",
        ),
    )

    out = qwen35_module.Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(
        vllm_config
    )

    assert out == (torch.bfloat16, torch.bfloat16)
    assert captured == {
        "model_dtype": torch.float16,
        "mamba_cache_dtype": "auto",
        "mamba_ssm_cache_dtype": "float32",
    }
