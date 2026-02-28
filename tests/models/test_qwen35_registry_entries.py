# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
