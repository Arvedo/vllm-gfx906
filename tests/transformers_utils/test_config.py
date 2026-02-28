# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`vllm.LLMEngine._get_eos_token_id`.
"""

from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils import config as transformers_config_module
from vllm.transformers_utils.config import try_get_generation_config


def test_get_llama3_eos_token():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]


def test_get_blip2_eos_token():
    model_name = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118


def test_rope_layer_types_compatibility_without_allowed_layer_types(monkeypatch):
    monkeypatch.setattr(transformers_config_module, "ALLOWED_LAYER_TYPES", None)

    assert transformers_config_module._rope_parameters_are_layer_typed(
        {"decoder": {"rope_type": "dynamic", "factor": 2.0}}
    )
    assert not transformers_config_module._rope_parameters_are_layer_typed(
        {"decoder": {"factor": 2.0}}
    )
