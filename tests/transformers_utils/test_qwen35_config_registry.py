# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.transformers_utils import configs
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig


def test_qwen35_config_registry_entries():
    assert _CONFIG_REGISTRY["qwen3_5"].__name__ == "Qwen3_5Config"
    assert _CONFIG_REGISTRY["qwen3_5_moe"].__name__ == "Qwen3_5MoeConfig"


def test_qwen35_configs_exported():
    assert hasattr(configs, "Qwen3_5Config")
    assert hasattr(configs, "Qwen3_5TextConfig")
    assert hasattr(configs, "Qwen3_5MoeConfig")
    assert hasattr(configs, "Qwen3_5MoeTextConfig")


def test_qwen35_rope_ignore_keys_are_set():
    qwen35_text_config = Qwen3_5TextConfig()
    qwen35_moe_text_config = Qwen3_5MoeTextConfig()

    assert isinstance(qwen35_text_config.ignore_keys_at_rope_validation, set)
    assert isinstance(qwen35_moe_text_config.ignore_keys_at_rope_validation, set)
