# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.transformers_utils import configs
from vllm.transformers_utils.config import _CONFIG_REGISTRY


def test_qwen35_config_registry_entries():
    assert _CONFIG_REGISTRY["qwen3_5"].__name__ == "Qwen3_5Config"
    assert _CONFIG_REGISTRY["qwen3_5_moe"].__name__ == "Qwen3_5MoeConfig"


def test_qwen35_configs_exported():
    assert hasattr(configs, "Qwen3_5Config")
    assert hasattr(configs, "Qwen3_5TextConfig")
    assert hasattr(configs, "Qwen3_5MoeConfig")
    assert hasattr(configs, "Qwen3_5MoeTextConfig")
