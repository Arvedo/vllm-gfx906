# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types
from importlib.metadata import PackageNotFoundError
from unittest import mock

from vllm.triton_utils.importing import TritonLanguagePlaceholder, TritonPlaceholder


def test_triton_placeholder_is_module():
    triton = TritonPlaceholder()
    assert isinstance(triton, types.ModuleType)
    assert triton.__name__ == "triton"


def test_triton_language_placeholder_is_module():
    triton_language = TritonLanguagePlaceholder()
    assert isinstance(triton_language, types.ModuleType)
    assert triton_language.__name__ == "triton.language"


def test_triton_placeholder_decorators():
    triton = TritonPlaceholder()

    @triton.jit
    def foo(x):
        return x

    @triton.autotune
    def bar(x):
        return x

    @triton.heuristics
    def baz(x):
        return x

    assert foo(1) == 1
    assert bar(2) == 2
    assert baz(3) == 3


def test_triton_placeholder_decorators_with_args():
    triton = TritonPlaceholder()

    @triton.jit(debug=True)
    def foo(x):
        return x

    @triton.autotune(configs=[], key="x")
    def bar(x):
        return x

    @triton.heuristics({"BLOCK_SIZE": lambda args: 128 if args["x"] > 1024 else 64})
    def baz(x):
        return x

    assert foo(1) == 1
    assert bar(2) == 2
    assert baz(3) == 3


def test_triton_placeholder_language():
    lang = TritonLanguagePlaceholder()
    assert isinstance(lang, types.ModuleType)
    assert lang.__name__ == "triton.language"
    assert lang.constexpr is None
    assert lang.dtype is None
    assert lang.int64 is None
    assert lang.int32 is None
    assert lang.tensor is None


def test_triton_placeholder_language_from_parent():
    triton = TritonPlaceholder()
    lang = triton.language
    assert isinstance(lang, TritonLanguagePlaceholder)


def test_no_triton_fallback():
    # clear existing triton modules
    sys.modules.pop("triton", None)
    sys.modules.pop("triton.language", None)
    sys.modules.pop("vllm.triton_utils", None)
    sys.modules.pop("vllm.triton_utils.importing", None)

    # mock triton not being installed
    with mock.patch.dict(sys.modules, {"triton": None}):
        from vllm.triton_utils import HAS_TRITON, tl, triton

        assert HAS_TRITON is False
        assert triton.__class__.__name__ == "TritonPlaceholder"
        assert triton.language.__class__.__name__ == "TritonLanguagePlaceholder"
        assert tl.__class__.__name__ == "TritonLanguagePlaceholder"


def _reload_triton_importing():
    sys.modules.pop("vllm.triton_utils", None)
    sys.modules.pop("vllm.triton_utils.importing", None)
    return importlib.import_module("vllm.triton_utils.importing")


def test_gfx906_distribution_enables_compat_probe_skip():
    with mock.patch("importlib.util.find_spec") as mock_find_spec, mock.patch(
        "importlib.metadata.version"
    ) as mock_version:
        mock_find_spec.side_effect = lambda name: object() if name == "triton" else None
        mock_version.side_effect = lambda name: (
            "3.5.0+gfx906"
            if name == "triton-gfx906"
            else (_ for _ in ()).throw(PackageNotFoundError(name))
        )

        importing = _reload_triton_importing()

        assert importing.HAS_TRITON is True
        assert importing.TRITON_IS_GFX906_VARIANT is True


def test_skip_active_driver_check_env_allows_partial_triton_install():
    with mock.patch.dict(
        "os.environ", {"VLLM_TRITON_SKIP_ACTIVE_DRIVER_CHECK": "1"}, clear=False
    ), mock.patch("importlib.util.find_spec") as mock_find_spec, mock.patch(
        "importlib.metadata.version"
    ) as mock_version:
        mock_find_spec.side_effect = lambda name: object() if name == "triton" else None
        mock_version.side_effect = PackageNotFoundError("distribution missing")

        importing = _reload_triton_importing()

        assert importing.HAS_TRITON is True


def test_partial_triton_install_without_compat_mode_falls_back():
    with mock.patch.dict(
        "os.environ", {"VLLM_TRITON_SKIP_ACTIVE_DRIVER_CHECK": "0"}, clear=False
    ), mock.patch("importlib.util.find_spec") as mock_find_spec, mock.patch(
        "importlib.metadata.version"
    ) as mock_version, mock.patch.dict(
        "sys.modules", {"triton": types.ModuleType("triton")}
    ):
        mock_find_spec.side_effect = lambda name: object() if name == "triton" else None
        mock_version.side_effect = PackageNotFoundError("distribution missing")

        importing = _reload_triton_importing()

        assert importing.HAS_TRITON is False
        assert importing.TRITON_IS_GFX906_VARIANT is False
