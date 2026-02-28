# Qwen 3.5 merge notes (gfx906 fork)

## Scope
Minimal code-level port of upstream Qwen 3.5 support into this fork, with no Docker or dependency pin edits in this phase.

## Upstream references used
Primary upstream commits reviewed for this merge set:

- `9562912ce` — `[MODEL] Adding Support for Qwen3.5 Models (#34110)`
- `5885e330e` — `[Misc] Port Qwen3.5 Configs (#34512)`
- `2f186635c` — `[Bugfix] Fix Qwen3.5 config loading (#34554)`
- `9521002f0` — `[Misc] fix qwen3.5 config (#34604)`
- `909b14719` — `[Bugfix] Fix prefix creation for Qwen3.5 (#34723)`
- `c0bd8b13d` — `[Bugfix] Redo Qwen3.5/Qwen3-Next GDN projector fusion (#34697)`
- `6fff24f30` — `[Bugfix] Qwen3.5 kv-scale weight remapping (#34719)`

## Files changed
### Added
- `vllm/transformers_utils/configs/qwen3_5.py`
- `vllm/transformers_utils/configs/qwen3_5_moe.py`
- `vllm/model_executor/models/qwen3_5.py`
- `vllm/model_executor/models/qwen3_5_mtp.py`
- `tests/transformers_utils/test_qwen35_config_registry.py`
- `tests/models/test_qwen35_registry_entries.py`

### Updated
- `vllm/transformers_utils/config.py`
- `vllm/transformers_utils/configs/__init__.py`
- `vllm/model_executor/models/registry.py`
- `tests/models/registry.py`

## Compatibility decisions for gfx906 / ROCm
- No changes were made to AMD-specific backend-selection paths (ROCm platform, attention backend selection, AITER/ROCm codepaths).
- Qwen 3.5 model implementation reuses existing vLLM core components (notably Qwen3-Next/GDN infrastructure) instead of introducing ROCm-specific behavior changes.
- This keeps existing gfx906 behavior intact while enabling model/config registration and loading for Qwen 3.5 classes.

## Runtime compatibility fixes
- `vllm/model_executor/models/qwen3_5.py` now guards import of `MambaStateCopyFunc` and `MambaStateCopyFuncCalculator` from `vllm/model_executor/layers/mamba/mamba_utils.py`.
- In environments where these symbols are absent, Qwen3.5 import now falls back to no-op mamba state copy callables, preventing architecture inspection/import-time crashes for `Qwen3_5MoeForConditionalGeneration`.

## Tests added/updated
- Added focused unit checks for Qwen 3.5 registration points:
  - `tests/transformers_utils/test_qwen35_config_registry.py`
  - `tests/models/test_qwen35_registry_entries.py`
- Added import-regression coverage in `tests/models/test_qwen35_registry_entries.py` to verify Qwen3.5 module import succeeds when mamba copy symbols are missing.
- Updated `tests/models/registry.py` with Qwen 3.5 example model entries and transformer-version gates.

Validation notes:
- `python -m py_compile ...` over all touched Python files completed successfully.
- Targeted `pytest` could not run in this environment due to missing `tblib` (`ModuleNotFoundError` from `tests/conftest.py`).

## Deferred items
Intentionally deferred to later phases (per subtask constraints):
- Dockerfile edits.
- Dependency/version pin updates (including Triton source override and Transformers major-version pinning).
- Broader refactors beyond minimum Qwen 3.5 integration.
