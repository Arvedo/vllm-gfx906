# Triton gfx906 integration note

## Scope

This change adds a narrow compatibility layer for using the `nlzy/triton-gfx906` package semantics in vLLM Triton probing/import logic.

It does **not** change Docker files, and does **not** modify global dependency pins (including Transformers).

## What changed

### 1) Triton probe compatibility (`vllm/triton_utils/importing.py`)

- Added distribution-level detection for gfx906 Triton variants:
  - `triton-gfx906`
  - `triton_gfx906`
- Added compatibility mode that skips strict `triton.backends` active-driver probing when:
  - gfx906 Triton distribution is detected, or
  - `VLLM_TRITON_SKIP_ACTIVE_DRIVER_CHECK=1` is set.
- Kept existing strict probe behavior for default Triton installs when compatibility mode is not active.
- Preserved safe fallback: if strict probing fails (e.g. partial install / missing `triton.backends`), `HAS_TRITON=False` is retained.

### 2) Exposed compatibility state (`vllm/triton_utils/__init__.py`)

- Re-exported `TRITON_IS_GFX906_VARIANT` so runtime code can introspect whether gfx906 variant detection matched.

### 3) Env wiring (`vllm/envs.py`)

- Added:
  - `VLLM_TRITON_SKIP_ACTIVE_DRIVER_CHECK` (bool, default `False` / `0`)

### 4) Usage telemetry env list (`vllm/usage/usage_lib.py`)

- Added `VLLM_TRITON_SKIP_ACTIVE_DRIVER_CHECK` to collected usage env keys.

## Expected package/module names

- Import module expected by vLLM remains:
  - `triton`
- gfx906 distribution names recognized for compatibility mode:
  - `triton-gfx906`
  - `triton_gfx906`

## Environment variables / toggles

- `VLLM_TRITON_SKIP_ACTIVE_DRIVER_CHECK`
  - `0` (default): use normal strict active-driver probe unless gfx906 distribution is detected.
  - `1`: force compatibility mode and skip active-driver probe.

## Fallback behavior

- If Triton module is missing: `HAS_TRITON=False` (unchanged).
- If Triton is present but strict probe fails and compatibility mode is off: `HAS_TRITON=False` (unchanged).
- If compatibility mode is on: active-driver probe is skipped to allow gfx906 Triton package semantics.

## Tests added/updated

Updated `tests/test_triton_utils.py` with mock-based unit tests for:

- gfx906 distribution detection enabling compatibility mode.
- env override (`VLLM_TRITON_SKIP_ACTIVE_DRIVER_CHECK=1`) allowing Triton presence without strict probe.
- safe fallback when strict probe remains active and `triton.backends` is unavailable.

## Limitations

- This change only addresses Triton detection/probing compatibility.
- It does not validate end-to-end kernel correctness/performance on gfx906.
- It does not change dependency pinning strategy (including Transformers 5.2), which is intentionally deferred to a later subtask.
