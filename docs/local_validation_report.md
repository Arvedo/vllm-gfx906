# Local Validation Report (Non-GPU)

Date (UTC): 2026-02-28
Workspace: `c:/Users/Arved/Desktop/vllm-gfx906`
Scope: Local sanity checks only (no GPU-required execution).

## Scope Performed
- Python syntax/compile checks on changed Python modules/tests.
- Lightweight pytest invocation on touched tests.
- Import smoke checks for touched modules.

## Commands Executed and Results

### 1) Identify touched files
```powershell
git status --short
```
Result: **PASS** (command executed successfully).

Observed touched Python areas included:
- `tests/test_triton_utils.py`
- `tests/models/test_qwen35_registry_entries.py`
- `tests/transformers_utils/test_qwen35_config_registry.py`
- `vllm/triton_utils/importing.py`
- `vllm/transformers_utils/config.py`
- `vllm/model_executor/models/qwen3_5.py`
- plus related registry/config/model files.

### 2) Python compile/syntax check over changed Python files
```powershell
$files = (git status --short | ForEach-Object { $_.Substring(3).Trim() }) | Where-Object { $_ -like '*.py' }; if ($files.Count -eq 0) { Write-Output 'No changed Python files detected.' } else { python -m py_compile $files }
```
Result: **PASS** (exit code 0, no syntax/bytecode compile errors).

### 3) Lightweight targeted pytest
```powershell
pytest -q tests/test_triton_utils.py tests/models/test_qwen35_registry_entries.py tests/transformers_utils/test_qwen35_config_registry.py
```
Result: **FAIL (blocked by environment dependency)**

Error:
- `ModuleNotFoundError: No module named 'tblib'`
- Source: `tests/conftest.py` import path before target tests could run.

### 4) Import smoke checks for key touched modules
```powershell
python -c "import importlib,sys; mods=['tblib','vllm.triton_utils','vllm.triton_utils.importing','vllm.transformers_utils.config','vllm.model_executor.models.qwen3_5'];
print('SMOKE_IMPORT_START');
failed=[];
for m in mods:
    try:
        importlib.import_module(m)
        print(f'OK {m}')
    except Exception as e:
        failed.append((m,repr(e)))
        print(f'FAIL {m}: {e!r}')
print('SMOKE_IMPORT_DONE');
print('FAILED_COUNT',len(failed));
"
```
Result: **PARTIAL PASS / PARTIAL FAIL**
- OK: `vllm.triton_utils`, `vllm.triton_utils.importing`
- FAIL: `tblib` missing
- FAIL: `vllm.transformers_utils.config` -> missing `gguf`
- FAIL: `vllm.model_executor.models.qwen3_5` -> missing `zmq`

Also observed non-fatal runtime warning:
- `RuntimeWarning: Failed to read commit hash: No module named 'vllm._version'`

## Pass/Fail Summary
- Compile/syntax checks on changed Python files: **PASS**
- Targeted pytest checks: **BLOCKED** by missing `tblib`
- Import smoke checks: **PARTIAL** (triton utils imports OK; config/model imports blocked by missing `gguf` and `zmq`)

## Blockers (Most Likely Root Causes)
From local logs, the most likely blockers are:
1. Incomplete local test/runtime dependency environment (`tblib`, `gguf`, `zmq` not installed).
2. Local workstation not provisioned as full vLLM dev/test environment for these modules.

## Deferred Actions
GPU/ROCm and MI50-specific validation is deferred to remote server execution (see `docs/mi50_remote_validation_checklist.md`).
