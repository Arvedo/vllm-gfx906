# Upstream Divergence Audit (gfx906 fork)

Date (UTC): 2026-02-28T12:38:13Z  
Repository: `c:/Users/Arved/Desktop/vllm-gfx906`

## Scope
Concrete divergence audit of current fork branch against:
1. Upstream release tag `v0.16.1rc0`
2. Upstream default/nightly baseline (`upstream_vllm/main` HEAD)

No code files were modified.

## Git state and compared refs

- Current branch: `gfx906/main`
- Current HEAD: `6dcb94a87ab10b2978a43e6b74d3ac1d4b40a491`
- Upstream release tag ref: `refs/tags/v0.16.1rc0` -> `3827c8c55aaa6622fd96b0c846a38b94444ebb80`
- Upstream nightly ref: `refs/remotes/upstream_vllm/main` -> `c68e69f1449cc6d84f43137fcc36c142de1c8fd3`
- Common merge-base (both comparisons): `d8c6210eeaa7f3b474e50cf74926f77a8dc79adf`

## Remote/fetch actions performed

- Existing remotes found:
  - `origin` -> `git@github.com:Arvedo/vllm-gfx906.git`
  - `upstream` -> `git@github.com:nlzy/vllm-gfx906.git`
- Added official upstream remote as required:
  - `upstream_vllm` -> `https://github.com/vllm-project/vllm.git`
- Fetch status: **successful** (`--tags --prune`), including tag `v0.16.1rc0`
- Default branch for official upstream resolved as: `upstream_vllm/main`

## Command snippets used

```powershell
git remote -v
git branch -vv
git status --short --branch
git rev-parse --short HEAD
git rev-parse --abbrev-ref HEAD
```

```powershell
$target='https://github.com/vllm-project/vllm.git'
if (-not (git remote | Select-String '^upstream_vllm$')) { git remote add upstream_vllm $target }
git fetch upstream_vllm --tags --prune
git remote set-head upstream_vllm -a
git show-ref --verify --quiet refs/tags/v0.16.1rc0
git symbolic-ref --short refs/remotes/upstream_vllm/HEAD
```

```powershell
git rev-list --left-right --count HEAD...refs/tags/v0.16.1rc0
git rev-list --left-right --count HEAD...refs/remotes/upstream_vllm/main
git merge-base HEAD refs/tags/v0.16.1rc0
git merge-base HEAD refs/remotes/upstream_vllm/main
```

## Quantified divergence summary

### Versus `v0.16.1rc0`

- Ahead (fork-only commits): **113**
- Behind (missing upstream commits): **2387**

### Versus upstream nightly (`upstream_vllm/main`)

- Ahead (fork-only commits): **113**
- Behind (missing upstream commits): **2489**

Interpretation:
- Fork has a stable local delta (113 commits) and is substantially behind both release candidate and nightly.
- Nightly introduces **102** additional commits beyond `v0.16.1rc0` relative to fork.

## High-level changed file areas and conflict hotspots

Area grouping was estimated from path patterns on diffs since merge-base.

### Fork-side modified files (67 total)
- `other`: 38
- `build-deps`: 14
- `model-config-engine`: 11
- `attention`: 4

### Upstream nightly modified files (2669 total)
- `other`: 1519
- `model-config-engine`: 616
- `entrypoints-serving`: 320
- `build-deps`: 142
- `attention`: 62
- `docker`: 10

### Direct path overlap (potential conflict concentration)
- Exact overlapping file paths: **20**
- Overlap by area:
  - `model-config-engine`: 8
  - `build-deps`: 7
  - `attention`: 3
  - `other`: 2

Representative overlap files:
- `CMakeLists.txt`
- `csrc/moe/moe_ops.h`
- `csrc/moe/torch_bindings.cpp`
- `csrc/ops.h`
- `csrc/quantization/gptq/q_gemm.cu`
- `csrc/torch_bindings.cpp`
- `requirements/rocm-build.txt`
- `vllm/_custom_ops.py`
- `vllm/attention/layer.py`
- `vllm/config/model.py`
- `vllm/model_executor/layers/fused_moe/fused_moe.py`
- `vllm/model_executor/layers/quantization/awq.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`
- `vllm/model_executor/layers/quantization/gptq.py`
- `vllm/model_executor/layers/quantization/kernels/mixed_precision/__init__.py`
- `vllm/model_executor/layers/utils.py`
- `vllm/model_executor/models/glm4.py`
- `vllm/platforms/__init__.py`
- `vllm/platforms/rocm.py`
- `vllm/v1/attention/backends/rocm_aiter_fa.py`

## Estimated merge risk by area

- **Attention / ROCm kernels**: **High**
  - Reason: overlap in ROCm and attention backend files plus large upstream evolution in attention stack.
- **Model/config/engine**: **High**
  - Reason: largest overlap concentration and heavy upstream churn in model-executor/config paths.
- **Build/deps/csrc**: **High**
  - Reason: overlap in `CMakeLists.txt`, `csrc/*`, and ROCm dependency file (`requirements/rocm-build.txt`).
- **Entrypoints/serving CLI**: **Medium**
  - Reason: high upstream churn but currently low direct path overlap in fork delta.
- **Docker**: **Low-Medium**
  - Reason: upstream changed Docker assets; no direct overlap observed in fork-side 67-file delta.

## Recommended merge order (minimal breakage for gfx906)

1. **Base sync to `v0.16.1rc0` first** (not nightly), resolve broad API/layout shifts with lower moving target.
2. **Build/deps/csrc layer**, keeping gfx906 compile/runtime viability early.
3. **Platform/ROCm + attention backend layer** (`vllm/platforms/rocm.py`, attention backends, custom ops).
4. **Model/config/quantization layer** (fused MoE, quantization kernels, config/model plumbing).
5. **Entrypoints and CLI/integration surfaces** after core runtime stabilizes.
6. **Optional nightly follow-up rebase/merge** to absorb post-`v0.16.1rc0` ~102 extra upstream commits.

Rationale: this order prioritizes restoring buildability and backend correctness before user-facing interfaces.

## Notes on uncertainty

- Fetch and tag resolution succeeded; no upstream connectivity gaps for this audit.
- Risk estimates are based on commit/path divergence and overlap heuristics, not full semantic conflict simulation.
