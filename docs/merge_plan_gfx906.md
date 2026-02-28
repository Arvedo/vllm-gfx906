# Merge Plan for gfx906 Fork Toward Upstream vLLM Nightly Capability

## Scope and Intent
This document defines a phased integration strategy for this `vllm-gfx906` fork to approach upstream `v0.16.1rc0` and nightly feature parity required for Qwen 3.5 workloads, while preserving AMD MI50 `gfx906` operability.

This plan is implementation-oriented and intentionally separates local workstation tasks from remote MI50 server validation.

## Goals
- Preserve functional inference support for AMD MI50 `gfx906` throughout the merge process.
- Incrementally rebase or merge fork deltas onto upstream `v0.16.1rc0` baseline with controlled conflict resolution.
- Reach feature parity needed for Qwen 3.5 usage paths, including tokenizer, model loading, runtime, and serving behavior.
- Establish a repeatable validation gate for each phase.
- Prepare a follow-on path for Triton `gfx906`, `transformers` `^5.2`, and Ubuntu MI50 Docker installation workflow.

## Non-Goals
- No immediate full performance optimization for every AMD SKU.
- No broad refactor unrelated to upstream parity or `gfx906` compatibility.
- No production hardening of deployment automation in this phase.
- No dependency uplift execution in this subtask; only planning and gating.

## Constraints
- Remote GPU MI50 server is **not available from this workstation** for direct execution during local planning and initial merge iterations.
- Merge operations must remain bisectable and auditable.
- Upstream nightly can change rapidly; pinning and controlled sync windows are required.
- CI coverage for `gfx906` may be incomplete relative to CUDA paths.

## Phased Strategy

### Phase P0 - Baseline Inventory and Delta Mapping
**Objective:** Build a precise map of fork-specific changes and upstream divergence.

**Actions:**
- Identify fork-only commits and categorize by area: ROCm kernels, build/tooling, model adapters, test changes.
- Compare current fork HEAD against upstream `v0.16.1rc0` and a selected nightly reference commit.
- Label each delta as: keep, upstream-equivalent, drop, or redesign.

**Exit Criteria:**
- Signed-off delta matrix with ownership and disposition for every fork-specific patch.

### Phase P1 - Upstream Alignment Foundation
**Objective:** Establish a stable integration branch aligned to upstream release baseline.

**Actions:**
- Create merge integration branch from current fork.
- Merge or rebase onto upstream `v0.16.1rc0` with minimal semantic drift.
- Resolve conflicts using a compatibility-first rule set for `gfx906` critical paths.
- Record all non-trivial conflict decisions in decision log.

**Exit Criteria:**
- Clean build metadata state and reproducible branch state with documented conflict resolutions.

### Phase P2 - Qwen 3.5 Capability Gap Closure
**Objective:** Close functional gaps required for Qwen 3.5 model support.

**Actions:**
- Validate model registry and architecture mapping coverage for targeted Qwen 3.5 variants.
- Validate tokenizer and chat template behavior for target inference modes.
- Reconcile runtime feature dependencies needed by Qwen 3.5 against preserved `gfx906` paths.
- Introduce compatibility guards where upstream assumptions exceed MI50 capabilities.

**Exit Criteria:**
- Gap list fully resolved or explicitly deferred with rationale and workaround.

### Phase P3 - gfx906 Compatibility Hardening
**Objective:** Ensure upstream-introduced changes do not regress MI50 operability.

**Actions:**
- Audit kernel/backend selection changes affecting ROCm and fallback paths.
- Validate compile-time and runtime feature gating for `gfx906`.
- Define fallback policy for unsupported kernels or operators.
- Add or update targeted regression tests in plan backlog for later implementation mode.

**Exit Criteria:**
- Compatibility checklist complete with no unresolved critical blockers.

### Phase P4 - Validation and Regression Gate
**Objective:** Enforce deterministic go/no-go criteria before advanced upgrades.

**Actions:**
- Run local non-GPU and static validation gates.
- Package remote MI50 validation suite for deferred execution.
- Compare behavior against baseline scenarios and document deltas.

**Exit Criteria:**
- Validation report split by local vs remote with pass/fail disposition and open issues.

### Phase P5 - Forward Compatibility Extensions
**Objective:** Prepare post-parity upgrade path for Triton `gfx906`, `transformers` `^5.2`, and Ubuntu MI50 Docker flow.

**Actions:**
- Define prerequisite checks and incompatibility risks for Triton on `gfx906`.
- Define dependency transition constraints for `transformers` `^5.2`.
- Define Ubuntu MI50 Docker installation plan and validation points.
- Sequence these upgrades behind validated merge baseline.

**Exit Criteria:**
- Approved extension execution order with explicit rollback points.

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation | Trigger/Signal |
|---|---|---|---|---|---|
| R1 | Upstream runtime assumptions exceed MI50 capability | High | Medium | Add capability guards and fallback execution paths; defer non-essential features | Runtime errors on MI50-only paths |
| R2 | Conflict resolution introduces silent behavior drift | High | Medium | Require decision-log entries and targeted regression cases per non-trivial conflict | Unexpected output or API behavior changes |
| R3 | Nightly churn invalidates integration assumptions | Medium | High | Pin nightly checkpoints; integrate in bounded sync windows | Frequent rework from moving upstream target |
| R4 | Missing remote MI50 access delays true validation | High | High | Split local and remote gates; pre-package remote test matrix and scripts | Local green but remote unknown |
| R5 | Dependency uplift breaks ROCm build/runtime | High | Medium | Isolate dependency upgrades into P5 with rollback checkpoints | Build failure or import/runtime incompatibility |
| R6 | Qwen 3.5 support depends on features not yet stable on fork | Medium | Medium | Track minimum required feature set and acceptable temporary workarounds | Incomplete generation or tool-calling behavior |

## Validation Strategy

### Local Workstation Validation
Focus on checks possible without remote MI50 server access.

- Merge integrity checks: branch topology, conflict audit completeness, reproducibility notes.
- Static checks: lint, import validation, configuration schema sanity, targeted unit subsets not requiring MI50.
- Functional smoke checks using CPU or available fallback environments where feasible.
- Artifact readiness for remote execution: scripts, test matrix, expected outputs, log capture format.

### Remote MI50 Server Validation
Execute after local gate is clean and remote access is available.

- Build/install validation on MI50 environment.
- End-to-end inference smoke tests for targeted Qwen 3.5 models.
- ROCm backend and kernel-path verification for `gfx906`.
- Regression suite for previously fixed `gfx906` issues.
- Performance sanity comparison against pre-merge baseline.

## Assumptions
- Upstream `v0.16.1rc0` is the stable merge anchor before broader nightly sync.
- Required Qwen 3.5 capabilities can be decomposed into independently verifiable gaps.
- MI50 remote environment can be used later for execution of prepared validation assets.
- Fork maintainers will preserve compatibility-first priorities when conflicts arise.
- Non-critical feature regressions may be temporarily accepted only with documented follow-up items.

## Decision Log
Record architectural or merge decisions with rationale and consequences.

| Date | Decision ID | Context | Decision | Consequence |
|---|---|---|---|---|
| YYYY-MM-DD | D-001 | Example: upstream conflict in ROCm backend selection | Example: keep fork fallback path and wrap upstream path with capability guard | Example: temporary divergence to preserve MI50 until upstream-compatible fix lands |

## Deliverables From This Plan
- A phase-indexed execution checklist in `docs/todo_gfx906_merge.md`.
- A decision trail for merge and compatibility choices.
- Validation artifacts prepared for local-first then remote-MI50 execution.
