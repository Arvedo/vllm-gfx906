# TODO Checklist for gfx906 Merge Execution

This checklist is phase-aligned 1:1 with the strategy in `docs/merge_plan_gfx906.md`.

## P0 - Baseline Inventory and Delta Mapping
- [ ] **P0.1 Build fork-vs-upstream delta matrix**
  - Acceptance criteria:
    - All fork-specific commits are enumerated and grouped by subsystem.
    - Each entry has disposition: keep, upstream-equivalent, drop, or redesign.
    - Coverage check confirms no uncategorized fork-only patch remains.

- [ ] **P0.2 Define upstream anchors**
  - Acceptance criteria:
    - `v0.16.1rc0` anchor commit is pinned.
    - Selected nightly reference commit is pinned.
    - Anchor selections are documented with rationale and retrieval commands.

## P1 - Upstream Alignment Foundation
- [ ] **P1.1 Create integration branch and perform base merge/rebase**
  - Acceptance criteria:
    - Integration branch exists and is reproducible from documented refs.
    - Merge/rebase completes with conflict list captured.
    - No unresolved conflict markers remain.

- [ ] **P1.2 Resolve conflicts with gfx906 compatibility-first policy**
  - Acceptance criteria:
    - Every non-trivial conflict has a decision-log entry.
    - Critical ROCm/gfx906 paths are explicitly reviewed during conflict resolution.
    - Post-resolution sanity checks pass for repository integrity.

## P2 - Qwen 3.5 Capability Gap Closure
- [ ] **P2.1 Validate model and tokenizer readiness for target Qwen 3.5 variants**
  - Acceptance criteria:
    - Target Qwen 3.5 variants are listed with required runtime features.
    - Model registry and tokenizer/chat-template handling are verified against targets.
    - Any unsupported variant is recorded with reason and workaround/defer decision.

- [ ] **P2.2 Close or defer capability gaps with traceable outcomes**
  - Acceptance criteria:
    - Gap list is reduced to zero critical unresolved items.
    - Deferred items include blocker, mitigation, and phase for revisit.
    - Validation notes demonstrate expected functional behavior for closed gaps.

## P3 - gfx906 Compatibility Hardening
- [ ] **P3.1 Audit backend/kernel selection for MI50 safety**
  - Acceptance criteria:
    - ROCm backend selection paths relevant to gfx906 are reviewed.
    - Unsupported kernel/operator cases have defined fallback behavior.
    - No known critical incompatibility remains unmitigated.

- [ ] **P3.2 Define targeted regression coverage for gfx906-sensitive areas**
  - Acceptance criteria:
    - Regression matrix identifies tests needed for prior breakpoints.
    - Each test target has expected result and failure signal definition.
    - Backlog entries are ready for implementation mode without ambiguity.

## P4 - Validation and Regression Gate
- [ ] **P4.1 Execute local validation gate**
  - Acceptance criteria:
    - Local static checks and non-MI50 tests run with recorded outcomes.
    - Known limitations from missing remote MI50 access are explicitly listed.
    - Local gate disposition is documented as pass/fail with evidence links.

- [ ] **P4.2 Package and hand off remote MI50 validation suite**
  - Acceptance criteria:
    - Remote test matrix includes build, inference smoke, and regression checks.
    - Required scripts/commands, expected outputs, and log capture format are documented.
    - Go/no-go criteria for remote sign-off are explicit and measurable.

## P5 - Forward Compatibility Extensions
- [ ] **P5.1 Plan Triton gfx906 path after parity baseline**
  - Acceptance criteria:
    - Triton-on-gfx906 prerequisite and incompatibility list is documented.
    - Entry and rollback criteria are defined for Triton enablement attempts.
    - Dependencies on earlier phases are explicitly referenced.

- [ ] **P5.2 Plan Transformers ^5.2 and Ubuntu MI50 Docker path**
  - Acceptance criteria:
    - Dependency transition constraints for `transformers` `^5.2` are documented.
    - Ubuntu MI50 Docker installation flow is drafted with validation checkpoints.
    - Sequencing confirms these upgrades occur only after P4 sign-off.

## Cross-Phase Control Items
- [ ] **C1 Maintain risk register updates per phase**
  - Acceptance criteria:
    - Each phase updates risk status, trigger, and mitigation state.
    - New risks are added with owner and containment action.

- [ ] **C2 Maintain decision log updates per non-trivial merge decision**
  - Acceptance criteria:
    - Decision entries include context, chosen option, and consequence.
    - Decision IDs are referenced from relevant phase artifacts.
