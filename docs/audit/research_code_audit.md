# Research code growth audit

## Findings

The program is reproducible but experiment orchestration has accumulated duplication. Representative files alone contain 6,550 lines: `cli.py` 1,033; S1/S2/S3 runners 1,081/816/592; rolling evaluation 825; cached experiment 611; and four recent history builders 1,592.

| Area | Evidence | Risk |
|---|---|---|
| Experiment runners | S2/S3 import private helpers from S1; fold, metric, bootstrap and artifact logic repeat | fixes in one runner do not automatically propagate |
| Feature builders | sectional, pace, opponent and speed each implement variants of same-date batching, decay state and cache join | timestamp semantics may drift |
| Market oracle | inverse-odds normalization exists in pipeline, diagnostics and EDA code | population and probability rules can diverge |
| CLI | one 1,033-line parser/dispatch with more than twenty commands | growing merge and discovery cost |
| Artifacts | runner predictions/features/models copied across experiment directories; S2 local artifact is about 1.3GB | storage and identity ambiguity |
| Documentation | “formal control” and “development incumbent” diverged after SPEED-01 | agents can resume the wrong branch |
| EDA | some workstreams contain standalone execution and duplicate bootstrap/market logic | library tests do not cover all paths uniformly |

## Small common contracts proposed

1. read-only model-registry resolver using family + ordered feature hash + config hashes;
2. immutable deployment-bundle manifest;
3. common source-isolation/fold-role contract;
4. common choice-set and probability integrity adapter;
5. public complete-race inverse-odds normalization;
6. shared race-metric/date-block bootstrap/report schema;
7. generic time-decayed history state with an explicit emit-before-update hook;
8. command modules registered into a smaller CLI dispatcher;
9. content-addressed local artifact references instead of repeated large files.

## Phase 5C boundary

No large refactor is performed now. The new historical oracle uses existing public metrics and a strict artifact adapter, but it does not reorganize old runners. Before touching model code, prioritize only contracts needed for prospective activation: registry resolution, deployment bundle, as-of snapshots, health ledger, and shadow record schema. Large feature-builder or runner refactors wait until the prospective path is operational and covered by fixtures.
