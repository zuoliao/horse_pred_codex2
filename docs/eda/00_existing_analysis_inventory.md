# Existing Analysis Inventory

## Scope and status

This inventory freezes the evidence available before Phase 5A. Existing 2024-only results are not rerun or reopened. Dates used by new target-aware EDA stop at 2022-12-31.

| Analysis question | Existing coverage | Existing evidence | Missing analysis | New work required |
|---|---|---|---|---|
| Data health | Raw schema, coverage, flat/jump population, outcome status, known missing races | `baseline_data_health_20260830.md`; 629,967 declared runners, 44,761 races, 2013–2025 raw coverage | 2014–2022 structural vs accidental missingness, unit/category drift | Workstream A, without opening later targets |
| Missing race audit | Official comparison and known shortfall completed | 146 known missing races; 2024 limitation recorded | Temporal representativeness within EDA years | Year/venue/surface coverage tables |
| Semantic ablation | Seven groups retrained on 2024 | field-relative 15-column removal improved both families; connections strong | Multi-year direction, conditional redundancy, feature clustering | Reframe on pre-2023 evidence |
| SHAP/permutation | Completed on corrected 2024 baseline | Model use and retrained utility differ | Temporal stability and dependence limitations | Shallow/pre-2023 diagnostics only |
| Race-condition errors | 2024 slices completed | Weak conditions documented | Replication and transition-specific persistence | Workstreams G/H using pre-2023 OOT only |
| Binary vs LambdaRank | 2024 paired uncertainty plus rolling studies | Family difference unresolved | OOT disagreement taxonomy and target-information analysis | Workstreams B/H |
| Rating R0–R6 | Standalone and LightGBM integration | margin-aware actual improves calibrated standalone rating; frozen rating columns not adopted | Separate field quality from realized performance | Workstream E |
| PV-00–PV-05 | Clock audit, signed clock-gap, margin-aware rating and transforms | PV-01 Binary accepted; PV-02 raw rejected; PV-03 standalone supported; PV-04/05 not adopted | Time/margin semantics and target suitability | Workstreams B/D; no production mapping |
| PV-06 | Train-only token audit and 2022 gate | Equal-clock refinement inconclusive; 2024 unopened | Broader token reliability description | Workstream D only; experiment paused |
| PACE/SPEED | Completed rolling experiments retained as archived evidence | Early-position and prequential speed histories showed signal; later pace variants mixed | Systematic content loss, decay and interaction analysis | Workstreams C/D/G; no tuning |
| Market oracle | Final odds 2024 diagnostic completed and isolated | Oracle advantage quantified | Pre-2023 OOT gap decomposition | Workstream H, oracle-only join |
| Feature dictionary | Prefix allowlist, timestamp semantics, schema sidecars | IDs excluded; same-date emit-before-update implemented | Logical views and availability/schema hashes | Common EDA contract |
| Rolling origin | Four expanding 2020–2023 folds | Later selection optimism reduced | Phase 5A partitions and diagnostic folds | EDA cutoff and prediction contract |
| New-horse population | Binary fit exclusion ablation | Exclusion worsened all-race performance | Learning curve by history availability | Workstream C; do not repeat exclusion test |

## Inventory decision

**Observation:** the repository has strong experiment governance and several local signals, but much original diagnosis is 2024-centric or hypothesis-specific.

**Interpretation:** the missing layer is a common, temporally replicated account of data structure, targets, information loss, and residuals—not another one-column test.

**Hypothesis policy:** all Phase 5A ideas enter the registry. None becomes a production feature until the user reviews the final roadmap.
