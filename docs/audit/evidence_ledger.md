# Historical evidence ledger

## Policy

Every year from 2013 through 2025 is now classified as `development_archive`. No historical year is an untouched final holdout. Final validation starts only with 2026-or-later data collected and scored prospectively under the frozen protocol.

This is stricter than saying that a particular experiment used zero rows from a year. Once targets, market values, coverage, or metrics have been viewed anywhere in the research program, later isolation does not restore that year's final-holdout status.

## Year-level ledger

| Year | Roles already used | Important exposures |
|---:|---|---|
| 2013 | state warm-up; coverage/quality; PV-01 M0 audit | left-truncated historical state inspected |
| 2014 | train; EDA discovery; rating/normalizer fit; S1 definition audit | start of most expanding histories |
| 2015 | train; discovery; missing-race audit | outcomes and data quality inspected |
| 2016 | train; discovery; rating/feature-definition fit | development inputs and outcomes exposed |
| 2017 | train; discovery; missing-race audit; PV-03 tau definition | selection-related definition work |
| 2018 | train; PV-01 representation selection; rolling early stopping | already selection-exposed |
| 2019 | train; final discovery year; rolling eval/calibration/ES; S1 definition | multiple roles across folds |
| 2020 | train; EDA replication; rolling eval; later-fold ES/cal/train; S1/S2/S3 eval | heavily reused outcome year |
| 2021 | train; EDA replication; rolling eval; later-fold ES/cal; S1/S2/S3 eval | heavily reused outcome year |
| 2022 | baseline ES/validation; EDA confirmation; PV/GR gates; rolling and S1/S2/S3 eval | repeated confirmation and selection |
| 2023 | baseline calibration; rolling eval/confirmation; PV-03 calibration; M0 audit | Phase 5A/S1–S3 excluded it, but earlier work did not |
| 2024 | corrected development; ablation/SHAP/error; rating/PV/SPEED milestones; final-market oracle | repeatedly examined; known coverage limitation |
| 2025 | explicit Task16 retrospective metrics and final-market oracle; data audit | 3,236 races/44,874 runners viewed; supersession does not reseal it |

Rolling-origin use is many-to-many: the same calendar year may be train in one fold, early stopping in another, calibration in another, and evaluation in another. The machine ledger therefore records multiple roles rather than assigning one label per year.

## Program implications

- “2025 unused final” and similar wording is prohibited.
- 2023–2025 zero-row assertions remain useful leakage checks for a specific run, but not evidence of program-level holdout status.
- Historical block intervals quantify sampling variation conditional on the frozen fits; they do not correct research selection optimism.
- A prospective result must include snapshot timestamps and a prediction receipt created before the outcome. A later replay of 2026 data is not equivalent.

The machine-readable source is `experiments/program_audit_20260901/evidence_ledger.json`.
