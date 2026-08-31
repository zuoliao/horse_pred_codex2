# Project Handoff

## Operating Rule

The active phase is `Phase 5B: EDA-guided representation / objective research`. Phase 5A is complete, and the human user selected S1 Two-axis past-race value. Preregister and execute only S1 as a four-arm rolling experiment, then stop for human review. Do not run S2 or S3 automatically.

S1 may not use 2024/2025 outcomes or any market information for definition, parameter selection, or acceptance. Keep 2023 in its prior calibration role and outside S1 evaluation. Never commit raw data, cache, models, runner predictions, or recoverable runner-level views.

## Current Goal

Complete S1 definition audit, preregistration, PIT-safe two-axis feature implementation, leakage tests, and C0/C1/C2/C3 rolling comparison. Classify the performance axis, field-quality axis, and joint representation separately, update S2/S3 readiness, and stop.

## Current State

- Phase 5A EDA and all three reviews are complete. Phase 5B S1 is current.
- Frozen controls remain Binary `pv_001_candidate_signed_time_gap` with 254 features and conservative LambdaRank `abl_006_drop_field_relative` with 253 features.
- PV-00 through PV-05 are complete. PV-06's 2022 token gate was inconclusive and is `deferred_by_eda`.
- Margin-aware rating improved standalone calibrated predictions, but PV-04/PV-05/R6 did not establish incremental LightGBM value.
- Margin-rating arithmetic, old recent-opponent runner-relative work, and broad COND-01 are `superseded_by_eda`. Old field-relative restoration and new-horse exclusion are rejected. A1/A2/A3 remain deferred future candidates.
- Phase 5A used 2013 only for warm-up/quality, 2014–2019 for discovery, 2020–2021 for replication, and 2022 for confirmation. It newly accessed neither 2023/2024 targets nor 2025.
- New-horse race exclusion is not supported. Race-class debut and `0_observed_history` are distinct.
- The EDA does not establish production metric, ROI, or profit improvement.
- Priority order is S1 current, S2 queued, S3 queued. They must not be combined.
- S1 controls: Binary PV-01 254 features; LambdaRank lean 253 features. LambdaRank PV-01 254 is point-improved but interval-inconclusive and prospective-only.

## Implemented Artifacts

- EDA contract and protocol: `docs/eda/00_existing_analysis_inventory.md` through `docs/eda/02_eda_protocol.md`.
- Workstream findings: `docs/eda/03_data_quality_and_drift.md` through `docs/eda/11_external_eda_practices.md`.
- Independent review: `docs/eda/12_cross_review.md`; detailed local review records are under `artifacts/eda_20260901/reviews/`.
- Decisions: `docs/eda/13_hypothesis_catalog.md`, `docs/eda/14_eda_synthesis.md`, `docs/eda/15_next_research_roadmap.md`.
- Machine-readable source of truth: `experiments/eda_20260901/`.
- Reproducible runner: `tools/eda/run_phase5a.py`; workstream modules are under `tools/eda/workstreams/`.
- Common loader/CLI: `src/horse_pred/eda.py` and `horse-pred run-eda`.
- Full ignored artifact: `artifacts/eda_20260901/report.html`, aggregate tables, plots, logs, scripts, manifest, and hashes.
- Superseded pre-fix artifact is retained non-destructively at `artifacts/eda_20260901_pre_contract_fix/` and is not canonical.

## Data State

- Approved private raw SHA-256: `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`.
- Full private raw: 629,967 declared runners, 44,761 races, 2013–2025.
- Phase 5A raw-status population: 488,715 rows / 34,504 races through 2022-12-28.
- Flat historical performance: 471,557 rows / 33,240 races. Strict predictive/outcome view: 450,340 rows / 31,689 races.
- Predictive views contain zero obstacle rows and zero target dates after 2022-12-31.
- `market_oracle` is separate from `runner_pre_race`; final odds are joined only in named oracle diagnostics.

## Verification

Independent review checked common/workstream hashes and concluded PASS for PIT/leakage, statistics/validation, and domain semantics. Final repository verification on 2026-09-01: 239 tests passed, Ruff passed, and `compileall` passed. The final ignored artifact manifest/hash, report XML, and tracked-private-file audits are release checks run after the last documentation commit.

## Known Gaps

- 2014 state is left-truncated because only 2013 is available as warm-up.
- The raw class vocabulary drifts over time and does not provide a complete stable grade/condition taxonomy.
- Historical `published_at` is unavailable, so this is retrospective PIT-C rather than snapshot-proven PIT-A.
- Local JRA history omits some external/overseas history and lacks bloodline, training, exact weight/handicap conditions, and reliable T-close snapshots.
- 2022 is confirmation evidence already exposed to research, not an untouched final holdout.
- EDA associations do not prove S1 incremental value after retraining the current model.
- Final market gaps cannot identify the causal contribution of missing data versus model, target, calibration, or selection effects.

## Next Tasks

1. Reconcile the post-EDA queue and commit the Phase 5B state.
2. Audit/fix one performance residual and one race-constant pre-race field-quality definition using discovery data only; preregister fixed folds and gates.
3. Implement the two 90-day historical state columns and the ten required PIT/invariance tests.
4. Run C0/C1/C2/C3 for both frozen families through 2022 only; save aggregate evidence and ignored full artifacts.
5. Update S2/S3 priority/readiness from the S1 result, stop, and request human selection.

## Useful Commands

```bash
UV_CACHE_DIR=/tmp/horse-pred-uv-cache uv run python tools/eda/run_phase5a.py \
  --raw-path /path/to/race_results_merged.csv \
  --output artifacts/eda_YYYYMMDD \
  --max-date 2022-12-31 \
  --rolling-predictions artifacts/eval_roll_001_current_best_20260831/predictions_scoring.csv.gz
UV_CACHE_DIR=/tmp/horse-pred-uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/horse-pred-uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/horse-pred-uv-cache uv run python -m compileall -q src tests tools
git status --short --branch
```

## Handoff Notes

S1 asks whether past-race value should be split into horse performance and race-constant field quality. The latter must be identical for every runner in a race and use only frozen pre-race strength; a leave-one-out opponent mean is not field quality. Do not add interactions, alternate windows, uncertainty, or condition-specific variants. S2/S3 remain unauthorized until S1 is reported.
