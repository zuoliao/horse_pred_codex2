# Project Handoff

## Operating Rule

The active phase is `Phase 5B: EDA-guided representation / objective research`. Phase 5A and S1 Two-axis past-race value are complete. The human selected S3 Condition-adjusted performance target; S3 is the only current experiment. Do not run S2 or feature follow-ups automatically after it.

S1 used no 2023/2024/2025 rows or market information. Preserve this firewall for any follow-up selection work. Never commit raw data, cache, models, runner predictions, or recoverable runner-level views.

## Current Goal

Preregister and execute S3 condition-adjusted performance target because S1 supported performance residual. Keep S2 race-wise probability queued and stop for human review after S3.

## Current State

- Phase 5A EDA, all three reviews, and Phase 5B S1 are complete.
- Frozen controls remain Binary `pv_001_candidate_signed_time_gap` with 254 features and conservative LambdaRank `abl_006_drop_field_relative` with 253 features.
- PV-00 through PV-05 are complete. PV-06's 2022 token gate was inconclusive and is `deferred_by_eda`.
- Margin-aware rating improved standalone calibrated predictions, but PV-04/PV-05/R6 did not establish incremental LightGBM value.
- Margin-rating arithmetic, old recent-opponent runner-relative work, and broad COND-01 are `superseded_by_eda`. Old field-relative restoration and new-horse exclusion are rejected. A1/A2/A3 remain deferred future candidates.
- Phase 5A used 2013 only for warm-up/quality, 2014–2019 for discovery, 2020–2021 for replication, and 2022 for confirmation. It newly accessed neither 2023/2024 targets nor 2025.
- New-horse race exclusion is not supported. Race-class debut and `0_observed_history` are distinct.
- The EDA does not establish production metric, ROI, or profit improvement.
- S1 performance is `supported`; standalone field quality is `inconclusive`; joint C3 is `supported`. Field quality conditional on performance is Binary weak but LambdaRank rejected, so two independent axes are not established.
- S3 is `current / authorized`; S2 remains `queued`. They must not be combined.
- S1 controls: Binary PV-01 254 features; LambdaRank lean 253 features. LambdaRank PV-01 254 is point-improved but interval-inconclusive and prospective-only.
- Formal controls remain unchanged pending human review. Binary C1/C3 and LambdaRank C1/C3 are rolling candidates, not final-holdout winners.

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
- S1 preregistration: `docs/experiments/s1_two_axis_race_value_preregistration.md`.
- S1 conclusion: `docs/experiments/s1_two_axis_race_value_20260901.md`.
- S1 machine summaries: `experiments/s1_two_axis_race_value_20260901/`.
- S1 implementation: `src/horse_pred/two_axis_race_value.py`, `src/horse_pred/s1_two_axis_study.py`, and CLI `horse-pred run-s1-two-axis-study`.
- Full ignored S1 artifact: `artifacts/s1_two_axis_race_value_20260901/` with models, scoring predictions, diagnostics, aggregate tables, and 38-file manifest.

## Data State

- Approved private raw SHA-256: `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`.
- Full private raw: 629,967 declared runners, 44,761 races, 2013–2025.
- Phase 5A raw-status population: 488,715 rows / 34,504 races through 2022-12-28.
- Flat historical performance: 471,557 rows / 33,240 races. Strict predictive/outcome view: 450,340 rows / 31,689 races.
- Predictive views contain zero obstacle rows and zero target dates after 2022-12-31.
- `market_oracle` is separate from `runner_pre_race`; final odds are joined only in named oracle diagnostics.
- S1 source isolation retained 488,715 rows / 34,504 races through 2022-12-28 and excluded 47,672/45,696/47,884 rows from 2023/2024/2025 before normalization.
- S1 model frame contained 403,855 eligible runners / 28,498 races. Market columns were removed before feature resolution.

## Verification

S1 completed 24 fits over evaluation years 2020–2022. Its artifact manifest verified all 38 files. Field quality had zero race-constant violations; condition fits used zero early-stopping/calibration/evaluation rows; 2023/2024/2025 and odds usage were zero. Before the final documentation commit, 259 repository tests passed; rerun the final gates after this commit because S1 added further tests.

## Known Gaps

- 2014 state is left-truncated because only 2013 is available as warm-up.
- The raw class vocabulary drifts over time and does not provide a complete stable grade/condition taxonomy.
- Historical `published_at` is unavailable, so this is retrospective PIT-C rather than snapshot-proven PIT-A.
- Local JRA history omits some external/overseas history and lacks bloodline, training, exact weight/handicap conditions, and reliable T-close snapshots.
- 2022 is confirmation evidence already exposed to research, not an untouched final holdout.
- S1 intervals condition on fitted models and do not include model-refit uncertainty.
- Field quality is highly redundant with existing historical field-strength features and did not work standalone.
- C3's advantage over C1 is family-dependent; it does not establish two independently useful axes.
- Final market gaps cannot identify the causal contribution of missing data versus model, target, calibration, or selection effects.

## Next Tasks

1. Preregister S3 before reading its model metrics; freeze target, folds, controls, probability mapping, and acceptance logic.
2. Implement and run Huber regression on the paired Binary-254 and LambdaRank-253 feature scopes without adding S1 feature columns.
3. Record rolling evidence through 2022 only, update all project state, and stop for human review.
4. Keep S2 queued and A1 transition reliability, A2 connection compression, and A3 last3F relative deferred.

## Useful Commands

```bash
UV_CACHE_DIR=/tmp/horse-pred-uv-cache uv run python tools/eda/run_phase5a.py \
  --raw-path /path/to/race_results_merged.csv \
  --output artifacts/eda_YYYYMMDD \
  --max-date 2022-12-31 \
  --rolling-predictions artifacts/eval_roll_001_current_best_20260831/predictions_scoring.csv.gz
UV_CACHE_DIR=/tmp/horse-pred-uv-cache uv run horse-pred run-s1-two-axis-study \
  --raw-path /path/to/race_results_merged.csv \
  --output artifacts/s1_two_axis_race_value_YYYYMMDD \
  --preregistration experiments/s1_two_axis_race_value_20260901/preregistration.json
UV_CACHE_DIR=/tmp/horse-pred-uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/horse-pred-uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/horse-pred-uv-cache uv run python -m compileall -q src tests tools
git status --short --branch
```

## Handoff Notes

S1 found that condition-adjusted historical performance is the stable incremental signal. Field quality alone failed and its conditional increment conflicted by family. Do not promote C3 as proof of a two-axis encoder, tune S1 windows, or add interactions from this result. S3 is now authorized; S2 remains unauthorized until the post-S3 human decision.
