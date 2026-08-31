# Project Handoff

## Operating Rule

`Phase 5A: systematic exploratory data analysis and problem reformulation` is complete. Do not automatically resume PV-06, margin-token refinement, rating transformations, COND-01, other local feature work, data collection, betting optimization, or UI work. These tracks are paused, not rejected. Wait for the human user to choose one of the three experiments in `docs/eda/15_next_research_roadmap.md`.

No follow-up may use new 2024 evidence for hypothesis/parameter selection or use 2025. Keep 2023 in its prior calibration role. Final odds remain a physically isolated market-oracle diagnostic. Never commit raw data, cache, models, runner predictions, or recoverable runner-level views.

## Current Goal

Wait for human selection among the three Phase 5A priorities. When one is selected, preregister exactly one interpretable experiment and use the frozen common PIT contract and rolling-origin evaluation. Do not infer a selection from the roadmap order alone.

## Current State

- Phase 5A EDA, all workstreams A–I, and PIT/statistics/domain cross-review are complete. All three final review verdicts are PASS.
- Frozen controls remain Binary `pv_001_candidate_signed_time_gap` with 254 features and conservative LambdaRank `abl_006_drop_field_relative` with 253 features.
- PV-00 through PV-05 are complete. PV-06's 2022 token gate was inconclusive; continuation and mapping refinement remain paused.
- Margin-aware rating improved standalone calibrated predictions, but PV-04/PV-05/R6 did not establish incremental LightGBM value.
- Later PACE/SPEED results remain archived reproducible evidence and do not authorize local feature search.
- Phase 5A used 2013 only for warm-up/quality, 2014–2019 for discovery, 2020–2021 for replication, and 2022 for confirmation. It newly accessed neither 2023/2024 targets nor 2025.
- New-horse race exclusion is not supported. Race-class debut and `0_observed_history` are distinct.
- The EDA does not establish production metric, ROI, or profit improvement.
- The only prioritized next hypotheses are:
  1. `EDA-S01-RACE-VALUE-2AXIS`
  2. `EDA-S02-RACEWISE-CHOICE`
  3. `EDA-S03-PERFORMANCE-TARGET`

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
- EDA associations do not prove incremental value after retraining the current model.
- Final market gaps cannot identify the causal contribution of missing data versus model, target, calibration, or selection effects.

## Next Tasks

1. Wait for human selection of S1, S2, or S3; do not execute any of them automatically.
2. After selection, preregister one experiment, one transform/target/objective, its fixed rolling folds, guardrails, and multiplicity count.
3. Reuse the Phase 5A PIT contract and frozen Binary/LambdaRank controls.
4. Record negative or inconclusive results machine-readably and return for review before another hypothesis.

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

The main new conclusion is not that one engineered column won. Stable information exists in decayed horse history, connections, and same-condition persistence, while the larger unresolved gaps concern how a past race is represented and what the objective teaches. Preserve the separation between observation, interpretation, and hypothesis. The roadmap order is a recommendation for human review, not permission to start S1 automatically.
