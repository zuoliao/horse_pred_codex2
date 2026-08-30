# Project Handoff

## Operating rule

Read `README.md`, `AGENTS.md`, this file, and relevant `docs/` before changing model, feature, or split behavior. Preserve one interpretable hypothesis per experiment, strict PIT semantics, no odds in the primary model, and 2026+ prospective final. Raw data, caches, model binaries, runner predictions, and market artifacts stay outside Git.

## Current state

The corrected LightGBM baseline validation round is complete through data/evaluation health, 2024 uncertainty, semantic ablation, SHAP/permutation diagnostics, conditional errors, and three limited improvement experiments.

- Primary selection period: 2024 development, 3,051 races / 41,946 runners / 106 dates.
- 2025: 0 rows used for hypothesis, feature, calibration, model, or accept/reject decisions. Cached retrospective rows are removed immediately after load.
- Corrected model frame: 533,853 runners / 37,889 races / 268 default features.
- Baseline Binary and LambdaRank both stably beat Uniform/History-rate. Their family difference remains unresolved.
- Current best feature config: `abl_006_drop_field_relative`, 253 features.
- Point-estimate best model: LambdaRank on that config (NDCG@3 .4924, Log Loss 2.0847), but Binary has better Top-1/Brier and paired family intervals cross zero.
- Surface-conditioned Elo adds a supported LambdaRank ranking signal relative to the corrected 268-feature baseline, but is not the overall best point config.
- Final odds remain an oracle-only diagnostic. No executable ROI or betting claim exists.

## Critical correction and population limits

The original Task 16 run incorrectly included races whose `course_type` was turf/dirt but whose class was obstacle. Commit `810a003` fixed race-level flat classification and prevented obstacle results from updating flat horse/jockey/trainer/Elo state. Do not use the old `mvp_task16_20260830` metrics for current decisions.

Known selection limits:

- 2024 missing: 108 races concentrated in nine days of the 7th Kyoto meeting; 103 are flat. This is structural, not MCAR.
- 2024 official flat coverage: raw 96.90%; strict scored population 91.70%.
- 173/3,224 raw 2024 flat races contain a scratch/exclusion and are excluded as whole races because event timestamps are unavailable.
- The baseline is internally trustworthy for the observed strict PIT-C population, but external validity for full JRA/live populations is only moderate.

## Completed evidence

- Data health: `docs/experiments/baseline_data_health_20260830.md`
- Block uncertainty: `docs/experiments/baseline_uncertainty_20260830.md`
- Semantic ablation: `docs/experiments/semantic_feature_ablation_20260830.md`
- Importance/error diagnostics: `docs/experiments/baseline_diagnostics_20260830.md`
- Improvement preregistration/results: `docs/experiments/improvement_experiments_20260830.md`
- Integrated decision report: `docs/experiments/baseline_validation_conclusions_20260830.md`
- Tracked machine source of truth: `experiments/baseline_validation_20260830/`

Local complete artifacts are under `artifacts/` and intentionally ignored. Local model-frame caches are `data/model_frame_20260830_corrected.pkl` (268) and `data/model_frame_surface_elo_20260830.pkl` (271); both are ignored.

## Improvement outcomes

1. `imp_001_compact_form_relative`: reject. Adding one 90-day form race-percentile to the 253-feature drop control significantly worsened NDCG and Log Loss in both families.
2. `imp_002_surface_conditioned_elo`: Binary inconclusive; LambdaRank ranking path accepted. NDCG improvement +.00364 with 95% interval `[+.00005,+.00714]`, with proper scores non-worse. Cache control matched all old 268 values exactly.
3. `imp_003_field_size_band_temperature`: reject. Ranking was exactly unchanged and ECE improved, but Log Loss/Brier worsened for both families; medium-field Log Loss significantly worsened.

These are nominal intervals across three hypotheses and two model families. They retain 2024 selection optimism and require prospective confirmation.

## Exact next task

Preregister one independent experiment:

> Control = `abl_006_drop_field_relative` (253 features). Candidate = the same config plus only the surface-conditioned rating family.

This tests whether the two individually supported changes are complementary. Do not assume they compose. Use the same 2024 4-date moving-block bootstrap (10,000), do not inspect 2025, and keep Binary/LambdaRank results separate. The experiment must be a new config/artifact/commit and must not bundle rating formula tuning.

If that experiment is not authorized, stop at the current validated reference rather than moving to data expansion, DNN, purchase strategy, or UI.

## Useful commands

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run python -m compileall -q src tests

# Corrected baseline uncertainty
uv run horse-pred analyze-uncertainty \
  --predictions artifacts/mvp_baseline_20260830_corrected/predictions.csv.gz \
  --output artifacts/baseline_uncertainty

# Registered cached experiment; 2025 is removed by the runner
uv run horse-pred run-cached-experiment \
  --cache data/model_frame_20260830_corrected.pkl \
  --config configs/ablations/abl_006_drop_field_relative.json \
  --output artifacts/example_ablation

# Rebuild opt-in surface Elo cache (long-running)
uv run horse-pred run-mvp \
  --raw-path /path/to/race_results_merged.csv \
  --output artifacts/surface_cache_control \
  --model-frame-cache data/model_frame_surface_elo.pkl \
  --surface-conditioned-elo
```

## Verification baseline

- Improvement run commit: `e3bdbf406f933f4fda69245f285be93e9835d321`, `dirty=false`.
- Raw SHA-256: `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`.
- Surface-cache control: 533,853 rows; old 268 column names/order/values/NaN positions exact; mismatch count 0; max absolute difference 0.
- Latest verification before final documentation: 96 tests passed and Ruff passed. Rerun after documentation changes before handoff.
