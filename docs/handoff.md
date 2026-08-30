# Project Handoff

## Operating rule

Read `README.md`, `AGENTS.md`, this file, and relevant `docs/` before changing model, feature, or split behavior. Preserve one interpretable hypothesis per experiment, strict PIT semantics, no odds in the primary model, and 2026+ prospective final. Raw data, caches, model binaries, runner predictions, and market artifacts stay outside Git.

## Current state

The corrected LightGBM baseline validation and standalone rating-module R0–R6 are complete. This includes data/evaluation health, 2024 uncertainty, semantic ablation, diagnostics, five limited improvements, training-period-only rating selection, frozen PIT rating generation, and LightGBM integration evaluation.

- Primary selection period: 2024 development, 3,051 races / 41,946 runners / 106 dates.
- 2025: 0 rows used for hypothesis, feature, calibration, model, or accept/reject decisions. Cached retrospective rows are removed immediately after load.
- Corrected model frame: 533,853 runners / 37,889 races / 268 default features.
- Baseline Binary and LambdaRank both stably beat Uniform/History-rate. Their family difference remains unresolved.
- Current best feature config: `abl_006_drop_field_relative`, 253 features.
- Point-estimate best model: LambdaRank on that config (NDCG@3 .4924, Log Loss 2.0847), but Binary has better Top-1/Brier and paired family intervals cross zero.
- Surface-conditioned Elo adds a supported LambdaRank ranking signal relative to the corrected 268-feature baseline, but adding the same 3-column family to the lean 253-feature config was rejected for both families in IMP-004.
- IMP-005 added one 90-day decayed global-Elo expected-vs-actual race-value. Binary was inconclusive with all point metrics worse; LambdaRank was rejected. The feature was highly redundant with 90-day mean finish and is not adopted.
- R0 reproduced the existing global Elo on all 489,674 pre-2025 scored runners exactly at the stored float32 contract.
- R2 selected global pairwise Elo K=48/scale=200 using only annual 2018–2021 scores. R3 top-choice PL and R4 surface blending were rejected on 2022. R5 froze the selected global spec and fitted temperature 0.5187 on 2023 only.
- R5 standalone 2024 calibrated metrics: NDCG@3 .3533, Top-1 .1693, Log Loss 2.4015, Brier .89459, ECE .00774.
- R6 added the frozen five-column rating group to the lean LightGBM. Binary was rejected on the Brier guardrail; LambdaRank NDCG significantly worsened. The module remains a standalone baseline but is not in the current best LightGBM.
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

Local complete artifacts are under `artifacts/` and intentionally ignored. Rating artifacts are `artifacts/rating_module_r0_r5_20260830/`, `artifacts/r6_*`; the augmented cache is `data/model_frame_rating_module_r6_20260830.pkl` (273). Existing corrected/surface/race-value caches remain local and ignored.

## Improvement outcomes

1. `imp_001_compact_form_relative`: reject. Adding one 90-day form race-percentile to the 253-feature drop control significantly worsened NDCG and Log Loss in both families.
2. `imp_002_surface_conditioned_elo`: Binary inconclusive; LambdaRank ranking path accepted. NDCG improvement +.00364 with 95% interval `[+.00005,+.00714]`, with proper scores non-worse. Cache control matched all old 268 values exactly.
3. `imp_003_field_size_band_temperature`: reject. Ranking was exactly unchanged and ECE improved, but Log Loss/Brier worsened for both families; medium-field Log Loss significantly worsened.
4. `imp_004_lean_surface_conditioned_rating`: reject in both families. Binary NDCG improved +.00299 but its Brier guardrail failed; LambdaRank NDCG and Log Loss worsened. Individually supported changes were not complementary.
5. `imp_005_expected_actual_race_value`: Binary inconclusive, LambdaRank reject, not adopted. Binary NDCG/LL improvements were −.00158/−.00172; Ranker −.00392/−.00299. The new feature correlated −.938 Spearman with 90-day mean finish while taking 28.7%/42.6% of Binary/Ranker gain.

These are nominal intervals across five hypotheses and two model families. They retain 2024 selection optimism and require prospective confirmation.

## Rating module R0–R6 outcome

- Protocol/config: `docs/experiments/rating_module_r0_r6.md`, `configs/rating/rating_module_r0_r6.json`.
- Tracked summary: `experiments/rating_module_20260830/summary.json`.
- Frozen module: global pairwise Elo, initial 1500, K 48, scale 200, no surface blend. Algorithm/spec is fixed; state continues chronological race-by-race updates.
- R6 Binary improvement: NDCG +.00207 `[-.00352,+.00757]`, LL +.00042 `[-.00603,+.00707]`, Brier −.00121; reject.
- R6 Rank improvement: NDCG −.00430 `[-.00808,−.00047]`, LL −.00444; reject.
- Current best remains `abl_006_drop_field_relative`, 253 features, with Binary/LambdaRank both retained.

## Exact next task

Do not automatically retune or add more rating variants against 2024. R0–R6 is complete and the latest user goal has been achieved.

If the user authorizes another feature hypothesis, the cleanest untested candidate remains:

> A 90-day decayed opponent-only pre-race field-strength feature, separate from outcome residuals.

Before that experiment, correct the concept and naming: current `horse_history__career__mean_opponent_elo` stores a self-inclusive field mean. Define opponent-only values, add equal/unequal-rating PIT fixtures, preserve default 268 columns, and keep the candidate one-column opt-in. Because 2024 has already supported many selections, consider freezing further 2024 iteration and waiting for 2026+ prospective data. Do not move to data expansion, DNN, purchase strategy, or UI without a new phase decision.

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

# Rebuild opt-in expected-actual race-value cache (long-running)
uv run horse-pred run-mvp \
  --raw-path /path/to/race_results_merged.csv \
  --output artifacts/race_value_cache_control \
  --model-frame-cache data/model_frame_race_value_surprise.pkl \
  --expected-actual-race-value
```

## Verification baseline

- Latest preregistered run commit: `874eda938bddfc760282bc26bdc438a877e72e57`, `dirty=false`.
- Raw SHA-256: `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`.
- Surface-cache control: 533,853 rows; old 268 column names/order/values/NaN positions exact; mismatch count 0; max absolute difference 0.
- Race-value-cache control: 533,853 rows; old 268 column names/order/values/NaN positions exact; mismatch count 0; max absolute difference 0. The 253-feature control reproduced `abl_006` with prediction/metric max difference 0.
- IMP-005 full local comparison: `artifacts/imp_005_primary_comparison/comparison.json`; tracked aggregate: `experiments/rating_race_value_20260830/imp_005_summary.json`.
- R6 cache control: 533,853 rows; old 268 columns exact; new columns 5; 2025 nonmissing 0. The R6 control reproduced `abl_006` exactly.
- Formal R0–R5 run commit: `ce6d5ccb465d7462eeb3872887fa19891a1e732f`, dirty=false. R6 run commit: `3d530258dbe51d985f54f5460e4824cbf4ff80a0`, dirty=false.
- Latest verification before final documentation: 109 tests passed and Ruff passed. Rerun after documentation changes before handoff.
