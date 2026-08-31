# Project Handoff

## Operating rule

Read `README.md`, `AGENTS.md`, this file, and relevant `docs/` before changing model, feature, or split behavior. Preserve one interpretable hypothesis per experiment, strict PIT semantics, no odds in the primary model, and 2026+ prospective final. Raw data, caches, model binaries, runner predictions, and market artifacts stay outside Git.

## Current state

The corrected LightGBM baseline validation, standalone rating-module R0–R6,
time/margin stages PV-00 through PV-06, graded-label experiment GR-001, the
S0/S1 rolling program, and PACE-01 through PACE-04 are complete. This includes data/evaluation health, 2024
uncertainty, semantic
ablation, diagnostics, training-period rating selection, frozen PIT rating
generation, the accepted PV-01 time-history feature, margin-aware actual and
calibration studies, two one-column integrations, raw margin-token clock
refinement, and one independent LambdaRank-label test.

- Historical baseline selection period: 2024 development, 3,051 races / 41,946 runners / 106 dates. New hypotheses use rolling-origin folds first; 2024 is milestone-only and 2025 is not used for iterative selection.
- 2025: 0 rows used for hypothesis, feature, calibration, model, or accept/reject decisions. Cached retrospective rows are removed immediately after load.
- Corrected model frame: 533,853 runners / 37,889 races / 268 default features.
- Baseline Binary and LambdaRank both stably beat Uniform/History-rate. Their family difference remains unresolved.
- The 2024-confirmed Binary development incumbent is `pv_001_candidate_signed_time_gap`, 254 features. NDCG@3 .4976, Top-1 .2965, Log Loss 2.0787, Brier .82767. The accepted pre-2024 rolling candidate is PV-01 plus PACE-01, 255 features; no 2024 milestone was opened.
- The conservative 2024 LambdaRank reference remains `abl_006_drop_field_relative`, 253 features. The accepted pre-2024 rolling candidate is lean plus SEC-3F plus PACE-01, 255 features; no 2024 milestone was opened.
- EVAL-ROLL now provides four expanding 2020--2023 folds. Current rolling macro is Binary NDCG .47045 / LL 2.14231 and lean Rank NDCG .46746 / LL 2.15388. These are screening years with prior exposure, not untouched final holdouts.
- OPP-RECENT was not adopted: Binary was inconclusive; Rank Log Loss worsened by .00393 `[-.00590,-.00195]`.
- SEC-3F race-relative last-3F history passed the Rank rolling path: NDCG +.00256 `[+.00045,+.00465]`, with Brier improved in all four years. Binary was inconclusive. Lean+SEC-3F is the accepted pre-2024 rolling Rank candidate; no 2024 milestone was opened.
- HPO-01 retained both parameter sets. Binary had no eligible profile; Rank `feature_fraction=.75` failed its single 2023 confirmation with wholly adverse Log Loss/Brier intervals.
- Fixed 50:50 ENS-01 was rejected because Binary-relative Log Loss improvement was .00157, below the .002 minimum. No weight search occurred and 2023 confirmation stayed unopened.
- SHIMBA-FILTER-001 rejected removing new-horse races from Binary fitting: all-race improvement was LL `-.00097` and NDCG `-.00225`; keep new-horse races in training and evaluation.
- PACE-01 added one 90-day race-relative early-position history. Binary passed its probability path with LL +.00659 `[+.00333,+.00978]`; Rank passed both paths with NDCG +.00440 `[+.00197,+.00683]`. Both families improved LL/Brier in all four years; accept both rolling candidates.
- PACE-02 rival-only front pressure was inconclusive for both families and was not adopted.
- PACE-03 final-position history was Binary inconclusive and LambdaRank rejected: Rank NDCG −.00276 `[−.00473,−.00084]`, worse in all four years.
- PACE-04 transition-normalized position gain was inconclusive for both families and was not adopted. The pace route is closed with PACE-01 as its sole accepted feature.
- LIVE-DATA official-source archive groundwork is complete. The user deferred actual collection because JV-Link is Windows-only and the current machine is a Mac. Do not pursue unofficial Mac transport or JRA Web scraping.
- Binary versus LambdaRank on the 254-feature config remains unresolved: Binary ranking is higher and LambdaRank Log Loss is lower by .00084, but all paired family intervals cross zero.
- Surface-conditioned Elo adds a supported LambdaRank ranking signal relative to the corrected 268-feature baseline, but adding the same 3-column family to the lean 253-feature config was rejected for both families in IMP-004.
- IMP-005 added one 90-day decayed global-Elo expected-vs-actual race-value. Binary was inconclusive with all point metrics worse; LambdaRank was rejected. The feature was highly redundant with 90-day mean finish and is not adopted.
- R0 reproduced the existing global Elo on all 489,674 pre-2025 scored runners exactly at the stored float32 contract.
- R2 selected global pairwise Elo K=48/scale=200 using only annual 2018–2021 scores. R3 top-choice PL and R4 surface blending were rejected on 2022. R5 froze the selected global spec and fitted temperature 0.5187 on 2023 only.
- R5 standalone 2024 calibrated metrics: NDCG@3 .3533, Top-1 .1693, Log Loss 2.4015, Brier .89459, ECE .00774.
- R6 added the frozen five-column rating group to the lean LightGBM. Binary was rejected on the Brier guardrail; LambdaRank NDCG significantly worsened. The module remains a standalone baseline but is not in the current best LightGBM.
- PV-00 found complete parseable clocks for all 515,983 numeric finishers/demotions in 2013–2023 flat data; 25.66% of fourths were within .3 seconds. Last-3F and passing order also have near-complete coverage but remain separate hypotheses.
- PV-01 added one 90-day decayed signed time-gap feature. Binary passed both preregistered paths: NDCG +.00631 `[+.00145,+.01117]`, LL +.00685 `[+.00050,+.01312]`, Brier +.00217. LambdaRank was inconclusive with all point metrics better.
- PV-02 replaced only the R5 ordinal pair actual with `sigmoid(distance-normalized clock gap / .125)`. It significantly improved 2022 ranking but significantly worsened raw Log Loss, so it stopped before 2023/2024 as preregistered.
- PV-03 tested the identified calibration mechanism independently. Previous-year temperatures improved Log Loss in every 2019–2022 evaluation (macro +.00909); 2024 candidate LL was 2.39570 versus ordinal 2.40153, improvement +.00583 `[+.00197,+.00975]`. The calibrated margin-aware rating is the supported standalone replacement.
- PV-04 added only the absolute margin-rating raw score to PV-01. Binary was near-null; LambdaRank points improved but intervals crossed zero. The feature was highly redundant with existing Elo (`rho≈.97`) and was not adopted.
- PV-05 added only margin score minus same-spec ordinal score. Probability points improved (LambdaRank LL 2.07606, Brier .82641) but the PV-01 paired intervals crossed zero and the preregistered minimum was missed. It remains prospective-only and is not adopted.
- PV-06 used 2014–2021 raw `着差` tokens only to dequantize equal 0.1-second clocks, then stopped at its first 2022 gate. All calibrated points improved, including Top-1 +.00315 `[+.00092,+.00557]`, but primary Log Loss was only +.00003 `[-.00022,+.00028]`; the result is inconclusive and 2024 stayed closed.
- GR-001 independently collapsed second/third training relevance and added one upper-half bucket. On its restricted temporal gate, candidate Log Loss worsened by .01539 `[.01023,.02068]` and Brier by .00201 `[.00058,.00349]`; reject and retain the original top-three labels.
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
- Time/margin/sectional audit, PV-01 protocol and result: `docs/experiments/race_content_time_m0_m4_20260831.md`
- PV-02 through PV-05 protocols/results: `docs/experiments/pv_002_margin_aware_rating_20260831.md`, `docs/experiments/pv_003_margin_rating_temporal_calibration_20260831.md`, `docs/experiments/pv_004_margin_rating_integration_20260831.md`, `docs/experiments/pv_005_margin_rating_delta_20260831.md`
- PV-06 protocol/result: `docs/experiments/pv_006_margin_token_refinement_20260831.md`
- GR-001 protocol/result: `docs/experiments/gr_001_graded_lambdarank_20260831.md`
- S-rank integrated result: `docs/experiments/s_rank_model_research_conclusions_20260831.md`
- Rolling/feature/HPO/ensemble/new-horse details: `docs/experiments/eval_roll_001_rolling_origin_20260831.md`, `docs/experiments/opp_recent_001_20260831.md`, `docs/experiments/sec_3f_001_20260831.md`, `docs/experiments/hpo_01_lightgbm_rolling_20260831.md`, `docs/experiments/ens_01_fixed_5050_20260831.md`, `docs/experiments/shimba_filter_001_20260831.md`
- LIVE-DATA source/activation state: `docs/live_data_prospective_snapshot.md`
- PACE-01 through PACE-04 protocols/results: `docs/experiments/pace_01_early_position_20260831.md`, `docs/experiments/pace_02_field_pressure_20260831.md`, `docs/experiments/pace_03_final_position_20260831.md`, `docs/experiments/pace_04_position_gain_20260831.md`
- Tracked machine source of truth: `experiments/baseline_validation_20260830/`
- Tracked PV summaries: `experiments/race_content_20260831/summary.json` and `pv_002_summary.json` through `pv_006_summary.json`
- Tracked GR-001 summary: `experiments/graded_rank_20260831/summary.json`
- Tracked S-rank aggregate: `experiments/s_rank_model_research_20260831/summary.json`
- Tracked PACE summaries: `experiments/pace_01_20260831/summary.json` through `experiments/pace_04_20260831/summary.json`

Local complete artifacts are under `artifacts/` and intentionally ignored. Rating artifacts are `artifacts/rating_module_r0_r5_20260830/`, `artifacts/r6_*`. PV-01 artifacts are `artifacts/pv_001_*`, with paired comparison at `artifacts/pv_001_comparison/comparison.json`; its augmented cache is `data/model_frame_race_content_time_20260831.pkl` (269 total cache features). PV-06 uses `artifacts/pv_006_margin_token_audit_20260831_clean/` and `artifacts/pv_006_margin_token_refinement_20260831/`; GR-001 uses `artifacts/gr_001_graded_lambdarank_20260831/`. Existing corrected/surface/race-value/rating caches remain local and ignored.

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
- R6 did not change the then-current best; PV-01 later superseded the Binary config only.

## Race-content and margin-rating PV-00–PV-06 outcome

- Protocol/config: `docs/experiments/race_content_time_m0_m4_20260831.md`, `configs/performance/pv_001_race_content_time.json`.
- Frozen feature: sole winner gets positive gap to fastest nonwinner; nonwinners get negative gap to official winner; normalize per 1000m, clip to `[-5,+5]`, exclude demotion/DQ races, dead-heat winner 0, 90-day decayed historical mean.
- Representation was selected on annual 2018–2021 standalone ranking only. 2022 confirmed its direction; 2023 was reserved for model calibration; 2024 was one-shot development; 2025 content was not generated.
- Cache gate: all 533,853 rows and old 268 columns exact, mismatch 0, max diff 0; one new column; 2025 nonmissing 0. Control predictions exactly reproduced `abl_006`.
- Binary candidate: NDCG .49757, Top-1 .29646, LL 2.07867, Brier .82767. Accept for the current development baseline, subject to 2026+ confirmation.
- LambdaRank candidate: NDCG .49413, Top-1 .29056, LL 2.07783, Brier .82810. All points improved, but primary intervals crossed zero; retain as inconclusive.
- Feature gain share: Binary 3.60%, LambdaRank 6.44%. Spearman with existing 90-day mean finish is −.812, so it is related but less redundant than IMP-005 (−.938).
- PV-02 train-only tau: median positive adjacent-rank gap `.125 sec/1000m`; 26,563 clean 2014–2021 races, 268,522 positive gaps, 81,809 equal-clock gaps, zero inversions.
- PV-03 frozen standalone spec: global pairwise Elo, initial 1500, K48, scale200, logistic time-margin actual tau .125, same-date batch, own previous-year/2023 temperature. 2024 calibrated NDCG .35634, Top-1 .16798, LL 2.39570, Brier .89355.
- PV-04 absolute-score integration did not pass versus PV-01. Binary deltas NDCG/LL `+.00003/−.00009`; Ranker `+.00208/+.00054`, both primary intervals crossed zero.
- PV-05 delta integration did not pass versus PV-01. Binary deltas NDCG/LL `−.00131/+.00057`; Ranker `−.00099/+.00178`, both primary intervals crossed zero. Do not promote its favorable probability point metrics as a supported improvement.
- PV-06 train-only audit covered 26,563 clean races and 348,745 adjacent distinct-rank edges with 100% ordinary-token coverage and no clean clock inversions. The frozen equal-clock mapping is `ハナ=.02`, `アタマ=.04`, `クビ=.06`, `1/2=.08` seconds, with maximal block cap `.08`.
- PV-06 2022 candidate improvements were LL +.000029 `[-.000222,+.000277]`, Brier +.000015, NDCG +.000834, Top-1 +.003149 `[+.000925,+.005573]`. Primary LL crossed zero, so inconclusive; do not open 2024 or promote the mapping.
- Tracked machine summaries are `experiments/race_content_20260831/summary.json` and `pv_002_summary.json` through `pv_006_summary.json`. Local artifacts are `artifacts/pv_002_*` through `artifacts/pv_006_*`.

## Graded LambdaRank GR-001 outcome

- Control labels remained `1st=3, 2nd=2, 3rd=1, other=0`. Candidate labels were `1st=3, 2nd/3rd=2, 4th..ceil(field/2)=1, lower half/DNF=0`; `label_gain=[0,1,3,7]` stayed fixed.
- Train-only 2014–2021 audit changed 133,439/360,318 rows and increased differently labelled pairs by 82.87%. Features remained the lean 253-column scope.
- Temporal gate was fit 2014–2019, early stopping 2020, temperature 2021, and one 2022 evaluation. No 2023–2025 outcomes were used.
- Candidate versus control: LL −.015391 `[-.020677,−.010227]`, Brier −.002007 `[-.003486,−.000580]`, NDCG −.003986 `[-.008313,+.000301]`, Top-1 −.003778 `[-.010414,+.002901]`. Both acceptance paths failed; reject.
- Descriptively, Log Loss worsened in every field-size band. Small and 17+ fields had positive NDCG points, but these small, unqualified slices do not justify retuning. Retain the original top-three training labels.

## Next task after PACE-01 through PACE-04

`docs/model_research_priorities.md` remains the living source of truth. The
latest accepted result is `docs/experiments/pace_01_early_position_20260831.md`.

The next independent modeling task is SPEED-01. Freeze one transparent
condition-adjusted expected-time residual using pre-2022 evidence, then carry
one PIT-safe historical performance column into EVAL-ROLL. Expected-time fitting
must remain inside each temporal training boundary or use an equivalent
forward-only expanding state; evaluation-year outcomes, same-day later races,
and future-derived track variants are prohibited. Do not mix speed definition,
hyperparameter tuning, and multiple feature variants in one experiment.
Continue to keep 2024 milestone-only and 2025 outside iterative selection.

LIVE-DATA actual collection is intentionally deferred. Reopen it only if the
user explicitly supplies or authorizes a supported private Windows JV-Link
environment. Do not substitute JRA website scraping.

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

# Rebuild PV-01 opt-in cache; 2025 is removed before content normalization
uv run horse-pred build-race-content-cache \
  --raw-path /path/to/race_results_merged.csv \
  --baseline-cache data/model_frame_20260830_corrected.pkl \
  --output data/model_frame_race_content_time_20260831.pkl

# Reproduce PV-03 standalone temporal-calibration study
uv run horse-pred run-margin-rating-calibration-study \
  --raw-path /path/to/race_results_merged.csv \
  --cache data/model_frame_20260830_corrected.pkl \
  --output artifacts/pv_003_margin_rating_temporal_calibration_20260831

# Reproduce PV-06 raw margin-token audit and 2022 study
uv run horse-pred audit-margin-tokens \
  --raw-path /path/to/race_results_merged.csv \
  --output artifacts/pv_006_margin_token_audit
uv run horse-pred run-margin-token-rating-study \
  --raw-path /path/to/race_results_merged.csv \
  --cache data/model_frame_20260830_corrected.pkl \
  --config configs/performance/pv_006_margin_token_refinement.json \
  --output artifacts/pv_006_margin_token_refinement

# Reproduce GR-001 restricted temporal gate
uv run horse-pred run-graded-rank-study \
  --cache data/model_frame_20260830_corrected.pkl \
  --config configs/performance/gr_001_graded_lambdarank.json \
  --output artifacts/gr_001_graded_lambdarank

# Reproduce PACE-01 cache and rolling evaluation
uv run horse-pred build-pace-recent-cache \
  --raw-path /path/to/race_results_merged.csv \
  --baseline-cache data/model_frame_sec_3f_001.pkl \
  --config configs/features/pace_01_early_position.json \
  --output data/model_frame_pace_01.pkl
uv run horse-pred run-rolling-evaluation \
  --cache data/model_frame_pace_01.pkl \
  --config configs/evaluation/pace_01_rolling.json \
  --output artifacts/pace_01_rolling_20260831

# Reproduce the final PACE-04 one-column test from the PACE-01 incumbent
uv run horse-pred build-pace-gain-cache \
  --raw-path /path/to/race_results_merged.csv \
  --baseline-cache data/model_frame_pace_01.pkl \
  --config configs/features/pace_04_position_gain.json \
  --output data/model_frame_pace_04.pkl
uv run horse-pred run-rolling-evaluation \
  --cache data/model_frame_pace_04.pkl \
  --config configs/evaluation/pace_04_rolling.json \
  --output artifacts/pace_04_rolling_20260831

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

- PV-01 run commit: `92f9a268c1525b8d4fc3829eb580a39555d3faea`, `dirty=false`.
- Raw SHA-256: `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`.
- Surface-cache control: 533,853 rows; old 268 column names/order/values/NaN positions exact; mismatch count 0; max absolute difference 0.
- Race-value-cache control: 533,853 rows; old 268 column names/order/values/NaN positions exact; mismatch count 0; max absolute difference 0. The 253-feature control reproduced `abl_006` with prediction/metric max difference 0.
- IMP-005 full local comparison: `artifacts/imp_005_primary_comparison/comparison.json`; tracked aggregate: `experiments/rating_race_value_20260830/imp_005_summary.json`.
- R6 cache control: 533,853 rows; old 268 columns exact; new columns 5; 2025 nonmissing 0. The R6 control reproduced `abl_006` exactly.
- Formal R0–R5 run commit: `ce6d5ccb465d7462eeb3872887fa19891a1e732f`, dirty=false. R6 run commit: `3d530258dbe51d985f54f5460e4824cbf4ff80a0`, dirty=false.
- PV-01 cache SHA-256: `85e92160f0a79f7286409bd4c006a0f0a1310ff67ff845e6fdc75d1894834eb9`; old cache SHA-256 `8d2cd52aea7e77a5b8d3fbeed1436cdffd92699f8c668b26daaec16b70ada62f`.
- PV-03 run commit `79905d8fa6afa84b4b53d520984926df1f8e342b`, dirty=false. PV-04 cache/run commit `66158b5`, dirty=false. PV-05 cache/run commit `e161d4e`, dirty=false.
- PV-06 run commit `ea725d94821886b06dc717dad5adf5e50cb6a3cc`, dirty=false; mapping hash `4998d728af91250c3ae5c7d43c3904af4df6a8e02f140d73664756ebbf8fc4db`.
- GR-001 run commit `9d3eeabe49c3755d6b15d7bd5075e1af0a6306d4`, dirty=false; feature-column hash `fd8735cf6f8472a5c7322e3622c83fde1ff7720b022b75637745c96a1bc1062f`.
- PV-04 cache SHA-256 `525ccf39ae654d78e72d2909bcc4ec184165f86d91fa7828466793887b1e7b96`; PV-05 delta cache SHA-256 `e9c6c0ccd5c9b5aeb20c4dfc1f95ea92e121937dba000975c1535d6f04c1ccfd`.
- Final verification after S-rank synthesis and SHIMBA-FILTER-001: 196 tests passed, Ruff passed, compileall passed.
- PACE-01 run commit `074f54e86ace8a3d51bbf5f386166b72ba6e7db2`, dirty=false; cache SHA-256 `8b4480da6d6396d9bac54b92d947948708e0bccc30cfe0ac24be40e427d5e732`; existing 270 features exact; 2025 PACE values 0.
- PACE-04 run commit `517a685a88ccfa7b0c72be3cd3b4a6d0ee8d1d48`, dirty=false; cache SHA-256 `ddd1365a098c2b4b0b63dc742da352f05213f1b6ecf07f6a4f345d360f01775c`; existing 271 features exact; 2025 PACE-04 values 0.
- Final verification after PACE-01 through PACE-04 synchronization: 227 tests passed, Ruff passed, compileall passed.
