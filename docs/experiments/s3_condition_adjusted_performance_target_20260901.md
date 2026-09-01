# S3 Condition-adjusted performance target

## Decision

S3 is complete and **rejected** under the preregistered criteria. Condition-adjusted continuous performance was easier for Huber to fit as a numeric response, but its race ordering and calibrated winner probability were materially worse than the matched Binary and LambdaRank controls in every evaluation year.

The formal controls remain unchanged. S2 supervised race-wise probability is the recommended next hypothesis, but was not executed.

## Scope

- Evaluation years: 2020, 2021, 2022.
- Fits: 3 folds × 4 methods = 12.
- Binary control: PV-01 254 features.
- LambdaRank control: lean 253 features.
- Huber candidates: exactly the corresponding 254- and 253-feature scopes; no S1 performance feature was added.
- 2023/2024/2025 rows used: 0/0/0.
- Odds, popularity, final market, direct entity IDs: not used.

## Target

The target was `expected winner seconds per 1000m - runner seconds per 1000m`, positive-is-better and clipped to `[-5,5]`. The 51-dimensional condition normalizer used the frozen S1 effects and was fitted separately inside each fold on 2014 through the fold train end only. It was then frozen for early stopping, calibration, and evaluation.

Target coverage was high: 99.59–99.61% over the full fold materialization and 99.50–99.74% in the evaluation years. DNF/missing-clock runners and demotion/DQ races were not zero-imputed. The clip rate was about 0.47%, almost entirely the negative tail. Thus the negative result is not plausibly explained by broad target missingness.

## Primary results

All deltas are positive when Huber improves on its paired control. Values are unweighted macro averages across the three evaluation years.

| Comparison | Δ Log Loss | 95% date-block CI | Δ Brier | Δ NDCG@3 | 95% date-block CI | Δ Top-1 | Direction | Decision |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Huber 254 vs Binary 254 | −0.09299 | [−0.10285, −0.08317] | −0.01920 | −0.02235 | [−0.02749, −0.01732] | −0.02464 | all primary metrics worse 3/3 years | probability/ranking rejected |
| Huber 253 vs LambdaRank 253 | −0.08481 | [−0.09338, −0.07627] | −0.01728 | −0.02225 | [−0.02726, −0.01738] | −0.01683 | all primary metrics worse 3/3 years | probability/ranking rejected |

Macro metric levels:

| Method | NDCG@3 | Top-1 | Log Loss | Brier | Winner reciprocal rank |
|---|---:|---:|---:|---:|---:|
| Binary control | .46879 | .26532 | 2.14976 | .84294 | .46968 |
| Huber on Binary scope | .44645 | .24067 | 2.24276 | .86215 | .44475 |
| LambdaRank control | .46575 | .26080 | 2.16373 | .84568 | .46585 |
| Huber on LambdaRank scope | .44351 | .24397 | 2.24854 | .86296 | .44596 |

## Temporal replication and slices

Both Huber candidates worsened Log Loss, Brier, NDCG@3, Top-1, and winner reciprocal rank in 2020, 2021, and 2022. Every primary paired interval was entirely on the adverse side.

Across the six preregistered slices, both Huber models worsened NDCG in every slice. The Binary-scope Huber had a small positive Log Loss point delta in new races (`+.00261`) and positive Top-1 points in new/history-0 races, but their ranking deltas were negative and the full-population result was strongly adverse. Surface-switch races showed the largest degradation. Slice results were not used to modify the target.

## PIT and artifact audit

- Fold normalizers fitted 13,261 / 16,588 / 19,909 clean train races and ended on 2017-12-28 / 2018-12-28 / 2019-12-28.
- Early-stopping, calibration, and evaluation rows used by every normalizer: zero.
- Calibration and evaluation choice-set row counts were identical across all four methods in every fold.
- Race-softmax probabilities summed to one with maximum absolute numerical error `1.33e-15`.
- Market columns were physically removed before feature resolution; feature scopes contained zero target/current-outcome columns and zero direct entity IDs.
- The local artifact contains 23 hashed files and complete hash verification passed.

One non-selection diagnostic deviation remains: the preregistration listed aggregate Huber loss, but the artifact saved target coverage, clipping, MAE, and race-wise Spearman without a separate Huber-loss field. This does not affect the registered acceptance metrics or the rejection. Binary logits and LambdaRank utilities have arbitrary scales, so their target MAE values are not compared with Huber MAE.

## Interpretation

S1 and S3 answer different questions. S1 showed that a past condition-adjusted performance summary is useful **as input context** for future winner/ranking prediction. S3 shows that making realized clock-based performance the **sole supervision objective** is not a better replacement for winner or top-heavy race supervision under this simple Huber formulation.

The condition adjustment is race-constant, so within each race the continuous target order is effectively the clock order. Huber also assigns loss to every timed runner across races, while the downstream task is a one-winner choice set. It can therefore spend capacity fitting lower-field and cross-race clock variation that is weakly aligned with winner discrimination. Track/day variant, pace, and latent stochastic performance remain in the response. This is a plausible interpretation, not a causal decomposition.

The result does not invalidate latent-performance modeling in general, multi-task learning, alternative robust targets, or a future auxiliary performance head. It rejects this preregistered single-target Huber replacement and gives no basis for tuning target variants on the same 2020–2022 evidence.

Several Huber fits stopped near the fixed 500-tree ceiling. The result therefore estimates this fixed Huber configuration, not a fully tuned model-family ceiling; post-hoc tree-budget tuning on the same years is not authorized.

## Next task

Recommend **S2 supervised race-wise probability** because it directly matches the one-winner choice structure and tests the remaining objective-mismatch hypothesis. Do not automatically execute S2. Further performance-target engineering should be deferred unless a new, independently motivated target or evaluation period is approved.

Machine-readable source: `experiments/s3_condition_adjusted_performance_target_20260901/summary.json`.
