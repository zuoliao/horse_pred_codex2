# PV-05 Margin-specific rating delta in LightGBM

Status: preregistered after PV-04 and before building or evaluating the delta cache.

## Hypothesis

PV-04's absolute margin-aware score was highly correlated with existing ordinal Elo
(`rho≈.97`) and took high model importance without incremental 2024 improvement. A
tree model may substitute it for the existing rating without efficiently isolating
the smaller margin-driven change. PV-05 therefore adds exactly one derived PIT state:

```text
margin_rating__delta_vs_ordinal_score_pre
    = margin-aware R5 raw score before race
    - same-spec ordinal R5 raw score before race
```

Both histories use initial 1500, K=48, scale=200, global state, the same event stream
and same-date batching. The delta consequently changes only because of the continuous
time-margin actual. It is zero for cold starts and never uses the current race result.

## Comparison and gates

The primary control remains PV-01's 254 features; the candidate adds only the delta
for 255. The absolute PV-04 score is not present. Cache construction must preserve all
PV-01 rows/features exactly, cover every pre-2025 scoring row, leave 2025 missing, and
reproduce the PV-01 control exactly before candidate evaluation.

Model configs, 2014–2021 fit, 2022 early stopping, 2023 output calibration, strict
2024 population and paired 10,000-draw four-date bootstrap remain unchanged. The
PV-04 probability and ranking paths are reused. LambdaRank also needs to pass the
same applicable path versus lean 253 before promotion.

This hypothesis was formulated after observing PV-04's 2024 result. Accordingly,
nominal intervals understate total research-selection uncertainty. Even a pass is
only a development candidate for 2026+ prospective confirmation. No 2025 result,
odds, ROI, graded label, extra rating column or LightGBM hyperparameter is used.

## Result

Cache and control gates passed. The delta covered all 489,674 pre-2025 scoring rows,
was absent on every 2025 row, had standard deviation .0690, and was exactly zero for
49,133 rows. The PV-01 control prediction maximum difference was zero.

| Family | Candidate metrics (NDCG / Top-1 / LL / Brier) | Improvement vs PV-01 |
|---|---|---|
| Binary | .496257 / .292363 / 2.078098 / .827242 | -.001308 / -.004097 / +.000574 / +.000431 |
| LambdaRank | .493143 / .287938 / 2.076056 / .826408 | -.000989 / -.002622 / +.001775 / +.001696 |

Paired primary intervals:

- Binary NDCG `-.001308 [-.006493,+.004191]`; Log Loss
  `+.000574 [-.007231,+.008085]`.
- LambdaRank NDCG `-.000989 [-.005737,+.003863]`; Log Loss
  `+.001775 [-.003438,+.006795]`.

Both point Log Loss and Brier values improved, especially for LambdaRank, but the
Log Loss improvement missed the preregistered .002 minimum and both primary intervals
crossed zero. Neither probability nor ranking path passed, so PV-05 is **inconclusive
and not adopted**.

Against lean 253, LambdaRank again passed the probability path (Log Loss +.008606
`[+.000803,+.016658]`, Brier +.003540 `[+.000908,+.006236]`), but the primary
PV-01 comparison did not. This cannot establish that the delta adds information.

The delta was less redundant than the absolute score: Spearman correlation was
-.367 with PV-01 and -.570 with existing Elo. It took 1.76% of Binary gain (rank
12/255) and 1.98% of LambdaRank gain (rank 9/255). The combination of nontrivial model
usage, favorable probability points, and broad paired intervals makes it a plausible
2026+ prospective candidate, not a current replacement.

PV-04 and PV-05 together close this representation branch for 2024: do not continue
trying arithmetic variants selected from the same development outcomes. The next
time/margin hypothesis should instead address the documented 0.1-second clock
resolution with raw margin tokens, choose its mapping before 2024, and keep it
separate from last-3F and passing-position features.

Complete ignored artifacts are `artifacts/pv_005_*`; the tracked aggregate is
`experiments/race_content_20260831/pv_005_summary.json`.
