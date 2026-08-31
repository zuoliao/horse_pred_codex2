# SPEED-01 prequential condition-adjusted speed history

Date: 2026-09-01 JST  
Status: rolling complete; accepted for both families via probability path

## Hypothesis

PACE-01 shows that race-relative early position is useful, but it does not say
whether a past race was objectively fast for its conditions. SPEED-01 adds one
horse-history column:

```text
speed__decay_90d__mean_condition_adjusted_time_residual
```

For each completed clean flat race, a prequential ridge estimates the expected
winner clock in seconds per 1000 m using only earlier dates. Each timed starter
gets `expected winner clock - own clock`, so positive is faster than the
condition expectation. Clip this observation to `[-5,+5]`, then emit its
90-day half-life horse mean from prior dates with cold `NaN` and same-date
batching.

## Train-only audit and frozen expectation model

The 2014--2021 audit contains 26,585 flat races and 378,961 rows. Of these,
26,563 races have a clean winner clock; 22 demotion/disqualification races are
excluded. No clean race has a missing winner clock or a timed nonwinner faster
than its winner. Timed starters cover 376,447 rows.

An all-condition exact-cell mean is rejected: 2,760 cells have median prior
count 8, and only 46.58%/27.02% of observations have at least 10/20 prior cell
races. The frozen alternative is a 51-parameter additive ridge with intercept
and reference-coded dummies for course×surface, exact distance, going, class
tier, and race age restriction. Ridge alpha is 1, the intercept is unpenalized,
and no scaling or interaction is used. A day is predicted only after 510 clean
prior races; all races on that day are predicted before any update from the day.

This prequential model generates expectations for 26,019/26,563 audit races
(97.95%), first on 2014-03-02. Winner residual MAE/RMSE are `.580/.758`
seconds per 1000 m. It produces 368,097 runner observations. Raw runner residual
q1/median/q99 are `-4.131/-.667/+1.236`; `.442%` exceed absolute 5 and the
minimum is `-39.05`. Reusing PV-01's `[-5,+5]` bound prevents rare abnormal or
large-loss clocks from persisting in horse history without selecting a new
threshold. The unclipped 90-day history covers 325,811/369,237 target starters
(88.24%).

The exact transformation and category references are frozen in
`configs/features/speed_01_condition_adjusted.json`, canonical SHA-256:

```text
55f5e4a5eb9c7bb12368bed9faa0a617b35ade5cbae918ffa8046be33b2d5c43
```

Unknown conditions supply no observation and are not silently mapped to a
reference. Winner clock teaches the race condition baseline; individual clocks
create horse observations. Current target-race going is not added as a model
feature. Full-period fit, fold-future fit, same-date later results, day/course
track variants, odds, and popularity are prohibited.

## Frozen rolling comparison

Binary and LambdaRank each use their accepted PACE-01 candidate as control.
Candidate adds SPEED-01 only; PACE-02 through PACE-04 remain excluded.
Parameters and labels are unchanged, and 2024/2025 remain closed.

| Method | Count | Ordered-column SHA-256 |
|---|---:|---|
| Binary control | 255 | `a50da361280b6f892ef7dfbe017768bbaa1657a6e95cb744ef7df6417f03275c` |
| Binary candidate | 256 | `3b6104ec33f6bf2b02b64685bf3ebf6bb828f14683c262e922e696da43bb4940` |
| Rank control | 255 | `0bebb3ab682423318d239d338b370d1d6f162f0bbb8d4678f8bc0760b4c63d3e` |
| Rank candidate | 256 | `79761cff41a0fdc122a0e81c947c9c4bf10f1184e3e6c1bfff2f1188a45a9ba0` |

Prior selection-use counts are `39/39/43/16` for 2020--2023.

## Decision rule

Probability acceptance requires positive Log Loss CI, at least three improved
years, and Brier/NDCG/Top-1 point guardrails of `-.001/-.002/-.005`. Ranking
acceptance substitutes a positive NDCG CI with Log Loss/Brier/Top-1 guardrails
of `-.002/-.001/-.005`. A failed guardrail or wholly adverse primary interval
is reject; otherwise inconclusive. Passing a path does not automatically open
2024.

## Rolling result

The frozen run used clean commit
`f4b38b8744dc7a49d701b23c1c14badc6a663571`, no odds, and zero 2024/2025
rows. Cache control preserved all 271 PACE-01 cache features exactly and added
only the registered SPEED-01 column.

| Family | NDCG improvement | Top-1 improvement | Log Loss improvement | Brier improvement | Decision |
|---|---:|---:|---:|---:|---|
| Binary | +.00169 `[-.00069,+.00403]` | -.00012 `[-.00516,+.00481]` | +.00769 `[+.00457,+.01078]` | +.00228 `[+.00127,+.00329]` | accept: probability |
| LambdaRank | +.00172 `[-.00067,+.00416]` | +.00114 `[-.00375,+.00616]` | +.00823 `[+.00523,+.01128]` | +.00222 `[+.00121,+.00323]` | accept: probability |

Binary Log Loss and Brier improved in all four years; NDCG improved in three,
and all point guardrails pass. LambdaRank Log Loss, NDCG, and Top-1 improved in
three years and Brier in all four; its probability path also passes. The
candidate is therefore accepted for both rolling incumbents. LambdaRank does
not separately pass the ranking path because its NDCG interval includes zero.

SPEED-01 is the first accepted post-PACE independent signal and is materially
distinct from PV-01: it estimates the race's expected condition-adjusted winner
clock before comparing each runner's own time, rather than using only the
within-race winner gap. The transformation, clip, ridge, and feature scope must
not be changed in response to these results.

The result is strong enough to preregister one unchanged 2024 milestone. That
milestone is confirmation only: no mapping, parameter, feature, or calibration
choice may be made from its outcome, and 2025 remains closed.

Tracked rolling evidence is `experiments/speed_01_20260901/summary.json`; full
local artifacts are under `artifacts/speed_01_rolling_20260901/`.

## Preregistered 2024 milestone

Before opening 2024, freeze the one-shot contract in
`configs/evaluation/speed_01_2024_milestone.json`. Each family compares its
unchanged 255-column PACE-01 incumbent with the corresponding 256-column
SPEED-01 candidate. Fit remains 2014--2021, early stopping 2022, and temperature
calibration 2023.

The milestone is supported when the paired Log Loss CI is wholly positive and
Brier/NDCG/Top-1 point guardrails remain above `-.001/-.002/-.005`.
Positive Log Loss with passing guardrails but an interval crossing zero is only
directionally consistent. A wholly negative Log Loss interval or failed
guardrail is contradicted; other outcomes are inconclusive. This classification
does not authorize any refit or mapping change, and 2025 remains closed.
