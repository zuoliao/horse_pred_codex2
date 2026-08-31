# PACE-04 transition-normalized position gain

Date: 2026-08-31 JST  
Status: complete; inconclusive for both families, not adopted

## Hypothesis

PACE-01 established that a horse's early race-relative position is predictive.
PACE-04 asks a distinct running-content question: does a horse's tendency to
advance or retreat between the first and last recorded checkpoints add signal
beyond that accepted early-position history?

It adds exactly one column:

```text
pace_gain__decay_90d__mean_transition_normalized_position_gain
```

PACE-02 rival pressure and PACE-03 final-position level are excluded. This
isolates signed movement rather than adding several correlated pace variants.

## Train-only audit and frozen transformation

For each eligible historical race, compute early and final within-race
percentiles independently using ascending average tie ranks among eligible
starters:

```text
E = 1 - (early_rank - 1) / (n_valid - 1)
F = 1 - (final_rank - 1) / (n_valid - 1)
G = (F - E) / (segment_count - 1)
```

One is frontmost and zero rearmost. Positive `G` means the horse advanced
relative to the field; negative `G` means it retreated. Division by the number
of observed transitions prevents races with more passing-order checkpoints
from mechanically producing larger movements.

The 2014--2021 outcome-free audit found mean absolute unnormalized gain of
`.0515`, `.1052`, and `.1532` for two-, three-, and four-segment records. After
transition normalization these became `.05153`, `.05258`, and `.05106`, which
supports the fixed denominator. Raw signed gain was positive in `37.09%`,
negative in `35.54%`, and zero in `27.37%` of eligible observations.
Transition-normalized gain had split-half repeatability `.323`, weaker than
PACE-01 early position's `.682`; this motivates one controlled test, not a
family of movement variants.

The historical feature is the 90-day half-life decayed mean of prior `G`
values, with same-date batching and cold `NaN`. A started runner needs a
numeric hyphen-separated passing order with at least two segments. One-segment
races, including Niigata turf straight 1000 m, supply no observation.

The exact transformation is frozen in
`configs/features/pace_04_position_gain.json` with canonical SHA-256:

```text
271a348b74fbf45fece5c8686583bfae4ea8753f15b235049899202cc2106934
```

PACE-04 does not add final-position level, segment count, absolute movement,
discrete style, rival pressure, field interactions, or alternative decay
windows.

## Frozen rolling comparison

Binary and LambdaRank each use their accepted PACE-01 rolling candidate as
control. Candidate adds PACE-04 only. Parameters and labels are unchanged;
2024/2025 remain closed.

| Method | Count | Ordered-column SHA-256 |
|---|---:|---|
| Binary control | 255 | `a50da361280b6f892ef7dfbe017768bbaa1657a6e95cb744ef7df6417f03275c` |
| Binary candidate | 256 | `fd2bf29640f63aae5f69a17c74b18e29c3c99b6d1067af1a278e887113bd3d03` |
| Rank control | 255 | `0bebb3ab682423318d239d338b370d1d6f162f0bbb8d4678f8bc0760b4c63d3e` |
| Rank candidate | 256 | `4dcb9958defc695bbf0b1af821257ba7c81897a2de2edbdfe35902d87f1c19b8` |

The rolling folds evaluate 2020--2023. Prior selection-use counts are
`37/37/41/14` for those years. Neither 2024 nor 2025 may be used to revise the
mapping, decay, normalization, feature set, or accept/reject decision.

## Decision rule

Use the unchanged PACE paths. Probability acceptance requires positive Log
Loss CI, at least three improved years, and Brier/NDCG/Top-1 guardrails of
`-.001/-.002/-.005`. Ranking acceptance substitutes a positive NDCG CI with
Log Loss/Brier/Top-1 guardrails of `-.002/-.001/-.005`. A failed guardrail or
wholly adverse primary interval is reject; otherwise inconclusive. Passing a
path does not automatically open 2024.

## Rolling result

The frozen run used clean commit
`517a685a88ccfa7b0c72be3cd3b4a6d0ee8d1d48`, no odds, and zero 2024/2025
rows. Cache control preserved all 271 incumbent features exactly and added only
the registered PACE-04 column.

| Family | NDCG improvement | Top-1 improvement | Log Loss improvement | Brier improvement | Decision |
|---|---:|---:|---:|---:|---|
| Binary | +.00054 `[-.00164,+.00260]` | -.00051 `[-.00436,+.00338]` | +.00048 `[-.00219,+.00314]` | +.00009 `[-.00078,+.00095]` | inconclusive |
| LambdaRank | -.00189 `[-.00389,+.00017]` | -.00367 `[-.00730,+.00011]` | -.00114 `[-.00324,+.00103]` | -.00047 `[-.00118,+.00024]` | inconclusive |

Binary Log Loss and Brier improved in three of four years, but its Log Loss
interval includes zero. Its guardrails pass, so the result is inconclusive.
LambdaRank NDCG improved in only one year and is adverse at the point estimate,
but the interval narrowly includes zero and its point guardrails remain within
their frozen limits. It too is inconclusive rather than rejected.

PACE-04 is not adopted or tuned. The PACE sequence is now closed: PACE-01 early
position is accepted for both families; PACE-02 rival pressure and PACE-04
position gain are inconclusive; PACE-03 final position is Binary inconclusive
and LambdaRank rejected. PACE-01 remains the sole pace incumbent, and 2024 was
not opened for any PACE experiment.

Tracked evidence is `experiments/pace_04_20260831/summary.json`; full local
artifacts are under `artifacts/pace_04_rolling_20260831/`.
