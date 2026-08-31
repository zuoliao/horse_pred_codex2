# PACE-04 transition-normalized position gain

Date: 2026-08-31 JST  
Status: preregistered; model outcomes unopened

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
