# PACE-03 final recorded position history

Date: 2026-08-31 JST  
Status: preregistered; model outcomes unopened

## Hypothesis

PACE-01's early position was strongly supported, while PACE-02 field pressure
was inconclusive. PACE-03 returns to horse-level running content and asks whether
the final recorded passing position contains incremental information beyond
early position. It adds exactly one column:

```text
pace_final__decay_90d__mean_final_position_percentile
```

## Train-only audit and frozen transformation

The 2014--2021 audit has 373,891 eligible starter tokens in 26,387 JRA flat
races. The remaining 198 races with no two-or-more-segment token are all
Niigata turf straight 1000 m and supply no observation.

For each eligible race, parse the last token, rank positions ascending with
average ties, and transform to `1-(rank-1)/(n_valid-1)`. One is frontmost and
zero rearmost at the final recorded checkpoint. The feature is the horse's
90-day half-life decayed mean from prior dates with same-date batching and cold
`NaN`, identical to PACE-01 except for first versus last token.

Early and final observation percentiles have Spearman `.867` overall, `.964`
for two-segment, `.871` for three-segment, and `.736` for four-segment records.
Among horses with at least three observations, historical means correlate
`.912`. Final-position split-half repeatability is `.662`, close to PACE-01
early position's `.682` and above normalized gain's `.323`. These outcome-free
statistics support one final-position test before position gain.

The exact transformation is frozen in
`configs/features/pace_03_final_position.json` with canonical SHA-256:

```text
b0bb8efb6b696cabd52e7da2f9e72ce9d5bbbd6d3ca9b6a95180a643e66be52c
```

PACE-03 does not include PACE-02, position gain, transition count, absolute
movement, discrete style, field pressure, or interactions.

## Frozen rolling comparison

Binary and LambdaRank each use their accepted PACE-01 rolling candidate as
control. Candidate adds PACE-03 only. Parameters and labels are unchanged;
2024/2025 remain closed.

| Method | Count | Ordered-column SHA-256 |
|---|---:|---|
| Binary control | 255 | `a50da361280b6f892ef7dfbe017768bbaa1657a6e95cb744ef7df6417f03275c` |
| Binary candidate | 256 | `4085d7d892929b9714b4d459c0697ca467d8ff0ab36b23455606c1fb04f2da37` |
| Rank control | 255 | `0bebb3ab682423318d239d338b370d1d6f162f0bbb8d4678f8bc0760b4c63d3e` |
| Rank candidate | 256 | `c899d599ad785944381e9b65a2f3eecc47ff4212b6cf35f81099c47e1fd7c8c3` |

## Decision rule

Use the unchanged PACE paths. Probability acceptance requires positive Log
Loss CI, at least three improved years, and Brier/NDCG/Top-1 guardrails of
`-.001/-.002/-.005`. Ranking acceptance substitutes a positive NDCG CI with
Log Loss/Brier/Top-1 guardrails of `-.002/-.001/-.005`. A failed guardrail or
wholly adverse primary interval is reject; otherwise inconclusive. Passing a
path does not automatically open 2024.
