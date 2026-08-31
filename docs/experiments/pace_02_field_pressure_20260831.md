# PACE-02 rival front-running pressure

Date: 2026-08-31 JST  
Status: complete; inconclusive for both families, not adopted

## Hypothesis

PACE-01 showed that a horse's own historical early position is predictive.
PACE-02 asks a separate inter-horse question: does the amount of front-running
tendency among the other current entrants change prediction quality?

It adds exactly one column:

```text
pace_pressure__current_field__rival_front_excess_sum
```

For target runner `i`, let `x_j` be each other current runner's frozen PACE-01
history. Define

```text
front_excess(j) = max(x_j - 0.5, 0)
pressure(i) = sum(front_excess(j) for finite j != i)
```

The midpoint `.5` is fixed by the `[0,1]` PACE-01 percentile scale, not selected
from target metrics. The sum jointly represents the number and strength of
front-oriented rivals. Field size already exists as a separate context feature;
PACE-02 does not normalize or add another size column.

The target is excluded, so the value is not merely race-constant. Missing rivals
are unknown and do not contribute as slow horses. If there is no finite rival,
the result is `NaN`; if known rivals all have `x <= .5`, the result is a genuine
zero. No current-race result, passing order, odds, or popularity is used.

## Train-only audit

On 2014--2021 cache rows, 324,392/360,318 PACE-01 inputs were finite. The
frozen pressure is finite for 329,978 rows; 30,340 rows remain missing. Exactly
2,212/25,322 races have all PACE-01 histories missing, principally cold fields;
there are no races with exactly one known source.

Finite pressure has mean `1.4384`, standard deviation `.6998`, median `1.3651`,
90th percentile `2.3798`, 99th percentile `3.3633`, and range `[0,5.7961]`.
Its Spearman correlations are `.333` with field size and `.130` with own
PACE-01. This is expected: the feature intentionally measures accumulated rival
pressure while remaining distinct from the target's own style.

The exact transformation is frozen in
`configs/features/pace_02_field_pressure.json` with canonical SHA-256:

```text
59c5d5a818498561956b1101fc990944ad475d7c8645e59269c10c414c697285
```

Only this representation was screened. Opponent mean, hard front-runner count,
field total including self, multiple thresholds, final-corner history, position
gain, and interactions are not part of PACE-02.

## Cache and comparison contract

PACE-02 is derived only from the already-built PACE-01 model frame. It uses a
new closed `pace_pressure__` group so the PACE-01 control cannot accidentally
select the candidate column. Existing rows, identity, feature order, values,
and NaN positions must remain exact. All 2025 PACE-02 values must be missing.

Binary compares PV-01+PACE-01 against the same 255 columns plus PACE-02.
LambdaRank compares lean+SEC-3F+PACE-01 against the same 255 columns plus
PACE-02. Parameters and labels are unchanged.

| Method | Count | Ordered-column SHA-256 |
|---|---:|---|
| Binary control | 255 | `a50da361280b6f892ef7dfbe017768bbaa1657a6e95cb744ef7df6417f03275c` |
| Binary candidate | 256 | `28fa3413c8f5ac5477c81be83e111a2b3e1dce3ef87c09ff1d271711f131e776` |
| Rank control | 255 | `0bebb3ab682423318d239d338b370d1d6f162f0bbb8d4678f8bc0760b4c63d3e` |
| Rank candidate | 256 | `30aab7301e82ed52ea9a4ae53c27c6e85b1be758eee4ce89e4fa56166e60d6b3` |

## Decision rule

Use the same family-specific paths as PACE-01. Positive differences improve.

- Probability path: four-year macro Log Loss CI lower bound above zero; Brier
  point at least `-.001`, NDCG at least `-.002`, Top-1 at least `-.005`.
- Ranking path: four-year macro NDCG CI lower bound above zero; Log Loss point
  at least `-.002`, Brier at least `-.001`, Top-1 at least `-.005`.
- The path primary improves in at least three of four years.

Pass either path to accept. A failed guardrail or wholly adverse primary
interval is reject; otherwise inconclusive. This does not automatically open
2024, and 2025 remains prohibited.

## Rolling result

The frozen run used clean commit
`c1bf4f0d798a90897ebf4bab9266b936f146e840`, no odds, and zero 2024/2025
rows. The cache preserved all 271 existing features exactly, added one column,
and generated no 2025 value.

| Family | NDCG improvement | Top-1 improvement | Log Loss improvement | Brier improvement | Decision |
|---|---:|---:|---:|---:|---|
| Binary | +.00017 `[-.00193,+.00228]` | -.00229 `[-.00611,+.00158]` | +.00038 `[-.00221,+.00308]` | -.00008 `[-.00088,+.00073]` | inconclusive |
| LambdaRank | -.00081 `[-.00294,+.00129]` | -.00248 `[-.00652,+.00155]` | +.00079 `[-.00147,+.00302]` | +.00004 `[-.00069,+.00076]` | inconclusive |

Binary improved NDCG and Log Loss in only two of four years; Brier improved in
one and Top-1 in zero. LambdaRank improved NDCG in one year and Log Loss/Brier
in two. No primary interval excluded zero, while point guardrails did not force
a reject. Both results are therefore inconclusive under the frozen rule.

PACE-02 is not added to either rolling incumbent and is not tuned with another
threshold, mean, or field-size normalization. The strong own-history PACE-01
result remains intact. The next independent horse-level hypothesis is PACE-03
final-position history, compared against the PACE-01 incumbents without
PACE-02. No 2024 milestone was opened.

Tracked evidence is `experiments/pace_02_20260831/summary.json`; full local
artifacts are under `artifacts/pace_02_rolling_20260831/`.
