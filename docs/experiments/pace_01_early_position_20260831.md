# PACE-01 race-relative early-position history

Date: 2026-08-31 JST  
Status: complete; accepted for Binary and LambdaRank on rolling evidence

## Hypothesis

A horse's habitual early position may represent running style and positional
speed not fully preserved by finish, full-race time gap, last-3F, and rating
history. PACE-01 adds exactly one column:

```text
pace__decay_90d__mean_early_position_percentile
```

It does not add final-corner position, position gain, a discrete style label,
current-field front-runner counts, pace pressure, distance/surface interaction,
absolute passing positions across races, model parameters, or label changes.

## Train-only audit and frozen transformation

The 2014--2021 JRA-flat audit covered 377,615 starters in 26,585 races.
All 377,283 nonmissing passing-order values matched the numeric hyphen-separated
grammar; 332 starter rows were missing. Segment counts were 3,392 one-segment,
200,738 two-segment, 22,561 three-segment, and 150,592 four-segment rows.

Exactly 198 races had no starter with two or more segments. Every one was a
Niigata turf straight 1000 m race. These races supply no PACE-01 observation:
their single token is not relabelled as a corner. In mixed-coverage races, a
starter with fewer than two segments is skipped while eligible starters may
still update. No eligible first token was below 1 or above field size.

For each race, take the first token of every eligible starter. Rank these
integer positions ascending with average ranks for ties. With `n_valid >= 2`,
define

```text
observation = 1 - (rank - 1) / (n_valid - 1)
```

One means frontmost and zero means rearmost. The target feature is the horse's
90-day half-life decayed mean over prior dates. Every runner on a date is
emitted before that date updates state. Cold history remains `NaN`.

The exact transformation is frozen in
`configs/features/pace_01_early_position.json` with canonical SHA-256:

```text
9b9dcadf4080a0dc059900848f5b2587d7f17ce980e37e86e12db7003a934ca5
```

No model outcome, 2022--2025 outcome, odds, or popularity selected this
representation. Only one PACE-01 representation was screened.

## Frozen rolling comparison

The comparison uses the existing four EVAL-ROLL folds and closes 2024/2025.
Binary compares PV-01 against PV-01 plus PACE-01. LambdaRank compares the
accepted pre-2024 lean-plus-SEC-3F candidate against that same selection plus
PACE-01. SEC-3F is a fixed incumbent input, not an interaction with PACE-01.

| Method | Count | Ordered-column SHA-256 |
|---|---:|---|
| Binary control | 254 | `e5228bb4ffd605888b7266030d5e1e9f0931e8468b6fbbf124f3cea60905e51d` |
| Binary candidate | 255 | `a50da361280b6f892ef7dfbe017768bbaa1657a6e95cb744ef7df6417f03275c` |
| Rank control | 254 | `fff37fcb9c1b40611df1a1813908be8d4b9eba2fd5dcd9a575858cc1f4942183` |
| Rank candidate | 255 | `0bebb3ab682423318d239d338b370d1d6f162f0bbb8d4678f8bc0760b4c63d3e` |

## Decision rule

Judge families separately; positive differences mean improvement.

- Probability path: paired four-year macro Log Loss improvement interval lower
  bound above zero; Brier point at least `-.001`, NDCG at least `-.002`, and
  Top-1 at least `-.005`.
- Ranking path: paired four-year macro NDCG improvement interval lower bound
  above zero; Log Loss point at least `-.002`, Brier at least `-.001`, and
  Top-1 at least `-.005`.
- The path primary must improve in at least three of four evaluation years.

Passing either path is `accept`. If no path passes and a guardrail fails or a
primary interval is wholly adverse, mark `reject`; otherwise `inconclusive`.
This protocol does not authorize opening 2024 automatically.

## Rolling result

The frozen run used clean commit
`074f54e86ace8a3d51bbf5f386166b72ba6e7db2`, excluded all 2024/2025 rows,
and used no odds or popularity. The cache preserved all 270 baseline features
exactly and added the one registered column. It was nonmissing for
440,460/489,674 pre-2025 scoring rows (`89.95%`) and for zero 2025 rows.

| Family | NDCG improvement | Top-1 improvement | Log Loss improvement | Brier improvement | Decision |
|---|---:|---:|---:|---:|---|
| Binary | +.00037 `[-.00183,+.00253]` | +.00189 `[-.00203,+.00578]` | +.00659 `[+.00333,+.00978]` | +.00104 `[+.00017,+.00192]` | accept |
| LambdaRank | +.00440 `[+.00197,+.00683]` | +.00524 `[+.00065,+.00973]` | +.01537 `[+.01256,+.01808]` | +.00318 `[+.00232,+.00403]` | accept |

Binary passed the probability path: Log Loss and Brier improved in all four
years, its Log Loss interval was wholly positive, and every ranking guardrail
passed. NDCG itself improved in two years and remains statistically
inconclusive, so the evidence is primarily probability quality.

LambdaRank passed both registered paths. NDCG, Top-1, Log Loss, and Brier all
improved in every year, and both NDCG and Log Loss intervals were wholly
positive. The PACE column averaged 2.05% of Binary gain and 2.66% of
LambdaRank gain across folds.

The accepted pre-2024 rolling candidates are therefore PV-01 plus PACE-01 for
Binary and lean plus SEC-3F plus PACE-01 for LambdaRank, each with 255 features
and unchanged LightGBM parameters. No 2024 milestone was opened. This support
authorizes PACE-02 as a separate experiment; it does not authorize adding
final-corner, position-gain, or multiple field-pressure variants at once.

Tracked decision evidence is `experiments/pace_01_20260831/summary.json`;
complete local artifacts are under `artifacts/pace_01_rolling_20260831/`.

## Reproduction

```bash
uv run horse-pred build-pace-recent-cache \
  --raw-path /path/to/race_results_merged.csv \
  --baseline-cache data/model_frame_sec_3f_001.pkl \
  --config configs/features/pace_01_early_position.json \
  --output data/model_frame_pace_01.pkl

uv run horse-pred run-rolling-evaluation \
  --cache data/model_frame_pace_01.pkl \
  --config configs/evaluation/pace_01_rolling.json \
  --output artifacts/pace_01_rolling_20260831
```
