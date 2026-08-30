# PV-06 raw margin-token clock refinement

## Status and isolation

This protocol was frozen before any 2022 performance metric was run. It is a
standalone rating actual-score experiment and does not change LightGBM,
PV-01, last-3F, passing order, condition-specific tau, or ranking labels.
Only 2014--2021 raw outcomes may define the token mapping. The first and only
performance gate in this experiment is 2022; 2023--2025 outcomes remain closed.

## Train-only evidence

The approved raw fingerprint is
`270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`.
The 2014--2021 eligible-flat population contains 26,585 races. Twenty-two
demotion/disqualification races are excluded as whole races, leaving 26,563
clean races and 348,745 adjacent distinct-rank edges.

- Every clean edge has exactly one recognized ordinary margin token and a
  parseable adjacent displayed-clock gap; clean coverage is 100%.
- There are no clean clock-order inversions.
- Token order versus displayed clock gap has Spearman `.95737`; the 21 token
  means are strictly increasing.
- Equal clocks account for 81,347 edges (23.326%) and use only `ハナ`,
  `アタマ`, `クビ`, or `1/2`.
- All 11 compound `+` tokens and all three observed clock inversions occur in
  the 22 excluded demotion/disqualification races.
- Same official-rank dead heats remain neutral.

This is sufficient to use the four tokens as ordered dequantization evidence,
but not to claim a universal physical horse-length-to-seconds conversion.

## Frozen transformation

The control is the supported PV-03 global pairwise Elo: initial 1500, K=48,
scale=200, `tau=.125 seconds/1000m`, with previous-year temperature.

The candidate differs only when two distinct official finish groups share the
same displayed 0.1-second clock:

| Adjacent token | Provisional gap |
|---|---:|
| `ハナ` | .02 seconds |
| `アタマ` | .04 seconds |
| `クビ` | .06 seconds |
| `1/2` | .08 seconds |

Within each maximal equal-clock block, provisional adjacent gaps are scaled
proportionally only when their sum exceeds `.08` seconds, so the whole block
remains strictly inside the unresolved clock tick. A nonadjacent pair receives
the sum of intervening refined edges. The result is normalized per 1000m and
passed through the already-frozen logistic actual rule.

Unequal clocks remain bit-for-bit on the PV-03 rule. Same-rank pairs remain
neutral. Any missing/unrecognized boundary falls back to the PV-03 equal-clock
actual `.5`; demotion, disqualification, or unknown-status races retain the
existing whole-race ordinal fallback.

## Preregistered 2022 gate

Each arm is fitted chronologically through 2022, calibrated using 2021 only,
and evaluated on the common 2022 scoring population. Raw metrics are
descriptive; calibrated race Log Loss is primary.

The candidate is `go` only if the paired four-date moving-block bootstrap with
10,000 resamples and seed `20240830` has Log Loss improvement lower bound
strictly above zero, while Brier improvement is at least `-.001`, NDCG@3 at
least `-.002`, and Top-1 at least `-.005`. A failed guardrail or wholly negative
Log Loss interval rejects; otherwise the result is inconclusive. The study
always stops after 2022 regardless of result.

Machine-readable local outputs include the train audit, frozen mapping,
config, per-race 2022 metrics, runner predictions, metrics, and artifact
manifest.

## Frozen-run result

The clean run used commit `ea725d94821886b06dc717dad5adf5e50cb6a3cc`
with `dirty=false`. The mapping gate reproduced all train-only counts and all
eight checks passed. The common 2022 population was 3,176 races / 43,537
runners. Temperatures fitted on 2021 were `.5006867` for the control and
`.5004095` for the candidate.

| 2022 calibrated metric | Control | Candidate | Improvement |
|---|---:|---:|---:|
| Log Loss | 2.3839791 | 2.3839499 | +.0000292 |
| Brier | .8901650 | .8901497 | +.0000153 |
| NDCG@3 | .3656247 | .3664588 | +.0008341 |
| Top-1 | .1786839 | .1818325 | +.0031486 |

Paired 95% intervals were Log Loss
`[-.0002222,+.0002772]`, Brier `[-.0000365,+.0000673]`, NDCG@3
`[-.0005850,+.0021792]`, and Top-1 `[+.0009248,+.0055728]`.

The candidate therefore improved Top-1 significantly and every calibrated
point estimate, but the preregistered primary Log Loss interval crossed zero.
The result is **inconclusive**. Do not promote the refinement, do not reject it
as harmful, and do not open 2024 for this branch. The route is complete at its
specified 2022 gate.
