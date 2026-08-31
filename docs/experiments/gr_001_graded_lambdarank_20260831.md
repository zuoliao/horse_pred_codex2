# GR-001 graded field-half LambdaRank

## Status and hypothesis

This protocol was frozen before any 2022 model metric was run. It is
independent of PV-06 and changes only LambdaRank training relevance. Features,
LightGBM parameters, `label_gain`, truncation, evaluation labels, probability
calibration, and race population remain fixed.

The current target is `1st=3, 2nd=2, 3rd=1, other=0`. It directly distinguishes
second from third but provides no pairwise supervision within fourth and below.
The candidate tests one hypothesis: the second/third distinction is less useful
than recovering coarse latent-performance order below the top three.

## Frozen labels

| Official finish | Control | Candidate |
|---|---:|---:|
| 1st | 3 | 3 |
| 2nd | 2 | 2 |
| 3rd | 1 | 2 |
| 4th through `ceil(field_size/2)` | 0 | 1 |
| Lower half / nonfinish | 0 | 0 |

Dead heats use the same official-position mapping for every tied runner.
`label_gain=[0,1,3,7]`, `lambdarank_truncation_level=6`, and all other model
parameters remain fixed. Evaluation always uses the existing public
`1st=3,2nd=2,3rd=1,other=0` NDCG target; candidate training labels never
redefine success.

## Train-only label audit

Across the complete 2014--2021 train split (25,322 races / 360,318 runners),
the candidate changes 133,439 rows (37.03%) and increases differently labelled
within-race pairs from 929,632 to 1,700,035 (+82.87%). It adds 795,709 pairs and
removes 25,306, mainly the former second-versus-third comparisons. Comparable
gain-gap mass rises 27.33%. Every race has a valid winner; DNF/non-numeric rows
remain relevance zero; each dead-heat group maps consistently.

This is a materially different supervision signal, so it warrants one test.
It is deliberately coarser than full finishing-order relevance to avoid making
noisy lower-place differences dominate the win-oriented objective.

## Time isolation and fixed model scope

- Model fit: 2014--2019.
- Early stopping: 2020.
- Temperature calibration: 2021.
- First and only gate: 2022.
- 2023--2025 outcome rows used: zero.

The conservative `abl_006_drop_field_relative` feature scope is fixed at 253
columns with ordered-column SHA-256
`fd8735cf6f8472a5c7322e3622c83fde1ff7720b022b75637745c96a1bc1062f`.
PV-01 and other later features are not added. The restricted temporal fit is a
hypothesis gate, not a replacement production fit.

## Preregistered decision

Both arms receive separate 2021 temperature calibration. On the identical
2022 population, ranking acceptance requires paired NDCG@3 improvement 95%
lower bound above zero, with Log Loss improvement at least `-.002`, Brier at
least `-.001`, and Top-1 at least `-.005`. The existing probability path is
also retained: Log Loss point improvement at least `.002` with lower bound
above zero, Brier non-worse, NDCG@3 at least `-.002`, and Top-1 at least
`-.005`.

Uncertainty uses a four-date moving-block bootstrap, 10,000 resamples, seed
`20240830`. Passing either full path accepts. A point guardrail violation or a
primary interval wholly below zero rejects; otherwise the outcome is
inconclusive. Field-size slices are descriptive only. The experiment always
stops after 2022.
