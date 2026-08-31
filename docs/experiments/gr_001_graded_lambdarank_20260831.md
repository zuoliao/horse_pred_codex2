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

## Frozen-run result

The clean run used commit `9d3eeabe49c3755d6b15d7bd5075e1af0a6306d4`
with `dirty=false`. Both arms used the fixed 253-feature scope. The control
stopped at iteration 174 and used 2021 temperature `.8917951`; the candidate
stopped at iteration 89 and used temperature `.8508479`. The common 2022
population was 3,176 races / 43,537 runners.

| 2022 calibrated metric | Control | Candidate | Improvement |
|---|---:|---:|---:|
| Log Loss | 2.1256279 | 2.1410193 | -.0153914 |
| Brier | .8417831 | .8437899 | -.0020068 |
| NDCG@3 | .4744598 | .4704739 | -.0039859 |
| Top-1 | .2611776 | .2573992 | -.0037783 |

Paired 95% intervals were Log Loss `[-.0206774,-.0102267]`, Brier
`[-.0034862,-.0005797]`, NDCG@3 `[-.0083134,+.0003012]`, and Top-1
`[-.0104139,+.0029007]`. Thus the candidate significantly worsened both
proper probability scores, failed both acceptance paths, and is **rejected**.

The preregistered descriptive field-size check does not identify a defensible
rescue. Log Loss worsened in all four bands. NDCG@3 improved slightly in small
fields (`+.00064`, 318 races) and more in fields of at least 17 (`+.01020`, 235
races), but worsened in medium (`-.00367`) and large (`-.00705`) fields. These
post-run slices are neither uncertainty-qualified nor eligible for label
retuning.

Retain the existing top-three relevance labels. The useful conclusion is not
that all finer labels are impossible, but that this particular exchange--lose
second-versus-third order and add an upper-half bucket--diluted the current
win/top-three objective. No 2023--2025 outcome was opened for GR-001.
