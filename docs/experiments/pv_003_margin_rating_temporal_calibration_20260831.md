# PV-03 Temporal calibration of the margin-aware rating

Status: preregistered after PV-02 and before inspecting any annual calibrated
comparison or any 2023/2024 outcome for this experiment.

## Motivation and hypothesis

PV-02's continuous margin actual significantly improved 2022 NDCG@3 and Top-1 but
significantly worsened raw Log Loss. Its score standard deviation was smaller than
the ordinal control's, while an explicitly non-decision-grade same-sample temperature
check reversed the Log Loss comparison. The single hypothesis here is therefore:

> Margin-aware Elo improves ordering, but changes the rating-score scale; fitting
> the same one-parameter temperature on a strictly preceding year will recover a
> better out-of-period probability distribution than ordinal Elo.

The rating state, pair actual, `tau=0.125`, initial 1500, K=48, Elo scale=200,
same-date batching, scoring population and softmax family are frozen from PV-02.
Only the already-existing temperature calibration is applied independently to each
arm. No LightGBM feature is added.

## Pre-development temporal test

To avoid using 2020–2021 to determine a scale for earlier races, `tau` must first be
reproduced as 0.125 using only positive adjacent-rank gaps in 2014–2017. Then:

| Temperature fit | Out-of-period evaluation |
|---:|---:|
| 2018 | 2019 |
| 2019 | 2020 |
| 2020 | 2021 |
| 2021 | 2022 |

Each arm gets its own temperature, fitted with the frozen optimizer and race Log
Loss. Rating state remains prequential and is never reset at year boundaries.
Ranking metrics use raw scores and therefore must exactly reproduce the corresponding
PV-02 ordering; probability metrics use only the previous-year temperature.

The gate passes only if all of the following hold:

1. Candidate calibrated Log Loss improves in at least three of four evaluation
   years.
2. The unweighted annual-macro Log Loss improvement is strictly positive.
3. The annual-macro Brier improvement is at least -0.001.
4. On the latest 2022 evaluation, the paired four-date moving-block 95% interval
   lower bound for Log Loss improvement is strictly positive.

This is stricter than selecting on one favorable annual point. PV-02 already exposed
the 2022 raw comparison and motivated this hypothesis, so even a pass remains
development evidence rather than untouched final confirmation.

## Conditional 2024 stage

If and only if the temporal gate passes, reload raw data with a hard pre-normalization
cut at 2024, fit each frozen arm's temperature on 2023, and evaluate 2024 once.
Adoption for a later LightGBM integration experiment requires a strictly positive
paired Log Loss interval lower bound plus Brier, NDCG@3 and Top-1 guardrails matching
PV-02. Raw probability metrics, calibrated probability metrics, temperatures,
coverage and paired intervals are all retained.

If the gate fails, 2023 and 2024 remain unopened. In all cases, 2025 is excluded
before normalization and final odds/ROI are unused. A supported result would only
authorize a separate PV-04 LightGBM integration experiment; it would not prove that
the integrated model improves.
