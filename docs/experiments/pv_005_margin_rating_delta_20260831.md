# PV-05 Margin-specific rating delta in LightGBM

Status: preregistered after PV-04 and before building or evaluating the delta cache.

## Hypothesis

PV-04's absolute margin-aware score was highly correlated with existing ordinal Elo
(`rho≈.97`) and took high model importance without incremental 2024 improvement. A
tree model may substitute it for the existing rating without efficiently isolating
the smaller margin-driven change. PV-05 therefore adds exactly one derived PIT state:

```text
margin_rating__delta_vs_ordinal_score_pre
    = margin-aware R5 raw score before race
    - same-spec ordinal R5 raw score before race
```

Both histories use initial 1500, K=48, scale=200, global state, the same event stream
and same-date batching. The delta consequently changes only because of the continuous
time-margin actual. It is zero for cold starts and never uses the current race result.

## Comparison and gates

The primary control remains PV-01's 254 features; the candidate adds only the delta
for 255. The absolute PV-04 score is not present. Cache construction must preserve all
PV-01 rows/features exactly, cover every pre-2025 scoring row, leave 2025 missing, and
reproduce the PV-01 control exactly before candidate evaluation.

Model configs, 2014–2021 fit, 2022 early stopping, 2023 output calibration, strict
2024 population and paired 10,000-draw four-date bootstrap remain unchanged. The
PV-04 probability and ranking paths are reused. LambdaRank also needs to pass the
same applicable path versus lean 253 before promotion.

This hypothesis was formulated after observing PV-04's 2024 result. Accordingly,
nominal intervals understate total research-selection uncertainty. Even a pass is
only a development candidate for 2026+ prospective confirmation. No 2025 result,
odds, ROI, graded label, extra rating column or LightGBM hyperparameter is used.
