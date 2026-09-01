# S1 Two-axis past-race value

## Decision

S1 is complete. The historical **performance residual axis is supported**. The standalone **field-quality axis is inconclusive and not accepted**. The joint arm improved, but the field-quality increment on top of performance was weak in Binary and negative in LambdaRank; this is not evidence that both axes are independently useful.

The current formal controls are not automatically replaced. C1 is the clean supported candidate, while C3 remains a follow-up candidate whose apparent advantage may be driven mainly by performance. The next recommended research task is **S3: condition-adjusted performance target**. S2 and S3 were not executed.

## Scope and controls

- Discovery/selection outcomes: 2014–2022 only.
- Evaluation years: 2020, 2021, 2022.
- Binary C0: accepted PV-01 254-feature control.
- LambdaRank C0: conservative lean 253-feature control.
- C1: performance residual only.
- C2: race-constant full-field quality only.
- C3: both columns, without an interaction.
- Fits: 3 folds × 2 families × 4 arms = 24.
- 2023/2024/2025 rows used: 0/0/0.
- Odds, popularity, and final market used: false.

## Primary results

All deltas below are positive when the candidate improves on its family-specific C0 control. Values are unweighted macro averages across the three evaluation years.

| Family / arm | Δ Log Loss | Δ Brier | Δ NDCG@3 | Δ Top-1 | Year direction | Classification |
|---|---:|---:|---:|---:|---|---|
| Binary C1 | +0.00755 | +0.00239 | +0.00247 | +0.00399 | LL 3/3; NDCG 2/3 | probability supported; ranking weakly supported |
| LambdaRank C1 | +0.00872 | +0.00279 | +0.00247 | +0.00080 | LL/NDCG 3/3 | probability supported; ranking weakly supported |
| Binary C2 | -0.00051 | +0.00010 | -0.00048 | +0.00053 | LL/NDCG 1/3 | rejected |
| LambdaRank C2 | -0.00258 | -0.00048 | +0.00033 | -0.00141 | LL 1/3; NDCG 2/3 | probability rejected; ranking inconclusive |
| Binary C3 | +0.00838 | +0.00267 | +0.00362 | +0.00620 | all listed metrics 3/3 | supported |
| LambdaRank C3 | +0.00543 | +0.00247 | +0.00227 | +0.00447 | LL/NDCG/Top-1 2/3 | probability supported; ranking weakly supported |

Binary C1 Log Loss improvement had a 95% date-block interval of `[+0.00366, +0.01141]`; LambdaRank C1 was `[+0.00545, +0.01211]`. Binary C3 improved NDCG@3 by `+0.00362`, interval `[+0.00084, +0.00637]`.

## Axis separation

- Performance alone (C1−C0): supported in both families on the probability path. Ranking point effects were positive, but their intervals narrowly crossed zero.
- Field quality alone (C2−C0): no stable incremental signal. Binary worsened Log Loss and NDCG macro; LambdaRank worsened Log Loss with a fully negative improvement interval.
- Performance given field quality (C3−C2): supported. This confirms that performance contains information field quality cannot replace.
- Field quality given performance (C3−C1): weak in Binary and rejected in LambdaRank. The direction is model-family dependent.
- Joint (C3−C0): supported, but does not establish two independent axes because C1 already carries the stable improvement.

The resulting case is **Case A: performance supported**, not Case C.

## Fixed-slice guardrails

The slice results were not used to change the feature definition.

- Binary C1 improved Log Loss and Brier in all six slices. It improved NDCG in large fields, open/graded races, and surface switches, but NDCG declined for history-0/new-race and ≥400m distance-change slices.
- LambdaRank C1 improved strongly for surface switches. Open/graded Top-1 and probability metrics declined; ≥400m distance changes also had weaker ranking and Log Loss.
- C3 gave strong Binary open/graded and surface-switch gains, but its LambdaRank open/graded Top-1 remained worse and its distance-change probability guardrail was mixed.

The evaluation race counts were: large field 4,999; history-0 winner 905; new race 847; open/graded 744; winner distance change ≥400m 854; winner surface switch 695.

## PIT and leakage audit

- Future rows appended after a target do not change earlier features.
- Changing a target-race result does not change that race's pre-race features.
- Same-date race ordering and same-date results do not change same-date emitted features.
- Field quality had zero race-constant violations and matched the full-starter pre-race Elo mean to numerical tolerance.
- The performance normalizer used 0 early-stopping, calibration, or evaluation rows in every fold.
- No future opponent result, leave-one-out field mean, odds, market field, or direct entity ID was used as a model feature.
- The local artifact contains 38 hashed files; complete hash verification passed.

## Interpretation and next task

The useful new information is the condition-adjusted historical performance residual. That favors testing whether the same performance concept should supervise a separate target, so **S3 should precede S2**. S2 remains queued because a race-wise probability objective addresses a different problem and S1 did not test it.

What cannot yet be claimed:

- C3 is not proven better than C1 in a model-family-stable way.
- Field quality is not shown to be an independently useful axis under this definition.
- No conclusion applies to 2024, 2025, final-market comparison, production-final performance, ROI, or profitability.

Machine-readable source: `experiments/s1_two_axis_race_value_20260901/summary.json`.
