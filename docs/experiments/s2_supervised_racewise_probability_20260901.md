# S2 Supervised race-wise probability conclusion

**Experiment:** `s2_supervised_racewise_probability_20260901`  
**Status:** completed  
**Decision:** race-wise objective as current-control replacement `rejected`; S1 performance feature `objective_robust_supported`  
**Selection scope:** 2020–2022 rolling-origin only; 2023/2024/2025 and market unused

## Outcome

Transparent Linear Conditional Logit did not outperform the matched nonlinear LightGBM Binary control, with or without the S1 performance feature. The probability path was rejected in both matched comparisons. The formal controls therefore remain Binary PV-01 254 features and conservative LambdaRank lean 253 features.

The S1 historical performance residual did replicate: it was supported when added to LightGBM Binary and when added to Conditional Logit. This strengthens the representation conclusion—past performance residual is useful input context—but does not support replacing the objective.

A paired difference-in-differences check found no clear objective-specific feature increment: the calibrated Log Loss interaction was `−0.00005 [−0.00367, +0.00353]` and NDCG@3 was `+0.00046 [−0.00253, +0.00352]`. The feature benefit is therefore broadly portable across the two tested objectives rather than uniquely enabled by Conditional Logit.

## Capacity gate

The preregistered gate used only model-validation years. Linear Conditional Logit recovered `94.39%`, `96.32%`, and `96.20%` of Binary's improvement over uniform Log Loss. No fold was below the `.75` threshold, so the conditional nonlinear grouped-softmax Stage N was not triggered. This was decided before evaluation aggregation.

The capacity-matched linear Binary diagnostic was much weaker on validation (`2.463`, `2.451`, `2.448` Log Loss versus Conditional Logit `2.132`, `2.147`, `2.186`). Thus choice-set likelihood clearly helps this linear utility comparison. Because the auxiliary Binary omits a free intercept and no evaluation predictions were retained, it is descriptive rather than a clean objective-only causal attribution. The rejected claim is narrower: the registered Conditional Logit does not replace the current nonlinear Binary model.

## Main comparisons

All deltas are candidate improvement; Log Loss/Brier signs have already been reversed so positive is better.

| Comparison | Δ Log Loss | 95% date-block interval | Positive years | Δ Brier | Δ NDCG@3 | Δ Top-1 | Probability path | Ranking path |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Binary B1 − B0 | +0.00742 | [+0.00365, +0.01123] | 3/3 | +0.00194 | +0.00293 | −0.00168 | supported | supported |
| Linear R0 − Binary B0 | −0.01816 | [−0.02611, −0.01022] | 0/3 | −0.00390 | −0.00089 | −0.00378 | rejected | rejected |
| Linear R1 − Binary B1 | −0.01822 | [−0.02604, −0.01047] | 0/3 | −0.00352 | −0.00043 | +0.00026 | rejected | inconclusive |
| Linear R1 − R0 | +0.00736 | [+0.00510, +0.00965] | 3/3 | +0.00232 | +0.00339 | +0.00236 | supported | supported |

Native `T=1` Log Loss gives the same direction: B1−B0 `+0.00800`, R0−B0 `−0.01808`, R1−B1 `−0.01871`, R1−R0 `+0.00736`. Therefore the race-wise losses were not rescued or reversed only by temperature calibration.

Calibration slopes after the independently fitted prior-year temperatures stayed near one: Binary B0 `.956–.986`, Binary B1 `.969–1.007`, Linear R0 `.925–.993`, and Linear R1 `.923–.994` across evaluation years. Fixed-bin ECE was low for all methods (`.0019–.0060`) but remains only a descriptive calibration diagnostic; the proper-score rejection of the linear candidates takes precedence.

## Temporal stability and fixed slices

- Race-wise versus Binary Log Loss worsened in all three evaluation years in both feature scopes.
- Performance residual improved Log Loss in all three years under both objectives.
- Binary B1 improved Log Loss in five of six fixed slices. The exception was winner distance change at least 400m (`−0.00389`), supporting continued caution around transition reliability.
- Linear R0 was especially weak in history-0 (`−0.10368` Log Loss), new races (`−0.07848`), and open/graded races (`−0.03735`) versus Binary B0.
- Within Conditional Logit, performance improved open/graded, large-field, surface-switch, and large-distance-change slices, but not history-0 or new-race Log Loss. The feature is missing and train-median imputed for a history-0 runner; changes in those slices come from whole-model refitting, regularization, and opponents' histories rather than a direct value for that runner.

Slices are descriptive guardrails without slice-specific confidence intervals and carry multiple-comparison risk. They are hypothesis context, not a basis for modifying S2 after seeing results.

## Model roles after S2

- **Binary:** remains the accepted development control and the strongest coherent-probability model in this experiment. Binary + S1 performance remains a supported rolling candidate, but this rerun uses the same evaluation years as S1 and a separately sorted/subsampled matched run. It is a robustness reproduction, not an exact rerun or new temporal evidence; the formal control is not automatically replaced.
- **LambdaRank:** remains the conservative 253-feature ranking control. S2 did not refit it because that would introduce a feature-scope mismatch into the primary objective comparison. Existing S1 evidence still says performance input helps LambdaRank.
- **Conditional Logit:** retained as a transparent supervised choice-model baseline. It demonstrates the semantics and calibration of one-race one-choice-set likelihood, but is not competitive with current nonlinear Binary.
- **Inter-horse / race-set model:** not supported as the automatic next step. Conditional Logit couples runners only through its denominator and learns no inter-runner interaction. S2 neither tests nor rejects a true set-interaction model, but direct choice normalization alone did not close the gap.

## Protocol and quality

Two implementation amendments occurred before evaluation metrics were generated:

1. finite iteration-limit solutions with gradient norm `<= .01` were accepted as operationally converged, and grouped softmax was vectorized without changing its formula;
2. one nonconverged L2 candidate in the auxiliary linear-Binary diagnostic was excluded while converged grid candidates remained eligible.

Both amendments are recorded in the preregistration. No fold, L2 grid, feature scope, capacity threshold, evaluation metric, or acceptance rule changed. The final primary Conditional Logit fits all selected converged normally; invalid candidates occurred only in the auxiliary diagnostic.

- source: 488,715 pre-cutoff raw rows / 34,504 races
- model frame: 403,855 runners / 28,498 races
- evaluation: 9,532 races over 2020–2022
- 2023/2024/2025 rows used: `0 / 0 / 0`
- odds/final market/direct entity ID features: `0`
- native/calibrated race-probability maximum sum error: `1.22e-15 / 1.33e-15`
- four-arm choice-set identity: PASS
- local artifact manifest: 27 files, all hashes verified
- fitted work: 6 LightGBM fits plus 36 optimizer-candidate fits across 18 model units
- full 1.3GB artifact, models, fold tables, and runner predictions remain Git-ignored

Post-run, non-selection diagnostics added coverage/drift, standardized performance coefficients (`+0.219`, `+0.227`, `+0.235`), race-constant linear-column detection, probability integrity, and the paired difference-in-differences calculation. Race-aware permutation dependence and auxiliary linear-Binary coefficients/predictions were not generated; both omissions are recorded as non-decision-impacting protocol deviations. The bootstrap intervals condition on the fitted models and do not include model-refit uncertainty. Evaluation year 2022 has now been used for S2 selection and cannot serve as fresh confirmation for a follow-up.

## Decision and next candidates

Do not change the formal control automatically and do not run the preregistered nonlinear Stage N after the fact; its gate did not trigger. S2's negative replacement result and S1 feature replication are both retained.

Recommended human-review candidates:

1. **A1 transition reliability:** the focused condition-transition hypothesis remains valid, and the large-distance-change probability slice is the one Binary-performance guardrail that worsened.
2. **A2 connection compression:** history-0/new races remain difficult and depend strongly on connection information; test shrinkage/redundancy without direct IDs.
3. **True Inter-horse set interaction:** defer until a distinct interaction hypothesis is specified. S2 does not justify moving directly to a large set model or DNN.

No next experiment was executed. 2024/2025 remain closed for selection.
