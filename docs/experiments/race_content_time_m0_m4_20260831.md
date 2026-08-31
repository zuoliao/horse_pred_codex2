# Race-content time study M0–M4

**Registered:** 2026-08-31 (JST)  
**Raw fingerprint:** `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`  
**Primary config:** `configs/performance/pv_001_race_content_time.json`

## Question and boundaries

The study asks whether a dominant past win and a close past loss contain useful information that ordinal finish history and Elo discard. Current-race result fields remain forbidden inputs. Every content value is computed after a past race and becomes visible only on a later date. Odds are unused, 2025 is excluded before feature generation and selection, and 2024 is reserved for one frozen development comparison.

This is a staged study, not one bundled model change:

| Stage | Hypothesis | Change |
|---|---|---|
| M0 / PV-00 | The local time, margin, last-3F and passing-order fields can support PIT-safe historical content | Aggregate quality audit only |
| M1 / PV-01 | Signed past-race time gaps add information beyond finish/rating history | One opt-in historical feature |
| M2 / PV-02 | A continuous margin-aware pairwise actual improves the frozen standalone rating | Raw mapping rejected: ranking up, probability down |
| M3 / PV-03 | Temporal calibration repairs the changed rating-score scale | Supported standalone on rolling years and 2024 |
| M4 / PV-04 | Absolute margin-aware score adds incremental LightGBM information | Inconclusive versus PV-01; not adopted |
| M5 / PV-05 | Margin-minus-ordinal score isolates the margin-specific state | Probability points up, intervals cross zero; not adopted |
| M6 / PV-06 | Raw margin tokens refine equal-clock results | Inconclusive at the frozen 2022 gate; 2024 remained closed |

Raw margin refinement, last-3F, passing position, absolute speed figures and track variants are separate hypotheses. They are not mixed into PV-01.

## M0 evidence

The audit hard-gated rows to 2013–2023 before inspecting result content and used the existing flat-race definition (`芝`/`ダート`, no obstacle class). The scope was 519,414 runners and 36,569 races.

### Time and margin

- All 515,983 numeric finished/demoted rows had a parseable `M:SS.t` time. The 3,429 absent times were exactly DNF 1,611, scratch 846 and exclusion 972. Two disqualifications retained a clock.
- A winner-relative time gap was derivable in all 36,569 races. Runner gap median was 1.0 seconds, p99 6.0, p99.9 11.2 and maximum 83.6, so an unbounded historical mean is unsafe.
- 9,382 of 36,565 fourth-place rows (25.66%) were within 0.3 seconds of the winner. This is direct evidence that ordinal fourth place hides materially different performances.
- The first distinct nonwinner was at least 0.5 seconds behind in 5,453 races, at least 1.0 second behind in 943 and at least 2.0 seconds behind in 39. Dominant wins are therefore observable.
- The source contains 34 margin tokens. `ハナ`, `アタマ`, `クビ`, fractional lengths, `大` and `同着` dominate. Margin is the gap to the preceding arrival, not a cumulative winner gap. Thirteen compound `+` tokens and all four official-rank/time inversions occurred in demotion races.
- Times are rounded to 0.1 seconds: 11,291 nonwinners (2.36%) shared the winner's clock. Margin may refine these ties later, but converting lengths to seconds in PV-01 would add an arbitrary scale.

### Last 3F and passing order

- Last 3F was present for 515,980 of 515,983 numeric finishers and parseable wherever present. Every race had at least two valid values. Finish percentile versus last-3F percentile had 2014–2021 Spearman `.7361`, but the winner was fastest only 37.60% of the time. It is a promising distinct feature, not an outcome-label replacement.
- Every one of the 517,132 nonmissing passing-order strings matched a numeric hyphen-separated grammar; numeric finishers had 100% coverage. Segment counts vary from one to four. The one-segment case is the Niigata straight 1000m, whose value cannot safely be called a corner position. A later experiment should use first/last/position-gain semantics and explicit structural missingness.

### Interpretation

The evidence supports a time-derived within-race historical content feature first. It does not establish that raw absolute time is comparable across distance, surface, course, going, pace or track speed. The initial value therefore uses only within-race differences and distance normalization.

## M1 frozen definition

For race distance `d_r` metres and official time `t_ir` seconds, let `t_1r` be the official winner time and `t_2r` the fastest nonwinner time. For a clean race with one official winner:

```text
winner:     raw_ir = +(t_2r - t_1r) * 1000 / d_r
nonwinner:  raw_ir = -(t_ir - t_1r) * 1000 / d_r
content_ir = clip(raw_ir, -5, +5)
```

The sign is therefore interpretable: a larger positive value is a more separated win, a close fourth is just below zero, and a large defeat is negative. Dead-heat winners receive zero. Races containing demotion or disqualification are excluded from content updates because official order and physical clock order can diverge. DNF receives no content observation; workload history still updates. Scratch/exclusion behavior remains unchanged.

For target date `d`, the one candidate feature is the target horse's exponentially decayed mean over prior dates only:

```text
weight(r, d) = 2 ** (-(d - date_r) / 90)
race_content__decay_90d__mean_signed_time_gap_per_1000m
    = sum(weight * content) / sum(weight)
```

The fixed `5 sec / 1000m` cap is above the ordinary tail but prevents one stopped-yet-officially-finished observation from dominating the mean. Missing history remains missing, never zero-imputed by the feature builder.

### Pre-2024 representation screen

A standalone race-ranking diagnostic compared signed-time summaries without LightGBM. Selection used annual 2018–2021 metrics only; 2022 was confirmation and 2023 was descriptive. The 90-day decay was strongest among the signed candidates:

| Representation | 2018–2021 mean NDCG@3 | 2018–2021 mean Top-1 | 2022 NDCG@3 | 2022 Top-1 |
|---|---:|---:|---:|---:|
| last 1 | .30620 | .13770 | .31976 | .15113 |
| last 3 mean | .34209 | .17003 | .35430 | .18451 |
| last 5 mean | .34629 | .17485 | .35295 | .18388 |
| 90-day decay mean | **.35747** | **.18355** | **.36608** | **.18892** |

These are standalone ranking diagnostics, not LightGBM improvement claims. They fix one temporal representation before the 2024 comparison.

## M1 comparison protocol

- Control: exact `abl_006_drop_field_relative` semantic groups, 253 columns.
- Candidate: the same groups plus the one frozen `race_content_time` column, 254 columns.
- Model fit: 2014–2021; LightGBM early stopping: 2022; temperature only: 2023.
- Evaluation: one-shot 2024 development on the identical strict race population, with the known Kyoto coverage warning.
- Uncertainty: paired 4-race-date moving-block bootstrap, 10,000 resamples.
- Binary and LambdaRank are judged separately with the existing probability/ranking paths in the config. A single point metric cannot accept the feature.
- 2025 rows receive no content feature and are removed before cached experiment operations. Final confirmation remains 2026+ prospective.

## M2 and M4 design constraints

PV-02 through PV-05 executed this branch in smaller stages. The hard-actual replacement changed score scale, so PV-02's raw probability gate failed despite significantly better 2022 ranking. PV-03 then preregistered previous-year temperature calibration and supported the margin-aware rating standalone. PV-04 and PV-05 tested one absolute score and one margin-minus-ordinal score separately inside the PV-01 LightGBM; neither established incremental value.

Graded LambdaRank remains last because it requires integer relevance and a global `label_gain`; continuous margin cannot be passed directly. A fixed-bin experiment also changes the training objective away from the current win/top-three emphasis and applies only to LambdaRank. Its evaluation must retain the current fixed ranking metrics rather than redefining success around the new label.

## M1 result

The registered run used commit `92f9a268c1525b8d4fc3829eb580a39555d3faea` with a clean worktree. The augmented cache retained all 533,853 rows and all old 268 columns exactly: schema, identity, values and NaN positions had zero mismatches and maximum absolute difference zero. The one new column was nonmissing for 440,367 of 489,674 pre-2025 model rows and for zero 2025 rows. The 253-column control predictions reproduced `abl_006_drop_field_relative` exactly.

| Model | Features | NDCG@3 | Top-1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|
| Binary control | 253 | .49126 | .29122 | 2.08552 | .82985 |
| Binary candidate | 254 | **.49757** | **.29646** | **2.07867** | **.82767** |
| LambdaRank control | 253 | .49243 | .28810 | 2.08466 | .82995 |
| LambdaRank candidate | 254 | **.49413** | **.29056** | **2.07783** | **.82810** |

Paired improvement is positive when the candidate is better:

| Model | NDCG@3 | Top-1 | Log Loss | Brier | Decision |
|---|---:|---:|---:|---:|---|
| Binary | +.00631 `[+.00145,+.01117]` | +.00524 `[−.00429,+.01431]` | +.00685 `[+.00050,+.01312]` | +.00217 `[+.00017,+.00410]` | accept on probability and ranking paths |
| LambdaRank | +.00170 `[−.00370,+.00699]` | +.00246 `[−.00931,+.01395]` | +.00683 `[−.00034,+.01454]` | +.00184 `[−.00047,+.00431]` | inconclusive; all point estimates improve but primary intervals cross zero |

The feature took 3.60% of Binary gain and 6.44% of LambdaRank gain. Its pre-2025 Spearman correlation with the existing 90-day mean finish was `−.812`, materially below IMP-005's `−.938` redundancy but still substantial. Binary gains were positive on both turf and dirt and all field-size bands for NDCG; proper-score regressions remained in sprint Log Loss (`−.00056`) and very-large-field Log Loss/Brier (`−.00842/−.00056`). These slices are descriptive and were not used to alter the feature.

PV-01 is adopted as the current Binary development baseline and must be confirmed prospectively in 2026+. The conservative LambdaRank baseline remains the 253-feature control; the 254-feature point improvement is retained as an inconclusive prospective candidate. The Binary-versus-LambdaRank family comparison still has intervals crossing zero.

Machine-readable aggregate: `experiments/race_content_20260831/summary.json`. Full model, prediction and bootstrap artifacts remain local under `artifacts/pv_001_*`; the cache remains ignored under `data/`.

## Sources

- JRA, 競馬用語辞典（ハナ・アタマ・クビ・馬身・大差）, accessed 2026-08-31: https://www.jra.go.jp/kouza/yougo/c10010_list.html
- JRA, 成績表の見方, accessed 2026-08-31: https://www.jra.go.jp/datafile/seiseki/report/mikata3.html
- LightGBM parameters (`lambdarank`, integer labels and `label_gain`), accessed 2026-08-31: https://lightgbm.readthedocs.io/en/latest/Parameters.html
- Burges, *From RankNet to LambdaRank to LambdaMART*, Microsoft Research Technical Report MSR-TR-2010-82, 2010: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
- Kovalchik, *Extension of the Elo rating system to margin of victory*, International Journal of Forecasting 36(4), 2020, DOI: https://doi.org/10.1016/j.ijforecast.2020.01.006
