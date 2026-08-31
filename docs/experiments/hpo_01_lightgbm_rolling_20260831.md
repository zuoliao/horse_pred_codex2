# HPO-01 bounded LightGBM rolling search preregistration

Date: 2026-08-31 JST  
Status: complete; Binary no change, LambdaRank selected candidate rejected

## Hypotheses and fixed scope

HPO-01 is two independent experiments. `HPO-01B` asks whether one bounded
LightGBM parameter change improves the 254-feature PV-01 Binary incumbent.
`HPO-01R` asks the same question for the conservative 253-feature LambdaRank
incumbent. Features, targets, probability mapping, seed, learning rate,
maximum estimator count, and early stopping remain fixed. Feature addition and
parameter selection are not mixed.

The executable sources of truth are:

- `configs/performance/hpo_01b_binary_rolling.json`
- `configs/performance/hpo_01r_lambdarank_rolling.json`

Binary uses the PV-01 ordered 254-column hash
`e5228bb4ffd605888b7266030d5e1e9f0931e8468b6fbbf124f3cea60905e51d`.
LambdaRank uses the lean ordered 253-column hash
`fd8735cf6f8472a5c7322e3622c83fde1ff7720b022b75637745c96a1bc1062f`.
The Ranker retains the original top-three relevance labels,
`label_gain=[0,1,3,7]`, and `lambdarank_truncation_level=6`. Truncation tuning
is explicitly outside HPO-01.

## Bounded profiles

The control has `num_leaves=31`, unlimited depth, `min_child_samples=100`,
L1 `0`, L2 `1`, feature fraction `.9`, bagging fraction `.9`, and the
LightGBM default `max_bin=255`. Eleven one-at-a-time candidates change exactly
one setting:

| Profile | Change |
|---|---|
| `leaves_15` | `num_leaves=15` |
| `leaves_63` | `num_leaves=63` |
| `depth_6` | `max_depth=6` |
| `min_child_50` | `min_child_samples=50` |
| `min_child_200` | `min_child_samples=200` |
| `l1_1` | `reg_alpha=1` |
| `l2_5` | `reg_lambda=5` |
| `feature_fraction_075` | `colsample_bytree=.75` |
| `bagging_fraction_075` | `subsample=.75` |
| `max_bin_127` | `max_bin=127` |
| `max_bin_511` | `max_bin=511` |

This is not a Cartesian grid. Candidate settings will not be combined after
the result. Each family therefore records eleven selection comparisons, not
one generic HPO comparison.

## Temporal selection firewall

All profiles use three EVAL-ROLL folds for selection:

| Evaluation | Fit | Early stop | Temperature |
|---:|---:|---:|---:|
| 2020 | 2014--2017 | 2018 | 2019 |
| 2021 | 2014--2018 | 2019 | 2020 |
| 2022 | 2014--2019 | 2020 | 2021 |

The deterministic selector freezes at most one candidate per family. Only
that candidate and the control are then fitted and scored in the 2023 fold
(fit 2014--2020, early stop 2021, temperature 2022, evaluation 2023).
Non-selected candidates must have zero 2023 prediction rows. The cache is cut
at 2023 before feature resolution, and 2024/2025 outcome-use counters must
remain zero.

The 2020--2022 selection years and 2023 confirmation year have prior project
exposure and are not untouched holdouts. The firewall is local to HPO-01: it
prevents choosing the best of eleven profiles on 2023. Selection intervals are
descriptive and are not presented as multiplicity-adjusted evidence.

## Selection and confirmation

Binary primary is year-macro race Log Loss improvement. A profile is eligible
only if the improvement is at least `.002`, positive in at least two of three
selection years, Brier improvement is at least `0`, NDCG@3 at least `-.002`,
and Top-1 at least `-.005`.

Ranker primary is year-macro NDCG@3 improvement. It must be positive in at
least two selection years, with Log Loss improvement at least `-.002`, Brier
at least `-.001`, and Top-1 at least `-.005`.

Among eligible profiles, primary results within `1e-4` of the best are tied.
Binary then uses Brier, NDCG, Top-1, preregistered complexity rank, and profile
ID. Ranker uses Top-1, Log Loss, Brier, complexity rank, and profile ID. No
eligible profile gives `no_change` without exposing a candidate to 2023.

The 2023 confirmation uses a paired four-date block bootstrap with 10,000
resamples, seed `20240830`, and a 95% percentile interval. Binary accepts only
if its Log Loss improvement is at least `.002`, the interval lower bound is
strictly positive, and all Binary guardrails pass. Ranker accepts only if the
NDCG interval lower bound is strictly positive and all Ranker guardrails pass.
A wholly negative primary interval or a guardrail failure rejects; otherwise
the result is inconclusive. Only `accept` changes the family incumbent.

## Artifact and execution contract

The runner writes the config and resolved base configs, feature schema,
selection and confirmation metrics, both bootstrap stages, long scoring
predictions, race metrics, models, feature importance, run metadata, and a
hashed artifact manifest. It writes atomically and refuses overwrite. Odds and
popularity columns are forbidden from the prediction artifact.

The real-data runs must start from a clean preregistration commit. Before
interpreting a candidate, the control rows must reproduce EVAL-ROLL-001 under
the same cache SHA, ordered feature hash, code, and LightGBM version. The
existing EVAL baseline took about 90 seconds for eight fits. HPO-01 performs
`12*3 + 2*1 = 38` fits per family, so the two runs are expected to take roughly
14--20 minutes on the same machine, excluding cache reads and artifact I/O.

```bash
uv run horse-pred run-hpo-study \
  --cache data/model_frame_race_content_time_20260831.pkl \
  --config configs/performance/hpo_01b_binary_rolling.json \
  --output artifacts/hpo_01b_binary_rolling_20260831

uv run horse-pred run-hpo-study \
  --cache data/model_frame_race_content_time_20260831.pkl \
  --config configs/performance/hpo_01r_lambdarank_rolling.json \
  --output artifacts/hpo_01r_lambdarank_rolling_20260831
```

## Result

Both runs used clean commit
`87bde38e224b1fd21e3e22db906d217194f4b4f8`, the frozen PV-01 cache, no
odds, and zero 2024/2025 rows.

### Binary

No profile met the 2020--2022 eligibility rule, so no candidate was exposed
to 2023. `leaves_15` had the largest eligible-looking Log Loss point
improvement (`+.00181`, two of three years) but missed the preregistered
`+.002` minimum. `l2_5` improved Log Loss in all three years (`+.00152`) but
failed the NDCG guardrail. The decision is `no_change`; retain the incumbent
parameters and do not combine favorable OAT settings.

### LambdaRank

`feature_fraction_075` was selected on 2020--2022: NDCG improvement
`+.00179` in two of three years, with its selection guardrails passing. On the
single frozen 2023 confirmation it worsened every point metric:

| Metric | Improvement | 95% paired interval |
|---|---:|---:|
| NDCG@3 | -.00058 | `[-.00473,+.00363]` |
| Top-1 | -.00063 | `[-.01015,+.00913]` |
| Log Loss | -.00508 | `[-.00915,-.00099]` |
| Brier | -.00140 | `[-.00262,-.00017]` |

The NDCG confirmation failed and both probability guardrails were wholly
adverse. The profile is rejected and the incumbent parameters remain
unchanged. Non-selected profiles have zero 2023 prediction rows.

Tracked evidence is `experiments/hpo_20260831/summary.json`; complete local
artifacts are under `artifacts/hpo_01b_binary_rolling_20260831/` and
`artifacts/hpo_01r_lambdarank_rolling_20260831/`.
