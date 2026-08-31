# SHIMBA-FILTER-001: Binary fit-population ablation

## Status

Complete; the fit-only exclusion was rejected. No 2024 or 2025 outcome was
opened for this hypothesis.

## Question

New-horse races are a cold-start population: the entrant-history features that
normally carry much of the model signal are unavailable. The one tested
hypothesis is therefore:

> Removing new-horse races from the Binary model's gradient-fitting rows reduces
> label noise enough to improve all-race probability quality.

This is not a proposal to remove new-horse races from evaluation, purchase
eligibility, early stopping, calibration, or point-in-time feature-state
updates. It is also not a separate cold-start model experiment.

## Pre-2024 audit evidence

The audit used only 2014--2023 raw/cache rows.

- Exact race definition: after replacing non-breaking spaces and removing
  whitespace, `race_class` contains `新馬`.
- On 447,728 strict cache rows in 2014--2023, that definition was exactly equal
  to `context__class_tier == 0`: zero row mismatches and zero within-race class
  conflicts.
- Raw flat population: 2,925/33,245 races (8.798%) were new-horse races.
  Strict-cache population: 2,775 new-horse races.
- In the four rolling training populations, the candidate would exclude:

| Evaluation fold | Fit years | Excluded races | Excluded rows |
|---|---:|---:|---:|
| 2020 | 2014--2017 | 1,082 / 12,665 | 14,828 / 183,710 |
| 2021 | 2014--2018 | 1,360 / 15,815 | 18,669 / 228,208 |
| 2022 | 2014--2019 | 1,646 / 18,966 | 22,668 / 271,740 |
| 2023 | 2014--2020 | 1,935 / 22,150 | 26,706 / 316,282 |

- In 2020--2023 evaluation rows, new-horse entrants had no prior horse starts,
  initial global Elo, and missing PV-01 history by construction. Connection
  histories remained almost fully available.
- Existing rolling predictions show that new-horse races are harder, not
  unlearnable. Binary year-macro new-horse Log Loss was 2.2718 versus 2.1297 in
  non-new-horse races, while still materially beating a uniform field baseline.

These descriptive findings motivate only the fit-population ablation below.
They do not establish that new-horse results are noise.

## Frozen experiment

- Experiment ID: `shimba_filter_001_rolling`
- Model family: Binary only.
- Features: the current PV-01 incumbent, 254 columns.
- Ordered feature hash:
  `e5228bb4ffd605888b7266030d5e1e9f0931e8468b6fbbf124f3cea60905e51d`.
- Control: all training races.
- Candidate: remove a whole race from gradient-fitting rows if and only if its
  normalized `race_class` contains `新馬`; require exact agreement with
  `context__class_tier == 0` before fitting.
- PIT cache/state: identical and read-only for both methods. Prior new-horse
  outcomes continue to update already-materialized history features.
- Early stopping: all races in the registered early-stopping year.
- Temperature calibration: all races in the registered calibration year.
- Evaluation: all races in 2020, 2021, 2022, and 2023 rolling folds.
- Odds/popularity: prohibited.
- Candidate comparisons: one.

The control and candidate use the same model config, random seed, features, and
feature order. The gradient-fit membership is the only changed factor.

## Frozen decision rule

Primary population is all evaluation races. Accept only if every condition is
met:

1. Paired date-block-bootstrap Log Loss improvement has a 95% interval lower
   bound strictly above zero.
2. Log Loss improves in at least three of four evaluation years.
3. Year-macro Brier improvement is at least `-0.001`.
4. Year-macro NDCG@3 improvement is at least `-0.002`.
5. Year-macro Top-1 improvement is at least `-0.005`.

New-horse and non-new-horse metrics are saved as descriptive slice diagnostics.
They cannot override the all-race primary decision.

If a guardrail fails or the primary interval is wholly below zero, reject. Any
other non-accepting result is inconclusive.

## Artifacts and invariants

The runner records per-fold excluded fit row/race counts, training population
before/after counts, the resolved feature schema and hash, model parameters,
temperatures, predictions, all-race race metrics, slice race metrics, block
intervals, comparison accounting, and the final rule evaluation.

It must fail before fitting if:

- 2024/2025 survives the rolling firewall;
- the race-class definition disagrees with `context__class_tier == 0`;
- a race would be only partially excluded;
- the model family, feature count, or feature hash differs from the frozen
  Binary PV-01 incumbent;
- the early-stopping, calibration, or evaluation population changes; or
- an odds/popularity column reaches saved predictions.

## Reproduction command after review

Do not run this command as part of implementation-only verification. After the
preregistration is reviewed:

```bash
uv run horse-pred run-shimba-filter-study \
  --cache data/model_frame_race_content_time_20260831.pkl \
  --config configs/evaluation/shimba_filter_001_rolling.json \
  --output artifacts/shimba_filter_001_rolling_20260831
```

Focused implementation checks:

```bash
uv run pytest -q tests/test_shimba_filter_study.py tests/test_rolling_evaluation.py
uv run ruff check src/horse_pred/shimba_filter_study.py src/horse_pred/cli.py \
  tests/test_shimba_filter_study.py
```

## Result

The run used clean commit
`a63317a07d71abd24540dc1f26f057bec28f0c36`, the frozen 254-feature PV-01
cache, no odds, and zero 2024/2025 rows. Candidate fit counts matched the
audit exactly: 1,082 to 1,935 whole new-horse races were removed as the
expanding fit window grew, while early stopping, calibration, and evaluation
counts remained unchanged.

All-race candidate improvements were:

| Metric | Improvement | 95% paired interval | Improved years |
|---|---:|---:|---:|
| NDCG@3 | -.00225 | `[-.00468,+.00019]` | 0/4 |
| Top-1 | +.00052 | `[-.00442,+.00551]` | 3/4 |
| Log Loss | -.00097 | `[-.00392,+.00199]` | 1/4 |
| Brier | +.00014 | `[-.00078,+.00107]` | 3/4 |

The primary Log Loss test failed, and NDCG worsened in every year and crossed
the `-.002` guardrail. The decision is `reject`.

The descriptive mechanism slices do not rescue the hypothesis. In new-horse
races, Log Loss worsened by `.00720` and NDCG by `.01455`, both in all four
years. In non-new-horse races, Log Loss worsened by `.00037` and NDCG by
`.00105`; both improved in only one of four years. Removing the difficult
population therefore did not improve learning for ordinary races either.

Retain new-horse races in Binary gradient fitting. Their predictions remain
harder and should continue to be reported as a cold-start subgroup, but this
experiment does not support removing them from evaluation, prediction, or a
future betting scope. Those are separate estimand/decision-layer questions.
Tracked evidence is `experiments/shimba_filter_20260831/summary.json`; complete
local artifacts are under `artifacts/shimba_filter_001_rolling_20260831/`.
