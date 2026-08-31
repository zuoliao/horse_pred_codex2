# EVAL-ROLL-001 rolling-origin evaluation preregistration

Date: 2026-08-31 JST  
Status: implementation and frozen real-data baseline run complete

## Purpose

The project has repeatedly inspected 2024 development results.  EVAL-ROLL-001
therefore establishes a reusable pre-2024 screening surface before OPP-RECENT,
SEC-3F, HPO-01, or ENS-01.  It characterizes the already frozen current-best
no-odds Binary and conservative LambdaRank methods; it does not select a new
model and does not make 2020--2023 untouched holdouts.

The executable source of truth is
`configs/evaluation/eval_roll_001_current_best.json`.

## Frozen folds

| Fold | Fit | Early stopping | Temperature calibration | Evaluation |
|---|---:|---:|---:|---:|
| `roll_2020` | 2014--2017 | 2018 | 2019 | 2020 |
| `roll_2021` | 2014--2018 | 2019 | 2020 | 2021 |
| `roll_2022` | 2014--2019 | 2020 | 2021 | 2022 |
| `roll_2023` | 2014--2020 | 2021 | 2022 | 2023 |

The early-stopping year remains validation-only; models are not refitted after
iteration selection.  The cache is cut at 2023 immediately after load and
before feature resolution.  Outcomes from 2024 and 2025 must remain at zero
used rows in both metrics and run metadata.

## Frozen methods

- `binary_current`: `pv_001_candidate_signed_time_gap`, 254 features, ordered
  feature hash
  `e5228bb4ffd605888b7266030d5e1e9f0931e8468b6fbbf124f3cea60905e51d`.
- `lambdarank_current`: `pv_001_control_lean`, 253 features, ordered feature
  hash
  `fd8735cf6f8472a5c7322e3622c83fde1ff7720b022b75637745c96a1bc1062f`.

Feature configs must use an explicit `include` selection.  `drop` configs are
not accepted because a newly augmented cache could otherwise silently add a
feature family.  Feature count and ordered-column hash are executable gates.

Binary marginal probabilities are converted to logits and then mapped to one
coherent race probability vector by a temperature fitted only on the fold's
calibration year.  LambdaRank uses its raw score in the same temperature map.
Ranking metrics use these utilities; probability metrics use calibrated race
probabilities.  Odds, popularity, and final-market fields are forbidden.

## Evaluation and uncertainty

Primary metrics are NDCG@3, exact Top-1 winner mass, race Log Loss, and race
Brier.  Each metric is first averaged over races within evaluation year.  The
four-year result is the unweighted macro average of those year values, so years
with more races do not receive more selection weight.

For every registered comparison the artifact reports year-level signed
improvement, improved/worsened/tied year counts, direction consistency, and the
worst yearly improvement.  Positive always means candidate improvement.

The primary interval is a paired moving-date-block bootstrap.  Four-date blocks
are sampled independently within each evaluation year, the sampled races are
averaged within year, and the four yearly values are then averaged equally.
The fixed settings are 10,000 resamples, seed 20240830, and 95% percentile
intervals.

These intervals condition on the fitted models.  They do not include model
refit uncertainty, overlapping-fit-window dependence, year drift, or prior
selection optimism.  Four yearly directions are descriptive and are not an
independent significance test.  The config records a lower-bound inventory of
prior selection uses for each evaluation year and the number of candidate
comparisons in the current run.

## Artifact contract

The runner writes atomically and refuses overwrite:

- `metrics.json`
- `run_meta.json`
- `config.json`
- `resolved_configs.json`
- `feature_schema.json`
- `predictions_scoring.csv.gz`
- `race_metrics.csv.gz`
- `feature_importance.csv`
- `models/<fold>/<method>.txt`
- `artifact_manifest.json`

The reusable prediction table contains both calibration and evaluation rows in
long form, including fold, role, method, model family, raw output, utility,
temperature-1 probability, and calibrated probability.  ENS-01 may reuse raw
calibration utilities without consulting evaluation outcomes.

## Reuse rules

- OPP-RECENT and SEC-3F add one precomputed PIT family to a new augmented cache.
  Each family gets separate control/candidate methods; Binary retains the
  254-feature control and LambdaRank retains the 253-feature control.
- HPO-01 keeps feature configs fixed and changes only model configs.  Binary
  and LambdaRank parameter searches are separate experiments, and every trial
  increments the recorded multiple-comparison count.
- ENS-01 consumes the stored calibration/evaluation utilities.  A fixed blend
  receives its own calibration-year temperature; blend weights cannot be
  selected using evaluation outcomes unless registered as rolling candidates.

## Execution gate

Implementation may be verified with fixtures before this protocol is committed.
The full cache run must occur only from a clean preregistration commit and must
write the commit hash, config/cache hashes, effective LightGBM parameters,
feature hashes, software versions, and 2024/2025 zero-use assertions.  This
document does not authorize opening 2024 or 2025 after the rolling result.

Reproduction command:

```bash
uv run horse-pred run-rolling-evaluation \
  --cache data/model_frame_race_content_time_20260831.pkl \
  --config configs/evaluation/eval_roll_001_current_best.json \
  --output artifacts/eval_roll_001_current_best_20260831
```

## Frozen-run result

The run used clean commit `0cdf659a934157ff0494986646f47b7ce22191c0`,
raw fingerprint `270923ce...c4db`, and cache SHA-256 `85e92160...4eb9`.
It used zero 2024/2025 rows and no odds. All four folds completed in 90.18
seconds.

| Method | Year-macro NDCG@3 | Top-1 | Log Loss | Brier |
|---|---:|---:|---:|---:|
| Binary PV-01, 254 features | .47045 | .26903 | 2.14231 | .84152 |
| LambdaRank lean, 253 features | .46746 | .26407 | 2.15388 | .84373 |

Descriptively, Binary minus LambdaRank was NDCG `+.00299`
`[+.00009,+.00583]`, Top-1 `+.00497` `[-.00057,+.01061]`, Log Loss
improvement `+.01157` `[+.00749,+.01560]`, and Brier improvement `+.00221`
`[+.00100,+.00340]`. Binary had lower Log Loss and Brier in all four years;
NDCG improved in three of four.

This is not a causal model-family comparison because Binary also has the
accepted PV-01 feature while the Ranker uses the lean feature scope. It selects
no new model. The result accepts EVAL-ROLL as the required screening
infrastructure and establishes the frozen rolling baseline for subsequent
within-family OPP-RECENT, SEC-3F, HPO-01, and ENS-01 comparisons.
