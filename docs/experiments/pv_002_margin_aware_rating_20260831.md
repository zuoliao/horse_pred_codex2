# PV-02 Margin-aware standalone rating

Status: preregistered; no validation or development result inspected at registration.

## Hypothesis

Replacing only the frozen R5 pairwise Elo's ordinal pair result with a continuous,
antisymmetric time-margin result preserves more information about how decisively a
horse beat an opponent. It may therefore improve the standalone rating's future-race
probability and ranking quality. This experiment does not add the rating to LightGBM.

## Frozen comparison

- Control: global pairwise Elo, initial 1500, K=48, scale=200, no surface blend,
  ordinal pair actual. This is the R5 algorithm before its 2023 temperature fit.
- Candidate: all control choices unchanged except the pair actual.
- For horses `i` and `j`, define signed margin `m_ij = (time_j - time_i) *
  1000 / distance` seconds per 1000m. Candidate actual is
  `sigmoid(m_ij / tau)` and the reverse pair is exactly `1 - actual`.
- `tau=0.125 seconds/1000m`, fixed before validation. It is the median strictly
  positive gap between adjacent, distinct official finish positions in eligible
  2014–2021 races. This is a distributional scale choice, not a metric-tuned value.
- Existing Elo expected scores and `K / (field_size - 1)` aggregation are unchanged.
  Consequently each eligible race remains zero-sum apart from floating error.

## Outcome exceptions

- Equal official finishes, including dead heats: actual 0.5 even if rounded clocks
  differ.
- A pair with a missing clock or nonnumeric finish: use the existing ordinal actual
  for that pair.
- A clock ordering that contradicts distinct official numeric ranks: use the
  existing ordinal actual for that pair.
- A race containing a demotion or disqualification: use the existing ordinal update
  for the whole race, because clock order and official order express different facts.
- Scratches and exclusions are not rating participants, matching the frozen scorer.
- Zero rounded clock gaps across distinct official finishes produce actual 0.5. A
  separate margin-token refinement may later test these ties; it is not mixed here.

## Temporal protocol

1. Subset raw rows by race-ID year before outcome normalization.
2. Derive and verify `tau` using only 2014–2021 eligible outcomes. 2013 is warm-up
   state only and is not part of the scale estimate.
3. Run both algorithms through 2022 and compare only the common accepted scoring
   population. Same-day updates remain batched, so another race on the same date
   cannot consume a result from that date.
4. Accept the candidate at the 2022 gate only if the paired 95% interval lower bound
   for raw coherent race Log Loss improvement is strictly positive, Brier point
   improvement is at least -0.001, NDCG@3 point improvement at least -0.002, and
   Top-1 point improvement at least -0.005. The interval uses a preregistered paired
   four-date moving-block bootstrap with 10,000 draws and seed 20240830.
5. If the gate fails, stop without inspecting 2023 or 2024. If it passes, rerun from
   raw data through 2024, fit one temperature per algorithm on 2023 only, and perform
   a one-shot 2024 comparison with paired four-date-block bootstrap intervals.
6. 2025 is never normalized, scored, calibrated, or used for a decision. Final odds
   and betting ROI are outside this experiment.

## Interpretation rule

A passing 2022 result merely authorizes the sealed 2024 comparison. A 2024 candidate
will be described across ranking, raw/calibrated probability quality, calibration,
and uncertainty; one favorable point metric is not sufficient for adoption. Even a
supported standalone result requires a separate PV-03 experiment before LightGBM
integration.

The machine-readable protocol is
`configs/performance/pv_002_margin_aware_rating.json`.

## Result

PV-02 stopped at the preregistered 2022 gate. The candidate improved ranking but
worsened the primary raw probability metric, so the overall decision is **reject**
for this exact fixed-probability-mapping specification. Neither 2023 nor 2024 was
opened by the runner.

| 2022 strict scoring population | Control | Candidate | Candidate improvement |
|---|---:|---:|---:|
| Races / runners | 3,176 / 43,537 | 3,176 / 43,537 | — |
| Race Log Loss ↓ | 2.436946 | 2.439476 | -0.002530 |
| Race Brier ↓ | .897987 | .898295 | -0.000308 |
| NDCG@3 ↑ | .360364 | .365625 | +0.005260 |
| Top-1 ↑ | .171757 | .178684 | +0.006927 |

Paired four-date-block 95% intervals, expressed as positive-is-candidate-better:

- Log Loss: `-0.002530 [-0.004223, -0.000678]`
- Brier: `-0.000308 [-0.000647, +0.000052]`
- NDCG@3: `+0.005260 [+0.001271, +0.009324]`
- Top-1: `+0.006927 [+0.000944, +0.012891]`

The result is therefore not a generic rejection of time margins. It says that
softening close results and strengthening large results improved ordering, while
the unchanged R5 score-to-probability mapping became significantly worse for Log
Loss. Candidate score dispersion was lower (`0.5577` versus `0.6009`), which is
consistent with an altered scale.

As a post-hoc mechanism check only, fitting each arm's one-parameter temperature on
the same 2022 rows gave temperatures 0.5120 (control) and 0.4683 (candidate), and
resubstitution Log Loss 2.39154 versus 2.38320. This is optimistic and is not an
acceptance result. It motivates a separate, preregistered out-of-period calibration
experiment before abandoning the margin-aware rating direction.

Train-only scale audit:

- 26,563 clean 2014–2021 flat races; 22 demotion/DQ races excluded.
- 350,331 adjacent-rank clock pairs: 268,522 positive, 81,809 equal-clock, zero
  clock-order inversions.
- Frozen `tau=0.125` was reproduced from the positive-gap median.
- Across 2022 raw flat events, 290,151 pairs used a positive continuous margin,
  11,112 used a zero margin, 1,741 used pair fallback, and 78 used whole-race
  fallback.

The complete ignored artifact is `artifacts/pv_002_margin_rating_20260831/`. The
tracked aggregate is `experiments/race_content_20260831/pv_002_summary.json`.
