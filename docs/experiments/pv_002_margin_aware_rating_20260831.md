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
