# Phase 5C historical final-market oracle diagnostic

## Scope

This was the single preregistered, non-selective historical oracle diagnostic. It joined the frozen 256-feature SPEED-01 Binary predictions to normalized inverse final odds on all 3,051 complete common 2024 races (41,946 runners, 106 dates).

The fixed combined probability was the normalized geometric mean of fundamental and market probabilities. No blend search, calibration refit, model selection, slice selection, ROI, or 2025 analysis was performed.

## Results

| Method | Race Log Loss | Race Brier | Fixed-bin ECE |
|---|---:|---:|---:|
| frozen fundamental | 2.06821 | .82534 | .00195 |
| final-market-only | **1.88276** | **.78153** | **.00271** |
| fixed 50:50 log-pool | 1.93364 | .79470 | .00574 |

Positive paired improvement means the candidate beats final market:

| Comparison | Δ Log Loss | 95% four-date block interval | Δ Brier | 95% interval |
|---|---:|---:|---:|---:|
| fundamental − market | −.18545 | [−.20339, −.16852] | −.04381 | [−.05018, −.03803] |
| combined − market | −.05088 | [−.05940, −.04302] | −.01317 | [−.01653, −.01023] |

## Interpretation

The fixed equal log-pool did not provide descriptive evidence of market-nonredundant value; it diluted the final-market oracle. This is negative evidence for this exact frozen mapping on an already exposed year.

It does not show that:

- cutoff-time market data will be as strong as final odds;
- every fixed or trained combination will fail;
- the fundamental model has no useful causal or conditional information;
- executable EV or ROI is negative;
- another model should be selected.

Final odds contain post-cutoff information and 2024 helped promote SPEED-01. The result is therefore not a model-adoption test or a prospective estimate. No follow-up combination was run.

Machine result: `experiments/program_audit_20260901/historical_oracle_summary.json`.
