# S-rank no-odds model research conclusions

Date: 2026-08-31 JST  
Status: S0/S1 complete

## Outcome

The S-rank program established a reusable 2020--2023 rolling-origin screen
and completed every registered S1 hypothesis without using 2024 or 2025 for
selection. The only supported new predictive signal was race-relative last-3F
history for LambdaRank.

| Work item | Decision | Main evidence |
|---|---|---|
| DOC-SYNC | complete | README, AGENTS, priorities, development plan, conclusions, and handoff synchronized |
| EVAL-ROLL | complete | Four expanding folds; year macro, direction, paired date-block CI, zero 2024/2025 use |
| LIVE-DATA | groundwork complete; activation blocked | Official JV-Link-only source gate and append-only archive implemented; no live collection without private Windows host, contract, and key |
| PV-06 | inconclusive | 2022 Log Loss `+.00003 [-.00022,+.00028]`; stopped before 2024 |
| OPP-RECENT | Binary inconclusive; Rank reject | Rank Log Loss `-.00393 [-.00590,-.00195]`; not adopted |
| SEC-3F | Binary inconclusive; Rank accept | Rank NDCG `+.00256 [+.00045,+.00465]`, 3/4 years; Brier 4/4 years |
| HPO-01 | retain parameters | Binary no eligible profile; Rank selected feature fraction `.75` then failed 2023 confirmation |
| ENS-01 | reject | Fixed blend missed Binary Log Loss minimum; 2023 confirmation not opened |

Binary remains the 254-feature PV-01 development incumbent. The conservative
2024 LambdaRank reference remains the lean 253-feature model because no new
2024 milestone was opened. For future rolling/prospective Rank work, the
accepted candidate is lean plus the single SEC-3F column with unchanged
LightGBM parameters and original top-three relevance labels.

The user-added SHIMBA-FILTER-001 check was completed alongside synthesis. New-
horse races are substantially harder, but removing them only from Binary fit
worsened all-race Log Loss by `.00097` and NDCG by `.00225`; retain them in
training. Evaluation or betting exclusion remains a separate decision-layer
question and was not made.

## What not to do next

- Do not tune OPP-RECENT variants, HPO combinations, or ensemble weights from
  these results.
- Do not remove new-horse races from training or hide them from evaluation.
- Do not use 2024/2025 to rescue rejected or inconclusive S1 candidates.
- Do not combine SEC-3F with pace interactions in the same experiment.

## Next modeling task

Proceed to PACE-01 as the next independent A-rank hypothesis: freeze a small
horse-level passing-position/style history representation, explicitly handling
one-section straight 1000 m races. Only if that signal is supported should
PACE-02 add field pace-pressure interactions. In parallel, LIVE-DATA can move
from archive groundwork to actual snapshots only after a private supported
Windows JV-Link environment, user contract/key, and scheduled transport exist.

Tracked aggregate evidence is
`experiments/s_rank_model_research_20260831/summary.json`. Individual machine
summaries and full local artifacts remain the source for exact intervals and
predictions.
