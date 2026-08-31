# Phase 5A Systematic EDA Protocol

## Goal

Describe predictable structure in grouped JRA outcomes, test temporal replication, identify information discarded by current representations, and reformulate the next research questions. This is not a 2024 metric search. It ends with at most three proposed experiments and human review.

## Frozen scope

| Role | Dates | Target use |
|---|---|---|
| Warm-up / quality | 2013 | no target claims |
| Discovery | 2014–2019 | exploratory |
| Replication | 2020–2021 | replicated |
| Confirmation | 2022 | prioritization only |
| Calibration reserve | 2023 | excluded |
| Known development | 2024 | no new access |
| Forbidden | 2025 | no use |
| Prospective final | 2026+ | no retrospective use |

The canonical loader asserts both the requested cutoff and every retained target date are at most 2022-12-31.

## Common rules

- Separate Observation, Interpretation, and Hypothesis.
- One race is one choice set. Race-macro effects are primary; intervals resample race dates or blocks.
- Report runner/race counts, years, denominator, missingness, effect size, interval/dispersion, temporal sign consistency, exploration status, multiple-comparison risk, and PIT risk.
- Prefer race-relative percentiles, within-race contrasts, race-wise softmax/conditional associations, and yearly effects over global correlation.
- Classify missingness as structural, status-related, history/cold-start, era/source, or accidental.
- Feature use, permutation importance, and retrained ablation are different estimands.
- Diagnostic models are shallow/pre-2023 and are not production candidates.

## Workstreams

| ID | Scope | Document | Artifact namespace |
|---|---|---|---|
| A | quality, coverage, missingness, drift | `03_data_quality_and_drift.md` | `workstreams/a_quality` |
| B | targets and stochastic race structure | `04_target_and_race_structure.md` | `workstreams/b_target` |
| C | horse history, decay, workload | `05_horse_history_and_temporal_dynamics.md` | `workstreams/c_history` |
| D | past-race performance content | `06_past_race_performance_content.md` | `workstreams/d_content` |
| E | opponent and field structure | `07_opponent_and_field_structure.md` | `workstreams/e_opponent` |
| F | connections and entity stability | `08_connections_and_entity_stability.md` | `workstreams/f_connections` |
| G | context, transitions, interactions | `09_context_suitability_and_interactions.md` | `workstreams/g_context` |
| H | pre-2023 OOT errors and market gap | `10_model_errors_and_market_gap.md` | `workstreams/h_errors` |
| I | external EDA practices | `11_external_eda_practices.md` | `workstreams/i_external` |

Agents edit only assigned documents/namespaces. The lead owns shared code/config/registries/synthesis. Raw data never leaves the local workspace.

## Reproducibility

```bash
uv run horse-pred run-eda \
  --raw-path /path/to/race_results_merged.csv \
  --output artifacts/eda_20260901 \
  --max-date 2022-12-31
```

The command is deterministic, refuses incompatible cutoff/config, records config/git/raw hashes, supports workstream selection/resume, and saves plot-source aggregates. Full artifacts are ignored; tracked summaries are non-recoverable aggregates.

## Review and stop gates

Independent PIT/leakage, statistical, and domain reviews follow all workstreams. Major issues are fixed before synthesis. Every idea enters the registry, including rejected-by-EDA ideas. The phase completes only when documents, machine-readable aggregates, HTML, reviews, cutoff tests, lint/tests/compile, and one-command reproduction pass. No prioritized experiment is executed.
