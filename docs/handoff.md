# Project Handoff

## Operating Rule

The active phase is `Phase 5A: systematic exploratory data analysis and problem reformulation`. Use `docs/eda/02_eda_protocol.md` as the authoritative queue. Pause all local production-feature experiments until EDA is complete; pause does not mean reject. Do not access new 2024 targets, use 2025, tune betting, scrape, or promote an EDA finding into production. Finish with no more than three proposed experiments and wait for human selection.

## Current Goal

Complete reproducible pre-2023 EDA, all nine workstreams, three independent reviews, a machine-readable hypothesis registry, synthesis, and roadmap. The central loader must enforce `target_date <= 2022-12-31`.

## Current State

- Safety checkpoint on 2026-09-01 found clean `main`, no untracked files, HEAD `b3d2b79`; no checkpoint branch was required.
- PV-00 through PV-05 are complete. PV-06 was completed as an inconclusive 2022 token gate, but every continuation/refinement is paused.
- Frozen Phase 5A references: Binary `pv_001_candidate_signed_time_gap` with 254 features; LambdaRank `abl_006_drop_field_relative` with 253 features.
- Previously known 2024 Binary evidence is NDCG@3 .4976, Top-1 .2965, Log Loss 2.0787, Brier .82767. It must not be reopened for EDA selection.
- Margin-aware rating improved calibrated standalone predictions, but PV-04/PV-05 and R6 did not establish incremental LightGBM value.
- Later PACE/SPEED experiments remain archived reproducible evidence. They do not change the conservative EDA reference or authorize further local search.
- Binary versus LambdaRank remains unresolved. GR-001 rejected broad upper-half relevance; original `3/2/1/0` labels remain historical pending EDA reformulation.
- Removing new-horse races from Binary fitting was rejected; EDA instead analyzes history-availability learning curves.
- 2025 is not used for model selection. 2023 remains reserved for prior calibration and excluded from Phase 5A target EDA.
- LIVE-DATA collection is user-deferred on Mac. No collection or scraping occurs in Phase 5A.

## Implemented Artifacts

- Phase state/inventory: `README.md`, `AGENTS.md`, `docs/development_plan.md`, `docs/eda/00_existing_analysis_inventory.md`.
- Contract/preregistration: `docs/eda/01_data_contract.md`, `docs/eda/02_eda_protocol.md`, `configs/eda/phase_5a.json`.
- Existing evidence: `experiments/`, `docs/experiments/`; full models/predictions remain ignored.
- Existing PIT pipeline: `src/horse_pred/data.py`, `features.py`, `pipeline.py`, and leakage/split tests.

## Data State

- Approved raw SHA-256: `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`.
- Full private raw: 629,967 declared runners, 44,761 races, 2013–2025. Phase 5A retains no target later than 2022-12-31.
- Raw, caches, models, predictions, and recoverable runner-level files are ignored and must not be committed.
- Odds/popularity exist only in `market_oracle` and require an explicit oracle diagnostic join.

## Verification

Safety commands completed:

```bash
git status --short --branch
git branch --show-current
git log -8 --oneline --decorate
git ls-files --others --exclude-standard
```

Result: clean `main`, ahead of `origin/main`, no untracked files, HEAD `b3d2b79`. Last pre-EDA verification: 234 tests passed, Ruff passed, compileall passed. Phase 5A verification is pending.

## Known Gaps

- `run-eda` CLI and physical cutoff assertions are not implemented.
- Logical views and machine-readable summaries are not generated.
- Workstreams A–I, OOT diagnostics, HTML, and aggregate plots are pending.
- Three-way cross-review, registry, synthesis, and roadmap are pending.
- Historical periods have prior exposure; no result is an untouched holdout claim.

## Next Tasks

1. Commit Phase 5A synchronization and Wave 0 protocol.
2. Implement `run-eda`, views, cutoff/market isolation tests, manifest, resume, tables, plots, and HTML.
3. Run workstreams A–I in waves using separate namespaces.
4. Build pre-2023 OOT diagnostic predictions without production promotion.
5. Perform independent PIT, statistics, and domain reviews; fix major findings.
6. Produce registry, synthesis, at most three next hypotheses, run full verification, and stop.

## Useful Commands

```bash
uv run horse-pred run-eda --raw-path /path/to/race_results_merged.csv --output artifacts/eda_20260901 --max-date 2022-12-31
uv run pytest -q
uv run ruff check .
uv run python -m compileall -q src tests
git status --short --branch
```

## Handoff Notes

Do not reinterpret archived PACE/SPEED acceptance as an active queue. Phase 5A freezes conservative PV-01/lean references while retaining later evidence. Separate observation, interpretation, and hypothesis; all final recommendations require human review.
