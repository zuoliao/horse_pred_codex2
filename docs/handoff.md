# Project Handoff

## Operating Rule

Read `README.md`, `AGENTS.md`, this file, and the relevant specification under `docs/` before changing model or split behavior. Preserve one interpretable hypothesis per experiment, strict point-in-time semantics, no odds in the primary model, and human ownership of experiment acceptance. Raw data, model binaries, runner predictions, and market artifacts remain outside Git.

## Current Goal

The user-designated Goal, tasks 1–16 through the LightGBM Binary baseline, is complete. LambdaRank, coherent probability calibration, integrated evaluation, and final-odds oracle diagnostics (tasks 17–20) were also completed as forward preparation. The next decision is DEC-01: the human user selects the next single hypothesis.

## Current State

- Research and free-data source assessment: complete.
- Data/coverage/outcome/PIT/split contracts: complete and frozen before metrics.
- PIT feature engine: complete, 268 numeric features in six logical groups.
- Full-data Binary and LambdaRank baseline: complete.
- Primary evaluation: 2024 development; 2025 was inspected once with explicit retrospective opt-in and must not be used to refit/reselect this baseline.
- Baseline interpretation: both LightGBM models clearly beat uniform/history-rate baselines; Binary and LambdaRank are effectively tied without uncertainty intervals; the final-odds market remains materially stronger.

## Implemented Artifacts

- Data contract and loader: `docs/data_contract.md`, `src/horse_pred/data.py`
- PIT feature contract and engine: `docs/feature_spec.md`, `src/horse_pred/features.py`
- Model/evaluation contracts: `docs/implementation_spec.md`, `docs/evaluation_spec.md`
- Models and calibration: `src/horse_pred/modeling.py`
- Metrics and oracle evaluation: `src/horse_pred/evaluation.py`
- One-command runner: `src/horse_pred/pipeline.py`, `src/horse_pred/cli.py`
- Frozen configs: `configs/data_manifest.json`, `configs/splits.json`, `configs/exp_001_binary.json`, `configs/exp_002_lambdarank.json`
- Tracked aggregate result: `experiments/mvp_task16_20260830/metrics_summary.json`
- Critical interpretation: `docs/experiments/mvp_task16_20260830.md`

## Data State

- Canonical local raw filename: `race_results_merged.csv`
- Runtime path used: `/Users/zuoliao/Documents/GitHub/horse_codex/data/data/raceinfo/race_results_merged.csv`
- SHA-256: `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`
- Raw rows/races: 629,967 / 44,761; 26 raw columns; 2013-01-05 through 2025-12-28.
- Known official shortfall: 146 races total, including 108 in 2024.
- The user explicitly approved use within this private, personal repository. No new scraping or network collection was used for implementation or the full experiment.
- `data/`, `datasets/`, `artifacts/`, `.venv/`, caches, models, predictions, and local market tables are intentionally untracked.
- Final verified local artifact: `artifacts/mvp_baseline_20260830_task16_v2/`; manifest SHA-256 `92e2807bcb7c29fb1d5139fc59abab9a69ae0e4adf45cf5c3650f679c9c6e60e`.

## Verification

- Full run commit: `b7e86ee44a3aa8aed4b93a867950ee1588e23404`, recorded as `dirty=false` in both run metadata files.
- Full run: 540,709 model runners, 38,472 races, 268 features, 1,005.7 seconds.
- 2024: 42,334 runners / 3,087 races. Binary+T NDCG@3 0.4898, Log Loss 2.0875; LambdaRank+T NDCG@3 0.4907, Log Loss 2.0866.
- 2025 opt-in retrospective: 44,874 runners / 3,236 races. Binary+T NDCG@3 0.4759, Log Loss 2.1340; LambdaRank+T NDCG@3 0.4757, Log Loss 2.1338.
- Strict JSON parsing passed; all 13 artifact checksums passed; primary predictions contain no final odds/popularity; first and second full-run metrics matched except intentional `Infinity` to JSON `null` normalization.
- Latest local checks: `uv run pytest -q` reports 58 passed; `uv run ruff check .` passes; compileall and `git diff --check` pass.

## Known Gaps

- No prospective PIT-A snapshot or executable closing-time odds history.
- 2,096 races containing scratch/exclusion events are excluded from PIT-C scoring, creating selection bias.
- No race/date block bootstrap or confidence intervals; tiny Binary/LambdaRank differences are not decision-grade.
- No ROI, profit, EV selection, staking, or payout settlement evaluation.
- 2024 has 108 known missing official races; missingness is not shown to be random.
- Feature importance is split-count diagnostic only; no ablation or causal interpretation.
- The LightGBM `eval_at` duplicate-warning path was removed after the verified run; tests cover the no-warning behavior, but it does not change model settings or prior metrics.

## Next Tasks

1. DEC-01 (requires human choice): accept the baseline as a reference and select one next hypothesis.
2. Recommended candidate: one feature-group ablation experiment, starting with `connections_pit` or `rating_strength`, using the same frozen split and adding race/date block uncertainty.
3. Keep 2025 closed to iterative tuning. Use 2024 development for the next registered hypothesis and reserve 2026+ for prospective final evidence.
4. Separately plan LIVE-01 before making any executable market or ROI claim.

## Useful Commands

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run python -m compileall -q src tests

# Default: 2024 development only
uv run horse-pred run-mvp \
  --repo-root . \
  --raw-path /path/to/race_results_merged.csv \
  --output artifacts/mvp_baseline

# 2025 is non-sealed and requires explicit opt-in
uv run horse-pred run-mvp \
  --repo-root . \
  --raw-path /path/to/race_results_merged.csv \
  --output artifacts/mvp_baseline_with_2025 \
  --include-retrospective-test
```

## Handoff Notes

Do not rerun or reinterpret 2025 to choose features, calibrators, thresholds, or hyperparameters for this baseline. Do not add odds/popularity to primary model features. Any data acquisition beyond the approved local raw needs a separate source/terms gate. The user explicitly allowed autonomous technical judgments but retained the final model/experiment accept-reject decision.
