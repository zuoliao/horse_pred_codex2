# Project Handoff

## Operating Rule

The active phase is `Phase 5C: program audit and prospective validation pivot`. New feature, target, and model experiments are frozen. Do not run A1/A2/A3, nonlinear race-wise, Inter-horse, DNN, historical blend tuning, or additional feature search unless the human starts a later phase.

Treat all 2013–2025 data as `development_archive`. No historical year is an untouched final holdout. Final validation begins only with timestamped 2026-or-later prospective snapshots and predictions durably recorded before outcomes.

Never commit raw data, JV-Link payloads, credentials, caches, model binaries, runner-level predictions, or reconstructable runner-level views. Do not implement unofficial Mac/Wine JV-Link workarounds or unlicensed scraping.

## Current Goal

Wait for the human to decide whether to activate the official JV-Link prospective path. Repository-only activation infrastructure can be implemented after authorization, but live operation additionally requires a private supported Windows host, contract/key, official SDK, storage/topology choices, and an operations owner.

The prospective primary question is whether a fixed combination of the frozen fundamental score and cutoff-market probability improves paired out-of-time Log Loss, Brier, and calibration over cutoff-market-only. Profit is a later fixed-policy shadow diagnostic, not the model success definition.

## Current State

- Phase 5A and Phase 5B S1/S3/S2 are complete historical research.
- The canonical frozen fundamental candidate is LightGBM Binary `PV-01 + PACE-01 + SPEED-01`, 256 features, ordered-feature SHA-256 `3b6104ec33f6bf2b02b64685bf3ebf6bb828f14683c262e922e696da43bb4940`.
- Binary PV-01 254 is the formal historical comparison control. It is not the accumulated incumbent.
- LambdaRank lean 253 is the conservative formal control. LambdaRank SPEED-01 256 is only a rolling candidate; its 2024 result was directionally consistent, not a promotion.
- S1 performance was supported in Binary and LambdaRank/Conditional-Logit studies, but it overlaps SPEED-01 and was never combined with the 256-feature incumbent. Do not add it after the fact.
- S3 Huber performance target and S2 Linear Conditional Logit were rejected as replacements for the matched historical controls. These results do not reject every continuous target or nonlinear choice-set model.
- The model registry was reconciled using existing evidence only; no new model metric or retuning selected the canonical fundamental.
- 2013–2025 are all development archive. In particular, 2025 metrics and final-market results were viewed in the superseded Task16 work, so 2025 cannot be resealed as final.
- The one preregistered 2024 final-market oracle is complete. Final-market-only Log Loss was `1.88276`, frozen fundamental `2.06821`, and fixed equal log-pool `1.93364`. Combined was worse than market by `.05088`, with a paired 95% four-date-block interval `[.04302, .05940]` in the adverse direction.
- That oracle is descriptive negative evidence for one fixed mapping on an exposed year. It is not a cutoff-market, adoption, executable-EV, ROI, or profit test. No follow-up combination was executed.
- Official-source archive groundwork exists, but operational collection has not started and demonstrated real receipts are zero.
- Missing activation pieces are contract/key, private Windows 10/11 host in Japan, current official JV-Link/SDK, real adapter/parser, scheduler/cursors/retries, monitoring/health ledger, as-of materializer, immutable deployment bundle, shadow runner, and settlement ledger.
- The prospective protocol is designed but not active. A cutoff will be frozen only after at least two meetings of completeness/latency evidence and at least 99% complete eligible snapshots at that cutoff.

## Implemented Artifacts

- Phase 5C protocol: `docs/audit/phase5c_program_audit_protocol.md`
- Model registry: `docs/audit/model_registry.md` and `experiments/program_audit_20260901/model_registry.json`
- Evidence ledger: `docs/audit/evidence_ledger.md` and `experiments/program_audit_20260901/evidence_ledger.json`
- Revised objective: `docs/audit/revised_system_objective.md`
- Historical oracle report: `docs/audit/historical_oracle_diagnostic.md`
- Historical oracle preregistration/result: `experiments/program_audit_20260901/historical_oracle_preregistration.json` and `historical_oracle_summary.json`
- Activation plan: `docs/audit/prospective_activation_plan.md`
- Prospective protocol: `docs/audit/prospective_evaluation_protocol.md`
- Research-code audit: `docs/audit/research_code_audit.md`
- Summary inventory and program summary: `experiments/program_audit_20260901/summary_inventory.json` and `summary.json`
- Fixed diagnostic implementation and tests: `src/horse_pred/program_audit.py`, CLI `run-historical-oracle-diagnostic`, and `tests/test_program_audit.py`
- Earlier EDA/S1/S3/S2 artifacts remain unchanged under `docs/eda/`, `docs/experiments/`, and `experiments/`.

## Data State

- Approved private raw SHA-256: `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`.
- Full private raw: 629,967 declared runners, 44,761 races, 2013–2025.
- Historical oracle population: 3,051 complete common 2024 races, 41,946 runners, 106 dates.
- Historical final odds are isolated oracle data and are not an executable cutoff market.
- Prospective official receipts demonstrated: 0.
- Prospective cutoff: not selected.
- Prospective evaluation window: not started.

## Verification

The one-time oracle run used the preregistered configuration without blend search, recalibration, ROI, or 2025 analysis. Its paired uncertainty used a four-date circular block bootstrap with 10,000 draws and seed `20240830`.

Phase 5C final repository verification passed on 2026-09-01:

```text
UV_CACHE_DIR=/tmp/horse-pred-uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/horse-pred-uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/horse-pred-uv-cache uv run python -m compileall -q src tests
git diff --check
```

- `pytest`: 302 passed, 1 non-failing physical-core-detection warning.
- Ruff: passed.
- `compileall src tests`: passed.
- `git diff --check`: passed.

Tracked experiment JSON parsing, referenced artifact/config hashes, and staged-file policy were also checked before the final audit commits.

## Known Gaps

- Historical data are PIT-C; receipt-backed PIT-A evidence does not yet exist.
- The canonical model has historical development evidence, not prospective final evidence and not a production-profit claim.
- The fixed final-odds log-pool failed descriptively; a cutoff market may have different strength, but no alternative combination may be selected from the same historical archive in Phase 5C.
- The local frozen model artifact is not yet an immutable deployment bundle with full schema/preprocessing/calibration hashes.
- The groundwork rejects empty data batches even though operations need zero-record evidence. Add a separate health/error/zero-record ledger; do not weaken the raw envelope.
- JV-Link operation requires supported Windows and official subscriber software. The current Mac cannot activate it directly.
- O1 odds are snapshots, not guaranteed executable prices. Shadow ROI remains diagnostic.
- Stable-profit claims require the preregistered horizon, population, month-level evidence, drawdown, and uncertainty; a short black period is insufficient.

## Next Tasks

### Immediately executable after human authorization

1. Define the Windows adapter interface and conformance fixtures.
2. Add the separate health/error/zero-record event schema and audit CLI.
3. Implement fixture-only as-of materialization and completeness/latency auditing.
4. Define immutable deployment-bundle, decision-snapshot, shadow-policy, and settlement schemas.
5. Add dry-run monitoring and restart/cursor fixtures.

### Require user decision or provisioning

1. Confirm the individual/private JRA-VAN Data Lab contract and service key.
2. Choose/provision an always-on private supported Windows 10/11 host in Japan and install the current official JV-Link/SDK.
3. Decide private storage, encryption, backup, retention, and whether any encrypted Mac handoff is allowed.
4. Assign an operator and private alert channel for meeting-day monitoring.
5. Confirm individual/private shadow-only use; shared/public/commercial use requires a fresh terms review.

### Must wait for prospective data

1. Freeze a cutoff after at least two meetings using completeness and latency only.
2. Start the frozen fundamental-only, market-only, combined, and shadow-policy window.
3. Evaluate incremental proper scores, calibration, edge bands, fixed-stake shadow ROI, bet count, monthly stability, drawdown, and uncertainty after both 12 months and 2,500 complete races.
4. Reconsider A1/A2/A3, nonlinear race-wise, Inter-horse, DNN, or added features only after prospective readiness review; do not auto-run them.

## Useful Commands

```bash
UV_CACHE_DIR=/tmp/horse-pred-uv-cache uv run horse-pred run-historical-oracle-diagnostic \
  --repo-root . \
  --preregistration experiments/program_audit_20260901/historical_oracle_preregistration.json \
  --output experiments/program_audit_20260901/historical_oracle_summary.json

UV_CACHE_DIR=/tmp/horse-pred-uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/horse-pred-uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/horse-pred-uv-cache uv run python -m compileall -q src tests
git status --short --branch
```

The historical diagnostic command is for reproducibility only. Do not rerun it to search variants.

## Handoff Notes

The important conceptual change is that success is no longer “the no-odds model beats final market.” The frozen no-odds model is a fundamental layer. The valid future test is whether it adds information beyond an actually observable cutoff market under a frozen prospective protocol, followed separately by shadow betting evaluation.

The model-registry correction is equally important: PV-01 Binary 254 remains a stable matched-experiment anchor, while the already promoted SPEED-01 Binary 256 is the canonical frozen fundamental. Feature counts alone are not model identities, and unsupported branches must not be composed.
