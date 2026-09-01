# AGENTS.md

## Project role

This repository is a research-and-development project for a JRA-focused horse-racing prediction and betting-decision support system.

The long-term objective is to improve long-run betting return and profit, not raw hit rate. Profitability is a research target to be verified with robust out-of-sample evaluation; it is not an assumed or guaranteed outcome.

The final bet decision remains with the human user. Automated wagering is outside the current scope.

Before making architecture or modeling decisions, read `README.md` and all relevant documents under `docs/`.

## Current phase

The project is in **Phase 5C: program audit and prospective validation pivot**. New feature, target, and model experiments are frozen. Phase 5A and Phase 5B S1/S3/S2 remain completed historical research; do not resume A1/A2, nonlinear race-wise, Inter-horse, DNN, or additional feature work until prospective readiness has been reviewed by the human user.

Use `docs/audit/model_registry.md`, `docs/audit/evidence_ledger.md`, `docs/audit/revised_system_objective.md`, `docs/audit/prospective_activation_plan.md`, `docs/audit/prospective_evaluation_protocol.md`, `experiments/program_audit_20260901/summary.json`, and `docs/handoff.md` as the current decision state.

The canonical frozen fundamental candidate is LightGBM Binary `PV-01 + PACE-01 + SPEED-01`, 256 features, ordered-feature SHA-256 `3b6104ec33f6bf2b02b64685bf3ebf6bb828f14683c262e922e696da43bb4940`. Binary PV-01 254 is the formal **historical comparison control**, not the accumulated incumbent. LambdaRank lean 253 remains the conservative formal control. S1 performance is a supported but overlapping 254/255 branch and was never evaluated in union with the 256-feature incumbent, so do not compose it after the fact.

Every year from 2013 through 2025 is a `development_archive`; none is an untouched final holdout. Specific experiments may still report zero rows from a year as a leakage check, but that does not restore program-level holdout status. Final validation begins only with timestamped 2026-or-later prospective snapshots and pre-outcome prediction receipts.

The one preregistered 2024 final-odds oracle found the fixed 50:50 log-pool worse than final-market-only. This is negative descriptive evidence for that exact mapping on an exposed year. It is not a cutoff-market test and cannot support adoption, executable EV, ROI, or profit claims. Do not run another historical combination, recalibration, weight search, or model comparison during Phase 5C.

Operational collection is not active: official JV-Link archive groundwork exists, but real receipts are zero and the contract/key, supported private Windows host, official adapter, scheduler/monitoring, as-of materializer, deployment bundle, and shadow runner are missing. Do not implement unofficial Mac/Wine workarounds or unlicensed scraping. Never commit raw data, cache, model binaries, credentials, runner predictions, or reconstructable runner-level views.

## Core design decisions already agreed

Treat these as current project decisions unless research produces strong contrary evidence:

1. Initial scope is JRA central racing.
2. Fundamental prediction, cutoff-market prediction, their frozen combination, and betting decision are separate layers.
3. Odds remain excluded from the fundamental model; the market-only and combined layers use only timestamp-valid cutoff snapshots.
4. The primary prospective test is whether the frozen combined model improves Log Loss, Brier, and calibration over market-only on identical races.
5. The initial modeling phase is non-DNN and LightGBM-centered.
6. Build and compare at least:
   - LightGBM binary classification for `P(win)`
   - LightGBM LambdaRank for race-level relative ranking
7. More advanced probabilistic ranking models such as Plackett-Luce are later candidates.
8. Horse ID, jockey ID, and similar entity IDs should not be used directly as memorization features in the initial model.
9. Entity history may be summarized into point-in-time statistics and state features.
10. Do not arbitrarily restrict horse history to only a few recent races; preserve history and expose multiple time windows / conditional aggregates where feasible.
11. Strict point-in-time correctness and leakage prevention are mandatory.
12. Prospective betting evaluation is shadow-only, win-only, fixed-stake, and governed by a frozen policy; automated or real wagering is out of scope.
13. Profit is evaluated after proper scores and calibration, and a short profitable window must never be called stable profit.
14. Evaluation must be multidimensional: ranking, probability quality, calibration, subgroup behavior, market comparison, and backtest metrics.
15. Historical time splits remain useful development evidence, but only receipt-backed 2026+ prospective data can provide final validation.
16. One experiment should correspond, as a rule of thumb, to one interpretable hypothesis and one reproducible repository state.

## Long-term model hypothesis

Do not implement this as the MVP unless explicitly requested. It is a future architectural hypothesis:

```text
past race inputs
    -> Race-value encoder
    -> per-race representation
    -> Intra-horse aggregation
    -> horse representation
    -> Inter-horse interaction / attention
    -> race-specific latent strength
    -> probabilistic ranking head
```

The conceptual motivation is that an observed finish is a stochastic realization of race-specific latent performance, not an absolute label of horse ability.

An important research question is how to represent the value of a past performance while accounting for field strength, race class, margins, time, conditions, and opponent quality.

## Research-mode instructions

When asked to execute research:

- Use parallel sub-agents aggressively when workstreams are independent.
- Follow `docs/research_plan.md`.
- Prefer primary sources, official documentation, academic literature, and original datasets.
- Separate:
  - evidence
  - findings
  - uncertainty
  - implication
  - recommendation
- Cite sources with URLs and enough metadata to re-check them later.
- Record negative results, unavailable fields, cost barriers, and terms-of-use constraints.
- Never convert uncertainty into a confident recommendation without stating the assumption.
- Never treat scraping as permissible without evidence from applicable terms or official documentation.
- Reconcile conflicting sub-agent conclusions during synthesis.

Do not merely produce independent reports. Finish with `docs/research/research_conclusions.md`, which must translate research into concrete implementation decisions.

## Implementation-mode instructions

After the research phase has been accepted:

- Favor simple, reproducible baselines before advanced methods.
- Reuse a common point-in-time feature pipeline across model families.
- Ensure every feature has a documented timestamp semantics.
- Avoid hidden use of future results from opponents or future races.
- Make dataset construction deterministic and versionable where possible.
- Build a single experiment command that can run the relevant pipeline end to end.
- Keep feature groups separable to support ablations.
- Keep prediction models separate from betting strategy code.
- Keep market odds separate from the primary no-odds feature set.

## Experiment protocol

Default principle: **one experiment = one interpretable hypothesis = one reproducible code state**.

For each experiment, store at minimum:

- experiment ID
- config
- git commit hash
- timestamp
- data version / dataset fingerprint
- random seed where relevant
- model family
- feature set / feature groups
- metrics
- betting-backtest metrics where applicable
- predictions when practical
- feature importance / diagnostics where applicable

Machine-readable experiment artifacts are the source of truth. README experiment tables should be generated or updated from those artifacts rather than manually maintained as the only record.

A future experiment directory may resemble:

```text
configs/
  exp_001.yaml

experiments/
  exp_001/
    metrics.json
    run_meta.json
    predictions.parquet
    feature_importance.csv
    model.txt
```

Exact filenames may change after implementation planning.

## Evaluation principles

Do not reduce model quality to a single metric.

At minimum consider:

### Ranking
- NDCG or suitable race-group ranking metrics
- top-k behavior
- rank correlation where meaningful

### Probability
- Log Loss
- Brier score

### Calibration
- reliability / calibration curves
- calibration metrics with their limitations clearly stated

### Conditional diagnostics
Examples:
- odds bands
- favorite vs longshot
- race class
- distance
- turf / dirt
- field size

### Market comparison
Compare model probability estimates with market-implied probability in a technically appropriate way.

### Betting backtest
At minimum record:
- return / ROI
- total profit/loss
- number of bets
- hit rate
- drawdown
- dispersion / uncertainty where feasible

Do not tune repeatedly on the final untouched holdout period.

## Betting-strategy constraints for the initial phase

Keep the betting algorithm deliberately simple so changes in ROI remain interpretable as model changes.

Initial target:

- win bets first
- fixed stake
- expected-value threshold based on predicted win probability and observable odds
- optionally report several predeclared EV thresholds as diagnostics

Do not introduce Kelly sizing, portfolio optimization, complex multi-bet strategies, or automated wager placement until the prediction layer is sufficiently mature.

## Coding and change discipline

- Prefer small, reviewable changes.
- Do not mix several unrelated modeling hypotheses in one experiment.
- Preserve backward reproducibility when refactoring experiment infrastructure.
- Add tests for time leakage, feature timestamps, and dataset splits as soon as those systems exist.
- Avoid unnecessary framework complexity.
- Do not introduce DNN infrastructure simply because it may be useful later.
- Do not silently change evaluation splits or betting rules between experiments.

## Decision-making behavior

The human user may give either:

1. a concrete experiment specification, or
2. an abstract hypothesis such as "does fatigue information help?"

For a concrete experiment, implement and evaluate it faithfully.

For an abstract hypothesis, translate it into a small, interpretable experiment or feature-group comparison. You may choose reasonable concrete tests, but avoid bundling many unrelated changes. State the hypothesis and what was changed before reporting results.

Report what improved and what degraded. Do not label an experiment as successful solely because one metric improved.

The final accept/reject decision belongs to the human user unless explicitly delegated.

## Repository documentation

- `README.md`: project overview, agreed direction, current status, experiment summary once experiments begin
- `AGENTS.md`: operating instructions for Codex / coding agents
- `docs/research_plan.md`: research execution plan
- `docs/research/`: research outputs
- future implementation specifications should live under `docs/` rather than being buried in chat history

If a project decision changes, update the relevant repository documentation in the same change whenever practical.
