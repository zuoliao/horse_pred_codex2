# Research Plan

## Purpose

This document defines the research phase that should be executed before implementation decisions are finalized. The goal is not to collect generic horse-racing information, but to produce evidence that directly informs the architecture, dataset, evaluation protocol, and MVP scope of this repository.

Research should be executed in parallel with multiple sub-agents where useful. Each workstream must distinguish verified facts from interpretation and recommendations.

## Output contract

Each research workstream must produce:

1. **Questions investigated**
2. **Evidence / sources**
3. **Findings**
4. **Uncertainties / conflicts**
5. **Implications for this project**
6. **Recommendations**

Facts, assumptions, and recommendations must not be mixed together.

Where external information is used, record source URLs, titles, publication dates when available, and access dates. Prefer official documentation, primary sources, academic papers, and original datasets over secondary summaries.

## Research workstreams

### A. Data sources, access, cost, and terms

Investigate at minimum:

- JRA
- JRA-VAN / JV-Link and related services
- netkeiba
- JBIS
- other credible public or commercial sources that could materially improve the project

For each source, determine:

- obtainable fields
- historical coverage
- update frequency
- whether historical snapshots / point-in-time data are available
- odds coverage, including whether time-series odds are available
- race result fields
- horse, jockey, trainer, pedigree, workout/training, body weight, weather, going, pace, sectional/lap, payout fields
- access method: API, download, database, official software interface, scraping, etc.
- pricing
- rate limits or practical throughput constraints
- authentication requirements
- terms of use relevant to automated retrieval, storage, redistribution, and model development
- whether use in a private research tool is realistically feasible

Primary output: `docs/research/data_sources.md`

### B. Available features and point-in-time semantics

Build a structured inventory of potentially useful features and identify when each feature becomes known relative to race start.

Investigate:

- historical race information
- race grade / class
- finish position, margins, final time, sectional/lap information
- field strength / opponent quality
- course, distance, turf/dirt, going, weather
- draw, carried weight
- horse body weight and body-weight change
- layoff interval and race frequency
- jockey/trainer historical statistics
- pedigree and static horse attributes
- training/workout information
- scratches, jockey changes, race-day updates
- odds snapshots and final odds

For each field, record:

- source(s)
- availability period
- missingness
- point-in-time availability
- leakage risk
- expected usefulness

Primary outputs:

- `docs/research/available_features.md`
- `docs/research/point_in_time.md`

### C. Existing horse-racing prediction methods

Survey both academic and practitioner work, with emphasis on methods that are actually relevant to structured race data and modest dataset sizes.

Investigate:

- LightGBM / XGBoost / CatBoost baselines
- binary win-probability prediction
- top-k / place prediction
- regression approaches and their weaknesses
- learning-to-rank approaches
- neural methods only where they provide evidence useful for later phases
- published comparisons of model families
- important feature groups repeatedly found useful

Avoid assuming that newer or more complex models are better.

Primary output: `docs/research/horse_racing_prediction.md`

### D. Ranking, rating, latent strength, and race-value modeling

This workstream is central to the project hypothesis.

Investigate:

- LambdaRank / LambdaMART
- Plackett-Luce
- Bradley-Terry
- Thurstone-style models
- Elo
- TrueSkill
- dynamic rating systems
- context-specific ratings
- opponent-strength adjustment
- race-strength / field-strength estimation
- methods for evaluating the value of a past performance beyond raw finishing position
- approaches that can distinguish "strong field, lower finish" from "weak field, high finish"
- temporal decay and form modeling

Assess whether there are methods suitable for:

1. handcrafted LightGBM features in the MVP
2. a learned `Race-value encoder` later
3. probabilistic ranking and calibrated finishing distributions

Primary output: `docs/research/ranking_rating_and_strength.md`

### E. Form, fatigue, suitability, and interaction effects

Investigate evidence and feature engineering approaches for:

- layoff duration
- repeated starts / congested schedules
- cumulative racing load
- recent distance raced
- body-weight change
- age / development / decline
- distance suitability
- turf vs dirt
- going / weather suitability
- course-specific suitability
- draw effects
- running style
- likely pace
- composition of the field
- interactions such as multiple front-runners changing expected pace

Separate empirically supported effects from racing folklore.

Primary output: `docs/research/form_fatigue_and_interactions.md`

### F. Probability estimation and calibration

The betting layer needs probabilities, not only ranking scores.

Investigate:

- Log Loss and Brier score
- reliability diagrams
- Expected Calibration Error and its limitations
- Platt scaling
- isotonic regression
- beta calibration
- multiclass / ranking probability calibration where relevant
- calibration under dataset shift
- whether calibration should be global or conditioned on odds / race type / field size

Primary output: `docs/research/probability_and_calibration.md`

### G. Betting market and market-implied probability

Investigate:

- how JRA pari-mutuel odds should be interpreted
- takeout / deductions by bet type
- converting displayed odds to market-implied probabilities
- overround / normalization issues in pari-mutuel markets
- favorite-longshot bias in Japanese racing if credible evidence exists
- market efficiency literature
- how predictive models should be compared against market odds
- differences between using final odds and realistically available pre-race odds

Primary output: `docs/research/betting_market.md`

### H. Backtesting, leakage, and evaluation design

Investigate common failure modes in horse-racing ML backtests.

At minimum:

- temporal splits
- data leakage from future races
- opponent future-performance leakage
- use of final odds that were not known at the intended decision time
- revisions / corrected data
- survivorship-like biases
- tuning on the final holdout set
- repeated evaluation against the same year
- realistic transaction assumptions
- minimum sample sizes for ROI claims
- uncertainty / confidence intervals for ROI
- drawdown and variance metrics

Recommend a robust train / development backtest / untouched final backtest scheme for this project.

Primary output: `docs/research/backtesting_and_leakage.md`

## Integration phase

After all workstreams complete, run a separate synthesis pass.

The synthesizing agent must:

1. read all research outputs
2. identify contradictions and missing evidence
3. perform targeted follow-up research where necessary
4. distinguish high-confidence conclusions from open questions
5. convert findings into concrete project decisions
6. explicitly state which earlier project assumptions should be retained, modified, or rejected

Primary output: `docs/research/research_conclusions.md`

The conclusions document should cover at least:

- recommended data source(s)
- expected historical coverage
- realistic prediction timestamp(s)
- MVP feature set
- leakage-control rules
- LightGBM binary baseline design
- LightGBM LambdaRank baseline design
- probability calibration plan
- fixed initial betting rule
- evaluation split and metrics
- which advanced ideas should be deferred
- implementation risks

## Parallel execution recommendation

A reasonable parallel decomposition is:

- Agent A: data sources / cost / terms
- Agent B: available features / point-in-time semantics
- Agent C: horse-racing prediction baselines
- Agent D: ranking / rating / latent strength
- Agent E: form / fatigue / interactions
- Agent F: probability / calibration / betting market
- Agent G: backtesting / leakage / evaluation

Agents may launch narrower sub-agents where needed, but should avoid duplicating entire workstreams.

## Research quality rules

- Do not treat blog posts or betting-site marketing claims as evidence when primary sources exist.
- Do not infer API availability from undocumented examples.
- Do not claim scraping is allowed unless terms or official documentation support that interpretation.
- Do not use future information when proposing point-in-time features.
- Do not recommend a modeling method solely because it is sophisticated.
- Record negative findings when a desired data field is not practically obtainable.
- If sources disagree, record the disagreement rather than silently choosing one.
- Keep implementation recommendations proportional to the expected dataset size.

## Out of scope for the first research pass

Unless the research uncovers a compelling reason otherwise, defer:

- race video / image modeling
- paddock video analysis
- news / SNS / text sentiment modeling
- LLM-based handicapping from prose
- automated bet placement
- bankroll optimization beyond a simple fixed-stake EV rule
- large end-to-end Transformer architectures

These may be revisited after the structured-data baseline is established.
