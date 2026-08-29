# Backtesting, Leakage, and Evaluation Design

**Scope:** Research workstream H<br>
**Access date for web sources:** 2026-08-30<br>
**Status:** Research output; concrete calendar splits remain conditional on verified data coverage.

## 1. Questions investigated

1. Which leakage paths are specific to race histories and pari-mutuel betting?
2. What chronological train/development/final design is defensible?
3. How should model, market, calibration, and betting performance be separated?
4. What uncertainty measures are appropriate for ROI, drawdown, and repeated experiments?
5. Is there a universal minimum number of bets for a profitability claim?

## 2. Evidence and sources

- Tashman, L. J. (2000), “Out-of-Sample Tests of Forecasting Accuracy: An Analysis and Review,” *International Journal of Forecasting*, 16(4), 437–450, DOI: 10.1016/S0169-2070(00)00065-0. It discusses fixed and rolling origins/windows and the need for enough, diverse out-of-sample forecasts. [Publisher record](https://www.sciencedirect.com/science/article/pii/S0169207000000650).
- White, H. (2000), “A Reality Check for Data Snooping,” *Econometrica*, 68(5), 1097–1126, DOI: 10.1111/1468-0262.00152. Reusing one history to select among rules inflates the apparent performance of the best rule; the paper develops a bootstrap reality check. [Publisher page](https://doi.org/10.1111/1468-0262.00152); [PDF copy](https://bashtage.github.io/kevinsheppard.com/files/teaching/mfe/advanced-econometrics/White.pdf).
- Bailey, D. H., Borwein, J. M., López de Prado, M., and Zhu, Q. J. (2014), “The Probability of Backtest Overfitting.” The paper formalizes selection of an overfit configuration from many backtests. [PDF](https://carmamaths.org/resources/jon/backtest2.pdf).
- Bailey, D. H. and López de Prado, M. (2014), “The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality,” *Journal of Portfolio Management*, 40(5), 94–107. [SSRN PDF and metadata](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID2460551_code87814.pdf?abstractid=2460551).
- Politis, D. N. and Romano, J. P. (1994), “The Stationary Bootstrap,” *Journal of the American Statistical Association*, 89(428), 1303–1313, DOI: 10.1080/01621459.1994.10476870. It provides resampling for weakly dependent stationary observations. [Publisher page](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870); [PDF copy](https://users.ssc.wisc.edu/~behansen/718/Politis%20Romano.pdf).
- Cameron, A. C., Gelbach, J. B., and Miller, D. L. (2008), “Bootstrap-Based Improvements for Inference with Clustered Errors,” *Review of Economics and Statistics*, 90(3), 414–427, DOI: 10.1162/rest.90.3.414. [NBER working-paper record](https://www.nber.org/papers/t0344).
- JRA official odds and settlement evidence is documented in `betting_market.md`; it establishes that final odds and payouts can differ after exclusions/dead heats.

## 3. Leakage taxonomy and mandatory controls

| Leakage / bias | Failure example | Mandatory control |
|---|---|---|
| Random temporal mixing | future races of a horse help predict its earlier start | chronological race-level split only |
| Same-race split leakage | runners from one race appear across train/test | assign the entire race to exactly one split |
| Future self history | “career win rate” computed using races after prediction | as-of aggregation with strict event cutoff |
| Opponent future leakage | target race at time \(t\) uses an opponent result published after \(t\) | cap every re-evaluation at target cutoff; distinguish historical-race ex-ante strength from as-of-target re-evaluation |
| Entity-stat leakage | jockey/trainer season total includes later rides | time-stamped cumulative statistic shifted before current event |
| Target-derived aggregate leakage | course par time or class strength computed using outcomes after the target cutoff | fit on the training side or update sequentially using only records published before each cutoff; version the rule |
| Preprocessing leakage | imputer, categorical dictionary, scaler, feature selection fit on all years | fit every learned transform on training side only |
| Calibration leakage | calibrator sees development/final outcomes used for reporting | separate chronological calibration slice |
| Odds look-ahead | selection uses final odds unavailable at bet time | timestamped executable odds snapshot |
| Revision leakage | corrected historical values replace what was known live | preserve event time, publication time, ingestion time, revision version |
| Scratch/change leakage | later field or jockey state assumed at earlier forecast | explicit forecast timestamp and as-of racecard snapshot |
| Survivorship bias | only horses with later complete careers or clean records retained | construct cohorts from races known at the time; preserve one-start horses and missing histories |
| Result/settlement filtering | cancellations, refunds, dead heats, zero-return cases dropped | encode official status and settlement; audit reconciliation totals |
| Holdout adaptation | final year inspected repeatedly and model changed | access-controlled/sealed final result and trial ledger |
| Rule data snooping | best of many EV thresholds/subgroups reported | predeclare one primary rule; record all tried variants |

### Point-in-time invariant

For every feature value used for race \(r\) at forecast timestamp \(t_r\):

\[
\text{available\_at(feature value)} \le t_r.
\]

The underlying event date alone is insufficient. A result can occur before \(t_r\) but be published or corrected after it. At minimum the dataset should distinguish `event_time`, `available_at`, `ingested_at`, processing completion, and `source_revision` where the source permits. Executable evaluation also requires both feed receipt and feature/prediction completion before the decision cutoff.

### Two valid race-value semantics

For a past race \(h\) used to predict target race \(r\) at cutoff \(t_r\), preserve two meanings rather than calling all later re-evaluation leakage:

1. **Ex-ante-at-\(h\):** field strength and expected performance frozen at \(h\)'s own cutoff. This measures surprise relative to what was knowable then.
2. **As-of-\(t_r\) re-evaluation:** revalue \(h\) using information published after \(h\) but no later than \(t_r\), such as an opponent's intervening results. This is live-available at the target race, but must be recomputed/versioned for every target cutoff.

Information after \(t_r\) is forbidden in both. Keep the two feature groups separate so the second does not silently rewrite the semantic meaning of the first.

## 4. Recommended chronological evaluation scheme

Exact years cannot be fixed until `data_sources.md` verifies history and completeness. The following relative design should be committed before examining holdout outcomes.

This design applies to the long-history **prediction track**. Keep a separate **executable-market track** for timestamped odds: PIT-B supports source-time replay but not proof of local receipt/processing/purchase latency, while PIT-A supports prospective/shadow operational evaluation. Because official time-series-odds retention is only guaranteed for one year, do not force the prediction track's multi-year development/final rule onto the executable-market track or treat a short market history as profitability confirmation.

```text
earliest usable history
    warm-up/history-only period
    expanding model-training period
    chronological calibration slice
    rolling-origin development backtests
    untouched final period
latest complete date
```

### 4.1 Warm-up

Retain early races to initialize histories and ratings even if their sparse features make them unsuitable as scored examples. Do not discard them from state construction merely because they are outside model training.

### 4.2 Development

Use several rolling-origin folds, preferably covering complete seasonal cycles:

```text
fold k: train on all data before cutoff k
        fit calibrator on a later, non-overlapping slice
        score the next fixed development window
```

Expanding windows match the intended use of all past information. A fixed-window challenger may be tested later as one explicit drift hypothesis. Hyperparameters, feature groups, calibration method, and EV threshold are selected only from development folds.

### 4.3 Untouched final

Reserve the latest complete contiguous period and do not inspect it until the research configuration is frozen. A useful planning rule is at least one complete JRA annual cycle; if the verified history is long enough, two complete years give more race regimes and more betting observations. This is not a universal sample-size guarantee. The chosen period must be recorded before result access.

After unsealing, a disappointing result is still the final estimate for that frozen research round. It must not become a new development set without creating a newer, prospectively untouched period. Before unsealing, also freeze whether one fixed model is used through the final interval or whether it is refitted at a predeclared cadence. For a rolling deployment simulation, each prediction must store its training cutoff, calibration cutoff, model fingerprint, and data fingerprint.

## 5. Evaluation unit and metric hierarchy

### 5.1 Prediction metrics

- Compute group/ranking metrics within race, then macro-average races so large fields do not silently dominate.
- Report NDCG with a fixed relevance mapping, top-1, top-k recall, and rank correlation where defined.
- Report coherent race winner Log Loss and multiclass Brier plus binary-runner scores as documented in `probability_and_calibration.md`.
- Report reliability with race-aware uncertainty and subgroup sample counts.
- Compare on the intersection of races covered by every model/baseline; separately report coverage.

### 5.2 Market comparison

- Uniform field baseline.
- Timestamp-matched normalized market baseline.
- No-odds model.
- Final-market oracle diagnostic, clearly non-executable.

### 5.3 Betting metrics

At minimum store:

- total purchased stake, effective non-refunded stake, official return including refunds, and refunds;
- primary gross ROI = `official return including refunds / purchased stake` (100% break-even), and primary net profit = `official return including refunds - purchased stake`;
- diagnostic exposure ROI = `(official return - refunds) / effective non-refunded stake`, with zero-denominator cases handled only after aggregation;
- number of bets, races bet, fraction of races bet, hit rate;
- mean/median odds and concentration by odds band;
- cumulative profit by race date;
- maximum drawdown in currency and stakes, primary on purchased cash flow and diagnostic on effective exposure;
- worst losing streak and high-quantile loss period;
- results by predeclared subgroup and EV threshold;
- uncertainty interval for mean profit/ROI;
- difference versus market/simple strategy on matched races.

If multiple horses are selected in one race, their outcomes are mechanically dependent. Treating tickets as IID produces misleadingly narrow uncertainty.

## 6. ROI uncertainty and minimum sample size

### Finding

There is no universal defensible statement such as “1,000 bets proves profitability.” Required sample size depends on selection rate, odds/payoff distribution, within-race dependence, temporal regime variation, and the edge size. Longshots can produce high variance and very slow convergence.

### Recommended uncertainty analysis

1. Preserve every race, including no-bet races, multiple tickets, and refunds, in chronological order. Define both purchased stake and effective non-refunded stake; report both denominator conventions.
2. In every resample, aggregate selected blocks and recompute primary \(\sum \text{official return including refunds}/\sum \text{purchased stake}\) and diagnostic \(\sum(\text{return}-\text{refund})/\sum \text{effective stake}\). Exclude and count resamples whose relevant aggregate denominator is zero. Do not average per-race ROI ratios.
3. Use a race-cluster bootstrap only as a simple cross-sectional diagnostic, because it ignores serial dependence.
4. For the primary temporal robustness interval, resample contiguous race-day or week blocks; use a predeclared block-length sensitivity analysis. A stationary/block bootstrap is motivated by possible dependence across adjacent race days, but its stationarity approximation is imperfect under structural drift.
5. Report calendar-block results (monthly/quarterly/yearly profit) without resampling as a transparent stability view.
6. Report both pointwise intervals for the one primary rule and clearly labeled exploratory intervals for secondary thresholds. Do not read “95%” literally after selecting the best of many unrecorded trials.

A profitability claim should require, at minimum, a positive point estimate on untouched data, an uncertainty interval and its assumptions, adequate coverage across time and race regimes, and prospective confirmation. Early prediction improvements do not require ROI above 100%, consistent with the repository’s agreed objective hierarchy.

## 7. Repeated evaluation and experiment accounting

White (2000) and subsequent backtest-overfitting work imply that every tested feature set, hyperparameter search, calibrator, subgroup rule, threshold, and manual decision contributes to selection. The repository should maintain:

- an immutable experiment ID and timestamp;
- git commit and configuration;
- data fingerprint and split specification;
- all tried thresholds, not only the winner;
- primary metric/rule declared before running;
- relationship to the parent hypothesis;
- whether the final holdout was accessed.

Formal reality-check, SPA, or deflated-Sharpe-style analysis may become useful after many strategy variants exist, but importing a financial Sharpe statistic mechanically is not required for the MVP. The immediate control is experimental discipline and an untouched/prospective sample.

## 8. Realistic transaction and settlement assumptions

- A decision uses only the selected timestamp’s available racecard, conditions, body weight, changes, and odds.
- The selection cannot assume final odds. Actual payout is nevertheless the final official settlement.
- Preserve the as-of runner set. A cutoff-after scratch may remove the horse; a scratch announced after decision does not retroactively change the prediction or selection and is settled as a refund. Re-normalizing only over eventual starters is post-event diagnostic unless that set was known at decision time.
- Fixed stake is deducted for every valid selected ticket.
- Refunds return stake with zero profit and are counted/reconciled rather than silently removed.
- Scratches, exclusions, cancellations, dead heats, and special payout programs follow official settlement data.
- Pool impact is assumed negligible only while proposed stakes are genuinely small relative to pools. If future stakes grow, market impact must become an explicit assumption.
- Compute both per-ticket and per-race stake caps as diagnostics; do not optimize sizing in the initial phase.
- Tax is not included in model ROI and must be labeled as excluded.

## 9. Data-quality and revision tests required before modeling

These are specification requirements, not implementation performed in this research phase:

1. Unique race and runner keys; no race crosses a split.
2. Exactly one official outcome state per runner and reconciliation of dead heats/non-completions.
3. Feature `available_at` never exceeds forecast cutoff.
4. Sequential aggregates reproduce a small manually audited history and, for PIT-A/B records, are invariant to appending future races.
5. Rebuilding a past cutoff from the same source snapshot/manifest yields identical feature rows. PIT-C can change when a provider supplies a later correction without its original publication history; freeze source snapshots/fingerprints and label this as an as-was limitation rather than claiming invariance.
6. Odds snapshot exists at or before cutoff; no “nearest” join may select a later record.
7. Selection ledger reconciles exactly to official stake, refund, and payout totals.
8. Dataset fingerprint changes when any source revision or split membership changes.

## 10. Uncertainties and limitations

- Historical source feeds may expose only latest corrected values, making exact real-time vintage reconstruction impossible for some fields. Such fields need a documented “retrospective corrected” flag or exclusion from executable backtests.
- A one- or two-year final set may still be underpowered for rare high-EV/longshot bets; calendar length is not a substitute for bet-count and payoff-distribution analysis.
- Block-bootstrap intervals depend on block definition and approximate stability; multiple block lengths should be shown.
- Rules, race programs, data definitions, and bettor behavior may shift, so even an untouched historical holdout is weaker than prospective shadow deployment.
- No statistical correction rescues an undisclosed search history. Trial logging is mandatory.

## 11. Project recommendations

1. Retain the agreed time-based split and strengthen it with a separate chronological calibration slice and rolling-origin development folds.
2. Delay exact calendar years until source completeness is measured; preregister them before outcome evaluation.
3. Make race the indivisible split and base resampling cluster; add day/week blocks for temporal uncertainty.
4. Use timestamped provisional odds for decisions and official payout for settlement; label final odds as oracle-only.
5. Freeze one primary fixed-stake EV threshold; record every sensitivity threshold and attempted strategy.
6. Require point-in-time invariance tests before accepting any feature pipeline.
7. Preserve early history as rating/aggregate warm-up data.
8. Do not make a strong profitability claim from a fixed bet-count heuristic; require uncertainty, temporal breadth, untouched evaluation, and later prospective confirmation.
