# Prospective evaluation protocol

## Evidence identity

**Protocol ID:** `phase5c_prospective_shadow_v1`

**Eligible races:** JRA central flat races for which one complete pre-outcome decision snapshot exists

**Prospective evidence start:** 2026 or later; never historical replay

**Status:** preregistered design; activation blocked by missing live prerequisites

## Pre-window activation gate

Before the scoring window starts:

1. collect at least two meetings with durable official JV-Link receipts;
2. audit runner-card, scratch/change, scheduled-post, weather/going, body-weight, and O1 coverage plus receipt latency;
3. choose one cutoff from the audited candidates using completeness and latency only;
4. require at least 99% complete eligible race snapshots at the chosen cutoff and record every excluded race/reason;
5. freeze hashes for the model bundle, feature schema, preprocessing, calibration, cutoff, combination rule, policy, and code commit.

T-10 versus T-15 is not selected before real receipts exist. Outcome, proper score, odds edge, ROI, or profit may not influence the cutoff.

## Snapshot contract

Eligibility is based on `ingested_at <= cutoff`; a source record published before cutoff but received later is excluded. O1 additionally requires `published_at <= cutoff`. Each immutable `decision_snapshot_id` records:

- race and latest known scheduled-post version;
- complete eligible runner set after known scratches/exclusions;
- source/observed/ingested timestamps and staleness;
- exact feature/model/calibration/config hashes;
- fundamental-only, market-only, and combined probabilities on the same runners;
- shadow decisions and policy version;
- no outcome, final odds, popularity, payout, or future correction.

Later results and official settlement are appended under a separate timestamped record; the decision snapshot is never overwritten.

## Frozen prediction arms

| Arm | Rule |
|---|---|
| Fundamental-only | canonical 256-feature Binary bundle; no odds |
| Market-only | normalized inverse cutoff O1 win odds |
| Combined | normalized equal-weight log pool: `sqrt(p_fundamental * p_market)` |

The combined rule is intentionally the same transparent fixed mapping used by the historical oracle. It is not retuned after that negative oracle result. A different combination requires a future protocol and a new prospective window, not an in-window amendment.

## Frozen shadow policy

- ticket: win only;
- stake: ¥100 for every eligible runner satisfying the rule;
- rule: `combined_probability * displayed_cutoff_win_odds >= 1.10`;
- multiple qualifying runners in one race remain separate ¥100 shadow tickets;
- no Kelly sizing, caps chosen from results, portfolio optimization, manual overrides, or real wagering;
- official payouts/refunds/dead-heat rules are used only at settlement;
- all zero-bet races/months remain in reporting.

This threshold is a prospective operational rule, not a profitability claim or a recommendation to wager.

## Frozen-window discipline

During the primary window, do not change:

- model or feature state semantics;
- calibration or probability mapping;
- cutoff or completeness requirements;
- combination weight;
- eligibility, edge threshold, ticket type, or stake;
- missing-data handling;
- primary metrics or minimum reporting horizon.

Operational bug fixes that can change predictions require a new protocol version. The affected period remains reported separately and is not silently discarded.

## Metrics

Primary comparison is `combined − market-only` on exactly paired races:

- race-macro Log Loss;
- race-macro Brier;
- race-balanced reliability/ECE and calibration slope/intercept where identifiable;
- paired date/week block uncertainty and monthly direction consistency.

Secondary diagnostics:

- fundamental-only proper scores;
- snapshot completeness, exclusion rate, latency and O1 staleness;
- edge-band calibration using fixed bands for `combined / market`: `<.80`, `.80–.95`, `.95–1.05`, `1.05–1.20`, `>=1.20`;
- fixed-stake shadow ROI, total profit/loss, ticket count, hit rate, month-by-month result, maximum drawdown, and block/bootstrap uncertainty.

The primary report opens only after **both** at least 12 calendar months and at least 2,500 complete eligible races. If the threshold is not reached, collection continues. Interim reports are operational dashboards and cannot claim stable profit or model success.

## Interpretation

- Combined proper-score improvement over market-only is evidence of incremental fundamental information at the executable data cutoff.
- It does not by itself establish positive betting value.
- Positive shadow ROI without proper-score/calibration support is treated cautiously as threshold/sample noise.
- A short profitable run is never called stable profit.
- A negative v1 result applies to the frozen model/cutoff/combination/policy only and does not authorize post-hoc historical rescue.
