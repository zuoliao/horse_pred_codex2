# Phase 5C program audit protocol

**Phase:** `Phase 5C: program audit and prospective validation pivot`  
**Date:** 2026-09-01 JST  
**Status:** active audit; new feature, target, and model experiments frozen

## Objective

Phase 5C connects the accumulated prediction research to the long-run profit objective without treating repeatedly exposed historical results as fresh validation. It does not optimize a new model. It freezes one fundamental candidate, records how every historical year has been used, runs one non-selective final-market oracle diagnostic, and makes a prospective shadow evaluation operationally specifiable.

The system is separated into four layers:

1. a no-odds fundamental model;
2. a cutoff-time market model;
3. a market-plus-fundamental combined model;
4. a separately frozen betting decision policy.

The primary prospective research question is whether adding the frozen fundamental score improves out-of-time Log Loss, Brier, and calibration over market-only on the same cutoff snapshots. Profit is evaluated only afterward with a fixed shadow policy.

## Historical evidence boundary

- All data and outcomes from 2013 through 2025 are the **development archive**.
- No year in 2013–2025 may be called an untouched final holdout.
- The next final validation population starts with genuinely timestamped 2026-or-later prospective snapshots.
- Existing metrics may be inventoried and reconciled. No historical parameter, feature, target, blend weight, cutoff, calibration, or bet-rule search is allowed.
- Final odds remain post-event descriptive oracle information. They cannot support executable EV, ROI, profit, or model-adoption claims.

## Frozen model-state rule

Existing evidence—not a new metric run—selects one canonical prospective fundamental candidate. Historical/formal controls and rolling candidates remain registered as evidence anchors rather than competing “current best” labels. Feature branches that were never evaluated together are not composed after the fact.

## One historical oracle diagnostic

The exact diagnostic is machine-preregistered in `experiments/program_audit_20260901/historical_oracle_preregistration.json` before any combined result is calculated.

- Population: complete 2024 development races common to the frozen SPEED-01 Binary predictions and corrected final-market oracle table.
- Fundamental: the already calibrated 2024 probability from the frozen 256-feature Binary candidate.
- Market: normalized inverse final win odds within each complete race.
- Combined: fixed equal-weight log-probability mean, equivalently normalized geometric mean `sqrt(p_fundamental * p_market)`.
- Metrics: race-macro Log Loss and Brier, race-balanced fixed-bin reliability/ECE, plus paired four-date block intervals for market minus fundamental and market minus combined.
- Decision use: none. No weight search, calibration refit, slice selection, bet threshold, ROI, or follow-up model is permitted.

## Prospective pivot

Collection activation is distinct from archive groundwork. A cutoff will be fixed only after at least two meetings provide receipt-based completeness and latency evidence. During the subsequent frozen evaluation window, fundamental-only, market-only, and combined predictions must be emitted for the same race population before results are known. Model artifacts, calibration, cutoff, combination rule, and betting rule cannot change during the window.

## Stop rule

After the audit documents, machine-readable registry/ledger/oracle summary, prospective activation plan, and protocol are synchronized and verified, stop for human decisions. A1/A2, nonlinear race-wise models, Inter-horse models, and additional feature search remain deferred.
