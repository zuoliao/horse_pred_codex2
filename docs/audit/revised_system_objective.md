# Revised system objective

## Success criterion

The program no longer defines success as a no-odds model beating the market by itself. The no-odds model is the fundamental-information layer. The central prospective question is whether its frozen score adds information to the market snapshot available at the decision cutoff.

```text
frozen fundamental model
             \
              -> frozen combined model -> frozen betting decision -> shadow settlement
             /
cutoff market model
```

| Layer | Input boundary | Output | Primary responsibility |
|---|---|---|---|
| Fundamental | PIT-safe race/history information, no odds | coherent win probabilities and score | encode non-market racing information |
| Cutoff market | last complete eligible odds snapshot at the frozen cutoff | normalized market win probabilities | represent information aggregated by the observable market |
| Combined | frozen fundamental and cutoff-market probabilities | coherent combined win probabilities | test incremental fundamental value over market-only |
| Betting decision | combined probability, displayed cutoff odds, fixed rules | shadow tickets and stakes | translate predictive edge into a separately auditable policy |

## Evaluation order

1. Primary: combined versus market-only out-of-time race Log Loss.
2. Co-primary guardrail: race Brier and calibration/reliability.
3. Diagnostics: fundamental-only, ranking, edge-band calibration, missingness, and cutoff completeness.
4. Only after prediction evidence: fixed-stake shadow ROI, number of bets, profit, monthly stability, drawdown, and uncertainty.

A profitable short period is not stable profit. No return claim is made until the preregistered minimum duration/population is reached, all months—including zero-bet months—are retained, and uncertainty is reported.

## Separation rules

- Final odds are descriptive oracle data, not the cutoff market.
- The fundamental model is never recalibrated using market outcomes during a frozen prospective window.
- Market or combined calibration is fitted/frozen only before the window according to the protocol; no rolling rescue based on live scores.
- Bet selection, staking, and settlement are separate from model acceptance.
- A1/A2 and all new model/feature/target work remain frozen until the prospective pipeline and governance are operational.
