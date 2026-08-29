# JRA Betting Market and Market-Implied Probability

**Scope:** Research workstream G<br>
**Access date for web sources:** 2026-08-30<br>
**Status:** Research output; no wagering automation is proposed.

## 1. Questions investigated

1. How do JRA pari-mutuel win odds and payouts work?
2. What deduction/takeout applies, and how should displayed odds be converted to a market probability baseline?
3. How should an EV rule use provisional odds versus final payouts?
4. What evidence exists for favorite–longshot bias and market efficiency in Japanese racing?
5. How should a no-odds prediction model be compared fairly with the market?

## 2. Official market mechanics

### Evidence

- JRA, “具体的な払戻計算式を知りたいのですが？” The official formula defines the distributable pool and lists the standard payout rates effective from 2014-06-07: win 80.0%, place 80.0%, bracket quinella/quinella/wide 77.5%, exacta/trio 75.0%, trifecta 72.5%, WIN5 70.0%. [JRA FAQ](https://www.jra.go.jp/faq/pop03/1_17.html).
- JRA, “オッズ（競馬用語辞典）.” JRA describes displayed odds as an approximate payout multiple per ¥100 and notes that final odds may still change after a subsequent exclusion and may differ from payout in a dead heat. [JRA glossary](https://www.jra.go.jp/kouza/yougo/w406.html).
- JRA, “馬券のルール.” Payout follows the official finish once the race is confirmed; scratches/exclusions can cause refunds, and dead heats have special settlement. [JRA betting rules](https://www.jra.go.jp/kouza/baken/index.html).
- Japan Horse Racing Law, Article 8, and the Ordinance for Enforcement define the statutory framework within which JRA sets payout rates. [JRA-hosted Horse Racing Law PDF](https://www.jra.go.jp/company/about/law/pdf/01.pdf); [enforcement ordinance PDF](https://www.jra.go.jp/company/about/law/pdf/03.pdf).

### Verified findings

JRA win betting is pari-mutuel, not fixed-odds bookmaking. For a single winner, ignoring breakage, special additions, refunds, and dead heats, the win payout multiple approximately follows

\[
o_i \approx R\frac{T}{B_i}=\frac{R}{s_i},
\]

where \(T\) is valid pool amount, \(B_i\) is the amount on horse \(i\), \(s_i=B_i/T\) is its betting share, and ordinary win payout rate \(R=0.80\). Thus displayed odds describe the current pool and can move until sales close. They are not a guaranteed execution price at the earlier decision timestamp.

Special payout promotions, minimum-return rules, rounding/breakage, refunds, exclusions, and dead heats can break the simple approximation. Backtests must use the actual official payout for realized profit.

## 3. Converting win odds to market probabilities

For displayed decimal odds \(o_{i,t}\) at timestamp \(t\):

### 3.1 Raw reciprocal

\[
u_{i,t}=1/o_{i,t}.
\]

This is not a coherent probability vector because it embeds the pool deduction and display rounding. The reciprocal sum will generally differ from one.

### 3.2 Normalized market baseline

\[
q^{mkt}_{i,t}=\frac{1/o_{i,t}}{\sum_{j\in r}1/o_{j,t}}.
\]

For an ordinary single-winner pool under the approximation above, normalization removes the common payout rate and estimates the bet/pool share. It also guarantees probabilities sum to one. This is the recommended **minimal market forecast baseline**, not an objectively fair probability merely by construction.

### 3.3 Pool-share reconstruction

If the data source supplies timestamped vote totals and the necessary pool fields, reconstructing shares from the official pool quantities is preferable to reciprocal odds because displayed odds are rounded. JRA-VAN’s TARGET manual states that Data Lab time-series data include odds and bet-type total votes at roughly 5–10 minute intervals for win, place, bracket quinella, and quinella; this is product documentation, not evidence that every historical year or service tier is available without verification. [JRA-VAN TARGET manual](https://targetfaq.jra-van.jp/faq/detail?hot_list=true&id=667&site=SVKNEGBV).

### Distinction that must be preserved

- Use \(q^{mkt}\) for a **market forecast baseline** and market-vs-model Log Loss/Brier comparisons.
- Use the observable displayed \(o_{i,t}\), not normalized \(q^{mkt}\), in the snapshot score \(\hat p_i o_{i,t}\). Because that pari-mutuel price is not locked, call this a **snapshot EV proxy**, not the true conditional expected return at execution.
- Use actual official payout for realized backtest profit.

These are three different quantities.

## 4. Prediction and execution timestamps

### Evidence

- JRA warns that even the displayed final odds can change after exclusion and that dead heats can change settlement. [JRA odds glossary](https://www.jra.go.jp/kouza/yougo/w406.html).
- Hanyu, H., Ishii, S., Otani, S., and Teramoto, K. (2025), “When Final Odds Are Not Sufficient: Last-Minute Market Movements and Return Predictability in Parimutuel Betting,” *Journal of Behavioral Economics and Finance*, 18 special issue, S1–S4, DOI: 10.11167/jbef.18.S1.pp.S1-S4. This four-page conference special-issue paper, which states that a full article is under consideration, reports preliminary evidence from JRA-VAN interim JRA odds through one minute before post time that last-minute changes are related to realized returns and attenuate the favorite–longshot pattern. Treat it as recent direct but preliminary evidence, not a settled effect size. [J-STAGE PDF](https://www.jstage.jst.go.jp/article/jbef/18/Special_issue/18_18.S1.pp.S1-S4/_pdf/-char/en).

### Finding

A backtest that selects bets with final odds commits look-ahead unless the decision could actually observe those odds. This remains true even if the final payout, rather than final odds, is the correct realized return.

### Recommendation

Maintain at least two explicitly named evaluations:

1. **Executable-timestamp backtest:** features and provisional odds frozen at a predeclared time; selections use that snapshot; realized proceeds use official payout.
2. **Final-market oracle diagnostic:** final odds used only to assess market information and the sensitivity of results to late movement. Never label this as realizable ROI.

Candidate operational timestamps must be confirmed against actual feed availability. A practical research design is an earlier “day-before/entry” no-body-weight forecast and a race-day forecast at a fixed lead time (for example, 10 or 15 minutes before the then-published scheduled post), but no exact lead time should be finalized until snapshot completeness, feed receipt/processing latency, and betting cutoff semantics are verified.

## 5. Favorite–longshot bias and market efficiency

### Evidence

- Thaler, R. H. and Ziemba, W. T. (1988), “Anomalies: Parimutuel Betting Markets: Racetracks and Lotteries,” *Journal of Economic Perspectives*, 2(2), 161–174, DOI: 10.1257/jep.2.2.161. [AEA article](https://www.aeaweb.org/articles?id=10.1257/jep.2.2.161).
- Snowberg, E. and Wolfers, J. (2010), “Explaining the Favorite–Long Shot Bias: Is It Risk-Love or Misperceptions?” *Journal of Political Economy*, 118(4), 723–746, DOI: 10.1086/655844. [Publisher page](https://doi.org/10.1086/655844).
- Ottaviani, M. and Sørensen, P. N. (2023), “Pari-Mutuel Betting Markets: Racetracks and Lotteries Revisited,” *Annual Review of Financial Economics*, 15, 641–662, DOI: 10.1146/annurev-financial-053122-021925. [Annual Reviews article](https://www.annualreviews.org/content/journals/10.1146/annurev-financial-053122-021925).
- Okamoto, K. and Fukushige, M. (2022), “Favourite–Longshot Biases in a Pari-Mutuel System without Cross Arbitrage,” *Economics Bulletin*, 42(1), 203–207. This JRA study reports differing bias in separately pooled bracket-quinella and quinella markets. [RePEc record](https://ideas.repec.org/a/ebl/ecbull/eb-21-00684.html).
- Hanyu et al. (2025), above, supplies recent Japanese central-racing evidence that interim dynamics matter.

### Findings

The general favorite–longshot bias is well established in the international literature, and some Japanese evidence exists. However:

- magnitude varies by period, jurisdiction, bet type, and market design;
- separately pooled bet types need not share one price;
- a behavioral or informational explanation is not uniquely identified by the existence of the bias;
- historical average return by final-odds band is not itself an executable trading strategy;
- recent last-minute JRA evidence makes static final-odds analysis incomplete.

### Recommendation

Treat favorite–longshot behavior as a benchmark and subgroup diagnostic, not as an assumed permanent edge. Estimate it afresh on training/development periods only and report it on final holdout without retuning.

## 6. Fair model-versus-market comparisons

On the exact same races, timestamp, scratches, and field definition, compare:

1. uniform \(1/n_r\) probability;
2. normalized timestamped market probability \(q^{mkt}_{i,t}\);
3. primary no-odds model probability;
4. a predeclared blend or odds-aware model only in a later, separately labeled experiment.

Report race winner Log Loss, race Brier, top-1/top-k, reliability, and subgroup results. A model can rank differently from the market while having worse probabilities, or improve probability score without achieving positive post-deduction ROI. Both facts should be reported.

For incremental information, a development-stage logistic/conditional-logit diagnostic can compare market log probability with and without the frozen model score. It is a statistical diagnostic, not part of the primary no-odds model.

## 7. Initial betting rule

The agreed initial rule remains appropriate with one clarification:

```text
bet type:              win only
stake:                 fixed amount per selected horse
selection score:       frozen coherent P(win) × observable snapshot odds
score meaning:         snapshot EV proxy; the displayed price is not locked
thresholds:            predeclared (e.g. >1.0, >1.1, >1.2 as diagnostics)
realized return:       official payout / stake, including refunds and settlement rules
primary threshold:     choose exactly one before final evaluation
```

The threshold of 1.0 is only a model-estimated snapshot break-even proxy, not the true conditional EV or a guarantee. The true decision-time return depends on the distribution of closing pool payouts. Estimation error, adverse late odds movement, rounding, and pool changes can turn an apparent edge into a loss. Record snapshot-to-final payout movement and the fraction of selected tickets whose proxy crosses back below the threshold as slippage diagnostics. To avoid choosing the luckiest threshold, designate one primary threshold on development data and treat the other fixed thresholds as labeled sensitivity diagnostics.

## 8. Uncertainties and negative findings

- No official source found in this workstream states that a bettor can lock a displayed JRA pari-mutuel price before pool close.
- Public JRA pages establish payout mechanics but do not by themselves provide a replayable historical pre-race odds-snapshot dataset.
- The JRA-VAN manual supports the existence of some 5–10 minute time-series odds, but exact historical coverage, retention, licensing, cost, and machine throughput must be confirmed in `data_sources.md` before source selection.
- The current evidence does not justify assuming a stable, exploitable JRA favorite–longshot edge after realistic timing and multiple-testing correction.
- Taxes are outside the model ROI metric and depend on the user’s circumstances; JRA notes that payouts can be taxable. Net personal profit therefore cannot be inferred from pre-tax backtest ROI. [JRA betting rules](https://www.jra.go.jp/kouza/baken/index.html).

## 9. Project recommendations

1. Use normalized reciprocal timestamped win odds as the minimum market probability baseline; use vote shares if reliably reconstructable.
2. Store `odds_observed_at`, scheduled post, actual post if available, feed receipt time, and source revision/version.
3. Store the exact snapshot used for every prediction and bet decision.
4. Use final payout only for settlement, and label final-odds selection as an oracle analysis.
5. Keep odds out of the primary base model and primary calibrator; test odds-aware or blended models later as explicit comparison experiments.
6. Preserve refunds, exclusions, dead heats, special payout campaigns, and cancellations rather than dropping losing-looking exceptions.
7. Freeze one primary EV threshold before final holdout and count every tried rule in the experiment ledger.
8. Preserve the as-of field at decision time. A horse scratched after the decision remains in the decision snapshot and its ticket is settled as an official refund; post-event renormalization to actual starters is a separate diagnostic, not an executable backtest.
