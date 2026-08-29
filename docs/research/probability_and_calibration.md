# Probability Estimation and Calibration

**Scope:** Research workstream F (probability estimation and calibration)<br>
**Access date for web sources:** 2026-08-30<br>
**Status:** Research output; no implementation has been started.

## 1. Questions investigated

1. Which metrics assess win-probability quality, and what does each omit?
2. How should reliability diagrams and Expected Calibration Error (ECE) be used?
3. What are the trade-offs among sigmoid/Platt, isotonic, and beta calibration?
4. How can a race-level probability vector be obtained from binary or ranking scores?
5. How should calibration be fitted and evaluated under chronological dataset shift?
6. Should calibration be global or conditioned on odds, race type, or field size?

## 2. Evidence and sources

### 2.1 Proper scoring rules

- Gneiting, T. and Raftery, A. E. (2007), “Strictly Proper Scoring Rules, Prediction, and Estimation,” *Journal of the American Statistical Association*, 102(477), 359–378, DOI: 10.1198/016214506000001437. The paper defines proper and strictly proper scoring rules and treats logarithmic and quadratic/Brier scores as important examples. [Publisher page](https://doi.org/10.1198/016214506000001437); [author-hosted PDF](https://www.eecs.harvard.edu/cs286r/courses/fall10/papers/Gneiting07.pdf).
- Brier, G. W. (1950), “Verification of Forecasts Expressed in Terms of Probability,” *Monthly Weather Review*, 78(1), 1–3, DOI: `10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2`. [NOAA-hosted record](https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml).

**Verified fact.** Log Loss and Brier score are proper probability scores: in expectation, truthful probabilities are optimal. They measure the combined quality of discrimination and calibration, not calibration alone.

**Interpretation.** Log Loss penalizes a confidently wrong winner probability much more strongly than Brier score. Brier is bounded and easier to decompose/interpret, but in a one-winner-per-race, highly imbalanced runner table it can look deceptively good for low, uninformative probabilities. Neither should be used alone.

### 2.2 Reliability diagrams and scalar calibration errors

- Arrieta-Ibarra, I. et al. (2022), “Metrics of Calibration for Probabilistic Predictions,” *Journal of Machine Learning Research*, 23(351), 1–54. The paper analyzes reliability diagrams and the limitations of scalar summaries of them. [JMLR article and PDF](https://www.jmlr.org/papers/v23/22-0658.html).
- Kumar, A., Liang, P., and Ma, T. (2019), “Verified Uncertainty Calibration,” *NeurIPS 2019*. The paper shows that commonly used empirical calibration error estimates can be biased and proposes verified estimates. [NeurIPS paper](https://proceedings.neurips.cc/paper/2019/hash/f8c0c968632845cd133308b1a494967f-Abstract.html).
- Widmann, D., Lindsten, F., and Zachariah, D. (2019), “Calibration Tests in Multi-Class Classification: A Unifying Framework,” *NeurIPS 2019*. It connects calibration measures to test statistics and emphasizes that multiclass calibration has multiple non-equivalent definitions. [NeurIPS paper](https://proceedings.neurips.cc/paper/2019/hash/1c336b8080f82bcc2cd2499b4c57261d-Abstract.html).

**Verified facts.** Histogram reliability plots and ECE depend on bin count, bin boundaries, and finite-sample occupancy. A single ECE value can hide the direction and location of miscalibration. “Confidence calibration,” “classwise calibration,” and full multiclass calibration are not interchangeable.

**Implication.** This project must store the binning rule and sample count per bin. ECE is a descriptive diagnostic, not the model-selection objective or a proof of calibration.

### 2.3 Post-hoc calibration methods

- Platt, J. (1999/2000), “Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods,” in *Advances in Large Margin Classifiers*. It fits a sigmoid mapping on held-out predictions. [PDF copy hosted by University of Waikato](https://www.cs.waikato.ac.nz/~eibe/pubs/platt.pdf).
- Niculescu-Mizil, A. and Caruana, R. (2005), “Predicting Good Probabilities with Supervised Learning,” *ICML 2005*, 625–632, DOI: 10.1145/1102351.1102430. Their experiments compare Platt scaling and isotonic regression and show that boosted-tree scores can benefit from post-hoc calibration. [ICML PDF](https://icml.cc/Conferences/2005/proceedings/papers/079_GoodProbabilities_NiculescuMizilCaruana.pdf).
- Zadrozny, B. and Elkan, C. (2002), “Transforming Classifier Scores into Accurate Multiclass Probability Estimates,” *KDD 2002*, 694–699, DOI: 10.1145/775047.775151. This develops binary calibration/recombination approaches for multiclass estimates. [ACM DOI](https://doi.org/10.1145/775047.775151).
- Kull, M., Silva Filho, T., and Flach, P. (2017), “Beta Calibration: a Well-Founded and Easily Implemented Improvement on Logistic Calibration for Binary Classifiers,” *AISTATS 2017*, PMLR 54:623–631. Beta calibration includes the identity mapping; the authors note that isotonic regression can overfit smaller calibration sets and that ordinary logistic calibration can uncalibrate an already calibrated classifier. [PMLR article and PDF](https://proceedings.mlr.press/v54/kull17a.html).

**Findings.**

| Method | Shape / flexibility | Data need | Main risk | MVP role |
|---|---|---:|---|---|
| Sigmoid / Platt | raw marginまたは`logit(p)`へのtwo-parameter sigmoid | low to moderate | cannot represent every distortion | mandatory simple candidate |
| Beta calibration | three-parameter family; includes identity | low to moderate | still a global parametric map | preferred parametric challenger |
| Isotonic regression | flexible monotone step function | higher | overfits sparse tails; unstable extrapolation | challenger only when calibration sample is sufficient |
| No post-hoc map | identity | none | raw GBDT probabilities may be distorted | required control |

No cited study establishes one universally best method for JRA races or LightGBM under this project's exact split. Selection therefore remains an empirical development-set comparison.

### 2.4 Calibration under shift

- Park, S. et al. (2020), “Calibrated Prediction with Covariate Shift via Unsupervised Domain Adaptation,” *AISTATS 2020*, PMLR 108:3219–3229. It explicitly studies the failure of ordinary source-distribution calibration under covariate shift. [PMLR article](https://proceedings.mlr.press/v108/park20b.html).
- Gong, Y. et al. (2021), “Confidence Calibration for Domain Generalization Under Covariate Shift,” *ICCV 2021*, 8958–8967. [CVF paper and PDF](https://openaccess.thecvf.com/content/ICCV2021/html/Gong_Confidence_Calibration_for_Domain_Generalization_Under_Covariate_Shift_ICCV_2021_paper.html).

**Verified fact.** Calibration on one distribution is not a guarantee of calibration after covariate or concept shift.

**Project interpretation.** JRA rule changes, changing field composition, new equipment/data definitions, and temporal changes in the population or market can all make an old mapping stale. The appropriate response in the MVP is chronological validation and monitoring, not a complex domain-adaptation method.

## 3. Race-specific probability semantics

Exactly one runner normally wins a completed race, apart from dead heats. Therefore coherent single-winner probabilities should satisfy

\[
0 \le p_{i,r} \le 1, \qquad \sum_{i \in r} p_{i,r}=1.
\]

An independently trained binary classifier does not enforce the sum constraint. LambdaRank scores are not probabilities at all.

Three different outputs must not be conflated:

1. **Raw binary probability:** LightGBM binary output for each runner. Directly optimizes a runner-level Bernoulli objective but may not sum to one in a race.
2. **Race-normalized binary output:** a deterministic normalization such as \(p_i/\sum_j p_j\). This is coherent but can change marginal calibration and has no general optimality guarantee.
3. **Race-softmax score probability:** \(q_i(T)=\exp(s_i/T)/\sum_j\exp(s_j/T)\), with temperature \(T\) fitted on a later calibration set by winner negative log likelihood. This is a Plackett–Luce-like first-place mapping; it is a useful empirical baseline, not proof that the full Plackett–Luce model is true.

Dead heats require an explicit target and scoring policy. The data pipeline should retain official dead-heat status. For the initial probability score, if (m) horses share first place, use a **winner-mass convention** (y_i=1/m) for each tied winner and zero otherwise. The race cross-entropy is then (-\sum_i y_i\log q_i). This is equivalent to scoring a pseudo-outcome that selects one tied winner uniformly; it is not a model of the joint dead-heat event or each ticket's settlement probability. Report dead-heat counts and an exclude-dead-heats sensitivity. Betting settlement remains separate and uses the actual official payout.

## 4. Recommended metrics and plots

### 4.1 Primary probability metrics

- **Race winner Log Loss:** \(-\sum_i y_{i,r}\log q_{i,r}\), macro-averaged by race, using a coherent race probability vector and the winner-mass convention above. Store the probability clipping epsilon used for numerical evaluation.
- **Race multiclass Brier:** \(\sum_i(q_{i,r}-y_{i,r})^2\), without a one-half factor, macro-averaged by race. Because the uniform baseline is field-size dependent, report paired differences on the same races, uniform skill, and field-size strata.
- **Runner binary Log Loss and Brier:** retained to evaluate the direct binary model. Store both runner-micro and race-macro versions, state the training/evaluation weights, and report the within-race probability-sum distribution.
- **Market-relative skill:** report score difference against a preregistered market-only probability baseline; do not compare raw scores on incompatible scales.

Log Loss and Brier must both be shown. The market baseline and simple uniform \(1/n_r\) baseline make the scale interpretable.

### 4.2 Calibration diagnostics

- Reliability plot with fixed, versioned binning; equal-frequency bins are preferable for the main plot because wins are sparse in high-probability regions.
- Per-bin number of runners, number of wins, mean prediction, empirical win rate, and uncertainty interval.
- Runner-level logistic calibration intercept/slope diagnostics with the weighting definition recorded. A coherent vector has predicted and observed total winner mass one by construction, so comparing those race totals is not an informative calibration-in-the-large test.
- ECE as a secondary summary, with number and type of bins in the artifact.
- Race probability-sum histogram for unnormalized binary output.
- Subgroup diagrams for field size, surface, class, prediction timestamp, and probability/odds band only where sample sizes permit.

Pooling runners ignores within-race dependence and makes large fields contribute more. Store runner-weighted and race-weighted reliability summaries explicitly. Confidence intervals should resample at least by race; for temporal uncertainty, resampling contiguous race-day or week blocks is safer (see `backtesting_and_leakage.md`).

## 5. Calibration data protocol

### High-confidence recommendation

Use strictly chronological, non-overlapping stages:

```text
model-fitting period
    -> out-of-time calibration slice
    -> development backtest
    -> untouched final backtest
```

- Fit the base model only on the model-fitting period.
- Generate truly out-of-sample scores for the calibration slice.
- Fit each runner calibrator only on those scores and outcomes, using `1/n_asof` per-runner weights and weighted binary Log Loss. Use the same weights for isotonic regression. This weighting is a race-balanced engineering choice, not a JRA-optimal result established by literature.
- Apply Platt scaling to the saved raw margin or `logit(clip(p, eps))`, beta calibration to `log(clip(p, eps))` and `log(1-clip(p, eps))`, and isotonic regression to raw `p`; store `eps` and all extrapolation rules.
- Apply the complete mapping—including any binary calibration map and subsequent race normalization—to form the final coherent vector. Because normalization can change marginal calibration, select candidates by the final vector's race Log Loss/Brier plus reliability, never by the intermediate map or ROI alone.
- Freeze base model, calibrator family, subgroup policy, and betting thresholds before one-time final evaluation.

For reproducibility, constrain temperature (T>0), fit it by race winner NLL with one fractional winner target per race, and record whether its input is a raw binary margin, binary logit, or ranking score. Beta calibration must specify clipping of input probabilities at zero/one and whether monotonicity is constrained. Isotonic calibration must specify its behavior outside the observed score range and at step values of zero/one. Intermediate calibrated-but-unnormalized outputs are diagnostic only; only the coherent final vector may feed the EV proxy.

If more efficient use of older training data is needed, rolling-origin out-of-fold predictions may train the calibrator, but every calibration observation must come from a model trained strictly before that race. Random cross-validation is not acceptable.

For final evaluation, first select the base recipe, mapping family, and threshold using only rolling development out-of-time predictions. Reserve a fresh calibration window immediately before final: fit the final base only before that window, generate out-of-time scores on it, fit the calibrator there, then freeze both throughout final. If development data are added to the final base fit, another base-unseen calibration window must remain. Never fit a calibrator on rows seen by its base model.

After model selection, a production refit may use the same predeclared chronological recipe. Its operational performance is prospective and must not be confused with the frozen research holdout result.

## 6. Global versus conditional calibration

### Findings

- Global calibration has the best effective sample size but can conceal subgroup errors.
- Field size changes the probability base rate and the race-sum constraint; it is a plausible nonmarket conditioning variable.
- Race class, surface, and distance may have distinct distributions, but small cells make flexible calibration unstable.
- Odds are downstream market information under the agreed primary design.

### Recommendation

1. Start with one global calibrator and subgroup diagnostics.
2. Add a field-size-aware or broad race-type calibration experiment only if development diagnostics show a stable, material error and the group has enough wins.
3. Do not train the primary calibrator on odds or odds bands. Doing so creates an **odds-aware prediction system**, even if the base LightGBM model excludes odds. It may later be tested as a clearly labeled comparison model.
4. Do not choose subgroup boundaries after inspecting the final holdout.

## 7. Uncertainties and conflicts

- Published calibration comparisons are not JRA-specific and do not establish the best map for the eventual data volume.
- The best way to turn binary scores into a coherent race distribution is an empirical question. Simple normalization, temperature-softmax, and direct calibrated binary probabilities can disagree.
- ECE variants have materially different finite-sample behavior; no single scalar calibration metric is authoritative.
- The required calibration-window length depends on temporal drift and the number of races/winners, neither of which is known until a source and coverage are fixed.
- Dead heats and cancelled/non-completed races need data-source-specific semantics.

## 8. Concrete recommendation for the first experiment family

Keep these as separate, interpretable hypotheses:

1. **Binary raw baseline:** raw LightGBM binary probabilities.
2. **Parametric recalibration:** compare identity, Platt-on-margin/logit, and beta calibration on the same chronological calibration predictions.
3. **Nonparametric challenger:** isotonic only if development calibration counts support stable tails.
4. **Coherence experiment:** compare direct calibrated binary probabilities with race-normalization and one-temperature race softmax.
5. **LambdaRank probability mapping:** treat ranking scores as non-probabilistic until a separately fitted race-softmax/Plackett–Luce-like mapping has passed Log Loss and reliability evaluation.

The fixed initial betting rule must consume the frozen probability output selected by probability metrics. Selecting a calibrator because it maximizes development ROI would entangle prediction calibration with strategy tuning.
