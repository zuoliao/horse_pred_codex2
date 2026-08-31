# External EDA Practices for Grouped Temporal Racing Data

## 1. Questions

This workstream asks which established exploratory practices from tabular prediction, temporal validation, grouped data, sports ranking, and discrete-choice modeling transfer to JRA race prediction, and which require modification.

The specific questions are:

1. How should missingness, drift, leakage, high-cardinality entities, redundancy, interaction, calibration, learning curves, and error slices be diagnosed?
2. Which validation unit is defensible when runners share a race, horses and connections recur, and information accumulates over time?
3. What can SHAP, permutation importance, partial dependence (PD), accumulated local effects (ALE), adversarial validation, and model disagreement establish—and what can they not establish?
4. How should an exploratory finding be promoted to a later experiment without turning EDA into repeated development-set selection?
5. How do race-wise ranking and choice-probability formulations change ordinary row-wise tabular practice?

## 2. Data scope

This is a methods review, not a target-aware analysis of project data. No JRA raw, cache, model, prediction, or runner-level data was read, transmitted, or submitted to an external service. The review uses public papers and official software documentation accessed on 2026-09-01. Its evidence table has 25 topic rows; runner count, race count, missing count, and JRA year coverage are therefore not applicable.

The project-side scope is fixed by the Phase 5A contract:

- 2013: coverage and state warm-up only;
- 2014–2019: discovery;
- 2020–2021: temporal replication;
- 2022: confirmation and prioritization;
- 2023: excluded from this EDA and retained for its prior calibration role;
- 2024: no new access;
- 2025: forbidden.

The unit of choice is one race, not one runner. Consequently, methods below are evaluated for a grouped, temporally ordered panel in which runners within a race are mutually dependent and horse, jockey, and trainer identities recur between races.

## 3. Definitions

- **Exploratory / replicated / confirmed**: discovered in 2014–2019, directionally reproduced in 2020–2021, and retained after a single 2022 confirmation, respectively. “Confirmed” here means confirmed for prioritization, not a production-valid causal or predictive claim.
- **Target leakage**: target-related information that would not legitimately be available at prediction time enters model fitting, preprocessing, feature selection, or evaluation. This includes current-race results, future opponent results, full-period normalization, and target statistics calculated with the row being encoded.
- **Entity leakage**: repeated horse, jockey, trainer, or other entity information crosses a validation boundary in a way inconsistent with deployment, or an entity-level target encoding uses future/current outcomes.
- **Covariate shift**: the input distribution changes while the conditional outcome mechanism is assumed unchanged. A time discriminator detects separability of periods; it does not establish covariate shift, concept drift, or harm by itself.
- **Adversarial validation**: a diagnostic classifier distinguishes an earlier period from a later period. Its AUC measures separability under that classifier and feature set, not predictive model degradation.
- **Conditional association**: an association evaluated within a race/choice set, such as a race-stratified contrast or conditional-logit coefficient. It is not a causal effect.
- **Model reliance**: loss change when an already-fitted model is perturbed. It differs from the marginal value of a feature group measured by retraining without it.
- **Choice model**: a model whose denominator is the current race's eligible runner set. Conditional logit assigns utilities within that set; Plackett–Luce extends the idea to ordered outcomes.

## 4. Methods

### 4.1 Search and source selection

The search covered the required EDA topics and favored original papers, publisher records, author-hosted manuscripts, and official scikit-learn, LightGBM, CatBoost, and SHAP-related sources. Kaggle-style practices were treated as informal prompts only: popularity, leaderboard success, and copied notebook code were not evidence. No external horse-racing result was generalized to JRA.

The following sources are the re-checkable evidence base. Publication metadata is included where available; all were accessed 2026-09-01.

| Topic | Primary or official source | Relevance |
|---|---|---|
| Missing-data mechanism | Donald B. Rubin, “Inference and Missing Data,” *Biometrika* 63(3), 1976, pp. 581–592, [DOI 10.1093/biomet/63.3.581](https://academic.oup.com/biomet/article-abstract/63/3/581/270932) | Missingness mechanism cannot be ignored without assumptions; motivates cause-specific missingness audit. |
| Leakage | Kaufman, Rosset, Perlich, and Stitelman, “Leakage in Data Mining,” *ACM TKDD* 6(4), 2012, [DOI 10.1145/2382577.2382579](https://doi.org/10.1145/2382577.2382579) | Formalizes leakage as information illegitimately available to the learner. |
| Preprocessing leakage | scikit-learn, “Common pitfalls and recommended practices,” stable documentation, [section 11.2](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) | Fit transforms and feature selection on training partitions only. |
| Grouped and temporal CV | scikit-learn, “Cross-validation: evaluating estimator performance,” stable documentation, [user guide](https://scikit-learn.org/stable/modules/cross_validation.html) and [`GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) | Ordinary K-fold assumes approximately IID rows; group and time constraints solve different problems. |
| Time-series CV caveat | Bergmeir, Hyndman, and Koo, “A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction,” *CSDA* 120, 2018, pp. 70–83, [DOI 10.1016/j.csda.2017.11.003](https://robjhyndman.com/publications/cv-time-series/) | Standard K-fold validity requires restrictive error assumptions; it does not justify random folds for this nonstationary panel. |
| Dependent uncertainty | Politis and Romano, “The Stationary Bootstrap,” *JASA* 89(428), 1994, pp. 1303–1313, [DOI 10.1080/01621459.1994.10476870](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870) | Supports block resampling rather than treating rows as independent. |
| Dataset shift | Rabanser, Günnemann, and Lipton, “Failing Loudly,” NeurIPS 2019, [proceedings](https://papers.neurips.cc/paper_files/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html) | Domain classifiers help characterize shift; detection does not determine harmfulness. |
| Covariate shift | Sugiyama, Krauledat, and Müller, “Covariate Shift Adaptation by Importance Weighted Cross Validation,” *JMLR* 8, 2007, pp. 985–1005, [paper](https://jmlr.org/papers/v8/sugiyama07a.html) | Distinguishes input shift from a change in the conditional target mechanism. |
| Ordered target statistics | Prokhorenkova et al., “CatBoost: Unbiased Boosting with Categorical Features,” NeurIPS 2018, [proceedings](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html) | Demonstrates prediction shift/leakage from ordinary target statistics and an ordering principle. |
| Model selection bias | Cawley and Talbot, “On Over-fitting in Model Selection and Subsequent Selection Bias,” *JMLR* 11, 2010, pp. 2079–2107, [paper](https://www.jmlr.org/papers/v11/cawley10a.html) | Repeatedly optimizing a noisy validation estimate overfits the selection criterion. |
| Nested evaluation | Varma and Simon, “Bias in Error Estimation When Using Cross-Validation for Model Selection,” *BMC Bioinformatics* 7:91, 2006, [DOI 10.1186/1471-2105-7-91](https://link.springer.com/article/10.1186/1471-2105-7-91) | Selection and error estimation need separated levels or an untouched evaluation set. |
| PD / ALE | Apley and Zhu, “Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models,” *JRSS B* 82(4), 2020, pp. 1059–1086, [DOI 10.1111/rssb.12377](https://doi.org/10.1111/rssb.12377) | PD can extrapolate under correlated inputs; ALE reduces that extrapolation. |
| SHAP | Lundberg and Lee, “A Unified Approach to Interpreting Model Predictions,” NeurIPS 2017, [proceedings](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) | Defines a local additive attribution framework for a fitted prediction function. |
| SHAP interpretation | Janzing, Minorics, and Blöbaum, “Feature Relevance Quantification in Explainable AI: A Causal Problem,” AISTATS 2020, PMLR 108:2907–2916, [paper](https://proceedings.mlr.press/v108/janzing20a.html) | Shows that feature dependence and intervention semantics make causal readings of attributions unsafe. |
| Permutation reliance | Hooker, Mentch, and Zhou, “Unrestricted Permutation Forces Extrapolation,” *Statistics and Computing* 31:82, 2021, [DOI 10.1007/s11222-021-10057-z](https://link.springer.com/article/10.1007/s11222-021-10057-z) | Independent permutation can create unrealistic points when features are dependent. |
| Correlated features | scikit-learn, “Permutation Importance with Multicollinear or Correlated Features,” stable example, [documentation](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html) | Importance can be diluted among substitutes; Spearman clustering is a diagnostic, not a final selection rule. |
| Calibration | Niculescu-Mizil and Caruana, “Predicting Good Probabilities with Supervised Learning,” ICML 2005, pp. 625–632, [paper](https://icml.cc/Conferences/2005/proceedings/papers/079_GoodProbabilities_NiculescuMizilCaruana.pdf), and scikit-learn [calibration guide](https://scikit-learn.org/stable/modules/calibration.html) | Reliability curves require support counts; calibrators need data independent of model fitting. |
| Error slicing | Chung et al., “Slice Finder: Automated Data Slicing for Model Validation,” ICDE 2019, [Google Research record](https://research.google/pubs/slice-finder-automated-data-slicing-for-model-validation/) | Useful slices should be interpretable, problematic, and sufficiently supported. |
| Feature-selection stability | Nogueira, Sechidis, and Brown, “On the Stability of Feature Selection Algorithms,” *JMLR* 18(174), 2018, pp. 1–54, [paper](https://www.jmlr.org/papers/v18/17-514.html) | Selection stability across resamples/periods is distinct from predictive score. |
| Multiple comparisons | Benjamini and Hochberg, “Controlling the False Discovery Rate,” *JRSS B* 57(1), 1995, pp. 289–300, [DOI 10.1111/j.2517-6161.1995.tb02031.x](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1995.tb02031.x) | Formal testing needs a declared family; exploratory screening still needs multiplicity records. |
| Preregistration | Nosek et al., “The Preregistration Revolution,” *PNAS* 115(11), 2018, pp. 2600–2606, [DOI 10.1073/pnas.1708274114](https://doi.org/10.1073/pnas.1708274114) | Separates prior predictions from post-hoc explanations and improves interpretability of evidence. |
| Choice sets | McFadden, “Conditional Logit Analysis of Qualitative Choice Behavior,” in *Frontiers in Econometrics*, 1974, pp. 105–142, [author-hosted PDF](https://eml.berkeley.edu/reprints/mcfadden/zarembka.pdf); Train, *Discrete Choice Methods with Simulation*, 2nd ed., 2009, [Berkeley page](https://eml.berkeley.edu/books/choice2.html) | Runner utilities must be normalized within the available alternatives; standard logit has IIA limitations. |
| Ordered outcomes | Plackett, “The Analysis of Permutations,” *Applied Statistics* 24(2), 1975, pp. 193–202, [DOI 10.2307/2346567](https://rss.onlinelibrary.wiley.com/doi/abs/10.2307/2346567) | Provides a probability distribution over full rankings; assumptions need testing before adoption. |
| Learning-to-rank grouping | LightGBM, [`LGBMRanker` official API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html) and [parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html) | Ranking requires explicit query groups; gain/truncation encode target priorities. |
| Learning curves | scikit-learn, “Validation curves: plotting scores to evaluate models,” stable documentation, [learning-curve guide](https://scikit-learn.org/stable/modules/learning_curve.html) | Training-size curves distinguish data-limited from representation/model-limited behavior only under a valid splitter. |

### 4.2 Transfer criteria

A practice was judged transferable only if it can preserve all four constraints: target-time availability, same-date emission-before-update, race grouping, and pre-2023 evaluation. A practice that violates any constraint may still be useful as a non-predictive quality audit, but not as predictive evidence.

## 5. Descriptive findings

The statements in this section are observations from the reviewed methods literature. They are not observations about JRA data.

### 5.1 Recommended application matrix

| Practice | Ordinary tabular use | Phase 5A adaptation | Main failure if copied directly |
|---|---|---|---|
| Distribution / ECDF / outliers | Inspect each column globally | Show year and race-relative distributions; preserve genuine domain tails; include availability counts | Global distributions confound field size, class, distance, and era |
| Missingness | Rate, indicator, matrix, imputation | Split status/structural/history/era/source missingness; prove indicator availability at prediction time | Missingness may encode a future status or source correction |
| Univariate target plot | Quantile vs event rate | Quantiles within discovery years; race-macro win/top-3 rate; within-race percentile and conditional association | Runner weighting lets large fields dominate and ignores the one-winner constraint |
| Drift / adversarial validation | Compare train and test feature distributions | Compare 2014–2019 vs 2020–2021 and 2014–2021 vs 2022 with date/race grouped validation; audit feature importance and missingness | AUC is mistaken for harmful concept drift or a feature opportunity |
| CV | Random K-fold | Fixed expanding-origin years; group whole races; fit all transforms/state/calibration inside each temporal fold | Future and same-choice-set information leak into training |
| Entity handling | One-hot, counts, target encoding | IDs remain keys; use prior-only smoothed state, effective sample size, uncertainty, cold-start flag | Identity memorization and future target statistics |
| Redundancy | Correlation heatmap or VIF | Spearman and missingness clusters by source family and year; distinguish deterministic derivation, fitted reliance, and retrained group ablation | A substitute feature looks unimportant, or a harmful group looks important in the fitted model |
| PD / ICE | Vary one feature over observed range | Prefer ALE and support/rug counts within replicated periods; use 2D views only for preregistered domain pairs | Impossible feature combinations and causal over-interpretation |
| SHAP | Global/local fitted-model explanation | Compute on OOT folds; aggregate by race and year; cluster correlated families; compare sign/stability | Attribution is treated as causal effect or incremental OOS value |
| Permutation importance | Shuffle one held-out column | Permute within defensible blocks/choice sets when semantics permit; group correlated features; compare with retrained ablation | Cross-era/race impossible rows, extrapolation, diluted substitute importance |
| Calibration | Reliability diagram / ECE | Race-coherent probabilities; OOT bins with race/date bootstrap; field-size slices; show bin support; Log Loss/Brier guardrails | Runner bin counts imply excess precision; ECE hides binning and resolution |
| Learning curve | Random subsets of rows | Expand training years or number of whole dates/races; evaluate fixed future years | Random runners break history state and choice sets |
| Error slicing | Search all feature conjunctions | Use domain-defined slices first; require race count, effect interval, time replication, and multiplicity label | Small, post-hoc slices manufacture apparently large errors |
| Interaction screening | All pairwise PDP/SHAP interactions | Limited domain pairs in discovery, 2D ALE/shallow tree diagnostic, then replication | Combinatorial search and correlated-feature artifacts |
| Model disagreement | Cross-tab predictions | Compare Binary/Ranker at race level on common OOT folds: selected winner, entropy, top-1 margin, loss | Disagreement is mistaken for ensemble complementarity |
| Hypothesis registry | Informal idea list | Record evidence period, replication, support, PIT/redundancy/cost, and one fixed later validation plan | EDA silently becomes feature selection on confirmation data |

### 5.2 Missingness and high-cardinality entities

**Observation.** Rubin's framework makes the missingness mechanism part of the inferential assumptions; a low missing rate does not make missingness ignorable. Ordered target statistics were introduced specifically because ordinary category target means can use the encoded observation's target and induce prediction shift.

**Interpretation.** In this project, no-history, status-related missingness, and source-era missingness are semantically different. Horse/jockey/trainer IDs are especially risky: cumulative target summaries can approach an identity fingerprint even when the raw ID is absent.

**EDA implication.** Report missing rate and missingness-conditioned outcomes by time regime, but do not use a missing indicator predictively unless it is available before the target race. For entities, inspect prior-start support, shrinkage, effective sample size, year-to-year rank stability, and cold-start behavior. Any target-derived state must be prequential: earlier dates only, with all same-date predictions emitted before update.

### 5.3 Drift and adversarial validation

**Observation.** A domain classifier can reveal that periods are distinguishable and identify the variables associated with separability. The dataset-shift literature explicitly separates detection, characterization, and harmfulness.

**Interpretation.** High adversarial AUC may reflect legitimate rule changes, population composition, field size, or source coverage. It can also exploit missingness artifacts. It does not say that the win mechanism changed or that the discriminator's important variables should become model features.

**EDA implication.** Validate the discriminator with whole races/dates, exclude outcome and market columns, report AUC with date-block uncertainty, and compare against univariate period effects. Then test whether drifted variables co-vary with OOT model loss. Do not importance-weight or adapt the production model during this EDA.

### 5.4 Grouped temporal validation and leakage

**Observation.** `GroupKFold` prevents a group from appearing in both train and validation, while temporal splitting prevents future-to-past training. Neither enforces both properties alone. Standard K-fold arguments for some stationary autoregressive settings do not justify random folds for recurrent entities and changing race composition.

**Interpretation.** A JRA splitter must be custom: calendar order is primary, every race stays intact, state features are recomputed within the fold, and same-day races cannot update one another. Grouping by horse across all time would answer a different cold-start question and may discard the intended deployment scenario in which a horse's past is legitimately known.

**EDA implication.** Use expanding-origin OOT folds and an explicit feature-availability audit. When measuring new-entity generalization, add a separately named cold-start diagnostic; do not substitute it for temporal deployment evaluation.

### 5.5 Feature effects, redundancy, and interaction

**Observation.** PD may evaluate correlated features outside their joint support. ALE reduces this extrapolation by accumulating local changes, but remains an explanation of the fitted response surface. SHAP allocates a fitted prediction among features under a background-distribution convention; it does not by itself identify causes. Unrestricted permutation can also create off-support combinations, and correlated substitutes can divide or mask apparent importance.

**Interpretation.** Horse-history aggregates are intentionally correlated across windows and source families. Therefore, a single importance ranking cannot distinguish information content from representation redundancy or harmful flexibility.

**EDA implication.** Use four complementary estimands: (1) race-stratified univariate association, (2) OOT fitted-model attribution, (3) grouped/permitted perturbation reliance, and (4) retrained feature-group ablation in a later registered experiment. Use Spearman, missingness, and deterministic-lineage clustering before interpreting individuals. For interactions, require a domain rationale and temporal replication; a 2D plot is a hypothesis generator, not evidence of incremental value.

### 5.6 Calibration, learning curves, slicing, and disagreement

**Observation.** Reliability diagrams depend on binning and support. Proper scores combine calibration, discrimination/resolution, and irreducible uncertainty. Calibration fitted to in-sample predictions is biased. Automated slice finding explicitly balances effect, interpretability, and support. Learning curves are meaningful only under a splitter that represents intended deployment.

**Interpretation.** Runner-level calibration bins are not independent and base win probability changes with field size. Binary/Ranker disagreement can locate races with different inductive biases, but it does not prove that blending improves OOT probability quality.

**EDA implication.** Produce OOT, race-coherent probabilities; show reliability by year and predeclared field-size bands with race/date-block intervals and bin counts. Construct learning curves by adding past years or complete date blocks. Summarize residuals per race and prioritize interpretable, supported slices. Compare model disagreement using paired race losses and winner ranks on identical folds; ensemble weight selection is out of scope.

### 5.7 Ranking and choice modeling

**Observation.** Conditional logit normalizes utilities across the alternatives available in one choice set. Standard logit entails restrictive substitution/IIA behavior. Plackett–Luce provides a sequential probability model over permutations. LightGBM ranking similarly requires explicit query groups, while label gains and truncation encode which ranks matter.

**Interpretation.** Winner Binary, top-heavy LambdaRank, top-3, full order, and performance targets retain different outcome information. Treating rows independently obscures the one-winner denominator; treating full finish order as equally reliable may over-weight noisy lower placings and non-finish statuses.

**EDA implication.** Use shallow conditional-logit or race-softmax diagnostics only to compare target information and within-race association. Record full-order and performance-target candidates in the hypothesis registry; do not change the production objective in Phase 5A.

## 6. Temporal replication

The literature does not supply replication evidence for this JRA dataset. The transferable replication design is therefore procedural:

1. Freeze transformations, bins, slice definitions, missingness taxonomy, and interaction pairs after 2014–2019 discovery.
2. Recompute effects independently for 2020 and 2021, reporting both years and their macro average rather than only a pooled result.
3. Require direction consistency and adequate race support; an average effect driven by one year remains exploratory.
4. Use 2022 once for confirmation/prioritization. Do not revise a mapping, quantile boundary, lag window, or interaction after viewing 2022.
5. For diagnostic models, fit preprocessing, imputers, encoders, feature selection, early stopping, and any calibration only on earlier periods assigned to those roles.
6. Report temporal stability of feature-family attribution/selection, not just stability over random seeds. The feature-selection literature treats sampling stability as its own property; here, sign/rank stability across years is at least as important.

No reviewed source establishes a universal threshold for “stable.” Phase 5A should report effect sizes, intervals, support, and number of years with the same direction, leaving acceptance thresholds to preregistered follow-up experiments.

## 7. Uncertainty

- **Review scope.** This was a targeted practices review, not a systematic review or meta-analysis. Search and publication availability can favor well-documented methods; source inclusion is qualitative and no pooled effect size is claimed.
- **Documentation drift.** Stable software documentation URLs can change with releases. The access date and the accompanying original-paper metadata are retained so that conclusions can be rechecked.
- **Sampling unit.** Primary uncertainty resamples races or contiguous date blocks, never independent runners. Recurrent horse and connection dependence may remain; sensitivity using longer date blocks should be reported for major findings.
- **Sparse slices.** Every slice needs runner count, race count, years covered, missing count, denominator, and aggregation unit. Suppress or mark unstable any slice with inadequate race/year coverage instead of relying on a narrow interval formula.
- **Multiplicity.** EDA comparisons are exploratory. Record the number and family of examined variables, bins, slices, and interactions. If formal p-values are reported, define the family and an error-control procedure; BH does not turn post-hoc EDA into confirmation, especially under dependent tests.
- **Model randomness.** Separate seed variability from temporal variability. Stable seeds with unstable years are not temporal replication.
- **Attribution uncertainty.** SHAP/PDP/ALE/permutation variability across folds and years should be shown. None has a generally valid causal confidence interpretation here.
- **Shift uncertainty.** Adversarial AUC needs date/race-grouped uncertainty. Classifier misspecification can hide a real shift; a strong classifier can detect a harmless one.
- **Calibration uncertainty.** Reliability bands should respect races/dates, and displayed bin counts must accompany curves. ECE alone is too dependent on binning and mixture composition.

## 8. Failure cases

1. **Random row split:** runners from one race or future entity state cross the boundary, inflating apparent performance.
2. **Pre-split preprocessing:** imputation, normalization, category pooling, feature selection, or target encoding sees replication/confirmation data.
3. **Same-date update:** an earlier-card result updates a later same-day prediction despite the fixed emission-before-update contract.
4. **Adversarial-AUC reification:** high AUC is called concept drift, or discriminator importance is directly added as a production feature.
5. **Identity through statistics:** hundreds of unsmoothed cumulative connection rates reproduce entity identity and historical target noise.
6. **Off-support explanations:** PD or unrestricted permutation combines a race context and historical state that do not co-occur.
7. **Correlated-feature erasure:** one of several history windows appears useless under permutation, although the family is jointly important; conversely, a fitted model heavily uses a feature group whose removal improves OOT performance.
8. **SHAP causality:** attribution sign is described as what would happen if the horse's condition were intervened on.
9. **Slice mining:** a rare venue × class × transition conjunction is selected because its post-hoc loss is extreme.
10. **Improper calibration reuse:** the same OOT year selects the model and fits/evaluates its calibrator.
11. **Row learning curve:** random runners are added, breaking complete race groups and historical state construction.
12. **Choice-set mutation:** scratches or ineligible runners are inconsistently included in the probability denominator.
13. **Lower-rank certainty:** DNF/DQ/demotion and noisy lower placings are treated as clean equal-interval targets.
14. **Confirmation redesign:** a 2022 anomaly causes a mapping, lag, bin, or slice threshold to be edited and rechecked.

## 9. Leakage / PIT considerations

The external practices do not override the repository's stricter data contract. For every target-aware analysis:

- require `target_date <= 2022-12-31` and refuse later retained rows;
- use current-race context known at the declared prediction timestamp, but isolate current result, clock, margin, last 3F, passing order, final odds, and popularity;
- make historical outcomes available only when `performance_date < target_date`;
- emit all races on a date before updating horse, opponent, jockey, or trainer state;
- calculate opponent strength from opponent pre-race state, never later results;
- fit all data-dependent transforms inside the discovery/training side of a split;
- keep IDs as keys and never numeric model inputs;
- calculate entity/target statistics from strict historical prefixes with smoothing and effective sample size;
- keep `market_oracle` physically separate and join only for a named diagnostic;
- never use market gaps to select a feature, target, calibration map, or acceptance decision;
- save only non-recoverable aggregates in Git; private runner rows and OOT predictions remain ignored local artifacts.

A useful audit is a “future append invariance” test: appending post-cutoff rows must not change any earlier pre-race view or aggregate. Another is a same-date permutation test: reordering races within a date must not change emitted pre-race state.

## 10. Modeling implications

These are method implications, not production decisions.

| Area | Phase 5A implication | Status |
|---|---|---|
| Binary vs LambdaRank | Compare on identical race/date OOT folds and paired race-level losses; disagreement is diagnostic | Retain both frozen references |
| Alternative targets | Compare retained information, missingness, stability, and probability mapping with shallow diagnostics | Redesign candidate, not implemented |
| Race-wise probability | Conditional logit/race softmax is the transparent diagnostic baseline for the one-winner constraint | Candidate for later experiment |
| Full ranking | Plackett–Luce/full-order targets may use more information but amplify lower-order/status assumptions | Defer pending target EDA |
| Connections | Prefer prior-only smoothed rates, deviations, uncertainty, and effective sample size over many fingerprint-like aggregates | Simplify/redesign candidate |
| History features | Interpret correlated windows as families; compare decay and condition specificity temporally | Retain, then simplify/redesign |
| Drift | Diagnose source/era composition and relation to loss before considering adaptation | Diagnostic only |
| Explainability | Use OOT SHAP/ALE/permutation to generate hypotheses; require retrained rolling ablation later | Diagnostic only |
| Calibration | Evaluate coherent race probabilities with proper scores and supported reliability slices | Retain separate calibration stage |
| Ensemble | Model disagreement can motivate but cannot validate blending | Defer |

The strongest methodological conclusion is that LightGBM family choice is not yet the main question. A valid comparison depends first on the same strict-PIT representation, choice-set denominator, temporal folds, and target semantics. No external practice justifies skipping those controls in favor of a more complex model.

## 11. Candidate hypotheses

These are proposals for the central hypothesis registry, not experiments to execute now.

### I-H01 — Race-conditional association changes the apparent univariate ranking

- **Question:** Do feature rankings based on global runner-level curves differ materially from race-stratified/race-softmax associations?
- **Evidence:** Choice-model theory requires normalization within the available alternatives; field size changes the base win rate.
- **Expected direction:** Some global associations weaken or reverse after race conditioning.
- **PIT / leakage risk:** Low if only strict pre-race features are used.
- **Validation plan:** Discover in 2014–2019, freeze bins/model form, replicate 2020 and 2021, confirm once in 2022; date-block intervals.
- **Proposed priority:** A; high expected knowledge, low implementation cost.

### I-H02 — Temporal stability should precede feature-importance magnitude

- **Question:** Are high fitted importances concentrated in one period or unstable among correlated source families?
- **Evidence:** Feature-selection stability, correlated permutation failure, and model-selection bias literature all show that a large point estimate is insufficient.
- **Expected direction:** Family-level rankings are more stable than individual correlated features; some large individual importances fail year replication.
- **PIT / leakage risk:** Low for OOT predictions; medium if background/permutation samples cross periods or races.
- **Validation plan:** OOT family-level SHAP/permutation by year plus Spearman/missingness clusters; no production acceptance without a later registered retrained ablation.
- **Proposed priority:** A; high expected knowledge, medium cost.

### I-H03 — Period separability is partly source/coverage drift, not necessarily harmful concept drift

- **Question:** Which feature families distinguish time regimes, and is separability associated with later race-level loss?
- **Evidence:** Dataset-shift work separates detection from harmfulness; missingness can reveal acquisition regimes.
- **Expected direction:** Missingness/context variables explain some adversarial AUC, while only a subset relates consistently to loss.
- **PIT / leakage risk:** Low if outcome/market columns are excluded and splits are date grouped.
- **Validation plan:** Two predeclared period classifiers, family ablations, date-block AUC interval, then a separately reported association with OOT race loss.
- **Proposed priority:** A; diagnostic value, medium cost.

### I-H04 — Connection aggregates contain redundant identity-like state

- **Question:** Does connection-family simplification preserve temporal signal while reducing instability and cold-start disparity?
- **Evidence:** Ordered target-statistic literature identifies leakage/prediction shift, and correlated families can distribute fitted importance misleadingly.
- **Expected direction:** Smoothed rate + effective sample size + uncertainty retains most replicated association with fewer unstable columns.
- **PIT / leakage risk:** Medium; prequential construction and same-date blocking are mandatory.
- **Validation plan:** EDA-only family clustering and year stability first; any simplification must be a later one-hypothesis rolling ablation.
- **Proposed priority:** B pending Workstream F evidence.

### I-H05 — Binary/Ranker disagreement identifies target-representation weakness, not automatic ensemble gain

- **Question:** Are disagreements concentrated by history availability, field competitiveness, or transition type, with temporally stable paired-loss patterns?
- **Evidence:** The objectives retain and weight outcome information differently; disagreement analysis is an error-localization tool.
- **Expected direction:** Stable slices exist where one family ranks the winner better, but blend superiority is not implied.
- **PIT / leakage risk:** Low using common OOT folds and no market-based selection.
- **Validation plan:** Race-level disagreement matrix and paired date-block intervals by discovery/replication/confirmation.
- **Proposed priority:** A; directly informs target reformulation.

## 12. What not to conclude

- A high adversarial-validation AUC does **not** prove concept drift, model degradation, or that a discriminator feature should enter production.
- A low adversarial AUC does **not** prove distributional identity; the classifier may be weak or the shift may be in the conditional target mechanism.
- Missingness associated with outcomes does **not** make a missing indicator safe; the cause and prediction-time availability must be established.
- SHAP, PD, ALE, mutual information, and univariate curves do **not** identify causal effects.
- ALE reduces PD's correlated-feature extrapolation problem; it does **not** remove confounding or make extrapolation/low-support concerns disappear.
- Permutation importance does **not** equal the improvement obtainable by adding a feature, and fitted importance does **not** equal the loss from retraining without it.
- Stable feature importance across random seeds does **not** establish temporal stability.
- A strong global runner-level correlation does **not** establish useful race-relative discrimination.
- A supported error slice does **not** establish that a condition-specific model will improve it.
- Good ranking does **not** imply coherent or calibrated race probabilities; good calibration does **not** imply strong ranking.
- Binary/Ranker disagreement does **not** establish that a 50:50 blend is better.
- Conditional logit or Plackett–Luce being structurally coherent does **not** establish that its IIA/order assumptions fit JRA outcomes.
- Full-order labels using more result columns do **not** necessarily contain more reliable learnable signal, especially for lower ranks and exceptional statuses.
- External tabular or horse-racing benchmarks do **not** establish JRA performance or permit a model-family decision.
- Multiple-comparison adjustment does **not** convert a post-hoc feature search into preregistered confirmation.
- None of this review authorizes new 2024/2025 access, production feature implementation, model replacement, calibration selection, or betting optimization.
