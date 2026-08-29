# Form, fatigue, suitability, and interaction effects

**Workstream:** E<br>
**Scope:** JRA central flat racing; research/specification only<br>
**Research date / source access date:** 2026-08-30 (JST)<br>
**Status:** Evidence synthesis for later MVP specification. No model or data-collection implementation is included.

## 1. Questions investigated

1. What evidence supports using layoff duration, congested schedules, cumulative racing load, and recent distance raced as predictors of the next performance?
2. What can and cannot be inferred from race-day body weight and body-weight change?
3. How should growth, maturity, aging, and survivor/retirement selection be represented?
4. What evidence supports individual suitability for distance, turf/dirt, going, weather, and racecourse?
5. Is the draw effect real, and under which course, distance, field-size, surface, and running-style conditions should it be modeled?
6. How should running style, likely pace, and field composition be represented using information available before the race?
7. Which common racing claims are evidence-backed, which are plausible but unverified, and which should not be encoded as rules?
8. Which features are realistically point-in-time (PIT) safe at a pre-race prediction timestamp?
9. How should sparse interactions and external-validity limits be handled in a JRA model?

## 2. Evidence interpretation rules

This report uses the following distinctions throughout.

- **Evidence:** what an official source, primary study, or systematic review directly measured or specified.
- **Finding:** a synthesis supported by that evidence.
- **Uncertainty:** what the evidence does not identify, including confounding and external-validity limitations.
- **Implication:** what the result means for this repository.
- **Recommendation:** a proposed, testable project decision. A recommendation is not an empirical fact.

Evidence strength is described as:

- **High for JRA prediction:** large JRA race dataset or current JRA/JRA-VAN official specification directly addressing the field.
- **Moderate:** Thoroughbred field data from another jurisdiction or a smaller JRA study with a directly relevant outcome.
- **Mechanistic only:** controlled exercise physiology, biomechanics, or injury evidence that makes a hypothesis plausible but does not estimate next-race winning probability.
- **Weak / folklore:** practitioner assertion without reproducible primary evidence for the claimed interaction.

The target of this project is **prediction**, not causal treatment selection. A feature can be predictively useful despite confounding. Conversely, a plausible physiological mechanism does not establish predictive value after the market, class, and horse ability are controlled.

## 3. Executive findings

| Topic | Best-supported conclusion | Evidence level for JRA next-race prediction | MVP disposition |
|---|---|---:|---|
| Layoff / days since last start | Observable and likely predictive, but it is not equivalent to detraining or rest; unseen training and injury/placement decisions are major confounders | Moderate | Include flexibly, never as a fixed “fresh/stale” rule |
| Congested starts / cumulative load | Racing workload is a biologically credible state signal; injury literature shows non-linear and sometimes conflicting relations because exercise also induces adaptation | Moderate-to-mechanistic | Include multiple recent windows and distance totals; do not collapse to one fatigue score |
| Recent distance raced | Race distance contributes to modeled bone fatigue and acute energetic demand, but race-only distance omits most training load | Mechanistic | Include rolling distance and distance × interval, with explicit proxy limitations |
| Body-weight change | Large changes were associated with slower times in a large Korean dataset; JRA weight also changes systematically with age, sex, and season | Moderate, direct JRA context for baseline but not performance effect | Use within-horse normalized deviation; test late-information model separately |
| Age | JRA average speed rises through early age four, while older-starter averages are distorted by retirement and weight assignment | High for population pattern, not an individual causal curve | Include exact age and nonlinear age/career interactions |
| Distance suitability | Strong biological and genetic evidence that individual distance aptitude exists | Moderate-to-high for existence; historical race performance remains confounded by placement | Use shrunk, PIT historical conditional performance; do not require DNA |
| Turf/dirt suitability | JRA surfaces have different speed and mechanical properties; individual surface preference is plausible, but raw surface win rates mix ability and race selection | Moderate | Use surface-specific history with shrinkage and minimum support |
| Going | JRA track condition materially changes race time, differently on turf and dirt | High for race-wide effect; limited for individual affinity | Include current going and prior performance × going; prioritize race-wide context before horse affinity |
| Weather / heat | Heat and humidity affect thermoregulation; JRA heat-illness prevalence varies with summer climate | Mechanistic-to-moderate | Include actual pre-race weather when timestamped; keep horse-specific weather affinity experimental |
| Course | JRA course geometry varies materially; bends, slope, straight length, and surface influence optimal speed | High for context, limited for individual “course specialist” claims | Include structural course features; add shrunk course-history only after baseline |
| Draw | JRA assigns draw by computerized randomization; effects should be estimated conditionally, especially by field size and course layout | High identification potential; JRA effect sizes still need estimation | Include normalized draw and a small predeclared set of context interactions |
| Running style | Real-race tracking establishes that pace, drafting, pack position, and distance-dependent tactics matter | Moderate | Derive continuous past-position style with recency and uncertainty |
| Multiple front-runners / pace collapse | Plausible and widely asserted, but no adequate JRA primary study directly validating this simple rule was found | Weak for the specific rule | Treat as an explicit experiment, not a fact or hard-coded adjustment |

## 4. Form, rest, fatigue, and load

### 4.1 Layoff duration is observable; actual training status is not

**Evidence.** JRA defines race interval as the time between starts (`連闘` for the following week, `中1週` for the week after that) [S1]. The public race record therefore gives exact days since the last start. It does not, by itself, reveal whether a horse was turned out, confined, cantered, galloped, rehabilitated, or intensively prepared.

Controlled detraining studies show that physiological response depends on activity during the nominal layoff:

- Mukai et al. randomized 27 trained Thoroughbreds after 18 weeks of treadmill training to cantering, walking, or stall rest for 12 weeks. Mass-specific maximal oxygen uptake and cardiovascular measures fell across groups, but the cantering group better maintained several performance-related variables than walking or stall-rest groups [S2].
- In six young Thoroughbreds, 10 weeks with pasture access after training maintained whole-body aerobic capacity despite changes in mass-specific values, demonstrating that “not in formal training” need not mean physiological inactivity [S3].
- Equine muscle adaptations can reverse during detraining, but magnitude and timing depend on breed, training type, and detraining protocol [S4].

**Finding.** `days_since_last_start` is a valid observable predictor but an invalid literal measure of loss of fitness. Two horses with the same interval can have very different workloads, health histories, and readiness.

**Uncertainty.** Public racing data incompletely observe the trainer’s reason for the break. Long layoffs are often caused by injury or poor form, while trainers also deliberately give high-class horses longer campaigns. These opposing pathways can make the marginal relationship non-monotone or change by class and age.

**Implication.** A universal rule such as “long rest is bad” or “fresh horses run better” would encode selection effects as physiology. The feature is nonetheless likely useful predictively because trainer placement, prior problems, and campaign design are themselves informative, even if not causal.

**Recommendation.** Preserve:

- exact days, `log1p(days)`, and broad predeclared bins;
- debut / first-record / foreign-last-start / missing-history indicators;
- horse’s prior performance after comparable intervals, only with strong shrinkage;
- interactions with age, class, prior injury-like absence proxy, and observable workout recency if available.

Do not use a single “freshness optimum” learned from all races, and do not interpret a SHAP dependence curve for interval as a causal rest prescription.

### 4.2 Racing frequency and “optimal” interval evidence

**Evidence.** Morrice-West et al. surveyed 66 Victorian Thoroughbred trainers about intended training workload and rest practices, then associated trainer-level programs with official trainer success. Reported mean racing frequency was about 2.3 weeks. Their multivariable results suggested success increased up to roughly 2.5–3 weeks between starts and declined beyond it [S5]. However, the exposure was a trainer’s typical intended program, not verified workload for each horse; the outcome was trainer-level win/place/prizemoney rate; and the study was Australian, not JRA.

**Finding.** The study supports testing a nonlinear race-interval feature. It does **not** establish a causal three-week optimum for a JRA horse.

**Uncertainty.** Trainer quality, stable population, race programming, travel, medication rules, climate, surface, and campaign structure differ across jurisdictions. Ecological association at trainer level can differ from the horse-start-level association needed here.

**Recommendation.** Use it only to justify nonlinear representation and predeclared interval interactions. Let JRA out-of-time evidence determine usefulness; never center a hand-designed score at 21 days merely because this study found a trainer-level optimum.

### 4.3 Congested starts and cumulative racing load

**Evidence.** Three relevant evidence streams must be separated:

1. **Acute fatigue within a race.** High-speed video from Japanese Derby races showed fatigue-related changes in stride frequency and step structure, with both stride frequency and stride length contributing to late-race speed reduction. The analyzed biomechanical sample was 23 horses / 71 strides, so it demonstrates mechanism, not a next-start effect [S6].
2. **Bone loading per start.** A model using GPS/accelerometer-derived stride data from 25,234 Tasmanian starts estimated more subchondral-bone fatigue-life use with longer distance, firmer turf, older age, and other horse/race factors, with large between-horse variation. Joint load itself was not measured; speed was used as a proxy and fatigue equations were scaled over assumed loads [S7].
3. **Injury epidemiology.** A systematic review/meta-analysis found total career high-speed exercise distance and average high-speed distance per day associated with musculoskeletal injury, but other workload metrics were inconsistent. The authors described an apparent protective adaptation at lower exposure before potential harm and substantial methodological heterogeneity [S8]. A Kentucky matched case-control study even found **less** recent high-speed exercise among injured horses and explicitly noted disagreement with California findings [S9].

**Finding.** Racing load is not a monotone “more starts = more fatigue = worse result” process. Work produces acute stress and cumulative damage, but also fitness and bone adaptation. Injury risk and next-race performance are different endpoints.

**Uncertainty.** JRA race histories omit most daily training load. A horse with few races may have accumulated extensive fast work; a horse may have rested because early symptoms were detected. Healthy-worker/survivor bias is severe: only horses deemed fit enough start.

**Implication.** Counts and distances from official races are low-cost proxies for state and campaign intensity, not validated measures of total physiological load.

**Recommendation.** Keep multiple separable features instead of a single fatigue index:

- starts in the previous 14, 30, 60, 90, 180, and 365 days;
- total race distance and distance weighted by recency over 30/60/90/180 days;
- days since last start, shortest interval in the last 90 days, and number of short-interval transitions;
- last-race distance, last-two-race distance, and previous distance × days-since-start;
- exponentially decayed race count and distance;
- campaign length since a long break, and starts in current campaign;
- missing workout / no workout coverage flags, if workout data are joined.

Counts, distance, and interval should remain a named feature group so an ablation can answer one interpretable question: **does race-history load add out-of-time prediction after ability, class, age, and recent form?**

### 4.4 Recent distance raced

**Evidence.** Longer race distance increased estimated per-start bone fatigue in the stride-based model [S7]. Pace and energy-demand studies also show that the appropriate velocity profile changes with distance [S10, S11]. These are mechanisms, not proof that a horse that ran farther last time will underperform next time.

**Finding.** Recent distance belongs in load history, but raw metres are an incomplete exposure. Pace, going, surface, carried weight, actual speed, and training between starts alter demand.

**Recommendation.** Test both raw and context-weighted variants, but keep the MVP interpretable. A reasonable first experiment is rolling race-distance totals plus last-race distance and interval interactions. A later hypothesis may weight distance by speed relative to race/course, going, and carried weight. Do not infer load from finishing position alone.

## 5. Body weight and body-weight change

### 5.1 What is empirically established

**Evidence.** JRA-specific population data are strong for the baseline process:

- Takahashi and Takahashi analyzed 632,540 JRA flat-race body-weight measurements from 2002–2014. Body weight changed with age and season, differed by sex, and increased by about 30 kg across the athletic career; males/geldings were lightest in summer, females in spring [S12].
- A separate Japanese Thoroughbred GWAS reported mean body weights increasing from age two to four, confirming growth and substantial stable between-horse variation [S13].
- JRA warns that body weight changes with training, transport, feeding, and defecation, and that a training-centre measurement can differ from race-day weight [S14].

The most directly relevant performance study located was Korean. Cho et al. analyzed 155,656 KRA records from 8,197 Thoroughbreds. Race time was slower when change from the prior start exceeded about ±10 kg; ±20 kg was about 0.3 s slower than ±5 kg, and performance decreased at changes greater than about ±2.5% of body weight [S15]. This is an observational model in a different racing system and should not be converted into a JRA threshold rule.

**Finding.** A fixed absolute change such as “minus 10 kg is bad” is misspecified. The same change has different meaning by baseline mass, age, sex, season, interval, transport, and measurement circumstance. Direction alone is insufficient; both unusually large loss and gain may carry information.

**Uncertainty.** Weight is not body composition, hydration, gut fill, muscle, or fitness. The Korean association may include illness, intentional conditioning, seasonal change, class/placement, travel, and elapsed-time confounding. No equally large JRA study isolating the next-race performance effect of within-horse weight deviation was found in this search.

### 5.2 Point-in-time constraint

**Evidence.** JRA states that ordinary race-day body weight is measured around 80 minutes before post time on a standard schedule [S14]. JRA-VAN Data Lab describes速報馬体重 delivery at roughly 60 minutes before post [S16]. The 2026 hot-weather program changes the assembly and announcement schedule for some meetings, so a fixed clock offset cannot safely be assumed [S17]. JRA also publishes training-after weight for all G1 races and two New Year graded races, normally Thursday at 17:00, but explicitly says it is not race-day weight [S18].

**Finding.** Race-day weight is unavailable to an earlier forecast and cannot be silently backfilled. The measurement time and data-delivery time are distinct.

**Recommendation.** Maintain at least two prediction products/feature sets:

1. **Early model:** no current race-day weight; may use prior weights and, only where consistently available, the explicitly timestamped training-after weight.
2. **Late model:** current race-day weight, activated only after the actual feed arrival timestamp is logged.

For the late model, test:

- current weight;
- absolute and percentage change from last comparable measurement;
- deviation from a recency-weighted within-horse baseline;
- robust z-score relative to the horse’s own history;
- deviation from age × sex × calendar-month population trend;
- interval since prior measurement and prior-measurement context;
- asymmetric loss/gain splines or bins;
- missing/first-start/foreign-prior-weight indicators.

Do not hard-code ±2.5% as a decision boundary. Test it, if desired, as one diagnostic bin fixed before evaluation.

## 6. Age, development, and decline

**Evidence.** Takahashi analyzed JRA flat races from 2002–2010 on selected common distances and firm/standard going. Average speed increased through the first half of age four. After that, population average speed stayed roughly constant. The paper explicitly attributes later stability partly to retirement of weaker/declining horses and reduced carried weight; an earlier longitudinal US analysis had found decline after a peak around age 4.25–4.75 [S19]. JRA body-weight analysis found growth continuing toward age five [S12].

**Finding.** The population curve is not an individual aging curve. JRA has informative attrition: slower horses leave, so conditioning only on horses still starting can mask decline. Age is also entangled with race eligibility, class, carried-weight rules, experience, and cohort quality.

**Uncertainty.** “Peak age” varies by sex, distance aptitude, training history, injury, and selection into racing. Calendar-year JRA age is not exact biological age unless birth date is available.

**Recommendation.** Include:

- exact age in days when birth date is available, otherwise clearly named calendar age;
- nonlinear age (bins or tree-learned effects), not a monotone penalty;
- career starts, days since debut, and age at debut;
- age × sex, age × distance band, age × surface, and age × career-stage candidates;
- cohort/year controls or rolling normalization to avoid confusing program changes with aging.

Report performance for two-year-olds, three-year-olds, age four, age five-plus, and sparse senior ages separately. Do not claim causal growth/decline from model feature importance.

## 7. Suitability: distance, surface, going, weather, and course

### 7.1 Distance suitability

**Evidence.** A genome-wide association study in elite Thoroughbred winners found the MSTN region to be the strongest predictor of optimum distance. Mean optimum distance differed substantially across C:C, C:T, and T:T genotypes (approximately 6.2, 9.1, and 10.5 furlongs in that study), with known muscle-mass differences [S20]. This establishes that biologically meaningful individual distance aptitude exists. It does not imply that genotype is necessary or sufficient for race prediction, and the elite-winner phenotype creates selection limitations.

Tracking and optimal-control work also finds distance-dependent pace and energetic strategies [S10, S11]. JRA race-speed analysis shows speed changes with distance and with the number/tightness of turns [S19].

**Finding.** Distance is not merely a linear race-level covariate. A horse × distance relationship is justified, but naïve win rate at a distance confounds aptitude with class, opposition, track, pace, and trainer selection.

**Recommendation.** Estimate aptitude from PIT performance values that have already adjusted, as far as practical, for field strength and race context. Candidate representations:

- recency-weighted performance by continuous distance using kernels or broad overlapping bands;
- difference between today’s distance and distances of the horse’s best prior adjusted performances;
- within-horse slope/curve of adjusted performance over distance, with strong population shrinkage;
- surface-specific distance history;
- pedigree-derived population priors only if they can be produced PIT-correctly without direct sire/dam ID memorization.

Keep broad bands in the MVP; defer free-form per-horse curves until minimum support and shrinkage are validated.

### 7.2 Turf versus dirt

**Evidence.** In 183,465 starts across all ten JRA courses (2000–2004), track condition affected time differently on turf and dirt; dirt-course racecourse effects exceeded the going effect in that analysis [S21]. JRA’s dirt construction and racing-speed patterns also differ from US dirt, limiting transfer from North American studies [S19]. Biomechanical studies report surface-dependent hoof forces and propulsion, but they do not directly estimate individual win-probability transfer [S19, S21].

**Finding.** Surface-specific ability is plausible and likely useful. Raw “turf wins” and “dirt wins” are very noisy for lightly raced horses and reflect trainer placement.

**Recommendation.** Use current surface as a core context feature and create shrunk historical summaries for each surface:

- starts/support;
- adjusted performance mean, best, variance, and recency-weighted mean;
- surface-switch flag and time since last start on the current surface;
- difference between current-surface and other-surface shrunk estimates;
- explicit missing/unseen-surface indicator.

Do not force unseen-surface aptitude to zero; back off to age, pedigree attributes if legally/technically available, and population priors.

### 7.3 Going and track measurements

**Evidence.** Maeda et al.’s JRA analysis found, after controlling course and distance, that turf times ordered from faster to slower as firm/good/yielding/soft, whereas dirt ordering differed and muddy dirt could be faster than nominally “fast” dirt [S21]. Thus “wetter always means slower” is false across surfaces.

JRA explains that the four-level track-state designation is a holistic judgment and is not mechanically determined by moisture percentage. It publishes moisture measurements from the day before and race-day morning and cushion values for turf, with planned publication around 09:30 on race day; historical public archives begin in July 2018 for moisture and September 2020 for cushion value [S22, S23]. The archive page is ordinarily updated after the meeting, so an archive copy alone does not prove the exact earlier publication state [S23].

**Finding.** Race-wide going effects are well supported. Horse-specific going affinity is less certain and will be sparse, especially after splitting by surface and distance. Current measurements can improve context but have shorter historical coverage and may change during the meeting.

**Recommendation.** MVP order:

1. current categorical going, surface, course, race number, and meeting-day context;
2. race-day objective moisture/cushion only in a later-coverage experiment with actual announcement timestamps;
3. shrunk horse-specific prior adjusted performance by broad going group;
4. horse-going interactions only when support counts are exposed and missingness is modeled.

Preserve each official weather/going change event and its `発表月日時分`; JRA-VAN’s specification provides initial state and subsequent change records [S24]. Do not use the final recorded going for a prediction made before a later change.

### 7.4 Weather and heat

**Evidence.** In a JRA study of 975,247 starters from 1999–2018, post-race exertional heat illness occurred in 387 cases (0.04% overall), was more prevalent in summer (0.086%), and varied with climate and racecourse [S25]. Controlled Thoroughbred experiments show hotter/humid conditions raise core temperature faster, while heat acclimation improves thermoregulatory and exercise responses [S26, S27].

**Finding.** Temperature and humidity are biologically relevant. Heat-illness incidence is an adverse-health endpoint, not proof of a material average win-probability effect. Acclimation and stable/training location are mostly unobserved.

**Recommendation.** If an auditable race-site observation is obtainable at the prediction time, include temperature, humidity or a defensible composite such as WBGT, precipitation, and wind as race-level features. Store source, station/site, observation versus forecast status, and retrieval timestamp. Start with global weather effects and weather × distance/surface interactions. Defer per-horse “hot-weather specialist” aggregates until sufficient repeats exist.

### 7.5 Racecourse and course layout

**Evidence.** JRA publishes official layouts: direction, distances, straight length, elevation difference, course variants, and widths. For example, Tokyo turf has a 525.9 m straight and 2.7 m elevation difference, while Nakayama has much tighter geometry and 4.5 m elevation difference [S28, S29]. JRA speed studies found turn count and smaller turning radius associated with slower speed [S19]. A mechanics/energetics model calibrated to France Galop tracking data likewise shows slopes and bends alter the entire optimal velocity profile, not only the local segment [S11].

**Finding.** Course is a real structured context. “Horse for course” can reflect actual geometry, but a horse’s raw course win rate also reflects race availability, home region, class, and small samples.

**Recommendation.** Represent both:

- course identity/current exact course variant; and
- structural attributes such as direction, turn count, turn radius proxy, straight length, elevation profile, first-turn distance, surface, rail setting, and start-on-turf segment where available.

Add shrunk horse-course history after the structural baseline. This allows some generalization to unseen courses and distinguishes layout compatibility from course-ID memorization.

## 8. Draw effects

### 8.1 Evidence and identification

**Evidence.** JRA states that, except for the public Arima Kinen draw, horse number/draw is decided automatically by computer; the gate order is the horse number from the inside outward [S30]. This makes draw unusually close to an exogenous pre-race assignment among declared runners. An older Canadian econometric study found post position added information beyond odds rank and that the effect grew with field size [S31]; it does not estimate modern JRA effects.

**Finding.** The physical effect cannot be summarized by raw win percentage by gate. Raw post frequencies must be conditioned on field size, because high posts cannot exist in small fields. Geometry, distance to first turn, direction, surface, rail placement, going, scratches, and running style can all moderate the cost or benefit.

**Causal caution.** Random draw strengthens causal interpretation, but the observed effect includes jockey adaptation and subsequent race dynamics. Post-draw scratches change relative spacing; rare exceptional procedures and data corrections must be handled. Odds may mediate market response but are intentionally absent from the initial prediction model.

### 8.2 Recommended representation

Include:

- absolute horse number and frame number;
- normalized draw `(horse_number - 1) / (field_size - 1)` with a safe value for one-runner anomalies;
- distance from inside/outside and coarse inner/middle/outer thirds;
- field size;
- distance to first turn / number of turns / course direction where available;
- predeclared interactions: normalized draw × field size, draw × course-distance-surface, draw × estimated early style, and draw × going.

Estimate effects out of time and report uncertainty by course-distance-surface cells. Because JRA draw is randomized, a separate descriptive causal audit using only pre-draw covariates is valuable: test covariate balance by draw and confirm the data pipeline has not reordered horses after scratches.

## 9. Running style, likely pace, and field composition

### 9.1 What the evidence supports

**Evidence.** Spence et al. tracked 44,803 horse-starts in 3,357 UK races once per second. Drafting and distance-dependent pacing were associated with performance. The authors explicitly cautioned that the relation could reflect horse capability/personality and pack constraints rather than a purely causal tactic [S10]. A France Galop optimal-control study based on roughly ten races showed that starts that are too fast relative to capacity can damage final performance; slope, bends, and distance alter optimal regulation [S11]. The Japanese Derby stride study provides direct JRA evidence of late-race biomechanical fatigue but only in a small, homogeneous G1 sample [S6].

**Finding.** Race outcome is interactive: position affects drafting, obstruction, distance traveled, and energy use. Historical running position can therefore be useful. But “style” is not a fixed horse trait; it is jointly chosen by horse, jockey, draw, pace, distance, tactics, and competition.

### 9.2 Constructing PIT running style

JRA-VAN provides past corner passing positions; straight races have no corner values, and different courses have different numbers of recorded corners [S32]. The current race’s actual pace and passing order are unknown before the race.

Recommended historical style features:

- normalize each recorded corner position by field size, not raw rank;
- keep first available corner, last corner, and change in relative position separate;
- recency-weight across multiple starts, while exposing support count and variance;
- condition summaries by sprint/mile/middle/long distance, surface, and one-turn/two-turn structure only when support permits;
- derive continuous `early_position_tendency` and `position_variability` before coarse labels such as front/prominent/mid/held-up;
- mark straight-course or unrecorded-corner histories explicitly rather than imputing a back-of-field position;
- exclude current-race result laps and any post-race narrative from a pre-race feature row.

### 9.3 Likely pace and the “multiple front-runners” claim

**Evidence.** Tracking and physiology establish that pace, drafting, and excessive early effort can affect performance [S10, S11]. Practitioner sources commonly assert that several front-runners create a speed duel and favor closers. This search did **not** locate a sufficiently controlled primary JRA study estimating the effect of the number of expected front-runners on each style’s win probability.

**Finding.** “Many leaders imply pace collapse” is a plausible interaction hypothesis, not an established universal law. Jockeys respond strategically; an apparent leader may miss the break or concede; field ability and course layout matter; labels derived from finish-time data can leak the outcome.

**Recommendation.** Test a minimal pace-map experiment based only on prior races:

- per-runner probability/tendency of taking an early position;
- field mean, maximum, variance, and entropy of early-position tendency;
- expected number above a predeclared high-early-speed threshold;
- gap between the top two early tendencies (a “lone leader” proxy);
- for horse `i`, own early tendency minus field mean, number of stronger early rivals, and own style × expected field pace;
- uncertainty/support summaries for all of the above.

Prefer continuous aggregates to a brittle count of manually labeled front-runners. Compare the feature group with a frozen no-interaction baseline. Report both improvement and degradation across log loss, Brier, ranking, calibration, and relevant subgroups.

### 9.4 Field composition beyond pace

The field is not a set of independent rows. Useful PIT composition features include:

- distribution of pre-race ability/rating and distance/surface suitability;
- number and quality of likely early-position rivals;
- relative draw among likely early runners;
- field size and concentration of ability near the focal horse;
- count/support of first-time surface/distance runners;
- heterogeneity of age, experience, and recent load.

These should be computed from PIT-safe per-horse estimates and then joined back to each runner. Do not use actual finish order, final sectional pace, or opponents’ future results.

## 10. Empirical effects versus folklore

| Claim | Classification | Reason |
|---|---|---|
| “A layoff of N weeks always improves/worsens form” | Folklore / unsupported as a universal rule | Interval does not identify actual exercise, injury, intent, or readiness |
| “Three weeks is the optimal interval” | Limited external association, not a rule | Australian trainer-level survey found a nonlinear optimum; wrong unit and jurisdiction for direct transfer |
| “More recent racing is always harmful” | Contradicted as a universal rule | Exercise creates both adaptation and damage; injury studies are nonlinear and regionally inconsistent |
| “A long race last time causes a poor next run” | Mechanistically plausible, unverified predictive rule | Per-start load rises with distance, but no direct JRA next-start causal estimate found |
| “Minus 10 kg is bad; plus 10 kg is good” | Unsupported directional rule | JRA weight varies by age/sex/season; Korean study found large change in either direction associated with slower time |
| “Older horses decline after age four” | Partly supported at individual level, biased in population averages | JRA starter averages are shaped by retirement and carried-weight selection |
| “Every horse has a best distance” | Substantively supported, exact estimate uncertain | MSTN and racing evidence support aptitude, but observed best distance is trainer-selected |
| “Wet turf is slower” | Supported as a race-wide JRA pattern | Large JRA race-time study |
| “Wet dirt is slower” | False as a universal JRA rule | JRA study found a different ordering on dirt, including fast times on muddy going |
| “Course specialists exist” | Plausible, raw course record is not proof | Geometry matters; repeat course form is sparse and selection-confounded |
| “Inside draw is always best” | Unsupported universal rule | Effect depends on field size, first-turn distance, surface, layout, rail, and style |
| “Several front-runners guarantee a pace collapse” | Plausible folklore, not adequately established for JRA | Mechanism exists, but the simple count rule lacks direct controlled JRA evidence |
| “A closer benefits from any fast expected pace” | Hypothesis | Traffic, drafting, ability, start, course, and jockey response can reverse it |

## 11. Causal versus predictive interpretation

| Feature | Major confounding / selection path | Permissible predictive statement | Causal statement that must not be made |
|---|---|---|---|
| Days since start | injury, deliberate campaign, class placement, unseen workouts | “Interval adds out-of-time predictive information” | “Changing this horse’s rest to X days will improve performance” |
| Starts / distance in window | fitness, stable strategy, soundness, survivor bias | “Recent race-load history improves prediction” | “One additional start causes lower win probability” |
| Body-weight change | season, age, transport, hydration, feeding, conditioning, illness | “Unusual within-horse change is associated with outcome” | “Gaining/losing Y kg causes the result” |
| Age | eligibility, weight scale, experience, attrition | “Age/career stage improves conditional prediction” | “Observed starter curve is biological decline” |
| Past surface/distance/course record | trainer selection, class, opposition | “Conditional history is predictive with shrinkage” | “Switching conditions would cause the estimated difference” |
| Draw | approximately randomized in JRA; scratches and tactics follow assignment | “Conditional draw effect is estimated from randomized assignment” | “Every part of a draw-model association is purely physical” |
| Running style | ability, jockey, draw, expected pace, realized break | “Prior-position tendency and field context predict outcome” | “Forcing this tactic would create the same model effect” |

Feature importance, split gain, and SHAP values are associational diagnostics. They do not resolve these causal paths.

## 12. Point-in-time and availability matrix

| Information | Earliest safe use | PIT risk | Recommendation |
|---|---|---|---|
| Prior start date/distance/result/passing positions | After prior result is finalized and ingested | corrected results; opponents’ later performance | Version records and use only knowledge available before target start |
| Rolling start count / race distance | Derived from prior finalized races | current race accidentally included; non-JRA/foreign history gaps | Assert event time `< prediction_time`; flag incomplete history |
| Age/sex/birth data | Static or registration update | retrospective corrections; calendar age vs exact age | Store source version and semantics |
| Draw / horse number / field size | After numbered racecard publication | Thursday/Friday timing differs by race; scratches change active field | Snapshot numbered declarations and subsequent changes [S33] |
| Current categorical weather/going | Official initial announcement and each change | final state backfilled into earlier prediction | Store all announcement timestamps, not only final record [S24] |
| Moisture / cushion | When JRA publishes the measurement | short history; measurement time differs from publish time; archive updated later | Store measurement and observed publication/retrieval times [S22, S23] |
| Race-day body weight | After速報馬体重 is actually delivered | unavailable to early model; variable summer schedule | Separate early and late models; log arrival [S14, S16, S17] |
| Training-after body weight | Thursday 17:00 for limited named races | restricted coverage; differs from race-day weight | Separate field and coverage flag [S18] |
| Workout times | When officially delivered | coverage differs by track/type/date; post-workout updates | Preserve workout and delivery timestamps; do not equate absence with no work [S34] |
| Current realized pace / current passing order / final laps | After/during race | direct target leakage | Never use in a pre-race row |
| Expected pace | After field/draw is known, derived only from earlier starts | style features accidentally derived from target result | Freeze history cutoff and expose support/uncertainty |
| Race-site forecast | At actual forecast retrieval time | later revised forecast or observed weather backfill | Store forecast vintage; observations only for a late model if available pre-race |

## 13. Sparse interactions and sample-size discipline

### 13.1 Unit of evidence

A JRA race contains mutually dependent outcomes: exactly one winner (excluding dead-heat complications). Therefore, 18 horse-rows in one race are not 18 independent races for assessing an interaction. Support must be reported as both **races** and **horse-starts**, with uncertainty clustered or bootstrapped by race and, where relevant, by meeting/day.

Takahashi’s JRA body-weight paper describes roughly 3,400 flat races and 48,000 starts per year in its study era [S12]. Even a decade of data becomes sparse after splitting by 10 courses, surface, exact distance, rail/route, going, field-size band, draw band, style, age, and field composition.

### 13.2 Illustrative precision, not a universal minimum

For an independent Bernoulli rate near 10%, a simple normal approximation needs about 864 observations for a 95% half-width of ±2 percentage points and about 3,457 for ±1 point. Race dependence, temporal drift, repeated horses, multiple testing, and class imbalance make those figures optimistic. They are illustrations of why small interaction cells cannot support folklore-style claims, not acceptance thresholds.

### 13.3 Recommended safeguards

- Predeclare a short list of interactions with a physical or tactical rationale.
- Expose support counts and calendar coverage for every conditional diagnostic.
- Pool toward broader course-layout, distance-band, surface, and going parents rather than fitting raw cell averages.
- Use regularization/tree complexity controls and compare with an additive/no-interaction baseline.
- Require consistency across rolling time folds, not significance in one pooled historical period.
- Use race-level bootstrap intervals for metric differences and conditional effects.
- Treat exploratory interaction mining as hypothesis generation; retest in a later untouched development period.
- Keep the final holdout untouched; do not select pace thresholds or condition bins from it.
- For rare configurations, prefer continuous features and partial pooling to “no evidence means zero effect.”

## 14. External validity to JRA

| Evidence source | Transferability | Main limitation |
|---|---:|---|
| Large JRA race-time, age, body-weight, track-condition studies | High | Older eras; observational selection; not always next-race win probability |
| Current JRA/JRA-VAN specifications | High for availability/PIT | Service behavior and schedules can change; actual timestamps still need capture |
| Korean body-weight performance study | Moderate | KRA environment, race program, surfaces, and model differ; no causal identification |
| Australian workload/trainer-success study | Low-to-moderate | trainer-level intended workload, different campaign and training system |
| Australian/US/GB injury epidemiology | Mechanistic-to-moderate | injury rather than performance endpoint; training exposure and surfaces differ |
| UK real-race tracking and drafting | Moderate for race mechanics | different courses, data era, tactics, and surface mix |
| France Galop optimal-control model | Mechanistic | about ten races, selected low-interaction horses, synthetic standard surface |
| Treadmill detraining/heat studies | Mechanistic | small samples and controlled protocols do not reproduce a JRA race or stable program |
| Practitioner pace maps and betting maxims | Weak | selection, odds, unpublished methods, and no controlled JRA validation |

No foreign threshold should be copied into the JRA model without a JRA time-split replication.

## 15. Implications for this project

1. **Retain the agreed multi-window history design.** The evidence argues against keeping only the last few starts and against one handcrafted fatigue number.
2. **Separate availability-time models.** Race-day weight and possibly updated going/weather belong to a late model; an early model must not backfill them.
3. **Use adjusted performance for suitability.** Raw condition win rates are dominated by opportunity, class, and opponent quality. Distance/surface/going/course history should aggregate a common PIT performance value or rating.
4. **Treat state and aptitude differently.** Recent load and weight deviation are transient state candidates; distance/surface/course tendencies are slower aptitude candidates. Mixing them into one “form” score reduces interpretability.
5. **Add field-aware features without jumping to attention/DNNs.** Field aggregates and own-minus-field features can test the interaction hypothesis in LightGBM.
6. **Exploit the randomized JRA draw for a strong audit.** Draw effects can be measured more credibly than most handicapping features, but must remain conditional and out-of-time.
7. **Keep odds out of these primary prediction features.** Odds can later test whether a finding is already priced by the market, but should not enter the initial no-odds prediction model.
8. **Do not conflate prediction with welfare/causality.** A load feature associated with poor results is not a training recommendation; injury-risk evidence does not directly estimate win probability.

## 16. Recommendations and proposed research experiments

These are specifications for later implementation, not implementation performed in this workstream.

### 16.1 Recommended MVP feature groups

**Core, high priority**

- exact race interval with nonlinear transforms and missing/debut states;
- rolling starts and race-distance totals over several windows;
- exact/nonlinear age, career starts, days since debut, sex;
- current course, exact distance, surface, going, field size, draw and normalized draw;
- PIT historical adjusted performance by broad distance band and surface, with count/shrinkage;
- simple course-layout attributes;
- past relative corner-position tendency and uncertainty;
- field summaries of ability and early-position tendency.

**Late-timestamp feature group**

- current race-day body weight and within-horse normalized deviations;
- current announced weather/going changes and objective track measurements when captured in real time.

**Experimental, not baseline assumptions**

- individual going/weather/course affinity;
- workload weighted by inferred race intensity;
- “lone leader” and multiple-front-runner pace interactions;
- complex draw × style × going × field-size interactions;
- genotype-like pedigree distance priors.

### 16.2 One-hypothesis experiment sequence

1. **E-FORM-01 — race-history load:** add interval, rolling starts, and rolling distance to the frozen ability/context baseline.
2. **E-BW-01 — late body weight:** compare the frozen early model with a late model using only the body-weight group at the actual availability time.
3. **E-SUIT-01 — condition history:** add shrunk distance/surface adjusted-performance summaries.
4. **E-DRAW-01 — contextual draw:** add normalized draw and the predeclared field-size/course-layout interactions.
5. **E-STYLE-01 — own running tendency:** add PIT normalized corner-position histories.
6. **E-PACE-01 — field composition:** add expected early-pace distribution and own-style × field-pace terms, without changing other features.
7. **E-GOING-01 — individual affinity:** only after race-wide going context is stable, add shrunk horse-going history.

Each experiment should report overall ranking/probability/calibration metrics plus condition-specific changes. For these feature groups, particularly inspect:

- newcomers and lightly raced horses;
- age and class bands;
- short versus long intervals;
- turf/dirt and distance bands;
- course-distance-surface cells;
- field-size and normalized-draw bands;
- expected pace/support bands;
- early versus late timestamp models.

Do not accept a feature merely because ROI improves under one EV threshold. It should show plausible, stable predictive or calibration value in development periods, and any degradation must be recorded.

## 17. Uncertainties and follow-up research needs

1. **JRA next-start load effect remains unestimated.** The strongest workload papers address physiology, injury, or trainer success, not next-start probability after ability and class adjustment.
2. **Training exposure coverage is decisive.** JRA-VAN offers official slope and woodchip times, but this workstream did not establish the exact historical coverage/missingness of every workout field; Workstreams A/B should determine whether workout load can move beyond a race-only proxy.
3. **No strong JRA validation of the simple multi-front-runner rule was found.** A repository experiment is warranted, but the null result must be retained if it does not generalize.
4. **Individual going/course affinities may be mostly noise.** Their incremental value should be tested after a strong context and opponent-adjusted baseline with shrinkage.
5. **Weather PIT snapshots may be unavailable historically.** Backfilled observed weather or the final official state cannot reproduce an earlier prediction timestamp without archived vintages.
6. **Course layout needs a versioned source.** Renovations, rail settings, and program changes can invalidate static geometry.
7. **Body-weight thresholds lack JRA direct validation.** The Korean ±2.5% result is a comparison diagnostic, not a JRA policy.
8. **Survivorship affects every career-stage feature.** Retirement, injury, transfer, and class rules should be considered when interpreting age and long layoffs.
9. **Jumps racing is not covered.** Evidence and recommended features target JRA flat racing; jumps require a separate study.

## 18. Sources

All URLs below were accessed **2026-08-30**.

### Official JRA / JRA-VAN sources

- **[S1]** Japan Racing Association. “レース間隔,” 競馬用語辞典. <https://www.jra.go.jp/kouza/yougo/c10010_list.html>
- **[S14]** Japan Racing Association. “有馬記念（GⅠ） 出走馬の『調教後の馬体重』,” 2025-12-25. Includes race-day weighing timing and cautions about transport/training/feeding/defecation. <https://www.jra.go.jp/news/202512/122505.html>
- **[S16]** JRA-VAN Data Lab. “データの詳細仕様.” Includes速報馬体重 at approximately 60 minutes before post and other delivery timing. <https://jra-van.jp/dlb/ddata.html>
- **[S17]** Japan Racing Association. “2026年度競馬番組等,” hot-weather assembly/weight announcement schedule changes, 2026. <https://www.jra.go.jp/keiba/program/2026/pdf/gai02.pdf>
- **[S18]** Japan Racing Association. FAQ: “『調教後の馬体重』の発表レース、日時などを知りたいのですが?” <https://www.jra.go.jp/faq/pop02/2_4.html>
- **[S22]** Japan Racing Association. “馬場状態およびクッション値に関する情報.” Measurement meaning and planned publication times. <https://www.jra.go.jp/keiba/baba/kaisetsu/index.html>
- **[S23]** Japan Racing Association. “過去の含水率・クッション値 / 2026年.” Includes public-coverage start dates and archive-update note. <https://www.jra.go.jp/keiba/baba/archive/>
- **[S24]** JRA-VAN. *JV-Data仕様書 Ver. 4.9.0.1*, record 102 “天候馬場状態,” including announcement timestamps and state changes. <https://jra-van.jp/dlb/sdv/sdk/JV-Data4901.pdf>
- **[S28]** Japan Racing Association. “コース紹介：東京競馬場.” <https://www.jra.go.jp/facilities/race/tokyo/course/index.html>
- **[S29]** Japan Racing Association. “コース紹介：中山競馬場.” <https://www.jra.go.jp/facilities/race/nakayama/course/index.html>
- **[S30]** Japan Racing Association. “5.競走（出走）（よくあるお問い合わせ）,” Q5-9, computerized draw assignment. <https://www.jra.go.jp/owner/members/faq/category_e.html>
- **[S32]** JRA-VAN Data Lab developer support. “新潟・芝1000mの通過順位,” 2016-10-11. Clarifies corner-position data and straight-course zero values. <https://developer.jra-van.jp/t/topic/336>
- **[S33]** Japan Racing Association. FAQ: “出馬表はいつごろ発表されるのですか?” Numbered racecard timing. <https://www.jra.go.jp/faq/pop02/2_2.html>
- **[S34]** JRA-VAN NEXT. “調教タイム.” Official slope and woodchip workout-time availability. <https://jra-van.jp/nx/data_tyokyo.html>

### Primary studies and systematic reviews

- **[S2]** Mukai, K., Hiraga, A., Takahashi, T., Matsui, A., Ohmura, H., Aida, H., & Jones, J. H. (2017). “Effects of maintaining different exercise intensities during detraining on aerobic capacity in Thoroughbreds.” *American Journal of Veterinary Research*, 78(2), 215–222. DOI: 10.2460/ajvr.78.2.215. <https://pubmed.ncbi.nlm.nih.gov/28140647/>
- **[S3]** Ohmura, H. et al. (2007). “Effect of detraining on cardiorespiratory variables in young Thoroughbred horses.” *Equine Veterinary Journal*. <https://pubmed.ncbi.nlm.nih.gov/17402420/>
- **[S4]** Serrano, A. L., Quiroz-Rothe, E., & Rivero, J. L. L. (2000). “Early and long-term changes of equine skeletal muscle in response to endurance training and detraining.” *Pflügers Archiv*, 441, 263–274. DOI: 10.1007/s004240000408. <https://pubmed.ncbi.nlm.nih.gov/11211112/>
- **[S5]** Morrice-West, A. V., Hitchens, P. L., Walmsley, E. A., Wong, A. S. M., & Whitton, R. C. (2021). “Association of Thoroughbred Racehorse Workloads and Rest Practices with Trainer Success.” *Animals*, 11, 3130. DOI: 10.3390/ani11113130. <https://pmc.ncbi.nlm.nih.gov/articles/PMC8614314/>
- **[S6]** Takahashi, Y., Takahashi, T., Mukai, K., & Ohmura, H. (2021). “Effects of Fatigue on Stride Parameters in Thoroughbred Racehorses During Races.” *Journal of Equine Veterinary Science*, 101, 103447. DOI: 10.1016/j.jevs.2021.103447. <https://www.sciencedirect.com/science/article/pii/S0737080621000770>
- **[S7]** Morrice-West, A. V. et al. (2022). “Relationship between Thoroughbred workloads in racing and the fatigue life of equine subchondral bone.” *Scientific Reports*, 12, 11528. DOI: 10.1038/s41598-022-14274-y. <https://pmc.ncbi.nlm.nih.gov/articles/PMC9262984/>
- **[S8]** Crawford, K. L., Ahern, B. J., Perkins, N. R., Phillips, C. J. C., & Finnane, A. (2020). “The Effect of Combined Training and Racing High-Speed Exercise History on Musculoskeletal Injuries in Thoroughbred Racehorses: A Systematic Review and Meta-Analysis.” *Animals*, 10, 2091. DOI: 10.3390/ani10112091. <https://pmc.ncbi.nlm.nih.gov/articles/PMC7696103/>
- **[S9]** Cohen, N. D., Berry, S. M., Peloso, J. G., Mundy, G. D., & Howard, I. C. (2000). “Association of high-speed exercise with racing injury in Thoroughbreds.” *Journal of the American Veterinary Medical Association*, 216(8), 1273–1278. DOI: 10.2460/javma.2000.216.1273. <https://pubmed.ncbi.nlm.nih.gov/10767969/>
- **[S10]** Spence, A. J., Thurman, A. S., Maher, M. J., & Wilson, A. M. (2012). “Speed, pacing strategy and aerodynamic drafting in Thoroughbred horse racing.” *Biology Letters*, 8(4), 678–681. DOI: 10.1098/rsbl.2011.1120. <https://pubmed.ncbi.nlm.nih.gov/22399784/>
- **[S11]** Mercier, Q., & Aftalion, A. (2020). “Optimal speed in Thoroughbred horse racing.” *PLOS ONE*, 15(12), e0235024. DOI: 10.1371/journal.pone.0235024. <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0235024>
- **[S12]** Takahashi, Y., & Takahashi, T. (2017). “Seasonal fluctuations in body weight during growth of Thoroughbred racehorses during their athletic career.” *BMC Veterinary Research*, 13, 257. DOI: 10.1186/s12917-017-1184-3. <https://pubmed.ncbi.nlm.nih.gov/28821254/>
- **[S13]** Tozaki, T. et al. (2017). “A genome-wide association study for body weight in Japanese Thoroughbred racehorses clarifies candidate regions on chromosomes 3, 9, 15, and 18.” *Journal of Equine Science*, 28(4). <https://pmc.ncbi.nlm.nih.gov/articles/PMC5735309/>
- **[S15]** Cho, K.-H. et al. (2008). “Effects of Change of Body Weight on Racing Time in Thoroughbred Racehorses.” *Journal of Animal Science and Technology*, 50(6), 741–746. Korean/English abstract. <https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE09211580>
- **[S19]** Takahashi, T. (2015). “The effect of age on the racing speed of Thoroughbred racehorses.” *Journal of Equine Science*, 26(2), 43–48. DOI: 10.1294/jes.26.43. <https://www.jstage.jst.go.jp/article/jes/26/2/26_1506/_pdf/-char/en>
- **[S20]** Hill, E. W., McGivney, B. A., Gu, J., Whiston, R., & MacHugh, D. E. (2010). “A genome-wide SNP-association study confirms a sequence variant in the equine myostatin (MSTN) gene as the most powerful predictor of optimum racing distance for Thoroughbred racehorses.” *BMC Genomics*, 11, 552. DOI: 10.1186/1471-2164-11-552. <https://pmc.ncbi.nlm.nih.gov/articles/PMC3091701/>
- **[S21]** Maeda, Y., Tomioka, M., Hanada, M., & Oikawa, M. (2012). “Influence of Track Surface Condition on Racing Times of Thoroughbred Racehorses in Flat Races.” *Journal of Equine Veterinary Science*, 32(11), 689–695. DOI: 10.1016/j.jevs.2012.02.012. <https://cir.nii.ac.jp/crid/1360848657204895872>
- **[S25]** Nomura, M., Shiose, T., Ishikawa, Y., Mizobe, F., Sakai, S., & Kusano, K. (2019). “Prevalence of post-race exertional heat illness in Thoroughbred racehorses and climate conditions at racecourses in Japan.” *Journal of Equine Science*, 30(2), 17–23. DOI: 10.1294/jes.30.17. <https://www.jstage.jst.go.jp/article/jes/30/2/30_1901/_article/-char/ja/>
- **[S26]** Kohn, C. W., Hinchcliff, K. W., & McKeever, K. H. (1999). “Effect of ambient temperature and humidity on pulmonary artery temperature of exercising horses.” *Equine Veterinary Journal Supplement*, 30, 404–411. DOI: 10.1111/j.2042-3306.1999.tb05256.x. <https://pubmed.ncbi.nlm.nih.gov/10659290/>
- **[S27]** Ebisuda, Y. et al. (2024). “Heat acclimation improves exercise performance in hot conditions and increases heat shock protein 70 and 90 of skeletal muscles in Thoroughbred horses.” *Physiological Reports*. DOI: 10.14814/phy2.16083. <https://pmc.ncbi.nlm.nih.gov/articles/PMC11126422/>
- **[S31]** Betton, S. (1987 study; reprinted 2008). “Post Position Bias: An Econometric Analysis of the 1987 Season at Exhibition Park.” In *Efficiency of Racetrack Betting Markets*, pp. 511–526. DOI: 10.1142/9789812819192_0050. <https://ideas.repec.org/h/wsi/wschap/9789812819192_0050.html>

## 19. Bottom line

The robust design is not to encode racing maxims. It is to expose **PIT-safe, multi-window state proxies**, **shrunk condition-specific performance**, **structural course/draw context**, and a **small, testable set of field-composition interactions**. The strongest immediate candidates are interval/load history, nonlinear age/career stage, race context, normalized draw, broad distance/surface suitability, and late race-day body weight in a separate timestamped model. The multiple-front-runner interaction is worth testing precisely because it is plausible and popular—but current evidence does not justify treating it as a fact.
