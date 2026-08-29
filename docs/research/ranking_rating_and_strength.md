# Ranking・rating・latent strength・過去走価値の調査

- 対象 workstream: D
- 調査日: 2026-08-30
- 対象: JRA中央競馬のpoint-in-time能力表現、race-level ranking、将来の順位確率分布
- 本文中の「事実」は出典が直接支える内容、「解釈」は本プロジェクトへの読み替え、「提案」は未実施の設計案である。

## 1. Questions investigated

1. LambdaRank/LambdaMART、Bradley–Terry、Thurstone、Plackett–Luceは何を仮定し、何を出力するか。
2. Elo、Glicko型dynamic paired comparison、TrueSkillは、多頭数・時間変化・不確実性をどこまで扱えるか。
3. 「強い相手に負けた走り」と「弱い相手に勝った走り」をどう区別できるか。
4. surface、distance、course等のcontext-specific ratingをどう作り、疎データをどう扱うか。
5. temporal decayとformを、履歴切捨てではなくどう表現するか。
6. 次の3用途を分けると何が適切か。
   - MVPの手作りLightGBM特徴
   - 将来のlearned `Race-value encoder`
   - calibrated finishing distributionを目指すprobabilistic ranking

## 2. Conceptual map

| Family | Observation model / objective | Typical output | Probability interpretation | Primary role here |
|---|---|---|---|---|
| LambdaMART | pair順序 + NDCG差で重み付けたgradient | race内score | **なし** | MVP ranking baseline |
| Bradley–Terry (BT) | pairwise logistic comparison | item strength、pairwise win probability | pairwiseのみ整合 | opponent-adjusted ratingの基礎 |
| Thurstone | latent normal performance差 | mean/variance、pairwise probability | probit / order-statistic | heteroscedastic latent performance候補 |
| Plackett–Luce (PL) | remaining setから逐次選択 | worth、full permutation probability | full rank分布 | 最初のprobabilistic rank baseline候補 |
| Elo | BT型期待scoreに対するonline correction | sequential point rating | pairwise expected score | 軽量なpoint-in-time feature |
| Dynamic BT / Glicko | time-varying state + uncertainty | rating + uncertainty | pairwise | cold-start / inactivityを含むfeature候補 |
| TrueSkill | Gaussian skill/performance factor graph | posterior mean/variance | multiplayer順位事象 | 多頭数Bayesian rating候補 |

## 3. Evidence / sources

### 3.1 Ranking and random-utility foundations

| Source | Verified evidence | Limitation for horse racing |
|---|---|---|
| Thurstone, “A Law of Comparative Judgment,” *Psychological Review* 34, 1927, 273–286 ([DOI](https://doi.org/10.1037/h0070288), accessed 2026-08-30) | 各対象のlatent discriminal processを仮定し、差の分布からpairwise choice probabilityを表す。normal latent performance / probit比較の基礎。 | 心理測定起源。race中の依存・pace・干渉を仮定しない。 |
| Bradley & Terry, “Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons,” *Biometrika* 39, 1952, 324–345 ([DOI](https://doi.org/10.2307/2334029), accessed 2026-08-30) | positive meritの比、同値にlog-strength差のlogisticでpairwise勝率を表す。未対戦pairもconnected comparison networkなら推定可能。 | static・pairwise・transitiveな単一強度が基本。多頭数race、時間、contextは拡張が必要。 |
| Plackett, “The Analysis of Permutations,” *Applied Statistics* 24, 1975, 193–202 ([DOI](https://doi.org/10.2307/2346567), accessed 2026-08-30) | remaining itemsから順次選択する積でpermutation分布を定義。 | independence of irrelevant alternatives (IIA) とstage-invariant worthが強い仮定。 |
| Turner, van Etten, Firth, Kosmidis, “Modelling rankings in R: the PlackettLuce package,” *Computational Statistics* 35, 2020, 1027–1057 ([DOI](https://doi.org/10.1007/s00180-020-00959-3), accessed 2026-08-30) | PLのfull/partial rankings、任意次数ties、finite estimateのためのpseudo-comparison、ranking-level covariate partitionを整理。PL probabilityは `prod(alpha_i / sum remaining alpha)`。 | packageは主にstatic item worth。horse abilityを時間・条件で変える設計は別途必要。 |
| Ma et al., “Learning-to-Rank with Partitioned Preference: Fast Estimation for the Plackett-Luce Model,” AISTATS 2021 ([PMLR PDF](https://proceedings.mlr.press/v130/ma21a/ma21a.pdf), accessed 2026-08-30) | PL-based listwise methodの計算と、IIAが現実では強すぎる場合があることを明示。 | 情報検索中心。IIA違反がJRAでどの程度かは実証が必要。 |
| Xia et al., “Listwise Approach to Learning to Rank: Theory and Algorithm,” ICML 2008 ([DOI](https://doi.org/10.1145/1390156.1390306), accessed 2026-08-30) | ListMLEのpermutation likelihoodはPL分布になり、listを学習単位にする。 | probability likelihoodを使っても、model scoreが正しくspecifyされなければ校正は保証されない。 |
| Burges, “From RankNet to LambdaRank to LambdaMART: An Overview,” MSR-TR-2010-82, 2010 ([PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf), accessed 2026-08-30) | LambdaMARTはNDCG changeを反映するlambda gradientでboosted treesを学習する。 | NDCG最適化でありgenerative ranking probability modelではない。 |
| LightGBM, “Parameters” ([official documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html), accessed 2026-08-30) | `lambdarank`、整数relevance、`label_gain`、top focusの`lambdarank_truncation_level`を実装。 | scoreを勝率へ変換する公式保証はない。 |

### 3.2 Horse-racing ordering probabilities

| Source | Verified evidence | JRA implication / limitation |
|---|---|---|
| Harville, “Assigning Probabilities to the Outcomes of Multi-Entry Competitions,” *JASA* 68, 1973, 312–316 ([DOI](https://doi.org/10.1080/01621459.1973.10482425), accessed 2026-08-30) | 335 thoroughbred racesで、win probabilitiesだけからcomplete order probabilityを作る逐次式を提示。式はPL型。 | later positionsも同じworth mechanismで逐次生成する仮定が強い。 |
| Henery, “Permutation Probabilities as Models for Horse Races,” *JRSS-B* 43, 1981, 86–91 ([DOI](https://doi.org/10.1111/j.2517-6161.1981.tb01153.x), accessed 2026-08-30) | PL/first-order modelとnormal order-statistics modelを比較し、normal modelの近似を提示。 | closed formと計算容易性はPLに劣る。現在JRAでの再検証が必要。 |
| Henery, “Place Probabilities in Normal Order Statistics Models for Horse Races,” *Journal of Applied Probability* 18, 1981, 839–852 ([DOI](https://doi.org/10.1017/S0021900200034197), accessed 2026-08-30) | independent normal latent running-time modelからplace probabilitiesを扱う。 | independent normal running timeもrace interactionを捨てる仮定。 |
| Lo, “Application of Running Time Distribution Models in Japan,” in *Efficiency of Racetrack Betting Markets*, 1994, 237–247 ([publisher record](https://ideas.repec.org/h/wsi/wschap/9789812819192_0024.html), accessed 2026-08-30) | 日本のrace dataでHarville(exponential)、Henery(normal)、Stern(gamma)をfitし、特定shapeのStern modelが優位、Harvilleにsystematic ordering biasを報告。 | データは1990–1991頃の983 racesとされ、現代JRA・現在データ仕様でない。全文は有料で再現詳細に制約。 |
| Lo & Bacon-Shone, “A Comparison Between Two Models for Predicting Ordering Probabilities in Multiple-entry Competitions,” *The Statistician* 43, 1994, 317–327 ([DOI](https://doi.org/10.2307/2348347), accessed 2026-08-30) | Harvilleは必ずしもHeneryより良くなく、ordering probabilityにsystematic差があることを競馬データで比較。 | win fractionsを入力とした古典model比較。no-odds learned strengthへ同じ結果が出るかは未確認。 |
| Lo, Bacon-Shone, Busche, “The Application of Ranking Probability Models to Racetrack Betting,” *Management Science* 41(6), 1995, 1048–1059 ([DOI](https://doi.org/10.1287/mnsc.41.6.1048), accessed 2026-08-30) | US/Hong Kong/JapanでHarville系とnormal/gamma ordering modelsを比較。US/HKで改良modelのfit/strategy改善、日本ではprofit差が小さい結果も報告。 | final betting data、zero computational cost等の仮定があり、現在のJRA予測model選定へ利益値を直結できない。 |
| Benter, “Computer Based Horse Race Handicapping and Wagering Systems: A Report,” 1994 ([PDF](https://gwern.net/doc/statistics/decision/1994-benter.pdf), accessed 2026-08-30) | Harville式は2・3着でbiasがあり、その主因をlater-position contestのrandomness増加を表さないことと説明。過去相手強度を重要factorに列挙。 | 実務報告で現在JRAへの再現保証なし。ただし「win probabilityだけからlater rankを機械的に作らない」警告は重要。 |

### 3.3 Sequential and dynamic ratings

| Source | Verified evidence | Limitation for horses |
|---|---|---|
| FIDE, “FIDE Rating Regulations effective from 1 March 2024” ([official handbook](https://handbook.fide.com/chapter/B022024), accessed 2026-08-30) | Elo型運用ではrating差をexpected scoreへ変換し、`K * (actual - expected)`で更新する。Kはdevelopment係数。 | chessはpairwise、drawあり、同一条件に近い。多頭数・surface/distance・horse agingを直接扱わない。 |
| Glickman, “Parameter Estimation in Large Dynamic Paired Comparison Experiments,” *Applied Statistics* 48, 1999, 377–394 ([DOI](https://doi.org/10.1111/1467-9876.00159), [author PDF](https://www.glicko.net/research/glicko.pdf), accessed 2026-08-30) | time-varying abilityをnonlinear state-space modelとして扱い、Eloよりparameter uncertaintyを明示する高速近似を提示。chessとtennisへ適用。 | pairwise。多頭raceをpairへexplodeするとpair依存を無視し、1 raceを過剰countする危険。 |
| Herbrich, Minka, Graepel, “TrueSkill: A Bayesian Skill Rating System,” NeurIPS 2006 ([paper](https://proceedings.neurips.cc/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html), accessed 2026-08-30) | Gaussian skillとperformance noiseをfactor graphで推論し、multi-player、team、draw、uncertaintyを扱う。 | game matchmaking向け。馬のcondition/context、pace interaction、margin/timeは標準model外。 |
| Dangauthier et al., “TrueSkill Through Time: Revisiting the History of Chess,” NeurIPS 2007 ([paper](https://papers.neurips.cc/paper_files/paper/2007/hash/9f53d83ec0691550f7d2507d57f4f5a2-Abstract.html), accessed 2026-08-30) | filteringでなくpastとfutureを使うsmoothingによりskill trajectoryを推定。 | **予測特徴にそのまま使うとfuture leakage**。retrospective analysis専用と区別が必要。 |
| Cattelan, “Dynamic Bradley–Terry Modelling of Sports Tournaments,” *Applied Statistics* 62, 2013 ([DOI](https://doi.org/10.1111/j.1467-9876.2012.01046.x), accessed 2026-08-30) | team abilitiesをtime-varyingにし、home effect等を含むdynamic BTをsportsへ適用。 | team head-to-head。horse-raceへはmultiway likelihoodとcondition effectの再設計が必要。 |
| Gorgi, Koopman, Lit, “Analysis and Forecasting of Tennis Matches by Using a High Dimensional Dynamic Model,” *JRSS-A* 182, 2019 ([DOI](https://doi.org/10.1111/rssa.12464), accessed 2026-08-30) | 17年のATP dataでbaselineとsurface-specific time-varying strengthsを同時に持つdynamic modelが予測を改善。 | tennisのpairwise evidence。JRAへ一般化はできないが、「global + context deviation」のshrinkage設計を支持。 |
| Dixon & Coles, “Modelling Association Football Scores and Inefficiencies in the Football Betting Market,” *Applied Statistics* 46, 1997, 265–280 ([DOI](https://doi.org/10.1111/1467-9876.00065), accessed 2026-08-30) | dynamic team performanceを扱い、古いmatchをexponential weightingするlikelihoodを用いた。 | football score model。horse-specific decay rateや最適half-lifeの証拠ではない。 |

### 3.4 JRA performance / latent effects

| Source | Verified evidence | Limitation |
|---|---|---|
| Nakakita & Nakatsuma, “Hierarchical Bayesian analysis of racehorse running ability and jockey skills,” *International Journal of Computer Science in Sport* 22(2), 2023 ([DOI](https://doi.org/10.2478/ijcss-2023-0007), accessed 2026-08-30) | JRA 2016–2018の1800m平地でhorse/jockey individual effects、age、track conditions等を同時推定し、大きな個体差を報告。 | retrospective fit、距離限定。full-data random effectsはpoint-in-time featureではない。 |
| Oda et al., “Assessing the predictability of racing performance of Thoroughbreds using mixed-effects model,” *Journal of Animal Breeding and Genetics*, 2024 ([DOI](https://doi.org/10.1111/jbg.12822), accessed 2026-08-30) | JRA-VAN data。race effectがaverage velocityの主要因で、timeがrace levelを完全には表さず、rating/earnings indexも候補と論じる。 | breeding-value predictionが主目的で、betting forecastではない。 |

## 4. Findings by model family

### 4.1 LambdaRank / LambdaMART

#### Evidence

- raceをquery、runnerをitemとして相対scoreを学習できる。
- NDCG changeで重要pairを強く学ぶため、winner/top-kへobjectiveを寄せられる。
- scoreのlocation/scaleは勝率ではなく、race間の絶対比較も保証されない。

#### Interpretation

MVPの「順位を使うが確率を名乗らない」baselineに最適である。winner=3, second=2, third=1, other=0等のrelevanceは、full orderを真実と仮定しない折衷になる。`label_gain`とcutoffを変えると別仮説なので、単独experimentにする。

### 4.2 Bradley–Terry and pairwise expansion

#### Evidence

BTは

\[
P(i \succ j)=\frac{\exp(s_i)}{\exp(s_i)+\exp(s_j)}
\]

と書け、相手強度を介して直接対戦していないpairもcommon opponents経由で比較する。Elo updateはこの期待score残差を逐次反映する簡易法とみなせる。

#### Interpretation

finish orderを全pairへ展開すれば、各過去raceは「誰を上回ったか」を残すためfield strength調整になる。しかしn頭raceから最大`n(n-1)/2` pairが生まれ、同じraceのpairは独立でない。全pairを独立sampleとしてlikelihoodへ入れると大頭数raceを過剰weightし、uncertaintyを過小評価し得る。MVPではBTを最終確率modelにせず、causal point-in-time rating featureの生成器として使うのが安全である。

### 4.3 Thurstone-style latent performance

#### Evidence

各馬のrace performanceを

\[
z_{i,r}=\mu_{i,r}+\epsilon_{i,r}, \qquad \epsilon_{i,r}\sim N(0,\sigma_{i,r}^2)
\]

とし、`z`の順序を観測rankとする。BT/PLのlogistic・extreme-value系と異なり、normal performance noiseとhorse/raceごとのvarianceを自然に拡張できる。Heneryのhorse-race研究はnormal order-statisticsを直接検討した。

#### Interpretation

READMEのlatent strength仮説に最も近い。ただしfull ranking likelihoodはPLより計算が重く、相関したperformance、dead heat、DNF、pace interactionを入れるとさらに難しい。MVPでは実装せず、PLのlater-rank biasがJRAで確認された場合の次候補とする。

### 4.4 Plackett–Luce

#### Evidence

race-specific score `s_i` からworth `alpha_i=exp(s_i)`を作ると、ordering `i_1,...,i_n`の確率は

\[
P(i_1,\ldots,i_n)=\prod_{k=1}^{n}
\frac{\alpha_{i_k}}{\sum_{j=k}^{n}\alpha_{i_j}}.
\]

このとき `P(i wins)=alpha_i/sum(alpha)` で、full order、top-k、各順位のmarginalを同一modelから得られる。partial rankingとtiesの拡張もある。

#### Interpretation

最初のprobabilistic ranking baselineとして明快で、GBDT/neural score headとも接続しやすい。一方、Harville/PL型のlater-position biasは日本を含む古典競馬dataで報告されている。したがって「PLを採用すれば複勝・連系確率も正しい」とは言えない。position別calibrationとordering likelihoodを必ず検証し、normal/gamma order-statisticやheteroscedastic latent performanceと比較する必要がある。

### 4.5 Elo, dynamic BT/Glicko, TrueSkill

#### Evidence

- Elo: 軽量・逐次・相手調整済みだが、Kは固定的なresponsiveness/varianceの代理。
- Glickman: rating uncertaintyと時間変化を明示し、inactive / sparse entityの不確実性を扱う。
- TrueSkill: multiplayer full/partial outcomeとuncertaintyを扱える。
- TrueSkill Through Time: futureを使うsmootherであり、historical explanationには良いがforecast featureには不適切。

#### Interpretation

初期は透明なElo/BT-style ratingをfeatureとして実装し、`pre_race_rating`, `expected_finish`, `rating_delta`, `rating_uncertainty proxy`, `starts_count`を保存するのが妥当。TrueSkill導入は、単純ratingのcold-start/uncertainty問題が実測された後に行う。どの方式でも、予測行には**race更新前**のsnapshotだけを結合する。

## 5. Opponent / field strength and past-performance value

### 5.1 What can distinguish strong-field losses from weak-field wins?

#### Evidence

- Benterは過去performance adjustmentとしてcompetition strengthを明示した。
- BT/Eloでは、強い相手への予想内の敗戦は小update、格下への勝利も小update、強い相手を上回るupsetは大updateになる。
- Oda et al.のJRA研究はraw timeだけではrace levelを表し切れないことを示す。

#### Interpretation

raw finishだけでなく、「そのrace前に予想されたfieldに対して、どれだけ上振れ/下振れしたか」を保存すれば、強いfieldの低着順と弱いfieldの高着順を連続量で比較できる。

### 5.2 Recommended causal definitions for the MVP

race `r` のstart直前snapshotだけを使い、以下を作る。

#### Field features

- `field_rating_mean`, `field_rating_sd`
- `field_rating_max`, `field_rating_top3_mean`
- 自馬を除いた `opponent_rating_mean/max/top3_mean`
- `own_minus_field_mean`, `own_percentile_in_field`
- rating coverage（rating済み頭数 / field size）とcold-start頭数

#### Past-race performance residuals

- `expected_pairwise_score = sum_j P(i beats j)`
- `actual_pairwise_score = number of opponents officially beaten`（field-sizeで0–1へscaleした版も保持）
- `pairwise_surprise = actual - expected`
- `expected_rank` と `actual_rank` の差。ただしrank residualはfield sizeでnormalizeした版を併存
- class、margin、point-in-time standardized time、carried weight、draw、going等を**別列**で残す

#### Rating update

- 同一race中の更新順序依存を避けるため、全馬のpre-race ratingから期待値を計算し、race終了後にbatch updateする
- rating deltaのrace内総和を0にするvariantを基準候補とする
- dead heatはtie、取消はno contest、DNF/失格は原因別flagを残し、単純最下位扱いと除外を別policyとして検証する

これらは提案であり、まだJRA dataで有効性を確認していない。

### 5.3 Leakage rule

過去race `r` のfield strengthは、`r`より後の各相手の成績で再評価してはならない。「後にG1馬になった相手」を当時から強かったものとして書き換えるのはfuture leakageである。offlineでdynamic modelをfitする場合も、各prediction dateごとにfiltering stateを再現する。smoothing posteriorやfull-history entity effectは研究用label/diagnosticとし、model inputへ入れない。

## 6. Context-specific ratings

### Evidence

tennisのdynamic研究ではbaseline strengthとsurface-specific deviationの組合せが、surface別に完全独立ratingを持つより情報共有できる。JRA研究でもdistance、turf/dirt、course/track conditionによるperformance差が示されている。

### Interpretation

馬ごとに「芝1600m東京良」等の細分ratingを完全独立に持つと、ほとんどが疎になりcold-startが悪化する。global ratingを土台に、条件別ratingまたはdeviationを階層的にshrinkingする必要がある。

### Recommendation

MVPで同時に多数を入れず、次の順で独立ablationする。

1. global all-race rating
2. surface rating（turf / dirt）とglobalとのblend
3. broad distance-band rating（sprint / mile / middle / staying等。境界はdata specificationで固定）
4. surface × broad distance deviation

course単位、going単位、jockey-horse pair ratingはdata density確認後に延期する。各ratingには`count`, `days_since_last_update`, `uncertainty/shrinkage weight`を併記し、rating値だけを信用させない。

## 7. Temporal decay and form

### Evidence

- dynamic paired-comparison/state-space modelはabilityの時間変化を明示する。
- sports研究ではexponential time weightingやscore-driven dynamicsが使われる。
- これらは「JRA馬の最適half-life」を与えていない。

### Interpretation

`last 3 races`のhard cutoffは境界で情報を不連続に捨て、長期能力と短期formを混同する。一つのdecay rateへ固定しても、成長期・休養明け・高齢馬で異なる可能性がある。

### Recommendation

過去履歴を保存したまま、以下を並列特徴にする。

- windows: 1, 3, 5, 10 starts、30/90/180/365 days、career
- exponential: `w=2^(-days/half_life)` の複数predeclared half-lives
- trend: recent weighted mean − long-term mean
- dispersion: weighted variance、best、worst、count、missingness
- rating state: dynamic rating + days inactive + uncertainty

half-lifeはdevelopment期間で選び、final holdoutに合わせない。time decayは「疲労」と「能力変化」の両方を混ぜるため、layoff/workload featuresとは分離する。

## 8. Suitability for the three project horizons

### 8.1 MVP handcrafted LightGBM features

**Suitable now**

- PIT-safe forward-only Elo/BT-style global rating
- surface / broad-distance rating（globalとのshrinkage/blend）
- pre-race field distribution and own-relative features
- past-race expected-vs-actual surprise
- class、margin、standardized time、pace/lap、weight、draw、goingを分解したpast-performance attributes
- multiple window / exponential-decay aggregates
- coverage、start count、days since update等のreliability features

**Not suitable as MVP defaults**

- retrospective smoothed TrueSkill trajectory
- every context combinationの独立rating
- learned horse/jockey ID embedding
- one handcrafted scalar “performance value”だけへの圧縮

### 8.2 Future learned `Race-value encoder`

将来候補の入力単位は「過去1走」で、少なくとも次を分ける。

- outcome: rank、field-size normalized rank、margin、time/lap residual
- expectation: pre-race own rating、field distribution、expected rank/pairwise score
- context: class、course、distance、surface、going、weight、draw、pace
- reliability: timing/margin missingness、incident/DNF flags、field rating coverage

#### Proposed training signals

1. **current-race ranking likelihood**: 過去走token列から今回latent strengthを作り、PL/ListMLEまたは別ranking headで学習
2. **future performance residual auxiliary loss**: 次走または将来windowのopponent-adjusted surpriseを予測
3. **multi-task auxiliary targets**: standardized time/margin、top-k、rating update

#### Guardrails

- encoder targetに使う「future」はtrain sampleのlabelとしてのみ使用し、validation/test feature生成へfit済みfuture情報を混入しない
- race-valueという名前だけで中間表現の意味は保証されない。auxiliary-task performanceとablationで検証する
- history padding/orderと今回field順序を混同しない。historyは時間順、current runnersはsetとして扱う
- Set Transformer (Lee et al., ICML 2019, [PMLR](https://proceedings.mlr.press/v97/lee19d.html), accessed 2026-08-30) のようなpermutation-aware designはinter-horse moduleの候補であり、MVP根拠ではない

### 8.3 Probabilistic ranking and calibrated finishing distributions

推奨研究順序は次である。

1. **PL/ListMLE baseline**: 実装・likelihood・samplingが簡潔。`P(win)`とfull orderを同一scoreから得る
2. **position calibration audit**: win/top2/top3/各rank marginal、field size、favorite/longshot、class別に検証
3. **heteroscedastic Thurstone/order-statistic model**: PLのlater-rank biasまたはIIA違反が確認された場合
4. **correlated / mixture latent performance**: pace styleや同厩舎、race shock等の依存が残る場合

PLをLambdaRank scoreのsoftmax後処理として使う場合、temperatureをdevelopmentでfitしても、それだけでfull-rank calibrationは保証されない。BinaryのcalibrationとPLのranking likelihoodは別評価する。

## 9. Uncertainties / conflicts / negative findings

1. **現代JRAでElo、TrueSkill、dynamic PLを同一条件比較した一次研究を確認できなかった。** rating方式の最終選択は実験課題である。
2. **Harville/PLの簡潔さと競馬でのfitが衝突する。** full permutationを生成できるが、日本を含む古典dataでlater-position biasがある。PLはbaselineであって最終正解ではない。
3. **pairwise expansionの統計依存が未解決。** 全pairを使うとデータ量は増えるが、有効標本数は増えず、race重み付けが必要。
4. **「race strength」のground truthはない。** grade、prize、pre-race opponent ratings、後続成績は異なる概念で、後続成績は予測時には使えない。
5. **context分割とsparsityはtrade-off。** tennisのsurface evidenceはJRAへの直接証拠ではない。芝/ダート、距離、courseの最適分解はJRA dataで決める。
6. **margin/timeは能力だけでない。** pace、position、jockey decision、馬場、計時条件を含む。単一scalarへ早期圧縮すると説明可能性とablation性を失う。
7. **retrospective hierarchical effectsはforecast featureでない。** JRA研究で個体効果の存在は支持されるが、全期間fit値を過去年へbackfillするとleakする。
8. **DNF/失格/同着の生成機序が通常rankと異なる。** 一律最下位labelは事故・裁定と能力を混同する。データ取得後に頻度と規則を調査する必要がある。

## 10. Implications for this project

### Retain / modify / reject

| Existing hypothesis | Assessment | Concrete implication |
|---|---|---|
| race-specific latent strength + noise | **Retain** | Thurstone/TrueSkill/order-statisticsと整合し、raw finishを絶対labelと見なさない根拠がある。 |
| LambdaRank as initial relative-ranking model | **Retain** | probability headと分離し、scoreとして評価する。 |
| Plackett–Luce as later candidate | **Retain with caution** | 最初のfull-rank probabilistic baselineにするが、日本でのHarville biasを理由にposition calibrationと代替分布比較を必須化。 |
| Elo/TrueSkill point-in-time rating candidates | **Modify** | まず透明なonline Elo/BT features、次にuncertainty-aware model。smootherは禁止。 |
| strong-field adjustment | **Retain strongly** | pre-race field ratingとexpected-vs-actual residualをMVP feature groupにする。 |
| single learned race-value scalar in MVP | **Reject / defer** | 複数の手作りcomponentを保持し、later encoderの必要性を実証してから学習する。 |

## 11. Recommendations

1. **最初のratingはPIT-safe forward-only Elo/BT-styleにする。** これは未来を使わない逐次更新を意味し、因果効果推定を意味しない。race前snapshot、batch update、versioned parametersを保存する。
2. **rating単体ではなく信頼度を併記する。** starts count、days inactive、context count、coverageをLightGBMへ渡す。
3. **field strengthは相手のpre-race ratingで作る。** future achievementsやretrospective smoothed ratingを使わない。
4. **past performance valueを一数値に固定しない。** expectation residual、margin/time、class、conditionsを別featureにし、LightGBMへ統合させる。
5. **global + context deviationを採用する。** surface、次にbroad distanceを一つずつablationし、細分し過ぎない。
6. **decayは複数窓と複数half-lifeを保持する。** hard recent-Nだけに限定しない。
7. **PLはadvanced baseline、Thurstoneはfollow-up。** PLのwin/full-rank likelihoodとposition calibrationを確認してから、normal/gamma/heteroscedastic alternativesへ進む。
8. **LambdaRank scoreを勝率と表示しない。** 確率が必要ならBinary calibrated probabilityまたは明示的probabilistic headを使う。
9. **rating parameterはdevelopmentだけで選ぶ。** K、initial rating、margin weight、context blend、half-lifeをfinal holdoutへ合わせない。
10. **各rating stateをartifact化する。** algorithm/version、race_id、as-of timestamp、pre/post rating、update reason、input result statusを保存し、再現とleakage auditを可能にする。

## 12. Concrete later experiments (not implemented here)

| Hypothesis | One change | Key metrics / diagnostics |
|---|---|---|
| D-H1: opponent adjustment adds signal | base history vs +global pre-race Elo features | Log Loss, Brier, NDCG, field-strength subgroup |
| D-H2: context rating helps | global rating vs +surface blend | same metrics, turf/dirt cold-start coverage |
| D-H3: distance specialization helps | global+surface vs +broad-distance deviation | distance subgroup、count/shrinkage sensitivity |
| D-H4: surprise is better than raw finish | raw past rank group vs +expected-minus-actual residual | ablation、class transfer、strong/weak field cases |
| D-H5: decay captures form | non-decayed aggregates vs +predeclared half-lives | temporal subgroup、layoff subgroup、stability |
| D-H6: PL full-rank assumption is adequate | PL vs heteroscedastic Thurstone on same score features | rank NLL, win/top-k Brier, position calibration, field size |

## 13. Bottom line

MVPに必要なのは高度なBayesian ranking systemではなく、**予測時点直前の相手調整済みstateを再現可能に作ること**である。単純Elo/BT rating、field distribution、expected-vs-actual surprise、複数時間窓をLightGBM特徴として先に評価する。将来はPLでfull-rank probabilityの最小baselineを作れるが、日本の競馬データでも古くからlater-position biasが報告されているため、PLを最終的な順位生成法と決め打ちせず、heteroscedastic Thurstone/order-statistic modelとposition別校正を次段階に置く。
