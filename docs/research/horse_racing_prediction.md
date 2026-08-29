# 競馬予測手法調査：構造化データ向けベースライン

- 対象 workstream: C
- 調査日: 2026-08-30
- 対象: JRA中央競馬の構造化データを用いる、オッズ非依存の初期予測モデル
- 本文中の「事実」は出典が直接支える内容、「解釈」は本プロジェクトへの読み替え、「提案」は未実施の設計案である。

## 1. Questions investigated

1. 中規模の構造化競馬データで、LightGBMを初期基準にする根拠はあるか。
2. `P(win)`、top-k/place、着順回帰、learning-to-rank (LTR) は何を学び、何を捨てるか。
3. 競馬でのモデル比較はBinaryとLambdaRankの併設を支持するか。
4. どの特徴群が複数の研究で繰り返し現れるか。
5. 海外・年代の異なる研究をJRAへ一般化できる範囲はどこまでか。
6. neural methodから将来設計に採用できる原理はあるか。ただしMVPへ導入すべきか。

## 2. Evidence / sources

### 2.1 一般的なGBDT・LTRの一次資料と公式文書

| Source | Verified evidence | Applicability / limitation |
|---|---|---|
| Ke et al., “LightGBM: A Highly Efficient Gradient Boosting Decision Tree,” NeurIPS 2017 ([paper](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), accessed 2026-08-30) | GOSSとEFBにより、公開ベンチマークで従来GBDTとほぼ同等の精度を保ちつつ学習を高速化した。 | システム・一般表形式データの証拠であり、競馬精度を直接保証しない。 |
| LightGBM, “Parameters” ([official documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html), accessed 2026-08-30) | `binary`はbinary log loss、rankingは`lambdarank`と`rank_xendcg`を提供する。ranking labelは整数で大きいほど高relevance。`label_gain`、`lambdarank_truncation_level`、`eval_at`を明示設定できる。`is_unbalance`と`scale_pos_weight`は個別クラス確率を悪化させ得ると公式に警告されている。 | 実装仕様として直接適用可能。デフォルト`label_gain`は指数的なので、着順番号を無検討に投入できない。 |
| LightGBM, “Quick Start” ([official documentation](https://lightgbm.readthedocs.io/en/stable/Quick-Start.html), accessed 2026-08-30) | rankingではquery/groupデータが必要。 | 1 race = 1 queryという対応を直接支える実装資料。 |
| Burges, “From RankNet to LambdaRank to LambdaMART: An Overview,” Microsoft Research Technical Report MSR-TR-2010-82, 2010 ([PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf), accessed 2026-08-30) | LambdaMARTはLambdaRankのgradientをMART（boosted trees）で学び、順位入替によるNDCG変化をgradientの大きさへ反映する。 | 主な実証はweb検索。競馬でのlabel/gain設計は別途検証が必要。 |
| Grinsztajn, Oyallon, Varoquaux, “Why do tree-based models still outperform deep learning on typical tabular data?”, NeurIPS 2022 ([paper](https://papers.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html), accessed 2026-08-30) | 45の表形式データセットと大規模なhyperparameter searchで、特に約1万例程度の中規模データではtree-based modelが強く、速度でも優位だった。 | 競馬データは含まれない。表形式一般の初期priorであり、JRAで比較実験は必要。 |
| Prokhorenkova et al., “CatBoost: unbiased boosting with categorical features,” NeurIPS 2018 ([paper](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html), accessed 2026-08-30) | ordered boostingとordered target statisticsでカテゴリ特徴のtarget leakage / prediction shiftを抑える設計を示した。 | CatBoostの比較候補として重要。ただし馬・騎手IDを直接使う根拠にはならない。 |
| McCullagh, “Regression Models for Ordinal Data,” JRSS-B 42, 1980, 109–142 ([DOI](https://doi.org/10.1111/j.2517-6161.1980.tb01109.x), accessed 2026-08-30) | ordinal outcomeをcardinal scoreへ変換せず扱う回帰モデルを提示した。 | 「着順をMSE回帰すると順位差を等間隔扱いする」という一般的な弱点の根拠。 |

### 2.2 競馬を対象とした一次研究

| Source | Data / design | Main evidence | JRAへの一般化限界 |
|---|---|---|---|
| Bolton & Chapman, “Searching for Positive Returns at the Track: A Multinomial Logit Model for Handicapping Horse Races,” *Management Science* 32(8), 1986, 1040–1060 ([DOI](https://doi.org/10.1287/mnsc.32.8.1040), accessed 2026-08-30) | 200 races、horse/jockey/race features、holdout評価 | raceをchoice setとして扱うmultinomial logitを用い、各馬の相対強度からrace内で整合する勝率を得た。 | 1980年代、開催地・市場・データ仕様がJRAと異なり、標本も小さい。利益主張は再現保証にならない。 |
| Benter, “Computer Based Horse Race Handicapping and Wagering Systems: A Report,” in *Efficiency of Racetrack Betting Markets*, 1994, pp.183–198 ([PDF](https://gwern.net/doc/statistics/decision/1994-benter.pdf), accessed 2026-08-30) | Royal Hong Kong Jockey Club、主に1986–1993、実運用報告 | multinomial logitはrace内合計1の勝率を生成。近況、過去着順・着差・標準化time、過去相手強度、斤量、騎手、枠の不利、距離・surface・馬場・track適性を列挙。 | 実務報告で完全な再現コード・特徴定義・現在市場への外挿はない。closed populationの香港は、転入・地方/海外歴を含むJRAと同一でない。 |
| Lessmann, Sung, Johnson, “Identifying winners of competitive events: A SVM-based classification model for horserace prediction,” *European Journal of Operational Research* 196(2), 2009, 569–577 ([DOI](https://doi.org/10.1016/j.ejor.2008.03.018), accessed 2026-08-30) | UK Goodwood 556 races / 5,947 runners (1995–2000)。400 races train、156 races chronological holdout | finish-position regressionよりwin/non-win分類が良好。著者らは下位着順の信頼性と、独立binary予測がrace内合計1にならない点を明示。race-wise normalizationとconditional logit段を用いた。 | 単一競馬場・古いbookmaker市場・小標本。下位着順ノイズの理由には当時英国の騎乗・賞金制度に依存する解釈も含む。JRAで「3着以下は無価値」とは結論できない。 |
| Lessmann, Sung, Johnson, “Alternative methods of predicting competitive events: An application in horserace betting markets,” *International Journal of Forecasting* 26(3), 2010, 518–536 ([DOI](https://doi.org/10.1016/j.ijforecast.2009.12.013), accessed 2026-08-30) | Hong Kong 1,000 races / 12,902 runners (2005–2006)、40 fundamental variables | standard classifierは競争相手構成を直接扱わないため、race-aware adaptationを施したRandom Forestがconditional logit等と比較された。within-race competitionの明示処理が重要。 | 香港・短期間・odds利用を含む設計。現在のJRA no-odds primary modelへ利益結果を移植できない。 |
| Chung et al., “Horse race rank prediction using learning-to-rank approaches,” *Korean Journal of Applied Statistics* 37(2), 2024, 239–253 ([DOI](https://doi.org/10.5351/KJAS.2024.37.2.239), [author summary](https://junhyoung-chung.github.io/2025/01/07/Horse-race-rank-prediction-using-learning-to-rank-approaches.html), accessed 2026-08-30) | Seoul racing、2013-01–2023-07。linear regression / RF / RankNet / XGBoost, LightGBM, CatBoost rankerを比較。2023-07–11の追加データでも順位別precisionを確認 | pairwise LTRは概ねpointwiseより良く、CatBoost Rankerがsingle/exacta/trifecta hit ratio、Spearman、Kendall、NDCGで最良。重要特徴に過去成績、前走標準化記録、発走訓練、診断回数。 | KRA SeoulでありJRAでない。論文記載のtrain/evaluation分割の厳密な時系列性と、全特徴のpoint-in-time作成法は公開要約から完全確認できない。絶対値の移植は禁止。 |
| So, Woo, Lee, “Machine Learning-based Learning-to-Rank Approach for Horse Race Prediction and Web Service Development,” *Journal of the Korea Society of Computer and Information* 30(11), 2025, 311–318 ([DOI](https://doi.org/10.9708/jksci.2025.30.11.311), accessed 2026-08-30) | KRA 9,140 runner records、2024-05–2025-04、LightGBM/XGBoost/CatBoost ranking | CatBoostがNDCGとMAPで最良だが、betting scenarioの的中ではLightGBM/XGBoostが良いと報告。直近5走平均順位、全体平均順位、斤量、年齢が重要。 | 期間1年・9,140 runner recordsと小さく、JRA外。NDCG最良と単勝意思決定最良が一致しないことは示唆的だが、再現性・split詳細の確認が必要。 |
| Armerin, Hallgren, Koski, “Forecasting Ranking in Harness Racing Using Probabilities Induced by Expected Positions,” *Applied Artificial Intelligence* 33, 2019, 171–189 ([DOI](https://doi.org/10.1080/08839514.2018.1536105), accessed 2026-08-30) | Swedish harness racing。ridge/NNでexpected positionを推定し、順位周辺分布へ変換 | winnerだけでなく各馬の各順位確率を構成する二段法を提示し、著者データでmultinomial logit等と比較。 | 繋駕速歩競走で、JRA平地競走と競技・展開・データ生成が大きく異なる。MVP根拠ではなく将来候補。 |
| Borowski & Chlebus, “Machine Learning in the Prediction of Flat Horse Racing Results in Poland,” University of Warsaw Working Paper 13/2021 ([PDF](https://www.wne.uw.edu.pl/files/2616/2436/3658/WNE_WP361.pdf), accessed 2026-08-30) | Poland 3,782 flat races (2011–2020)、CART/GLMnet/XGBoost/RF/NN/LDA、2020 out-of-time test | tree/linear/NNの優劣はhorse typeとbet taskで変わり、単一model familyの普遍優位を示さない。 | working paper、ArabianとThoroughbred混在、市場・クラス体系がJRAと異なる。ROI値を一般化できない。 |

### 2.3 JRAデータに直接関係する補助証拠

| Source | Verified evidence | Limitation |
|---|---|---|
| Oda et al., “Assessing the predictability of racing performance of Thoroughbreds using mixed-effects model,” *Journal of Animal Breeding and Genetics*, 2024 ([DOI](https://doi.org/10.1111/jbg.12822), accessed 2026-08-30) | JRA-VAN提供の1986年以降データを用いた日本サラブレッド研究。average velocityではrace effectが最大で、race levelが高くても必ずしもtimeが速いとは限らず、time指標だけではperformanceを十分表さない。horse age、jockey、course、distance等の影響を検討。将来期間のgraded winner予測は多くの区分でAUC < 0.6。 | 主目的は遺伝能力・mixed effectsであり、馬券向けGBDT比較ではない。 |
| Nakakita & Nakatsuma, “Hierarchical Bayesian analysis of racehorse running ability and jockey skills,” *International Journal of Computer Science in Sport* 22(2), 2023, 1–25 ([DOI](https://doi.org/10.2478/ijcss-2023-0007), accessed 2026-08-30) | JRAの2016–2018年、障害を除く1800m、4,063 horses / 143 jockeys。horse/jockey効果を同時推定し、個体差と実績に強い関係を報告。 | 1800m限定、retrospective hierarchical fit。予測時点で全期間をfitすれば未来情報になるため、そのまま特徴量化できない。 |

## 3. Findings

### 3.1 GBDTをMVP基準にすること

#### Evidence

- LightGBMはbinary log lossとrace groupを用いるLambdaRankを同じtree boosting基盤で提供する。
- 一般の中規模表形式benchmarkではtree modelが強い。一方、これは競馬特有の優位を証明しない。
- 韓国競馬の2研究ではGBDT rankerが実用的で、CatBoostがranking指標で優位な例がある。しかしLightGBM/XGBoostがbetting-oriented hit metricで良い例もあり、指標ごとに結論が変わる。

#### Interpretation

LightGBM中心という既決事項は保持できる。理由は「競馬で必ず最強」だからではなく、同一特徴・同一splitでwin probabilityとrace rankingを比較でき、表形式データで強く、学習・診断コストが低いからである。CatBoostの証拠はLightGBMを置換する決定ではなく、カテゴリ処理を変える独立比較実験の根拠になる。

### 3.2 Binary `P(win)`

#### Evidence

- binary targetはwinner=1、others=0で、下位着順を捨てる。
- 1 raceにwinnerは1頭なのでrunner labelは強く不均衡だが、LightGBM公式はclass weightingが確率推定を悪化させ得ると警告する。
- 独立binary modelは、同一raceの各馬確率が合計1になる保証がない。Lessmann et al. (2009)もこの問題を明示し、race-aware conditional-logit段を用いた。

#### Interpretation

Binaryは単勝EVへ最短で接続するが、「runnerを独立観測とみなす近似」である。class imbalanceをaccuracy改善の問題として重み付けすると、プロジェクトが必要とする確率を壊し得る。AUCやhit rateのみで採用してはならない。

### 3.3 Top-k / place prediction

#### Evidence

- `top 3 = 1`等のbinary targetはplace事象を直接学べるが、field sizeや払戻対象頭数により事象定義が変わり得る。
- `P(rank<=k)`をkごとに独立学習すると、`P(rank<=1) <= P(rank<=2) <= ...`の単調性や、race全体の整合性は保証されない。
- full finishing distributionを作る研究は存在するが、harness racingや特定の確率仮定に依存する。

#### Interpretation

top-k classifierをMVPの主modelに増やすと仮説が散る。まずBinaryとLambdaRankを確立し、top-kは診断指標として扱う。複勝確率が必要になった時点で、独立classifierより整合的な順位分布modelを比較する。

### 3.4 着順・time回帰の弱点

#### Evidence

1. finish rankはordinalであり、MSEで1着と2着、10着と11着を同じ距離1として扱う根拠はない。
2. field sizeが異なるため、同じ「5着」の相対的位置が異なる。
3. pointwise regressionはraceの相手構成を直接表現しない。
4. Lessmann et al. (2009)は英国データでminor placingsの信頼性に疑義を示した。
5. Oda et al. (2024)はJRAデータでrace effectがtime/velocityに大きく、上級raceほど必ず速いわけではないと報告した。

#### Interpretation

raw finish rank回帰とraw time回帰を主baselineにすべきでない。ただし「回帰は常に無価値」ではない。距離・course・going・pace等でpoint-in-time標準化したtime/margin residualは、過去走価値を表す補助特徴や将来のauxiliary targetになり得る。

### 3.5 LambdaRank / race-wise LTR

#### Evidence

- LambdaMARTはpairwise preferenceを、順位入替がNDCGへ与える影響で重み付けする。予測値は順位付けscoreであって確率ではない。
- 1 raceを1 query/groupにする必要がある。
- LightGBMのdefault `label_gain`は`0,1,3,7,15,...`で、整数label差を指数gainに変える。finish rankを反転した大整数labelへそのまま使うと、意図しないtop-heavy objectiveになる。
- Seoul racingの2024研究ではpairwise LTRがpointwise手法を概ね上回ったが、英国2009研究はminor placingsの情報価値に否定的だった。

#### Interpretation

LambdaRankの価値は「着順を全部使う」こと自体ではなく、raceを比較単位とし、どの位置の入替を重要視するか明示できる点にある。label relevance、gain、`eval_at`はmodel hyperparameterではなく、業務目的を定義する仕様である。

### 3.6 繰り返し現れる特徴群

#### Evidence

複数の競馬研究で重複した群は次の通りである。

- 過去performance: 着順、着差、標準化time、直近成績、勝/連対/複勝実績
- current condition: 前走からの日数、recent race、workout / start training、年齢、診断・疾病情報
- class / field: race class、賞金、過去相手強度、今回相手との相対値
- suitability: distance、surface、going、course / track
- race-day: carried weight、draw/post position、field size
- human: jockey / trainer performance

#### Interpretation

「頻出」は因果効果やJRAでの有効性を意味しない。特に累積勝率・平均着順は相手水準と選抜過程を混同する。全てpoint-in-timeで作成し、raw集計、条件別集計、opponent-adjusted ratingを別feature groupに分けてablationする必要がある。

### 3.7 Neural methodsの扱い

#### Evidence

- Grinsztajn et al.は中規模表形式データでtree優位を示した。
- Zaheer et al., “Deep Sets,” NeurIPS 2017 ([paper](https://papers.neurips.cc/paper/6931-deep-sets.pdf), accessed 2026-08-30) はpermutation-invariant set functionの構造を示した。
- Lee et al., “Set Transformer,” ICML 2019 ([PMLR](https://proceedings.mlr.press/v97/lee19d.html), accessed 2026-08-30) はset要素間interactionをattentionで扱うpermutation-invariant architectureを提示した。

#### Interpretation

将来の出走馬集合は入力順に意味がないため、inter-horse moduleにはpermutation equivariance/invarianceが必要である。ただし競馬でGBDTを上回る直接証拠ではなく、MVP導入理由にはならない。将来のRace-value encoderは、まず手作りの過去走特徴がどこで不足するかをablationで確認してから検討する。

## 4. Uncertainties / conflicts / negative findings

1. **JRAでの厳密なLightGBM Binary vs LambdaRank比較を発見できなかった。** JRAに直接関係する査読研究はmixed-effects / hierarchical Bayes中心で、同一point-in-time特徴・時系列splitのGBDT比較ではない。
2. **下位着順の価値は文献で衝突する。** UK 1995–2000ではminor placingの信頼性に疑義、Seoul 2013–2023ではLTRが有利。制度、騎乗、賞金、field size、metric、時代の差を分離できない。
3. **CatBoost優位はJRAで未確認。** Seoulの結果はカテゴリ変数処理の仮説を生むが、馬・騎手IDを除く本プロジェクトで同じ優位が残るか不明。
4. **feature importanceは研究間で比較不能。** 特徴定義、利用時点、競馬制度、model、importance算出法が異なる。SHAP上位であることを採用理由にしない。
5. **利益結果は再現根拠にならない。** 古い香港/英国研究は現在のJRA市場、takeout、利用可能時点、データ品質と異なる。oddsを入力した二段modelも本プロジェクトのprimary no-odds modelと目的が違う。
6. **公開要約だけではleakage監査不能な研究がある。** 累積成績やstandardizationを予測時点より後の全標本で作成したか否かが不明な場合、性能数値を設計根拠に格上げしない。

## 5. Implications for this project

### 5.1 Retain / modify / reject

| Existing decision | Assessment | Evidence-based implication |
|---|---|---|
| LightGBM-centered non-DNN MVP | **Retain** | 表形式baselineとして妥当。競馬固有の普遍優位ではないため比較可能な設計を維持する。 |
| Binary `P(win)` and LambdaRank | **Retain strongly** | 文献の対立そのものが両方を同一条件で比較する価値を示す。 |
| odds excluded from primary model | **Retain** | 多くの古典的利益研究はodds併用であり、fundamental signalとの切分けにはno-odds baselineが必要。 |
| direct horse/jockey IDs excluded | **Retain for MVP** | CatBoostや階層BayesはID効果を学べるが、cold startとmemorizationを解決しない。point-in-time履歴統計へ変換する。 |
| raw finish/time regression as main model | **Reject** | ordinal interval仮定、field size、race effect、within-race competitionの問題がある。補助target/featureに限定。 |
| large DNN / end-to-end now | **Reject for MVP** | 競馬固有の優位証拠がなく、表形式GBDTを先に確立すべき。 |

### 5.2 MVP model contract

#### Binary baseline

- unit: runner、ただしsplitはrace/date単位
- row set: prediction cutoff時点のeligibleなas-of field。cutoff前取消は除外し、cutoff後取消はrowへ残す
- label: official winner=1、other as-of runners=0。cutoff後取消も0、同着1着は各1のmarginal target。target semanticsをartifactへ保存する
- objective: `binary`; probabilityが目的なので初期は`is_unbalance=false`, `scale_pos_weight=1`
- primary features: oddsなし、IDなし、point-in-timeのみ
- outputs: raw probability、race内sum、rank、calibration input
- evaluation: race平均Log Loss / Brier、runner平均も併記、Top-1、NDCG診断、field-size別calibration
- coherence diagnostic: `sum_i p_i`の分布を必ず保存する。race内正規化版はraw版と別variantとして評価し、無言で後処理しない

#### LambdaRank baseline

- group: `race_id`; raceを跨ぐpairを作らない
- score: probabilityとして使用しない
- 最初のrelevance案: winner=3、2着=2、3着=1、他=0
- explicit parameters: `label_gain=[0,1,3,7]`、`eval_at=[1,3,5]`; `lambdarank_truncation_level`は目的kより少し大きい候補をdevelopmentだけで選ぶ
- full-order案（`field_size-rank` + linear gain）は**別実験**とし、winner-focused設計と混ぜない
- evaluation: NDCG@1/3/5、Top-1、winner mean reciprocal rank、Spearman/Kendallは完走馬のみの補助指標
- binary modelと完全に同一のfeatures / time splitを使う

## 6. Recommendations

優先順は次の通りである。

1. **BinaryとLambdaRankを同じdataset snapshotで実行する。** model family比較より先にtask formulation差を測る。
2. **probability目的のBinaryでは自動class weightingを使わない。** 必要ならweightingありを独立実験とし、calibration悪化を測る。
3. **LambdaRankのlabel/gainをconfigへ明記する。** default任せにせず、まずtop-3 graded relevanceを採用する。
4. **race-relative featureを追加する。** 各馬のabsolute point-in-time statisticと、同race内平均との差・percentile・max差を併存させる。これはfuture結果を使わずにwithin-race competitionを表せる。
5. **feature groupを分離する。** `basic_current_race`, `horse_history_windows`, `jockey_trainer_pit`, `suitability`, `rating_field_strength`, `form_workload`を個別ablation可能にする。
6. **CatBoost Rankerはbaseline後の単独比較にする。** Seoul証拠を検証するが、direct IDsは入れず、同じfeatures/splitで比較する。
7. **raw rank/time regressionは診断baselineに留める。** 主採用候補にせず、standardized time residualをpast-performance featureとして評価する。
8. **neural encoderを延期する。** GBDTでhistory aggregation、field-relative features、ratingの限界が観測され、十分な学習規模が確認された後に限る。

## 7. Concrete hypotheses for later experiments (not implemented here)

| Experiment hypothesis | Single change | Expected informative outcome |
|---|---|---|
| C-H1: winner-only signal is more reliable than lower placings | Binary vs LambdaRank, otherwise identical | probability qualityとranking qualityのtrade-offを測る |
| C-H2: top-heavy rank supervision is preferable | LambdaRank top-3 relevance vs full-order linear gain | 下位着順の追加情報がJRAで有効か検証 |
| C-H3: within-race relative context helps pointwise Binary | absolute-only vs race-relative derived features | 独立binaryの競争文脈不足を特徴で補えるか検証 |
| C-H4: categorical treatment matters | LightGBM vs CatBoost Ranker, same inputs | Seoul結果のJRA再現性を検証 |
| C-H5: standardized performance residual adds information | history base vs +point-in-time time/margin residuals | raw着順以外の過去走価値を検証 |

## 8. Bottom line

証拠は「LightGBMが競馬で常に最良」とは示さない。しかし、構造化データに強い低コストbaselineであり、`P(win)`とrace-wise rankingという相補的な仮説を同一基盤で比較できる。JRAに最も必要なのは複雑なmodel追加ではなく、同一の厳格なpoint-in-time dataset上でBinaryとLambdaRankの改善・悪化を多面的に測ることである。
