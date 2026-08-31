# Horse History, Form, Workload, and Temporal Dynamics

## 1. Questions

1. 過去走内容は何走前・何日前まで次走と関連し、その減衰は2014--2019から2020--2021、2022へ再現するか。
2. last 1 / 2 / 3 / 5 / 10 / career、calendar window、exponential decayのどれが、履歴の情報と古さを最も安定して表すか。
3. history 0 / 1 / 2--3 / 4--9 / 10+で、利用可能性と単変量signalはどう変わるか。履歴0を学習から除外すべき根拠はあるか。
4. days since last start、14/30/60/90/180日内start数、累積距離、連続週出走は、勝敗・走行内容と単調でない関係を持つか。
5. 過去performanceの持続性は、年齢、surface変更、距離変更、休養期間で変わるか。
6. 現行history表現のうちretain / simplify / redesignすべき部分は何か。

本章は履歴構造のEDAである。新しいproduction feature、モデル、重み付け、履歴0除外規則は実装・採択しない。

## 2. Data scope

approved local raw `race_results_merged.csv`（SHA-256 `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`）を中央loaderで`date <= 2022-12-31`へ物理的に制限した。保持された最大target dateは`2022-12-28`、2023--2025行は0である。race全行が芝/ダートで、かつ`race_class`に`障害`を含まないflat raceだけを対象にした。取消・除外runnerは履歴・targetから外し、実際にstartしたDNF/DQは残した。

| 区分 | 期間 | 用途 | starter runners / races | dates | unique horses | 欠損・denominator | 不確実性・区分 |
|---|---|---|---:|---:|---:|---|---|
| warm-up | 2013 | state形成のみ | 48,293 / 3,324 | — | — | target claim 0 | quality/state only |
| discovery | 2014--2019 | 探索 | 284,883 / 19,925 | 658 | 34,349 | history 0: 28,140 (9.88%) | race macro、exploratory |
| replication | 2020--2021 | 時間再現 | 92,732 / 6,660 | 215 | 16,183 | history 0: 9,498 (10.24%) | race macro、replicated |
| confirmation | 2022 | 優先順位確認 | 45,649 / 3,331 | 108 | 11,307 | history 0: 4,703 (10.30%) | race macro、confirmed（単年） |

2013の結果は2014年以降から見ればstrictly past observationなのでstateに利用したが、2013自体のtarget統計は報告していない。horse×date重複は0件だった。local private snapshotと集計sourceは`artifacts/eda_20260901/workstreams/c_history/`に置き、Gitへrunner-level dataを追加しない。

## 3. Definitions

- **strict prior history:** `past_date < target_date`を満たす同一horseのflat start。各日をemitしてからstateを更新する。
- **race-relative performance:** numeric finishなら`1-(finish-1)/(starter_count-1)`、DNF/DQは0。1が1着、0が最下位相当で、field sizeを概ね揃える。これはEDA用の単純な履歴内容であり、production targetではない。
- **last-N mean:** 利用可能な直近`min(N, history_starts)`走のrace-relative performance平均。履歴数は別に併記する。
- **lag-N:** N走前1走だけのrace-relative performance。
- **calendar mean:** target date前30/90/180/365日内の平均。該当走がなければmissing。
- **exponential decay:** 30/90/180日半減でstrict past performanceを重み付けした平均。履歴が1走以上あれば定義できる。
- **rest:** target dateと前走dateの差。`debut`はraw上の真の新馬ではなく、2013以降のflat historyが0というleft-truncated定義である。
- **workload:** 過去14/30/60/90/180/365日内start数・累積距離と、直前の連続7日区間にstartがある週数。
- **field-size-centered outcome:** winnerなら1、他0から`1/starter_count`を引いた値。Top3も`min(3,n)/n`を引く。field size構成によるbase-rate差を軽減する記述量であり、因果効果ではない。
- **race Spearman:** 同一race内で3頭以上signalがある場合の、past signalとcurrent race-relative performanceのSpearman相関をrace macro平均したもの。
- **univariate Top-1 lift:** signal最大馬を選んだTop-1（tieはfractional）からraceごとのuniform `1/n`を引いたrace macro値。signal missingはrank末尾とし、少なくとも1頭にsignalがあるraceだけを使う。
- **temporal direction consistency:** 各年で効果方向が同じ年数。CIはrace dateを500回bootstrapし、同日raceを同じblockに保つ。

## 4. Methods

### Prequential history construction

horseごとにdate順で一度だけ走査し、snapshotを出した後に当該raceをhistoryへ追加した。同日複数startが0件であることを確認し、future appendや同日更新を避けた。入力はhorse key、過去の公式結果、過去race context、target時点のage/surface/distance/classだけで、final odds・人気・current clock/margin/last 3F/passing orderは用いていない。

### History length and decay

lag 1/2/3/5/10、last 1/2/3/5/10 mean、career mean、30/90/180/365日mean、30/90/180日half-life meanを同じrace-relative performanceから作った。各signalについてcoverage、race-macro Spearman、Top-1 lift、NDCG@3、race内上位20%と下位20%のfield-size-centered win差を求めた。

last-N間の多数比較はexploratoryである。point estimateが最大の窓をそのまま選ばず、discovery / replication / confirmationの方向、interval、coverage、既存featureとの重複を重視する。

### History availability and learning curve

target runnerをhistory 0 / 1 / 2--3 / 4--9 / 10+へ固定し、population share、field-size-centered outcome、career-history signalのrace Spearmanを集計した。同一bandがrace内に3頭以上あるraceだけがband別Spearmanへ寄与する。履歴0にはhorse-history signalが定義できないため、モデル性能ではなくcoverage gapを報告する。

### Rest, workload, and transition slices

restは`debut / 1--13 / 14--27 / 28--55 / 56--89 / 90--179 / 180--364 / 365+`、30日start数は`0 / 1 / 2 / 3+`、90日start数は`0 / 1 / 2 / 3 / 4+`へ事前に区分した。age、current distance、classとの2軸表も作ったが、全組合せを探索せず、supportのあるdomain-driven sliceだけを保持した。

前走performanceの持続性はage band、same/switch surface、前走からの距離差、rest bandごとにrace Spearmanで比較した。sliceが小さい場合はsignal race数を別に示し、全runner数だけで精度を装わない。

### Local artifacts

- `history_window_signal.csv`: 全窓のcoverage、Top-1/NDCG、race Spearman、interval、年方向。
- `lag_decay_signal.csv`: 1/2/3/5/10走前の減衰。
- `history_learning_curve.csv`: history band別coverageとcareer signal。
- `rest_workload_curves.csv`: rest/start density/連続週の非線形curve。
- `persistence_slices.csv`: age・surface・distance・rest別の前走signal持続性。
- `workload_context_interactions.csv`: workload × age/distance/classの記述表。
- `lag_decay_race_spearman.svg`、`rest_non_linear_centered_win.svg`: 上記CSVから生成した可視化。図のdenominator・period・CIは対応CSVをsource of truthとする。

## 5. Descriptive findings

### 5.1 History availability is stable, but zero history is not the majority

| History starts | Discovery runners | Replication runners | Confirmation runners | Career-signal race Spearman D / R / C |
|---|---:|---:|---:|---:|
| 0 | 28,140 | 9,498 | 4,703 | undefined / undefined / undefined |
| 1 | 26,564 | 8,966 | 4,406 | 0.435 / 0.442 / 0.437 |
| 2--3 | 45,158 | 15,154 | 7,388 | 0.456 / 0.419 / 0.458 |
| 4--9 | 85,400 | 26,958 | 13,330 | 0.398 / 0.348 / 0.372 |
| 10+ | 99,621 | 32,156 | 15,822 | 0.336 / 0.289 / 0.325 |

**Observation.** history 0は各期間9.88% / 10.24% / 10.30%で安定し、約90%には少なくとも1走のflat historyがあった。historyがある全bandでcareer meanと次走内容は正に関連した。一方、band内Spearmanは2--3走までが高く、10+では低かった。

**Interpretation.** 「新馬・履歴0は全体を支配して学習をノイズ化する」というpopulation上の証拠はない。ただし約10%にはhorse-history signalがなく、connections/current contextへの依存が高い。10+の相関低下は高齢、class、band内分散縮小、career meanの鈍化を混ぜており、履歴が多いほど予測不能という意味ではない。

history bandのfield-size-centered winは0走で`-0.0224 / -0.0224 / -0.0222`、4--9走で`+0.0182 / +0.0292 / +0.0422`、10+で`-0.0033 / -0.0086 / -0.0181`だった。この非単調性はrace/class/ageへの配置を強く反映するため、trainingからの除外判断には使えない。

**Hypothesis.** 履歴0を除外する前に、共通OOT predictionでhistory band別のBinary/Ranker loss、connections availability、race構成を分解する必要がある。

### 5.2 Single-race information decays smoothly with starts ago

| Lag | Discovery race Spearman [95% CI] | Replication | Confirmation | Available runners D / R / C |
|---:|---:|---:|---:|---:|
| 1 | 0.402 [0.398, 0.406] | 0.382 [0.376, 0.389] | 0.399 [0.389, 0.408] | 256,743 / 83,234 / 40,946 |
| 2 | 0.309 [0.305, 0.313] | 0.282 [0.275, 0.290] | 0.298 [0.286, 0.310] | 230,179 / 74,268 / 36,540 |
| 3 | 0.246 [0.241, 0.251] | 0.217 [0.209, 0.224] | 0.246 [0.234, 0.260] | 206,191 / 66,162 / 32,580 |
| 5 | 0.177 [0.171, 0.182] | 0.147 [0.136, 0.159] | 0.180 [0.167, 0.194] | 166,337 / 53,029 / 26,197 |
| 10 | 0.110 [0.103, 0.117] | 0.080 [0.067, 0.093] | 0.099 [0.078, 0.120] | 99,621 / 32,156 / 15,822 |

**Observation.** 単走lagのrace内関連は1→2→3→5→10走前で滑らかに減衰し、全期間で正だった。Top-1 liftもlag 1の`0.109 / 0.110 / 0.120`からlag 10の`0.013 / 0.006 / -0.003`へ低下し、lag 10のconfirmation CIは0を含んだ。

**Interpretation.** 古い1走を単独で強く使う根拠は弱いが、古い履歴を捨てる根拠でもない。多数走を平均・減衰してstate推定へ使うことと、lag 10を独立signalとして使うことは別である。

**Hypothesis.** historyは固定本数切捨てではなく、全履歴を保持した縮約・時間減衰stateとして扱う方が問題構造に合う。

### 5.3 Decayed aggregates are stronger than a single lag, with stable replication

| History representation | Missing D / R / C | Race Spearman D / R / C | Top-1 lift D / R / C |
|---|---:|---:|---:|
| last 1 mean | 9.88% / 10.24% / 10.30% | 0.402 / 0.382 / 0.399 | 0.109 / 0.110 / 0.120 |
| last 3 mean | 同上 | 0.432 / 0.404 / 0.434 | 0.131 / 0.137 / 0.138 |
| last 5 mean | 同上 | 0.431 / 0.405 / 0.434 | 0.136 / 0.139 / 0.145 |
| last 10 mean | 同上 | 0.430 / 0.403 / 0.433 | 0.138 / 0.141 / 0.151 |
| career mean | 同上 | 0.428 / 0.394 / 0.424 | 0.138 / 0.141 / 0.152 |
| 30-day mean | 53.40% / 56.84% / 58.09% | 0.385 / 0.368 / 0.391 | 0.103 / 0.089 / 0.091 |
| 90-day mean | 23.71% / 25.42% / 25.27% | 0.427 / 0.399 / 0.422 | 0.126 / 0.124 / 0.134 |
| 365-day mean | 10.38% / 10.84% / 10.97% | 0.433 / 0.407 / 0.438 | 0.137 / 0.141 / 0.150 |
| decay 30d | 9.88% / 10.24% / 10.30% | 0.441 / 0.416 / 0.439 | 0.135 / 0.141 / 0.150 |
| decay 90d | 同上 | **0.457 / 0.430 / 0.459** | **0.145 / 0.149 / 0.160** |
| decay 180d | 同上 | 0.455 / 0.428 / 0.458 | 0.147 / 0.150 / 0.159 |

decay 90dのrace Spearman 95% CIは`[0.453,0.461] / [0.424,0.437] / [0.450,0.467]`、race内signal上位20%と下位20%のcentered win差は`0.152 / 0.150 / 0.161`だった。

**Observation.** 90/180日指数減衰は、同一のrace-relative performanceから作ったfixed-count、career、calendar窓より全期間で強かった。30日calendar窓はmissingが半数を超え、短期formだけではcoverageが不足した。90dと180dの差は小さく、point estimateで半減期を細かく選ぶ根拠はない。last 2を追加してもlast 3以上の明確な優位はなかった。

**Interpretation.** 直近性と全履歴coverageを両立する連続減衰は有望である。一方、現行pipelineにもcount/day/decay familyがあり、本表はrace-relative performanceという履歴内容自体も同時に変えている。したがって「90d新featureが現モデルを改善する」とはまだ言えない。

**Hypothesis.** 後続実験ではhalf-life searchをせず、EDAで固定したrace-relative performance × 90d decayを一仮説として、既存history familyへの増分または簡素化をrolling OOTで検証する価値がある。

### 5.4 Rest and workload are non-linear and composition-sensitive

| Rest band | Discovery centered win [95% CI] | Replication | Confirmation | Runners D / R / C |
|---|---:|---:|---:|---:|
| debut | -0.022 [-0.025,-0.019] | -0.022 [-0.028,-0.016] | -0.022 [-0.030,-0.014] | 28,140 / 9,498 / 4,703 |
| 1--13d | -0.011 [-0.015,-0.006] | -0.015 [-0.023,-0.007] | -0.027 [-0.036,-0.017] | 18,174 / 6,042 / 2,853 |
| 14--27d | +0.013 [0.011,0.015] | +0.007 [0.004,0.011] | +0.000 [-0.005,0.006] | 89,667 / 26,988 / 12,981 |
| 28--55d | +0.005 [0.002,0.007] | +0.007 [0.002,0.011] | +0.015 [0.008,0.022] | 69,658 / 20,541 / 10,204 |
| 56--89d | +0.007 [0.003,0.010] | +0.007 [0.002,0.012] | +0.013 [0.006,0.020] | 38,516 / 15,053 / 7,793 |
| 90--179d | -0.006 [-0.009,-0.003] | -0.002 [-0.007,0.004] | -0.003 [-0.010,0.004] | 30,561 / 11,161 / 5,379 |
| 180--364d | -0.021 [-0.026,-0.016] | -0.021 [-0.029,-0.012] | -0.016 [-0.029,-0.003] | 8,707 / 2,891 / 1,429 |
| 365d+ | -0.031 [-0.040,-0.020] | -0.029 [-0.045,-0.011] | -0.020 [-0.042,0.005] | 1,460 / 558 / 307 |

30日内start数は、1走でpositive、2走でnegativeが3期間で概ね再現した。discoveryのcentered winは`0走=-0.001、1走=+0.013、2走=-0.011、3+=-0.060`、replicationは`+0.003、+0.007、-0.012、-0.018`、confirmationは`+0.007、+0.001、-0.026、-0.009`だった。ただし3+は各期間270 / 53 / 15 runnersしかない。

90日内start数は単調でなく、2--3走付近がpositiveだったが、4+は`+0.006 / +0.019 / -0.004`でconfirmationが再現しなかった。連続週出走1週は各期間negativeだったが、2週以上はsupportが極小だった。

**Observation.** restもstart densityもU字・山型で、単純な「長いほど悪い」「多いほど疲労」という単調関係ではない。

**Interpretation.** race placement、age、class、能力、怪我・調整理由を観測していないため、これらは疲労の因果効果ではない。特にdebut/短間隔/長期休養は異なるpopulationである。workload × age/distance/class表でも方向と大きさが変わり、単一global cutoffの根拠は弱い。

**Hypothesis.** workloadを試す場合は、既存model OOT residualに対する低自由度のnon-linear curveを1案だけ固定し、age/class/history availabilityをguardrail sliceとして扱うべきである。

### 5.5 Past-performance reliability changes with transition and career stage

| Slice | Discovery lag-1 Spearman | Replication | Confirmation | Temporal direction |
|---|---:|---:|---:|---|
| same surface | 0.399 | 0.381 | 0.394 | positive 6/6, 2/2, 1/1 years |
| surface switch | 0.268 | 0.225 | 0.241 | positive 6/6, 2/2, 1/1 |
| distance change 0--100m | 0.381 | 0.360 | 0.365 | positive all years |
| distance change 200--399m | 0.342 | 0.325 | 0.346 | positive all years |
| distance change 400m+ | 0.316 | 0.299 | 0.350 | positive all years; C difference narrows |
| age 2 | 0.483 | 0.460 | 0.464 | positive all years |
| age 3 | 0.377 | 0.370 | 0.369 | positive all years |
| age 4 | 0.312 | 0.269 | 0.292 | positive all years |
| age 5 | 0.276 | 0.272 | 0.262 | positive all years |
| age 6+ | 0.286 | 0.259 | 0.276 | positive all years |

rest別lag-1 Spearmanも、1--13日の`0.399 / 0.404 / 0.369`から90--179日の`0.244 / 0.255 / 0.289`へ低下した。180日以上はrace内に同じsliceが3頭以上いるraceが少なく、365日+はdiscovery 16、replication 10、confirmation 4 signal racesしかなく不確実だった。

**Observation.** 過去走signalはsurface switch、距離変更、長い休養で消えず、ただし概ね弱くなった。年齢別にも正方向は一貫したが、若馬sliceのrace内相関が高かった。

**Interpretation.** 「適性が変わると過去走は無価値」ではなく、state estimateの信頼度が条件遷移で変わる構造が示唆される。age差は若馬raceの同質性、class、成長、signal分散を含み、成長率の直接証拠ではない。

**Hypothesis.** transitionごとに大量の平均を追加するより、performance stateと`surface switch / |distance delta| / rest / effective history`の少数のreliability interactionを先に検証する価値がある。

## 6. Temporal replication

| Finding | Discovery | Replication | Confirmation | Status |
|---|---|---|---|---|
| lag 1→10の単走signal減衰 | 全lag正、概ね単調 | 同じ順序 | 同じSpearman順序、lag10 Top-1は0近傍 | replicated; priority confirmed |
| decay 90/180d > fixed/careerのrace Spearman | 両方約0.456 | 約0.429 | 約0.458 | replicated; 90 vs 180は未決着 |
| history 0 share約10% | 9.88% | 10.24% | 10.30% | stable coverage fact |
| same surface > switchのlag-1持続性 | 0.399 > 0.268 | 0.381 > 0.225 | 0.394 > 0.241 | replicated/confirmed |
| rest増加でlag-1持続性低下 | 1--179dで概ね低下 | 同方向 | 同方向、長期はwide CI | replicated |
| 14--89dのcentered winが相対的に高い | positive | positive | 28--89d positive、14--27d null | partial replication |
| 180d+のcentered winが低い | negative 6/6 years | negative 2/2 | negative、365+はwide CI | replicated; causal meaningなし |
| 30日2走のcentered winが低い | negative 6/6 | negative 2/2 | negative | replicated; confounded |

confirmationは1年なので、ここでの`confirmed`はproduction有効性ではなく仮説優先順位の確認を意味する。2022の値を見てhalf-life、rest bin、age bandを変更していない。

## 7. Uncertainty

- CIはobserved race dateのresamplingで、同日race依存を保つが、同一horse・trainer・jockeyの長期依存、2015/2017 source欠損、未観測raceを完全には表さない。
- 約13種の履歴表現、複数lag、rest/workload/context sliceを比較したため、multiple-comparison riskは高い。p-valueによるwinner selectionはしていない。
- race-relative performanceはfinishを等間隔化し、DNF/DQを0とした単純表現である。着差・時計・class・opponent strengthは含まない。
- starter field sizeは結果後に定義できるtarget normalizationであり、current predictive inputではない。取消・除外の事前可用性を示すものではない。
- calendar windowのmissingは「悪いperformance」ではなく、その期間にstartがない構造的missingである。
- history 0は2013からの観測left truncationであり、真のcareer debutと完全には一致しない。2014前半ほどpre-2013履歴欠落riskが高い。
- slice内race Spearmanは3頭以上同じsliceがあるraceだけを使う。長期休養やrare interactionでは全runner数よりsignal race数が大幅に少ない。
- workload outcomeは能力、race selection、injury、training、class、ageによる交絡を持つ。記述的関連を疲労の因果効果へ変換できない。
- normalized performanceのdecay 90dと180dは極めて近く、今回のpoint estimateでhalf-lifeを最適化するとselection biasになる。

## 8. Failure cases

1. 障害raceをflat target/historyへ混ぜる初期集計を検出し、共通contractに合わせて全結果をflat限定で再構築した。初期値は採用していない。
2. last-N meanだけを比較すると、履歴1頭ではlast 2/3/5/10が同値になる。history countとband別signalを併記しないと窓差を誤読する。
3. 30日meanの高いmissing率を0補完すると「最近走っていない」と「最近走って悪かった」を混同する。
4. history 10+の低いband内相関を「古参馬は予測不能」と読むと、age/class/variance restrictionを無視する。
5. 365日+休養のpoint estimateは数百runnerでも、race内Spearmanに寄与する同slice raceが極少である。
6. 30日3+ startsや連続2週+は大きなnegative pointでもsupport不足で、cutoff候補にできない。
7. same-surfaceの高い相関をsurface適性の因果効果と読むと、馬の選択配置とrace構成を混同する。
8. decay 90dの強さを既存モデルへの増分価値と読むと、既存feature redundancyとretraining effectを無視する。
9. history 0の相対成績が低いことをtraining除外の根拠にすると、推論時に必ず存在する約10%のpopulationを捨てる。

## 9. Leakage / PIT considerations

- loaderはrequested cutoffが`2022-12-31`と一致し、retained最大日付がそれ以下であることをassertした。
- 各snapshotは`past_date < target_date`だけを使い、current race結果を追加する前にemitした。
- horse×date重複0を確認した。将来複数startが生じても、同日一括emit-before-updateへ拡張すべきである。
- race-relative performance、rest、calendar/decay stateは後続dateだけで利用可能とした。
- target raceのfinish・field-size-centered outcomeは分析labelであり、pre-race signalへ入れていない。
- horse IDはjoin/state keyのみ。ID値、jockey/trainer ID、final odds、人気、body weight、current clock/margin/last 3F/passing orderをsignalへ使用していない。
- surface/distance/classはcurrent contextとしてslice定義にだけ使い、sliceそのものをmodel feature化していない。
- private runner snapshotはignored local artifactで、tracked repositoryへcommitしない。tracked conclusionにはaggregate count/effectだけを残す。

## 10. Modeling implications

| Representation | Decision after EDA | Evidence | Limitation / action |
|---|---|---|---|
| Horse absolute history | **retain** | 約90%に履歴があり、全期間で強いrace内signal | outcome contentの改善余地は別workstreamで評価 |
| Last 1/3/5/10 windows | **retain, then simplify** | 複数走平均はlag1より強いがlast 3--10は近い | 個別importanceでなくfamily ablationが必要 |
| Last 2 | **do not add now** | last3以上を明確に上回らない | 同じ情報の算術変形を増やさない |
| Calendar windows | **retain availability semantics** | 90--365dは有効、30dはmissing 53--58% | missing indicator/state stalenessを分離 |
| Exponential decay | **redesign candidate** | race-relative 90/180dが最も安定 | 既存decayとの重複、half-life選択biasあり |
| Effective history / uncertainty | **redesign** | history bandとrestでsignal reliabilityが変化 | meanだけでなくsupport/stalenessを併記 |
| Condition transition | **redesign with few interactions** | surface switch・距離差で持続性が弱まる | 大量condition averagesは避ける |
| Workload | **defer production use** | non-linear関連は再現するが交絡が大きい | 既存OOT residualで増分診断が先 |
| History 0 / debut | **retain in population** | 約10%で安定し、推論時にも存在する | model errorとconnections情報を別途診断 |
| Separate model by history count | **defer** | univariate reliability差だけでは分割根拠不足 | common OOT error analysisが先 |

本結果はLightGBM、Binary、LambdaRank、multi-task、別targetの優劣を直接決めない。まず同じOOT foldでhistory availability/staleness別errorを結び付ける必要がある。

## 11. Candidate hypotheses

### C-H01 — Race-relative performance state with one fixed temporal decay

- **Question:** raw finish平均ではなくrace-relative performanceを90日half-lifeで縮約すると、既存history familyへ非冗長な増分があるか。
- **Evidence:** race Spearman `0.457 / 0.430 / 0.459`、全periodでfixed/careerを上回り、coverageはhistory 1+と同じ。
- **Expected direction:** ranking改善。ただしLog Loss/Brierを悪化させない。
- **PIT status / risk:** safe / low。past-date resultのみ。
- **Redundancy risk:** high。既存count/day/decay/history featuresと重なる。
- **Validation plan:** 90dを固定し、1 feature groupだけをrolling OOTでretrained ablation。2024を設計に使わない。
- **Proposed priority:** A（expected knowledge high、cost medium）。

### C-H02 — History reliability under condition transition

- **Question:** performance stateを、surface switch・absolute distance change・rest・effective startsによる少数のreliability表現で補正すべきか。
- **Evidence:** same/switch surfaceのlag-1相関差が`0.399 vs 0.268 / 0.381 vs 0.225 / 0.394 vs 0.241`で再現し、rest・distance差でも持続性が低下した。
- **Expected direction:** transition raceのranking/error改善。
- **PIT status / risk:** safe / low。current conditionとstrict prior stateのみ。
- **Redundancy risk:** medium-high。現行suitability/current contextとの重複を監査する。
- **Validation plan:** interactionを一度に増やさず、まずstate × reliability 1表現をpreregisterしrolling OOT評価。
- **Proposed priority:** A。

### C-H03 — Zero-history and sparse-history error audit before exclusion

- **Question:** history 0 / 1 / 2--3で、Binary/RankerのOOT loss、winner rank、calibration、connections依存はどう異なるか。
- **Evidence:** history 0は約10%で安定しhorse-history signalはundefined、history 1--3ではsignalが強い。
- **Expected direction:** cold-startで両modelの誤差が高く、connections/current contextの寄与が相対的に高い。
- **PIT status / risk:** safe / low。
- **Redundancy risk:** low。feature追加でなくerror reformulation。
- **Validation plan:** Workstream Hの同一OOT predictionをhistory bandで事前定義slice。除外・別modelはまだ試さない。
- **Proposed priority:** A（問題理解を直接進める、cost low）。

### C-H04 — Low-degree workload residual curve

- **Question:** restと30/90日start densityの非線形関連は、既存model予測後にも残るか。
- **Evidence:** 1--13d/180d+、30日2走でnegative方向が再現したが、単調でなくcomposition依存。
- **Expected direction:** residualに弱い非線形構造が残る可能性。
- **PIT status / risk:** safe / medium。scratch/true training intentは未観測。
- **Redundancy risk:** high。既存form/workload groupがある。
- **Validation plan:** feature実験より先にOOT residual plotを固定binで確認。残らなければrejected-by-EDA。
- **Proposed priority:** B。

## 12. What not to conclude

- history 0の成績が低いことから、新馬・履歴0runnerをtrainingやevaluationから除外すべきとは言えない。
- history 0は真のcareer debutと同義ではない。raw coverageが2013からなのでleft truncationを含む。
- decay 90dのunivariate signalが強いことから、LightGBMへ追加すればmetricが上がるとは言えない。
- 90dが180dより優れているとは言えない。差は小さく、今回選ぶとpost-hoc half-life tuningになる。
- last 10 meanがlast 3 meanよりTop-1で高いことから、10走だけに履歴を制限すべきとは言えない。
- lag 10が弱いことから、10走以前の履歴を捨てるべきとは言えない。集約stateには寄与し得る。
- 14--89日restの相対成績が高いことから、その休養が馬を強くするとは言えない。
- 30日2走・連続週出走のnegative関連から、疲労の因果効果や出走回避ruleは導けない。
- same surfaceでperformance持続性が高いことから、surface別model分割が必要とは言えない。
- 若馬でlag相関が高いことから、若馬の能力が安定している、または高齢馬のhistoryが無用とは言えない。
- 本章は2024/2025、market、ROI、production feature acceptanceを一切扱っていない。
