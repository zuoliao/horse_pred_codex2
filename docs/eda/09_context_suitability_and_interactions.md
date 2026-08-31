# Race Context, Suitability, Transitions, and Interactions

## 1. Questions

1. current raceのsurface、distance、venue、direction、class、field sizeは、過去走の情報が次走へ持続する度合いをどの程度変えるか。
2. same conditionとcondition switchを、単純な適性勝率ではなく「過去performanceの信頼度」として表す根拠はあるか。
3. class、age、rest、field-size transitionのraw win rateは、頭数base rateとrace placementを考慮しても安定した構造を持つか。
4. domain-driven interactionのうち、`surface × past surface performance`、`distance change × past performance`、`class change × opponent strength`、`rest × workload`、`age × starts`、`field size × rating spread`、`post × distance/course`、`jockey form × horse experience`、`pace pressure × style`には、別期間で再現する根拠があるか。
5. 現行Binary / LambdaRankが特に弱いtransitionはどこか。旧field-relative 15列を戻さずに説明できるか。

本章はEDAであり、新しいproduction feature、condition別model、interaction feature、margin mappingを実装・採択しない。Observation、Interpretation、Hypothesisを分離し、候補はhypothesis registryへ渡すまでに留める。

## 2. Data scope

approved local raw `race_results_merged.csv`（SHA-256 `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`）を共通loaderで`date <= 2022-12-31`へ制限した。最大target dateは`2022-12-28`、2023以降の分析行は0である。2013は前走stateのwarm-upだけに使い、target claimをしていない。

primary populationは、race全行が芝/ダート、非障害、取消・除外のない`pit_c_scoring_eligible` choice setである。

| 区分 | 期間 | 用途 | runner count | race count | date blocks | Previous-run missing | Aggregation / uncertainty |
|---|---|---|---:|---:|---:|---:|---|
| discovery | 2014--2019 | interaction discovery | 271,740 | 18,966 | 658 | 26,908 | runner outcomeは`win - 1/n`、persistenceはrace macro、exploratory |
| replication | 2020--2021 | temporal replication | 88,578 | 6,356 | 215 | 8,945 | 同じ定義、replicated |
| confirmation | 2022 | priority confirmation | 43,537 | 3,176 | 108 | 4,391 | 同じ定義、confirmedは単年上の優先順位のみ |

既存model errorには`artifacts/eval_roll_001_current_best_20260831/predictions_scoring.csv.gz`の`role=evaluation`だけを使用した。chunk loaderで2020--2022に物理filterし、264,230 runner-prediction rows、9,532 unique races、2 methodsを保持した。calibration role、2023 fold、2024、2025の保持は0行である。同着raceはco-winnerへ`1/m` massを配り、各raceの総winner weightを1に保った。

local aggregate sourceは`artifacts/eda_20260901/workstreams/g_context/`にある。`transition_summary.csv`、`bounded_interactions.csv`、`oot_error_slices.csv`、`manifest.json`は集計だけで、raw、runner-level data、prediction rowsをGitへ追加していない。

## 3. Definitions

- **strict prior start:** 同一horseで`previous_date < target_date`を満たす直前のflat start。同日raceを前走にしない。
- **current performance:** `1 - (finish - 1)/(starter_count - 1)`。numeric finish不能のstarted runnerは0。これはEDA用の順位percentileで、production targetではない。
- **persistence:** transition cell内で、前走performanceと現走performanceのSpearmanをraceごとに算出し、3頭以上定義できるraceをmacro平均する。
- **base-adjusted win:** runnerのwinner indicatorから`1/current starter field size`を引く。raw win rateのfield-size差を軽減する記述量で、因果効果ではない。
- **surface transition:** `turf_same`、`dirt_same`、`turf_to_dirt`、`dirt_to_turf`。
- **distance transition:** absolute change 200m以内、201--399m、400m以上延長、400m以上短縮。200m以内には同距離を含む。
- **class tier:** 未勝利=0、1勝/旧500万=1、2勝/旧1000万=2、3勝/旧1600万=3、open/graded=4。新馬は同tierに潰さず、`new_to_maiden` / `new_related_other`として独立transitionにする。それ以外はcurrent minus previousが正ならup、負ならdown。
- **rest:** 0--13、14--29、30--89、90--179、180日以上。prior startなしは`history0`として別扱い。
- **field-size transition:** 前走との差が±1以内、2--4増減、5頭以上増減。
- **OOT error:** co-winnerへ`1/m`を配ったwinner massで重み付けした負対数と、race内最大calibrated probabilityに対応するwinner massというTop-1。確率比較ではLog Lossを主に読み、Top-1はguardrailとする。
- **temporal direction:** discovery、replication、confirmationで大小関係または符号が一致するか。単一年のpointだけでinteractionを選ばない。

## 4. Methods

### Strict-PIT transition construction

horse内をdate順に並べ、targetをemitしてから当該raceをhistoryへ追加した。previous dateがtarget date以上の行が0であることをassertした。current raceの時計、着差、last 3F、passing order、final odds、人気は入力へ使っていない。race ID、horse IDはjoin/state keyだけである。interactionのcentered winはrunner micro平均ではなく、各race内cell平均を作ってからrace macro平均した。

各transitionについて、sample runners、unique races、date blocks、win rate、uniform base、base-adjusted win、race-macro persistenceを同じ固定定義で集計した。raw win rateはclass-up/downやfield-size transitionで著しくselection-biasedなので、base-adjusted winとpersistenceを併記した。

### Domain-driven interactions

全feature pairを探索せず、事前に意味のある次の経路へ限定した。

1. surface / distance / venue / direction transition × 前走performance persistence
2. rest × 30日start数
3. age × career starts
4. jockey 90日減衰win rate × horse experience
5. field-size band × pre-race rating spread
6. post position × distance bandの限定監査
7. class transition × opponent strengthの意味論監査
8. frozen PACE-01 style × PACE-02 rival pressureの既存証拠監査

jockey formとrating spreadのcut pointは2014--2019の三分位で固定し、2020--2021、2022へそのまま適用した。post/courseはvenue×distance×surface×frameの全cell探索を禁止し、まずdistance帯だけを見た。supportを見てcellを追加していない。

### Model-suggested path

新しいshallow modelで2022まで繰り返しsplitを探す代わりに、凍結済みBinary / LambdaRankのrolling OOT residualを使った。discoveryで固定したfield-size / rating-spread cut pointを2020--2021 evaluation予測へ適用し、2022で方向を確認した。これはmodel-errorが示唆するinteractionであり、feature採択実験ではない。ALE/SHAP interactionの全pair searchは実施していない。

## 5. Descriptive findings

### 5.1 Observation: same-conditionほど前走内容は安定して持続する

| Transition | Discovery runners / persistence | Replication runners / persistence | Confirmation runners / persistence | Missing / denominator | Temporal uncertainty |
|---|---:|---:|---:|---|---|
| dirt same | 107,606 / .415 | 36,220 / .398 | 18,050 / .410 | prior missingは別cell、全eligible runnerが分母 | 3期間positive、同程度 |
| turf same | 104,018 / .381 | 33,096 / .361 | 16,025 / .379 | 同上 | 3期間positive |
| dirt → turf | 13,527 / .253 | 4,074 / .197 | 2,029 / .210 | persistence race 1,987 / 571 / 294 | sameより一貫して弱い、小cellで幅広い |
| turf → dirt | 19,681 / .281 | 6,243 / .247 | 3,042 / .247 | persistence race 3,060 / 968 / 479 | sameより一貫して弱い |
| distance change ≤200m | 197,873 / .391 | 65,134 / .371 | 32,204 / .388 | persistence race 17,128 / 5,724 / 2,860 | 3期間で最大または最大級 |
| extend ≥400m | 19,693 / .339 | 5,847 / .286 | 2,818 / .399 | persistence race 2,646 / 749 / 347 | positiveだがCで差が縮み、不安定 |
| shorten ≥400m | 17,931 / .307 | 5,583 / .346 | 2,634 / .328 | persistence race 2,315 / 699 / 315 | positive、sameより概ね弱い |
| same venue | 76,335 / .403 | 26,045 / .389 | 13,399 / .399 | persistence race 10,393 / 3,725 / 1,917 | switchを全期間上回る |
| venue switch | 168,497 / .362 | 53,588 / .338 | 25,747 / .362 | persistence race 16,659 / 5,570 / 2,764 | positiveだが弱い |
| same direction | 153,642 / .389 | 48,275 / .370 | 22,601 / .382 | persistence race 16,194 / 5,437 / 2,702 | switchを全期間上回る |
| direction switch | 91,190 / .357 | 31,358 / .326 | 16,545 / .361 | persistence race 12,068 / 4,352 / 2,368 | 差は小さい |

**Observation.** surface、distance、venue、directionの全てで前走signalはswitch後にも正だが、same conditionの方が多くの期間で強かった。surface switchの低下が最も明瞭で、direction差は小さい。distance 400m以上延長の2022 persistenceは高く、単調な`|distance delta|`だけでは説明できない。

**Interpretation.** condition switchは過去走を無価値にするのではなく、過去stateの信頼度を弱める。大量のsame-condition平均を別々に追加するより、共通performance stateと少数のtransition / uncertainty表現を分ける方が自然である。

**Hypothesis.** `past performance state × surface switch`と`past performance state × predeclared distance-change band`を、各1仮説としてrolling OOTで検証する価値がある。両方を同時投入しない。

### 5.2 Observation: raw transition win rateには大きなselection effectがある

| Transition cell | Runners D / R / C | Win minus `1/n` D / R / C | Persistence D / R / C | Uncertainty / reading |
|---|---:|---:|---:|---|
| dirt → turf | 13,527 / 4,074 / 2,029 | -.0378 / -.0409 / -.0433 | .253 / .197 / .210 | 負方向は再現、horse placementとの交絡大 |
| turf → dirt | 19,681 / 6,243 / 3,042 | -.0209 / -.0138 / -.0176 | .281 / .247 / .247 | 負方向は再現 |
| extend ≥400m | 19,693 / 5,847 / 2,818 | -.0331 / -.0330 / -.0285 | .339 / .286 / .399 | winは一貫、persistence差は不安定 |
| class down | 5,998 / 1,182 / 597 | +.0461 / +.0389 / +.0665 | .331 / .234 / .412 | confirmation 597 runners / 44 persistence racesでwide |
| new → maiden | 20,085 / 6,942 / 3,341 | -.0078 / -.0038 / -.0045 | .441 / .447 / .432 | 新馬を未勝利と同一tierにせず独立 |
| class same | 197,412 / 64,232 / 31,565 | -.0004 / -.0009 / -.0023 | .432 / .402 / .423 | 大support、stable |
| class up | 19,633 / 6,663 / 3,332 | +.0039 / +.0122 / +.0220 | .399 / .313 / .332 | promotion selectionを含む |
| rest 0--13d | 17,348 / 5,780 / 2,749 | -.0132 / -.0150 / -.0227 | .399 / .404 / .368 | negativeは全期間、因果的疲労ではない |
| rest 14--29d | 108,744 / 32,324 / 15,560 | +.0091 / +.0049 / +.0014 | .371 / .358 / .381 | magnitude drift |
| rest 30--89d | 79,932 / 27,577 / 14,042 | -.0014 / +.0038 / +.0087 | .321 / .300 / .343 | outcome符号は非安定 |
| rest 180d+ | 9,702 / 3,277 / 1,673 | -.0207 / -.0221 / -.0182 | .212 / .247 / .305 | negative outcomeは再現、persistence uncertainty大 |

**Observation.** class downは大きなpositive、class upもpositiveだった。これは「昇級が有利」という意味ではなく、昇級できた馬が直前に強い結果を出しているselectionを反映する。restは非線形で、0--13日と180日以上がnegative、中央の最良帯は期間で動いた。

**Interpretation.** transitionのmain effectと、過去performanceがどれだけ移植可能かというinteractionは分離すべきである。class transitionは相手水準・過去内容・promotion selectionを同時に含むため、単純なclass-up/down dummyだけでは意味が曖昧である。

**Hypothesis.** class changeを試す前に、Workstream Eの`performance axis`と`field quality axis`を2軸のまま持ち、classとの冗長性を確認する。field meanはclassで大きく変わり、新馬では全て1,500なので、`class × opponent mean`を直接量産しない。

### 5.3 Observation: rest × workloadとage × startsは非加法的だが、因果解釈できない

| Interaction cell | Centered win D / R / C | Runners D / R / C | Direction / limitation |
|---|---:|---:|---|
| rest 14--29d, starts30=1 | +.0152 / +.0079 / +.0047 | 102,982 / 30,555 / 14,747 | positive、magnitude縮小 |
| rest 14--29d, starts30=2+ | -.0084 / -.0077 / -.0229 | 5,762 / 1,769 / 813 | negativeが再現、small C |
| rest 0--13d, starts30=1 | -.0101 / -.0148 / -.0285 | 12,891 / 4,403 / 2,120 | negativeが再現 |
| rest 180d+, starts30=0 | -.0226 / -.0218 / -.0154 | 9,702 / 3,277 / 1,673 | negativeが再現 |
| age 3, starts 4--9 | +.0223 / +.0360 / +.0551 | 55,290 / 18,504 / 9,069 | positive、career-stage selection |
| age 6+, starts 10+ | -.0438 / -.0384 / -.0518 | 22,476 / 8,414 / 3,634 | negativeが再現、class/ability交絡 |
| age 4--5, starts 10+ | +.0039 / -.0114 / -.0147 | 61,730 / 19,610 / 10,238 | discoveryから符号反転 |

**Observation.** 14--29日restで30日内2走以上のcellは、同restの1走cellより全期間低かった。age 3 / starts 4--9とage 6+ / starts 10+も方向が再現したが、age 4--5 / starts 10+は符号が反転した。

**Interpretation.** 前者は疲労仮説と整合するが、race selection、能力、class、怪我・調整理由を観測していない。後者はageとcareer startsがcareer stageを共同で表す一方、直接IDなしでもpopulation fingerprintになりうる。

**Hypothesis.** workloadは単調変換ではなく、`rest × recent starts`の低自由度1案を事前固定し、history availability・age・classをguardrailにする。age × startsは新feature候補より、現model residualを解釈するsliceとして先に使う。

### 5.4 Observation: jockey formのmain effectは強いが、horse experience interactionの増分は明確でない

| Jockey form × horse history | Centered win D / R / C | Runners D / R / C | Reading |
|---|---:|---:|---|
| high × history 0 | +.0374 / +.0502 / +.0453 | 9,281 / 2,906 / 1,752 | strong main effect、cold-history horseでもpositive |
| high × starts 1--3 | +.0590 / +.0723 / +.0576 | 22,001 / 6,926 / 4,023 | high form内差は小さい |
| high × starts 4+ | +.0488 / +.0553 / +.0577 | 59,161 / 18,437 / 10,725 | positive |
| low × history 0 | -.0495 / -.0516 / -.0528 | 9,102 / 3,039 / 1,556 | negative |
| low × starts 4+ | -.0323 / -.0328 / -.0367 | 57,003 / 18,061 / 9,079 | negativeだが多少縮小 |

**Observation.** discovery三分位で固定したjockey formのhigh/low差は全experience帯・全期間で同方向だった。一方、experienceによる差はjockey main effectより小さく、同じ順序を全期間で保たない。

**Interpretation.** connections stateのsignalは強いが、`jockey form × horse experience`のinteractionを追加すべき証拠ではない。既存connections 130列の冗長性とentity fingerprint riskを先に扱うべきである。

**Hypothesis.** Workstream Fの縮約・uncertainty表現を優先し、本interactionはdeferする。

### 5.5 Observation: field size × rating spreadはoutcome main effectではなくmodel-error interactionとして現れる

race全体を同じfield size / spread cellへ割り当てると、`sum(win - 1/n)=0`なのでoutcome associationは定義上ほぼ0になる。したがってfield-level interactionはrunner win rateでなく、OOT race errorで診断した。

| Field × discovery-frozen spread | Binary winner Log Loss R / C | Rank winner Log Loss R / C | Winner races R / C | Direction / uncertainty |
|---|---:|---:|---:|---|
| ≤11, high | 1.886 / 1.800 | 1.856 / 1.826 | 369 / 191 | high spreadはlowより悪くない |
| ≤11, low | 1.863 / 1.874 | 1.869 / 1.885 | 582 / 359 | high/low差はsmall・family差あり |
| 12--14, high | 2.180 / 2.103 | 2.171 / 2.100 | 418 / 250 | replicationとconfirmationでlowとの順序が変化 |
| 15+, high | 2.339 / 2.313 | 2.366 / 2.317 | 964 / 436 | lowより両family・両期間で高loss |
| 15+, low | 2.268 / 2.195 | 2.303 / 2.206 | 1,272 / 574 | large field内の比較対象 |

**Observation.** large fieldではhigh rating spreadのwinner lossがlow spreadより両family・両期間で悪かった。small fieldでは同じ方向でなく、field sizeとのinteractionが示唆された。Top-1はcellごとに揺れ、probability lossほど一貫しなかった。

**Interpretation.** rating spreadの高い大頭数raceで勝馬を過小評価している、ratingのscale / uncertaintyが条件に合わない、またはclass・history mixが残っている可能性がある。旧field-relative 15列を復活させる証拠ではない。

**Hypothesis.** `large field × rating uncertainty/spread`はfeature追加より先に、calibration・winner characteristics・rating coverageを分解するresidual studyとして登録する。race-constant spread 1列を単独で試す場合も、旧15列を戻さない。

### 5.6 Observation: normalized horse-number/courseとpace/styleは追加探索の根拠が不足する

horse-numberをfield内位置へ正規化した限定監査は実施したが、これは枠番ではない。venue×distance×surfaceへ分けたcell countとdate-block intervalをcanonical aggregateへ固定していないため、数値結果をevidenceとして採用せず、horse-number/course feature候補へ昇格させない。

PACEについては、own historical early positionのPACE-01は既存rollingで支持されたが、rival-only current-field pressure 1列のPACE-02はBinary NDCG `+.00017 [-.00193,+.00228]`、Rank NDCG `-.00081 [-.00294,+.00129]`で両family inconclusiveだった。PACE-02 train-only auditは2014--2021の329,978 finite rows、25,322 races中2,212 racesがall-missingである。ここからpressure × styleを追加探索すると同じ仮説への事後適合になるため、本EDAでは行っていない。

**Interpretation.** postはdistance/courseとのinteractionを持つ可能性があるが、枠順効果、field composition、開催コース、surfaceを分離していない。pace pressure単体の増分が未確認な状態でinteractionを作る根拠も弱い。

**Hypothesis.** post/courseは独立したpre-registered analysisとして、support閾値とcourse cellをtargetを見る前に固定する。pace × styleはPACE-02の再表現探索ではなく、より直接的なpace data取得後までdeferする。

## 6. Temporal replication

| Finding | Discovery | Replication | Confirmation | Status |
|---|---|---|---|---|
| same surface > switch persistence | .381--.415 vs .253--.281 | .361--.398 vs .197--.247 | .379--.410 vs .210--.247 | replicated / confirmed |
| distance ≤200m persistence high | .391 | .371 | .388 | replicated; ≥400mとの差はCで縮小 |
| same venue > switch | .403 > .362 | .389 > .338 | .399 > .362 | replicated / confirmed |
| same direction > switch | .389 > .357 | .370 > .326 | .382 > .361 | replicated, effect small |
| rest 0--13 / 180+ centered win negative | -.013 / -.021 | -.015 / -.022 | -.023 / -.018 | replicated; causal claimなし |
| rest 14--29 × starts30=2+ below starts30=1 | -.007 vs +.010 | -.010 vs +.006 | -.024 vs +.003 | replicated; selection risk high |
| age 6+ × starts10+ negative | -.039 | -.034 | -.041 | replicated; composition effect |
| jockey high/low direction across experience | all cells positive / negative | same | same | main effect replicated、interaction未確定 |
| 15+ field × high spread has higher OOT loss | discovery-frozen cut | Rで両family | Cで両family | model-error interaction replicated |

2022はpriority confirmationだけであり、bin、cut point、interactionを2022の値に合わせて変更していない。2023/2024/2025を追加確認に使っていない。

## 7. Uncertainty

- 主要transition表はD/R/Cの時間変動そのものをuncertainty診断とし、persistenceの有効race数をrunner数と分けた。rare class-downやlarge distance changeではintervalが広い。
- race-macro Spearmanは同一race内にcell runnerが3頭以上あるraceだけを使う。transition runner数が多くても、persistence race数は小さくなり得る。
- OOT errorはcoherent dead-heat massでrace単位に集計し、date-level intervalも保存したが、同一horse・jockey・trainerの長期依存を完全に区間化していない。表のsmall differencesを有意差と読まない。
- class、rest、age、starts、jockey formはrace placementと能力に強く交絡する。`win - 1/n`はfield-size base rateだけを補正し、因果交絡を除かない。
- transition / interactionを複数見ているのでmultiple-comparison riskはhighである。符号の時間一致、sample support、意味論を使って仮説を絞り、p-value winnerを選んでいない。
- discovery三分位は時代driftの影響を受ける。固定cutをreplicationへ移したことはthresholdのproduction妥当性を保証しない。
- frame/courseのcanonical cell counts / block CIが不足し、pace × styleは既存PACE-02がinconclusiveなので、いずれも今回のready-for-experiment候補にはしない。

## 8. Failure cases

1. surface switch、400m以上の距離変更、長期休養では前走signalが弱まるが0にはならない。missingや0へ置換すると有用な履歴を捨てる。
2. class-upのbase-adjusted winがpositiveなのはpromotion selectionを含み、「昇級が勝率を上げる」という逆因果を生む。
3. field-level cellでrunnerのcentered winを平均すると構造上0に近くなる。current field compositionはrace-level error / entropy / winner probabilityで評価する必要がある。
4. history0は約10%あり、surface/distance/venue transitionがundefinedである。unknownをsame/switchのどちらかへ混ぜない。
5. 直線1000mやdirection空欄は通常courseと意味が異なる。direction switchの単一dummyでは吸収できない。
6. post × distanceの方向は見えたが、venue/course cell、枠番と馬番、取消後のstarter alignmentを分離していない。
7. jockey form × experienceはjockey main effectに支配され、interactionを足すとconnections groupの冗長性を増やす可能性がある。
8. PACE-02のfield pressureはfield sizeとSpearman .333であり、interactionを追加するとfield sizeの再表現になり得る。

## 9. Leakage / PIT considerations

- 全previous contextは`previous_date < target_date`を満たし、同日raceを前走にしない。同日全prediction emit後にstate更新する契約を維持する。
- current raceのsurface、distance、venue、direction、class、declared field memberはpre-race contextだが、歴史rawにはpublication timestampがない。取消反映時刻まで含む厳密なprospective PITとは主張しない。
- current result由来のfinish、clock、margin、last 3F、passing orderをpredictive inputへ使っていない。前走performanceは完了済み過去raceとしてだけ利用した。
- opponent strengthは相手のpre-race ratingに限る。future opponent resultやtarget horse自身をopponent-only平均へ混ぜない。
- OOT prediction loaderは`role=evaluation`かつyears `{2020,2021,2022}`だけを保持し、max date assert後に集計した。2023 rowは読み取りsourceに存在してもanalysis frameへ0行である。
- final odds、人気、市場oracleはjoinしていない。horse/jockey/trainer IDはjoin/state keyだけで、featureではない。
- 全期間normalizationを避け、jockey form / rating-spread cutはdiscoveryだけで固定した。

## 10. Modeling implications

| Representation | Decision from EDA | Reason |
|---|---|---|
| current surface/distance/venue/direction | retain | pre-race contextとして必要。transitionの意味を条件付ける |
| same-surface / distance-change suitability | redesign | 大量condition平均より、past stateのreliability / uncertaintyとして少数化 |
| class transition | redesign | promotion selectionとopponent qualityを分離する |
| rest / workload | retain then simplify | 非線形性は再現するが細かいbin探索を避ける |
| age × career starts | diagnostic first | career-stage compositionが強く、feature増分は未確認 |
| jockey form × experience | defer | connections main effectと冗長、interaction evidence弱い |
| rating spread × field size | residual diagnosis first | large/high-spread OOT lossは再現したが原因未分解 |
| old field-relative 15 columns | keep removed | 本EDAは単純復活を支持しない |
| normalized horse-number × course | redesign / preregister | 枠番とは別。course-level audit不足 |
| pace pressure × style | defer | PACE-02単体がinconclusive、事後interaction risk |

BinaryとLambdaRankはいずれもsame-conditionよりsurface switch、large distance change、history0でwinner lossが高かった。例としてdirt→turfのBinary winner lossはreplication / confirmationで2.679 / 2.891、LambdaRankは2.672 / 2.860、same dirtは2.118 / 2.074と2.120 / 2.058だった。これは両objective共通のinformation / representation gapを示唆し、model familyだけを替えれば解決する証拠ではない。

## 11. Candidate hypotheses

本章からready-for-experimentへ上げる候補は最大3件に絞る。ただしEDA終了時点では実行しない。

1. **EDA-G-01: transition-aware performance reliability**  
   `surface switch`または事前固定した`distance-change band`の一方だけを、既存past-performance stateのreliability interactionとして検証する。main-effect適性列の大量追加はせず、rolling-originでBinary / Rank両方のLog Loss・NDCG guardrailを見る。PIT risk low、temporal replication strong。
2. **EDA-G-02: rest × recent workload, low freedom**  
   14--29日内2走以上と短間隔/長期休養の再現構造を、1つの低自由度non-linear表現として事前固定する。age、class、history availability別OOT residualをguardrailにし、疲労の因果featureとは呼ばない。PIT risk low、confounding risk high。
3. **EDA-G-03: large-field high-spread error decomposition**  
   featureを直ちに追加せず、15頭以上×high rating spreadでwinner lossが高い理由を、rating coverage、class、history mix、winner rating rank、calibrationへ分解する。旧field-relative列を復活させない。PIT risk low、expected knowledge high。

`class × opponent strength`、`jockey × experience`、`normalized horse-number × venue-distance`、`pace pressure × style`は、現証拠では`proposed/defer`でありtop-3に入れない。

## 12. What not to conclude

- surface/distance switchのraw勝率低下から、その条件替わり自体が敗因だとは言えない。
- class down/upの差からclass transition dummyのproduction有効性や因果効果を主張できない。
- 14--29日・30日2走以上のnegative associationを疲労と断定できない。
- age 6+ / starts 10+のnegative associationから高齢馬をtrainingから除外すべきとは言えない。
- jockey formの強いmain effectから、さらにinteraction列を増やすべきとは言えない。
- large field × high spreadのlossから、rating spread feature、condition別model、calibration parameterのどれが正しい解決かはまだ分からない。
- post × distanceの記述からvenue/courseごとの枠順featureを選択できない。canonical supportとPIT publication semanticsが不足する。
- PACE-01が支持されたことからPACE pressure × styleまで支持されたとは言えない。PACE-02は両family inconclusiveである。
- 本結果は旧field-relative 15列の単純復活、2024でのinteraction search、2025評価、production model変更を正当化しない。
