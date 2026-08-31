# Past-race Performance Content

## 1. Questions

1. Official finishだけでは失われる「圧勝」「僅差負け」「内容のある敗戦」を、clock、margin、last 3F、passing orderはどこまで記述するか。
2. 各result fieldのcoverage、例外、条件交絡、時間安定性は十分か。
3. 当該race内の説明力と、strictly later next startへの関連はどの程度違うか。
4. condition-adjusted time、track/day variant、margin dequantizationを後続仮説にできるか。どこからが恣意的mappingになるか。

## 2. Data scope

approved local raw（SHA-256先頭12桁`270923ce73c`）をchunkで読み、`date <= 2022-12-31`の行だけを正規化した。最大retained dateは`2022-12-28`、2023--2025 retained rowは0。2013はhistory warm-up/qualityだけで、target-aware tableには含めない。

| 区分 | 期間 | flat numeric finisher / race | flat starter / race | missing denominator | 集計単位 | 区分・不確実性 |
|---|---|---:|---:|---|---|---|
| discovery | 2014--2019 | 284,001 / 19,925 | 284,883 / 19,925 | status別に下記で分離 | runner・race・date | exploratory |
| replication | 2020--2021 | 92,444 / 6,660 | 92,732 / 6,660 | 同上 | runner・race・date | replicated |
| confirmation | 2022 | 45,519 / 3,331 | 45,649 / 3,331 | 同上 | runner・race・date | confirmed（単年） |

next-start分析は、target raceに取消・除外がなく、target horseがstarterであるchoice setに限定した。対象はdiscovery 271,740 runners / 18,966 races、replication 88,578 / 6,356、confirmation 43,537 / 3,176。過去performance自体は、target dateより前に完了したflat startだけから取得した。

## 3. Definitions

- **winner-relative clock gap**: numeric finisherの公式0.1秒時計から、そのraceの最速official winner時計を引いた秒数。
- **finish quality percentile**: race内official rankのaverage-tie rankを`1` best、`0` worstへ線形変換。DNF/DQはmissing。
- **signed time content diagnostic**: sole winnerは2番目の時計との差を正、nonwinnerはwinnerとの差を負にし、1000m当たりへ換算。demotion/DQ raceはmissing。本章ではlast-start診断にだけ用い、production featureを新規実装しない。
- **last-3F speed percentile**: race内で小さい秒数を速いとしてaverage-tie rankし、`1` fastest、`0` slowest。
- **early/final position percentile**: 2 segment以上のpassing stringの最初/最後の位置を、`1` frontmost、`0` rearmostへrace内rank化。
- **per-recorded-transition position change**: `(final percentile - early percentile) / (segment count - 1)`。正なら記録点あたりfieldに対して前進するが、記録点は等距離・等時間ではなくcourse間で同じ運動量を表さない。
- **next-start association**: horseの直前flat startの内容を、その後のtarget race内finish qualityと比較する。current target outcomeをfeature側へ戻さない。
- **clean order race**: demotion/disqualificationがなく、official orderとphysical clock orderの解釈が通常であるrace。

## 4. Methods

### Descriptive race structure

clock gapをofficial position別に比較し、4着がwinnerから0.3秒以内の割合、最速nonwinnerが0.5/1/2秒以上離れたrace、winnerと同時計のnonwinnerを集計した。marginは「直前の到達順位groupとの差」であり、cumulative winner gapとして扱わなかった。demotion/DQを除くdistinct-rank edgeで、token順序と表示clock gapのSpearmanを測った。

last 3Fとpassingは、raw秒数や絶対位置をrace間で直接比較せず、race-relative percentileを使った。当該raceのfinishとの関係は説明的なresult structureであり、predictive evidenceとは分離した。

### Strict next-start diagnostic

horseごとにstartを日付順に並べ、各content列を1 start shiftした。非missingなlagについて`previous_date < target_date`をassertした。同日flat複数startは0件だが、実装上は日次emit-before-updateを要求する。各target race内で、過去signalとcurrent finish qualityのSpearmanを求めrace macro平均した。単変量Top-1は、過去signal最大のtie集合にwinnerが含まれるmassを平均した。

### Expected-time diagnostic

discoveryのwinner timeだけでRidge診断をfitし、replication/confirmationへ固定適用した。distance-onlyと、`distance + venue + surface + going + canonical class + month`のcategorical modelを比較した。これはperformance target候補の残差構造を見るためのshallow diagnosticで、production speed figureではない。

### Uncertainty

race-level associationやbinary rateの95% intervalは、日付ごとの平均をseed `20260901`で300--500回block bootstrapした。大きなdescriptive tableのquantile/countは観測rawのexact censusとし、known missing raceやsource uncertaintyはinterval外である。多数のsliceはmultiple-comparison riskが高い。

## 5. Descriptive findings

### 5.1 Finish position compresses clock content

下表はwinner-relative gapの`median [q25, q75]`秒である。missing clockはnumeric finisherでは0（last 3Fのみdiscoveryで3件欠損）。

| Official position | discovery n / gap | replication n / gap | confirmation n / gap | aggregation・区分 |
|---:|---:|---:|---:|---|
| 2 | 19,937 / 0.2 [0.0, 0.3] | 6,654 / 0.2 [0.1, 0.3] | 3,333 / 0.2 [0.1, 0.4] | runner within race、replicated |
| 4 | 19,919 / 0.5 [0.3, 0.8] | 6,662 / 0.6 [0.4, 0.9] | 3,334 / 0.6 [0.4, 0.9] | 同上 |
| 8 | 19,695 / 1.1 [0.8, 1.6] | 6,538 / 1.2 [0.8, 1.8] | 3,258 / 1.2 [0.9, 1.8] | 同上 |
| 12 | 16,505 / 1.8 [1.3, 2.5] | 5,222 / 1.9 [1.4, 2.7] | 2,534 / 2.0 [1.4, 2.7] | 同上 |
| 16 | 8,960 / 3.0 [2.1, 4.3] | 2,634 / 3.1 [2.2, 4.4] | 1,211 / 3.2 [2.15, 4.5] | 同上 |

| Race-content event | discovery | replication | confirmation | date-block 95% CI / direction |
|---|---:|---:|---:|---|
| 4着がwinnerから0.3秒以内 | 6,070 / 19,919 = 30.47% | 1,626 / 6,662 = 24.41% | 817 / 3,334 = 24.51% | [29.80,31.15]→[23.26,25.22]→[22.96,25.91]、後期で低下 |
| first nonwinner gap >=0.5秒 | 2,648 / 19,925 = 13.29% | 1,170 / 6,660 = 17.57% | 626 / 3,331 = 18.79% | [12.80,13.78]→[16.58,18.43]→[17.88,20.37]、同方向増加 |
| 同 >=1.0秒 | 418 / 19,925 = 2.10% | 240 / 6,660 = 3.60% | 126 / 3,331 = 3.78% | exact observed、同方向増加 |
| 同 >=2.0秒 | 14 / 19,925 | 15 / 6,660 | 3 / 3,331 | rare、interval不安定 |
| winnerと同時計のnonwinner | 6,777 / 264,049 = 2.57% | 1,725 / 85,766 = 2.01% | 767 / 42,184 = 1.82% | [2.50,2.63]→[1.88,2.08]→[1.71,1.94]、低下 |

**Observation.** positionが下がるほどclock gapのlocationとspreadが増えるが、分布は大きく重なる。4着でも後期の約4頭に1頭は0.3秒以内である。同じ着順でもclose lossとwide-margin lossは別の実現値を持つ。

**Interpretation.** ordinal finishは安定した中心情報を持つ一方、raceの競争度とperformance magnitudeを捨てる。gapの時代差にはfield size、pace、condition、class compositionも混ざる。

**Hypothesis.** future performance targetでは、finishを捨てずにrace-relative clock residualを第2軸として扱うべきである。raw gap平均を無制限に使う根拠にはならない。

### 5.2 Margin token is ordered but not a physical clock

2013--2022 rawには32 nonblank tokenがあり、通常の`ハナ / アタマ / クビ / fractional lengths / 大`、`同着`、10種類（12 occurrences）のcompound `+` tokenを含む。2014--2022 numeric nonwinnerのblankは21 / 391,999で、demotion等の例外へ集中した。通常raceではtokenは前の到達groupとの差で、winner gapではない。

| Period | clean race / adjacent edge | equal-clock edge | equal-clock token counts | token-order vs displayed-gap Spearman | inversion / missing | 区分 |
|---|---:|---:|---|---:|---:|---|
| discovery | 19,909 / 263,242 | 62,598 (23.78%) | クビ28,913、ハナ20,889、アタマ12,796 | .9575 | 0 / 0 | exploratory |
| replication | 6,654 / 85,503 | 18,749 (21.93%) | クビ8,661、ハナ5,755、アタマ4,163、1/2 170 | .9570 | 0 / 0 | replicated |
| confirmation | 3,330 / 42,108 | 9,146 (21.72%) | クビ4,135、ハナ2,785、アタマ2,109、1/2 117 | .9542 | 0 / 0 | confirmed |

全numeric nonwinner denominatorはdiscovery 264,049、replication 85,766、confirmation 42,184。nonwinner margin blankは15、5、1、非blank vocabularyは29、24、22だった。rare tokenが後期に出ないためvocabulary sizeは減るが、主要tokenと順序関連は再現した。compound token countはdiscovery 9、replication 2、confirmation 0で、既存auditどおりdemotion/DQのorder anomalyと結び付く。

**Observation.** adjacent edgeのおよそ22--24%は0.1秒表示上同時計で、tokenはその内部に安定した順序情報を持つ。

**Interpretation.** tokenはclock quantizationを補完し得るが、horse lengthから秒への普遍的換算を同定しない。surface、pace、速度でlength-to-timeは変わり、cumulative gapには足し算規則も必要になる。

**Hypothesis.** marginは「ordered interval / uncertainty」として扱う余地があるが、連続秒mappingをproduction採用する証拠はない。既存PV-06も2022 Top-1 pointを改善した一方、primary Log Loss intervalが0を跨ぎinconclusiveだった。

### 5.3 Last 3F contains finish-related but distinct content

各値はraceごとのSpearmanをdate-blockでmacro集計した。対象raceでは少なくとも3頭が有効。numeric finisherのlast 3F欠損はdiscovery 3、replication 0、confirmation 0。

| Relation | discovery race / effect [95% CI] | replication | confirmation | temporal direction |
|---|---:|---:|---:|---|
| last-3F percentile vs finish quality | 19,925 / .7369 [.7347,.7409] | 6,660 / .7512 [.7475,.7560] | 3,331 / .7388 [.7318,.7456] | strong、stable positive |
| last-3F vs early frontness | 19,773 / -.3004 [-.3069,-.2958] | 6,614 / -.2764 [-.2865,-.2698] | 3,311 / -.2663 [-.2776,-.2526] | stable negative |
| last-3F vs final-corner frontness | 19,773 / -.1387 | 6,614 / -.1040 | 3,311 / -.1064 | negative、後期で弱い |
| last-3F vs position gain | 19,670 / .3183 | 6,568 / .3293 | 3,275 / .3075 | stable positive |

少なくとも1頭のwinnerがfield最小last 3F値を共有したraceは7,403 / 19,925 = 37.15% `[36.34,37.82]`、2,610 / 6,660 = 39.19% `[37.96,40.11]`、1,355 / 3,331 = 40.68% `[38.77,42.34]`。winnerが最小値を共有しなかったraceはそれぞれ62.85%、60.81%、59.32%である。0.1秒丸めによるtieを許す定義で、sole-fastest winner率ではない。winnerとnonwinnerが最小値を共有するraceでは両者がfastestになり得る。逆にwinnerがfield最大last 3F値を共有したのは16、6、5 raceだけだった。

winner-fastest rateは距離で安定して異なった。

| Distance band | discovery | replication | confirmation | race denominator / missing |
|---|---:|---:|---:|---|
| sprint <=1399m | 1,309 / 4,900 = 26.71% | 390 / 1,625 = 24.00% | 226 / 803 = 28.14% | race、last 3F missing 0 |
| mile 1400--1799m | 2,380 / 6,585 = 36.14% | 805 / 2,129 = 37.81% | 428 / 1,077 = 39.74% | 同上 |
| middle 1800--2199m | 3,122 / 7,203 = 43.34% | 1,193 / 2,476 = 48.18% | 587 / 1,241 = 47.30% | 同上 |
| long >=2200m | 592 / 1,237 = 47.86% | 222 / 430 = 51.63% | 114 / 210 = 54.29% | 同上；confirmation小標本 |

surface差はdiscovery dirt/turf 37.73/36.59%、replication 38.99/39.39%、confirmation 43.71/37.55%で、符号が安定しなかった。venue別にも33--44%程度の幅があり、距離・course compositionを調整していない。

**Observation.** last 3Fはfinishと強く関連するがwinnerだけの属性ではない。後方位置と速いlast 3F、相対的position gainと速いlast 3Fが共存する。

**Interpretation.** last 3Fは「能力」だけでなくpace配分、位置取り、展開を含む。絶対秒を距離/course間で比較するより、race-relative値とposition contextを分離した方が意味が明瞭である。

**Hypothesis.** sectional signalは単独のwinner proxyでなく、過去raceのrun-style/content axisとして検証すべきである。

### 5.4 Passing order has usable grammar and structural exceptions

| Period | starter / race | nonmissing / invalid | segment 1 / 2 / 3 / 4 runners | one-segment race | mixed-segment race | 区分 |
|---|---:|---:|---:|---:|---:|---|
| discovery | 284,883 / 19,925 | 284,626 / 0 | 2,608 / 152,635 / 16,778 / 112,605 | 231 | 209 | exploratory |
| replication | 92,732 / 6,660 | 92,657 / 0 | 784 / 48,103 / 5,783 / 37,987 | 70 | 61 | replicated |
| confirmation | 45,649 / 3,331 | 45,605 / 0 | 352 / 22,811 / 3,136 / 19,306 | 27 | 21 | confirmed |

missingは257、75、44 startersで、主にnon-finishに伴う。one-segment runnerのうちNiigata turf 1000mは2,521 / 2,608、759 / 784、344 / 352（96.7--97.7%）だった。残りはDNF等のpartial recordを含む。したがってone segmentの値を「first corner」と呼べない。mixed-segment raceも同じ理由を含み、segment count自体をcourse-independentな運動量へ変換できない。

| Current-race relation | discovery [95% CI] | replication [95% CI] | confirmation [95% CI] | race count |
|---|---:|---:|---:|---:|
| early frontness vs finish | .2170 [.2104,.2208] | .2215 [.2124,.2299] | .2445 [.2322,.2562] | 19,773 / 6,614 / 3,311 |
| final-corner frontness vs finish | .3929 [.3831,.3955] | .4053 [.3950,.4139] | .4177 [.4023,.4316] | 同上 |
| normalized gain vs finish | .3300 [.3225,.3318] | .3387 [.3287,.3473] | .3196 [.3046,.3335] | 19,670 / 6,568 / 3,275 |

final-corner positionはfinishに近いためearly positionより強いが、それ自体が将来にも強いとは限らない。position gainはfast last 3Fと相関するため、sectionalとの冗長性がある。

**Hypothesis.** stable semantics候補は、2+ segmentに限定したfirst-recorded position、last-recorded position、per-recorded-transition changeである。ただし記録点を無条件に序盤/cornerや等距離segmentと呼ばない。final/changeの追加価値は別々に検証し、straight 1000mはstructural missingにする。

### 5.5 Absolute time is condition-dominated; a residual remains

discoveryでfitしたshallow modelを固定してlater periodへ適用した。各periodのmissing winner clockは0。

| Diagnostic | discovery races / MAE / R2 | replication | confirmation | temporal behavior |
|---|---:|---:|---:|---|
| distance only | 19,925 / 1.884s / .9892 | 6,660 / 1.852s / .9896 | 3,331 / 1.928s / .9890 | distanceがvarianceの大半 |
| distance+venue+surface+going+class+month | 19,925 / 1.020s / .9963 | 6,660 / 1.092s / .9960 | 3,331 / 1.070s / .9962 | laterでも概ねreplicate |

condition residual meanはdiscovery 0.000s、replication -0.129s、confirmation -0.294sで、後期raceはdiscovery expectationより少し速かった。residual SDは1.394、1.474、1.395秒。venue-date平均residualのSDは0.550（1,726 venue-days）、0.629（577）、0.599（288）秒で、同一venue-day平均との差を引いたrunner-levelではなくrace-level残差SDも約1.28--1.33秒残った。

**Observation.** absolute winner timeは条件でほぼ決まるが、condition model後にもrace/dayと個別raceのvariationがある。

**Interpretation.** 高いR2はhorse predictionの成功ではなく、主にdistance scaleを回収した結果である。negative temporal biasはtrack speed、制度、composition、model misspecificationのいずれでも説明できる。

**Hypothesis.** expected-time residualは有望なperformance target候補だが、track/day variantは同じraceを自身のbaselineへ含めず、同日future raceをcurrent predictionへ流さない設計が先に必要である。

### 5.6 What persists to the next start

全signalは直前startからのみ取得した。`association`はtarget raceごとのSpearman macro mean、Top-1は少なくとも2頭にhistoryがあるraceでのtie-adjusted winner massである。Top-1のfield-size base rateは期間/coverageで異なるため、signal間の小差をformal testとして扱わない。

| Previous-race signal | discovery availability / association [95% CI] / Top-1 | replication | confirmation | temporal direction |
|---|---:|---:|---:|---|
| finish quality | 244,413 runners、17,309 races / .4166 [.4118,.4203] / .1813 | 79,489、5,784 / .3943 [.3857,.4009] / .1862 | 39,082、2,892 / .4124 [.4014,.4221] / .1959 | stable positive |
| signed time content | 244,203、17,318 / .3941 [.3889,.3976] / .1857 | 79,410、5,790 / .3690 [.3608,.3750] / .1883 | 39,072、2,895 / .3858 [.3699,.3944] / .2008 | stable positive |
| last-3F percentile | 244,412、17,318 / .3205 [.3170,.3253] / .1397 | 79,490、5,790 / .3089 [.3028,.3156] / .1455 | 39,082、2,895 / .3171 [.3043,.3295] / .1558 | stable positive |
| early frontness | 242,596、17,317 / .1117 [.1043,.1152] / .1040 | 78,938、5,790 / .1078 [.1002,.1154] / .1110 | 38,821、2,895 / .1191 [.1030,.1297] / .1178 | weak but stable |
| final-corner frontness | 242,596、17,317 / .1821 [.1758,.1854] / .1178 | 78,938、5,790 / .1790 [.1710,.1856] / .1286 | 38,821、2,895 / .1900 [.1714,.1990] / .1308 | stable positive |
| per-recorded-transition change | 242,596、17,316 / .1342 [.1306,.1388] / .0860 | 78,938、5,790 / .1314 [.1214,.1371] / .0915 | 38,821、2,894 / .1285 [.1170,.1367] / .0929 | weak, stable; course-independentではない |

signed time contentはfinishよりmean rank associationが少し弱い一方、univariate Top-1 pointは3期間ともわずかに高かった。last 3Fはそれらより弱いが独立に安定し、passing-derived signalも弱いながら同符号で再現した。

**Interpretation.** 過去race contentは次走まで残るが、ここではelapsed days、condition transition、opponent strength、field compositionを調整していない。incremental LightGBM valueやcausal abilityを示すものではない。

**Hypothesis.** race valueを単一ordinalへ潰すより、outcome/clock、sectional、position/styleを少数の分離軸として持つ検証価値がある。

## 6. Temporal replication

1. finish--clock関係は全期間で単調だが、close-fourthは30.5%から24.4--24.5%へ低下し、dominant-win thresholdは13.3%から17.6--18.8%へ増えた。固定gap thresholdの意味は完全にはstationaryでない。
2. margin tokenの順序--clock Spearmanは`.954--.957`、equal-clock edgeは21.7--23.8%で再現した。rare vocabulary countではなく主要token orderingが安定部分である。
3. last-3F current-race associationは`.737--.751`、next-start associationは`.309--.321`で方向・規模とも安定した。
4. passing-derived next-start associationはearly `.108--.119`、final `.179--.190`、gain `.129--.134`と弱いが符号は一致した。
5. discovery-fitted condition-time modelはlater MAE 1.07--1.09秒で持続したが、mean residualは負へdriftした。

したがって「race contentが存在する」はreplicatedだが、固定変換やproduction incremental valueはconfirmedしていない。

## 7. Uncertainty

- rawは観測censusだが、2015/2017のknown missing Kyoto raceと、その後のhorse history欠落を含まない。
- date-block bootstrapはrace-date clusterを扱うが、horseの反復、venue-day、jockey/trainer dependenceを同時には扱わない。
- 2022 confirmationは1年。surface/venue sliceの符号反転や小標本を一般化しない。
- current-race associationは同じoutcomeから作るdescriptive relationで、prediction evidenceではない。next-startだけが時間方向を持つ。
- next-start分析は直前start availabilityでselectionされ、debut、DNF、長期休養、condition switchを無調整で混ぜる。
- 6 signal、複数period、surface/distance/venueを探索したmultiple-comparison riskは高い。p-valueで候補を順位付けしていない。
- expected-time residualのmodel familyとcategory groupingは1案だけの診断で、best model探索をしていない。

## 8. Failure cases

1. **Demotion/DQ:** official rankとphysical clockの向きが反転し得る。compound `+` tokenもここへ集中し、通常mappingを適用しない。
2. **Dead heat:** duplicate official rank、winner複数、`同着` token carrierが非対称。winner separationは0とし、same-rank pairはneutralにする必要がある。
3. **DNF/nonstarter:** clock、last 3F、marginがnot-applicable。zero imputationは大敗と混同する。
4. **Extreme slow finisher:** 60秒超last 3F等を平均へ直接入れると支配的になる。source errorとも正当な完走とも未確定。
5. **One segment:** Niigata straight 1000mはcorner historyを持たない。single tokenをearly/final cornerへ二重使用しない。
6. **Mixed segment:** DNF/partial passingを通常の4-corner transitionと比較しない。
7. **Absolute seconds:** distance、surface、venue、going、class、season、paceを跨いで直接平均しない。
8. **Self-included track variant:** target raceまたは同日future raceの結果でcurrent raceを補正するとleakする。

## 9. Leakage / PIT considerations

- current raceのclock、margin、last 3F、passing、finishは本章のresult-structure記述にだけ使う。predictive viewでは、同日全runnerをemitした後にhistorical stateへ更新する。
- next-start lagは全非missing行で`previous_date < target_date`。current target resultはlag列へ入らない。
- condition-time modelは2014--2019だけでfitし、2020--2022へ固定適用した。full-period normalizationはない。
- completed past raceのgoing/clockを後日のtargetへ使うことは可能だが、current goingのavailabilityをこのrawだけで保証しない。
- venue-day variantを作るなら、全日結果は翌日以降にのみavailableとする。同日のlater raceをearlier race predictionへ入れない。
- odds/popularity、2023--2025、direct horse/jockey/trainer IDは使用していない。

## 10. Modeling implications

| Information | Decision | Evidence | Confidence | Limitation | Recommended action |
|---|---|---|---|---|---|
| official finish | retain | strongest last-start association `.39--.42` | high | magnitudeを捨てる | anchor outcomeとして維持 |
| within-race clock gap | retain, redesign around residual | close/wide outcomesとTop-1補完 | high | gap regime drift、pace交絡 | condition/time targetと比較 |
| raw margin token | defer mapping | ordered relationはstable、PV-06 inconclusive | high on order / low on seconds | physical scaleなし | interval/ordinal hypothesisに限定 |
| race-relative last 3F | retain as content axis | next-start `.309--.321`、distance pattern stable | high | pace/position交絡 | positionを分けた1仮説 |
| early position | retain as style axis | weak next-start `.108--.119` | medium | course/segment dependence | straight exceptionを固定 |
| final position / gain | simplify or defer | currentは強いがnext-startは弱い | medium | sectionalと冗長 | earlyと同時追加しない |
| absolute time | do not use raw | condition model MAEが約半減 | high | variant/PIT設計未固定 | simple expected residualをpreregister |
| venue-day variant | redesign before use | day component SD `.55--.63s` | medium | self/future inclusion | leave-one-race and next-date availability |
| multi-channel race value | test later | signal強度と意味が異なる | medium-high | incremental value未評価 | outcome/time/sectional/styleを分離 |

## 11. Candidate hypotheses

| ID | Question / later experiment | Evidence | PIT risk | Redundancy / multiple-search risk | Suggested status |
|---|---|---|---|---|---|
| EDA-D-01 | discovery-fitted condition expected-time residualはordinal finish/time gapを越えて安定するか | later MAE 1.07--1.09s、residual drift可視 | medium | PV-01/SPEED evidenceと重複 medium | ready-for-experiment候補 |
| EDA-D-02 | past last 3Fをposition/styleから残差化または2軸化すると次走signalを保てるか | current last3F--early `-.27---.30`、next-start `.31--.32` | low | SEC/PACEと重複 high | proposed、1表現に絞る |
| EDA-D-03 | finish/time、sectional、early styleの3軸race-value target/representationは単一rankより情報を保つか | 各軸のnext-start強度と意味が異なる | low | target設計の探索 high | proposed |
| EDA-D-04 | raw margin tokenをordered intervalとして扱う価値があるか | equal edge約22%、ordering stable、PV-06 LL inconclusive | medium | post-hoc mapping risk high | defer / low priority |
| EDA-D-05 | venue-day residualのleave-one-race estimateは翌日以降のperformance valueを改善するか | venue-day mean residual SD `.55--.63s` | high unless delayed | speed residualと重複 | conditional、PIT review必須 |

本章ではmapping、feature、target、modelを採用しない。全案をhypothesis registryへ送るだけとする。

## 12. What not to conclude

- last 3Fとfinishの`.74`相関から、current-race last 3Fを予測入力にできるとは言えない。これはresult同士の関係である。
- fastest last 3F horseが最も強い、またはslow winnerが誤記とは言えない。paceと位置取りが違う。
- signed time contentのunivariate Top-1がfinishを少し上回ることから、既存featureにincrementalであるとは言えない。
- condition-time modelのR2 `.996`を、winnerやrace outcomeの予測精度と解釈しない。distanceが支配するtarget scaleである。
- venue-day residualからcurrent-day track variantを無条件に作らない。self inclusionと同日future leakageがある。
- margin token orderから一意のseconds/length mappingを導かない。今回production mappingは採用していない。
- passing final/gainのcurrent associationを、そのまま将来predictive valueとみなさない。
- archived SEC/PACE/SPEED実験の存在は、Phase 5A中に局所feature searchを再開する指示ではない。
