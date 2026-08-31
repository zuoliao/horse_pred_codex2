# Data Quality, Coverage, Missingness, and Drift

## 1. Questions

1. 2022年末までのrawは、race/runner key、型、単位、status、field sizeの点で内部整合しているか。
2. 欠損は偶発的か、それともfinish status、race type、時代、source coverageにより構造化されているか。
3. 2014--2019 discoveryで見た分布は、2020--2021 replication、2022 confirmationでも維持されるか。
4. 時代を識別できるほどのdriftがある場合、それは予測可能な競馬構造の変化か、制度・開催・表記の変化か。

## 2. Data scope

approved local raw `race_results_merged.csv`（SHA-256先頭12桁 `270923ce73c`）をchunkで読み、文字列の`date <= 2022-12-31`を満たす行だけを結合してから正規化した。正規化後に最大日付が`2022-12-28`、2023--2025行が0であることをassertした。raw/private rowは保存していない。

| 区分 | 期間 | 用途 | raw runner / race | flat declared runner / race | 欠損・denominator | 集計単位 | 不確実性・区分 |
|---|---|---|---:|---:|---|---|---|
| warm-up | 2013 | qualityとstate coverageのみ | 50,044 / 3,454 | 48,431 / 3,324 | official race欠損0 / 3,454 | runner・race | observed census、target claimなし |
| discovery | 2014--2019 | 探索 | 295,348 / 20,682 | 285,917 / 19,925 | official race欠損38 / 20,720（99.817% coverage） | runner・race・date | exploratory；欠損は京都に集中 |
| replication | 2020--2021 | 時間再現 | 96,103 / 6,912 | 93,044 / 6,660 | official race欠損0 / 6,912 | runner・race・date | replicated |
| confirmation | 2022 | 優先順位確認 | 47,220 / 3,456 | 45,810 / 3,331 | official race欠損0 / 3,456 | runner・race・date | confirmed（1年のみ） |

flatは、race全行のsurfaceが芝/ダートで、かつ`race_class`に`障害`を含まないraceとした。障害raceを`course_type`だけで判定しない既存契約を踏襲した。flat runner数は取消・除外を含むdeclared populationである。

## 3. Definitions

- **declared field**: rawにあるrace行数。取消・除外を含む。
- **starter field**: `finished / demoted / did_not_finish / disqualified`の行数。
- **PIT-C eligible race**: 取消・除外が1頭もないflat race。本章ではselectionの品質を診断するだけで、結果targetの選択には使わない。
- **cold-start runner**: 2013以降、target dateより前のflat start数が0のrunner。2013以前の履歴はrawにないため、真のcareer debutとは限らない。
- **missing**: 空文字、`--`、`---`。意味上の空欄（例：winnerの着差）とsource failureを分ける。
- **class group**: 表記差を除くため、新馬、未勝利、1/2/3勝相当、openへ正規化した診断用group。元の65 `race_class` tokenも別に監査した。
- **temporal direction consistency**: discoveryからreplication、confirmationへの差の向き。confirmationは1年なので「confirmed」は効果確定ではなく、優先順位確認を意味する。

## 4. Methods

### Integrity and coverage

既存の全期間data-health auditを単に再実行せず、同じparser/flat契約を2022 cutoff内へ適用した。`(race_id, horse_id)`、`(race_id, horse_number)`、race内context一意性、status parser、時計parser、距離・年齢・枠・馬番のsupportを確認した。official race数はfrozen manifestと比較した。

### Missingness

期間別rateに加え、finish status別rateとmissing-indicator間のphi相関を求めた。これにより、status由来、race type由来、履歴left truncation、source欠損を分離した。market/body-weight列はsource品質の確認にだけ用い、primary predictive viewへjoinしていない。

### Drift and uncertainty

race-level contextは1 race 1 row、runner分布はeligible statusを保ったrunner rowで集計した。平均field size等の95% intervalは、raceを日ごとに平均した後、日付blockをseed `20260901`で500回bootstrapした。年別値と1--99 percentileをばらつきとして併記した。多数のsliceは記述的・exploratoryであり、p-valueは用いていない。

### Adversarial validation

1 race 1 rowで、venue、surface、direction、distance、month、**declared** field、field平均age、strict-prior starts、rest、cold-start share、classを入力したbalanced logistic classifierを用いた。post-scratch starter fieldは入力していない。5-fold `StratifiedGroupKFold`で同一日をfold間に分けず、次を比較した。

1. 2014--2019 vs 2020--2021（26,585 races、873 dates）
2. 2014--2021 vs 2022（29,916 races、981 dates）

raw `race_class`版と、意味を保って表記だけ正規化したclass-group版を別にした。これはdrift診断であり、era classifier自体をfeatureにはしない。

## 5. Descriptive findings

### 5.1 Internal integrity

| Check | 対象期間・population | runner / race denominator | 異常・欠損count | 集計単位 | 不確実性・区分 |
|---|---|---:|---:|---|---|
| duplicate `(race, horse)` | 2013--2022 raw | 488,715 / 34,504 | 0 | runner key | observed exact、quality |
| duplicate `(race, horse number)` | 同上 | 488,715 / 34,504 | 0 | runner key | observed exact、quality |
| race内date/venue/surface/distance conflict | 同上 | 34,504 races | 0 | race | observed exact、quality |
| unknown status / venue / surface | 同上 | 488,715 runners | 0 / 0 / 0 | runner | observed exact、quality |
| non-empty clock parse failure | 同上 | 488,715 runners | 0 | runner | observed exact、quality |
| nonpositive distance、flat 1000--3600m外 | 同上 | 488,715 / 473,202 flat runners | 0 / 0 | runner | observed exact、quality |
| age 2--15、frame 1--8、horse no. 1--18外 | 同上 | 488,715 runners | 0 / 0 / 0 | runner | observed exact、quality |
| race horse numberが1..declared fieldでない | 同上 | 34,504 races | 0 | race | observed exact、quality |
| 同一horseの同日flat複数start | 同上 | horse-date | 0 | horse-date | observed exact、PIT check |

**Observation.** frozen schemaとkey/単位の基本整合性は高い。flat numeric finisher 470,080頭のclockは全件parseでき、距離、年齢、枠、馬番に契約外値はなかった。一方、last 3Fが60秒を超える極端に遅い完走はdiscovery 12、replication 4、confirmation 3頭あった。該当runnerは総時計も極端に遅く、単純な小数点ずれとは断定できない。

**Interpretation.** 一般的なparse failureは主問題ではないが、performance値を平均へ直接入れる際は、極端に遅い完走を「source error」と自動削除せず、race outcome/statusと一緒にrobust化する必要がある。

**Hypothesis.** 時計/上がりの後続実験では、raw clippingではなく、race-relative rank、robust residual、異常flagを事前定義して比較する価値がある。本EDA中にはproduction mappingを作らない。

### 5.2 Coverage and exceptional outcomes

| 年 | observed / official all races | shortfall | flat declared runner / race | nonstarter race rate | dead-heat race rate | denominator・ばらつき | 区分 |
|---:|---:|---:|---:|---:|---:|---|---|
| 2014 | 3,451 / 3,451 | 0 | 48,714 / 3,326 | 4.51% | 3.85% | race macro、observed exact | discovery |
| 2015 | 3,452 / 3,454 | 2 | 48,378 / 3,324 | 4.90% | 3.22% | 同上；2 raceは京都 | discovery |
| 2016 | 3,454 / 3,454 | 0 | 48,491 / 3,326 | 4.57% | 3.22% | race macro | discovery |
| 2017 | 3,419 / 3,455 | 36 | 47,278 / 3,296 | 4.31% | 2.67% | 同上；36 raceは京都 | discovery |
| 2018 | 3,454 / 3,454 | 0 | 47,054 / 3,328 | 5.35% | 2.85% | race macro | discovery |
| 2019 | 3,452 / 3,452 | 0 | 46,002 / 3,325 | 5.23% | 2.35% | race macro | discovery |
| 2020 | 3,456 / 3,456 | 0 | 46,716 / 3,331 | 4.41% | 3.12% | race macro | replication |
| 2021 | 3,456 / 3,456 | 0 | 46,328 / 3,329 | 4.72% | 2.91% | race macro | replication |
| 2022 | 3,456 / 3,456 | 0 | 45,810 / 3,331 | 4.65% | 2.01% | race macro | confirmation |

2014--2022 flat全体では33,240 races、473,202 declared runners、471,557 startersだった。取消・除外を含むraceは1,551（4.67%）、dead-heatを含むraceは978（2.94%）。2015/2017の既知欠損38 raceはすべて京都に集中し、うち35 raceがflatである。

**Observation.** 取消・除外rateは各年4.31--5.35%で持続し、単一期間固有の問題ではない。known source shortfallはdiscoveryにだけ存在する。

**Interpretation.** PIT-C eligibleだけを評価すると、約5%のraceを毎年選択的に外す。source欠損と取消除外selectionは異なる問題で、bootstrap intervalはいずれも復元しない。

**Hypothesis.** Phase 5Aの全error sliceは`source observed`と`PIT-C eligible`のdenominatorを併記し、nonstarter raceはpost-scratch oracle感度だけに分けるべきである。

### 5.3 Missingness is mostly structural

| 列 | discovery missing / 285,917 | replication missing / 93,044 | confirmation missing / 45,810 | 年内ばらつき・集計 | 分類 |
|---|---:|---:|---:|---|---|
| `around` | 0 (0%) | 0 (0%) | 0 (0%) | exact runner census | flatではcomplete；rawの空欄は非flat構造 |
| clock | 1,915 (0.670%) | 599 (0.644%) | 291 (0.635%) | discovery年別0.620--0.798%、runner | status構造 |
| margin | 21,849 (7.642%) | 7,261 (7.804%) | 3,622 (7.907%) | runner exact | winner-reference + status構造 |
| passing order | 1,284 (0.449%) | 385 (0.414%) | 204 (0.445%) | runner exact | status構造 |
| last 3F | 1,918 (0.671%) | 599 (0.644%) | 291 (0.635%) | discovery clockとの差3件 | status構造 + numeric finisher 3件 |
| final odds / popularity | 1,034 each (0.362%) | 312 each (0.335%) | 161 each (0.351%) | runner exact | nonstarter構造、market隔離 |
| body weight / delta | 552 each (0.193%) | 164 each (0.176%) | 63 each (0.138%) | runner exact | nonstarter中心、当日情報 |

status別に見ると、finished 470,058頭のclock/passing orderはcomplete、last 3F欠損は3頭、body weight欠損は1頭だけだった。対してDNF 1,475頭はclock/last 3F/marginが100%欠損、passing orderが29.83%欠損だった。scratched 771頭はこれらとodds/body weightが100%欠損、excluded 874頭はclock/last 3F/margin/oddsが100%、passing order 99.20%、body weight 9.73%欠損だった。finishedのmargin空欄7.07%は主にwinnerのreference blankであり、source failureではない。

missing-indicatorのphiはbody weight--delta 1.000、odds--popularity 1.000、clock--last 3F 0.9995、clock--passing 0.8140だった。これは独立した複数source failureより、同じstatus/eventにより列群が同時に未定義になる構造と整合する。

**Observation.** discovery、replication、confirmationのmissing rateは非常に近く、numeric finisherの結果contentはほぼ完全である。

**Interpretation.** generic imputation indicatorを多数作ると、情報源の欠損ではなくDNF/nonstarter identityを重複表現しやすい。historical DNFは正当な過去結果だが、current-race statusは入力にできない。

**Hypothesis.** historical performance viewでは`status`、`not_applicable`、`source_missing`を分離し、marginのwinner-reference blankをNAとは別codeにする。

## 6. Temporal replication

### 6.1 Population and history drift

| 期間 | runner / race | mean declared field [date-block 95% CI] | nonstarter race [95% CI] | mean strict-prior starts [95% CI] | cold-start runner | 区分 |
|---|---:|---:|---:|---:|---:|---|
| discovery 2014--2019 | 285,917 / 19,925 | 14.350 [14.315, 14.439] | 4.81% [4.46, 5.30] | 8.732 [8.580, 8.800] | 9.84% | exploratory |
| replication 2020--2021 | 93,044 / 6,660 | 13.971 [13.918, 14.076] | 4.56% [4.17, 5.25] | 8.898 [8.767, 9.027] | 10.21% | replicated |
| confirmation 2022 | 45,810 / 3,331 | 13.753 [13.629, 13.898] | 4.65% [3.83, 5.27] | 8.678 [8.525, 8.805] | 10.27% | confirmed |

field sizeは14.350から13.971、13.753へ同方向に低下し、date-block intervalもdiscoveryと後期でほぼ分離した。年別でも2014の14.646から2019の13.835、2022の13.753へ概ね低下しており、2017の京都欠損だけでは説明できない。一方、nonstarter rateとcold-start shareの変動は小さく、prior startsのperiod intervalは重なる。

厳密なprior startsは2014だけ平均6.39で、2015 8.40、2016 9.35となる。これは能力構造の変化だけでなく、rawが2013から始まるleft truncationを含む。2013は平均2.68、cold-start 22.37%であり、target claimに使用してはならない。

entity populationも年別に監査した。各値は当年flat declared runner（年別46,002--48,714頭、3,296--3,331 races）に現れるID/labelのexact distinct countで、欠損は0である。

| 年 | unique jockey ID | unique trainer label | 集計・不確実性 | 区分 |
|---:|---:|---:|---|---|
| 2014 | 182 | 252 | runnerから年内distinct、observed exact | discovery |
| 2015 | 170 | 243 | 同上 | discovery |
| 2016 | 176 | 244 | 同上 | discovery |
| 2017 | 193 | 250 | 同上；京都欠損あり | discovery |
| 2018 | 188 | 253 | 同上 | discovery |
| 2019 | 185 | 231 | 同上 | discovery |
| 2020 | 173 | 229 | 同上 | replication |
| 2021 | 146 | 228 | 同上 | replication |
| 2022 | 181 | 230 | 同上 | confirmation |

trainer populationは2019以降おおむね228--231で安定した。jockeyは2021に146へ低下したが2022に181へ戻っており、単調な縮小ではない。これはentity qualityや将来成績の変化を示さず、開催・騎乗populationのdrift診断に限る。

### 6.2 Context composition

| race-macro share | discovery 19,925 races | replication 6,660 | confirmation 3,331 | missing | 時間方向・区分 |
|---|---:|---:|---:|---:|---|
| turf | 50.19% | 49.32% | 49.17% | 0 | 小幅減、replicated |
| allowance-1相当 | 31.31% | 28.23% | 28.01% | 0 | 同方向低下、replicated/confirmed |
| allowance-2相当 | 12.91% | 14.26% | 14.02% | 0 | 上昇後維持 |
| open | 7.29% | 7.69% | 7.90% | 0 | 小幅上昇 |
| middle 1800--2199m | 36.15% | 37.18% | 37.26% | 0 | 小幅上昇 |
| Kyoto | 15.80% | 6.46% | 0% | discovery source欠損35 flat races | 開催制度による大drift |
| Chukyo | 8.44% | 11.89% | 16.06% | 0 | Kyoto減少と逆方向 |
| Hanshin | 14.52% | 17.66% | 18.94% | 0 | Kyoto減少と逆方向 |

raw class名は2014--2018の約49.4--50.2%が`500万/1000万/1600万下`表記、2020--2022の約48.4--48.9%が`1勝/2勝/3勝クラス`表記だった。2019が移行年である。意味groupへ正規化しないと、同じclass tierが時代labelになる。

### 6.3 Continuous distribution

対象はflat numeric finishers（discovery 284,001、replication 92,444、confirmation 45,519）。last 3Fのdiscoveryで3件を除き欠損0。値は`q01 / median / q99`で、runner aggregationの分布幅を示す。

| 値 | discovery | replication | confirmation | 解釈・区分 |
|---|---|---|---|---|
| distance m | 1000 / 1600 / 2600 | 1000 / 1600 / 2600 | 1000 / 1600 / 2600 | support安定 |
| age | 2 / 3 / 8 | 2 / 3 / 8 | 2 / 3 / 8 | support安定 |
| total clock sec | 58.6 / 98.1 / 160.8 | 58.7 / 98.5 / 161.9 | 58.9 / 98.7 / 161.8 | 小幅上昇、条件交絡あり |
| last 3F sec | 33.2 / 36.9 / 43.1 | 33.4 / 37.1 / 43.3 | 33.3 / 37.0 / 43.4 | 小幅、単位jumpなし |
| body weight kg | 402 / 470 / 540 | 400.9 / 470 / 542 | 400 / 472 / 540 | 概ね安定、当日oracle情報 |
| prior starts | 0 / 6 / 37 | 0 / 6 / 41 | 0 / 6 / 39 | tailのみ変動 |
| rest days | 7 / 29 / 301 | 7 / 35 / 314 | 7 / 35 / 328 | medianが6日延長、replicated |

absolute clockの小変化はdistance、surface、going、class、venue、season構成の交絡を除いていないため、馬が遅くなった証拠ではない。

### 6.4 Adversarial validation

| 比較 | class表現 | races / dates | ROC-AUC mean ± fold SD | aggregation / missing | 区分・multiple-comparison risk |
|---|---|---:|---:|---|---|
| 2014--2019 vs 2020--2021 | raw token | 26,585 / 873 | 0.8666 ± 0.0083 | race、numeric median impute | exploratory diagnostic、高 |
| 同上 | canonical group | 26,585 / 873 | 0.6186 ± 0.0170 | race、同上 | replicated drift diagnostic |
| 2014--2021 vs 2022 | raw token | 29,916 / 981 | 0.7949 ± 0.0051 | race、同上 | confirmation diagnostic、高 |
| 同上 | canonical group | 29,916 / 981 | 0.6443 ± 0.0088 | race、同上 | confirmation diagnostic |

raw-token classifierでは旧/新allowance labelが最大係数を占めた。canonical版では、前者比較は京都減少、mean age、left-truncated prior starts、field size、restが主な識別要因だった。2022比較では京都0件が突出した。

**Observation.** 時代はかなり識別可能だが、大部分はclass表記変更と京都開催休止という制度的変化である。意味正規化後もAUC 0.62--0.64の中程度driftが残り、field size、age、rest/history coverageが寄与する。

**Interpretation.** random splitやraw category memorizationは楽観的になり得る。adversarial score自体をfeatureにしても、この問題は解決しない。

**Hypothesis.** stable semantic class、venue/schedule-aware year slice、history observation spanを用いたrolling validationを維持すべきである。

## 7. Uncertainty

- ここでのcount/rateは観測rawのcensusであり、抽出sampling errorはない。ただし2015/2017の38 missing racesと、それらを過去走に持つ後続runnerへのhistory欠落はintervalに入らない。
- date-block intervalは開催日clusterを扱うが、horse/jockey/trainerの複数年依存とsource selectionは扱わない。
- confirmationは2022単年で、符号の確認以上の強い一般化はできない。
- adversarial AUCの`±`は5 foldの標準偏差で、confidence intervalではない。候補比較を増やしたmultiple-comparison riskは高い。
- class-groupは診断用の粗い意味正規化で、gradeやrace条件の完全な同値性を保証しない。
- 2014のhistory stateは2013しか観測していない。`career`と呼ぶとleft truncationを能力差と誤解する。

## 8. Failure cases

1. **Surfaceだけのflat判定:** 障害classでも芝/ダート表記があり、既存auditで実害が確認済み。race-class併用が必須。
2. **Raw class token:** 2019の制度表記変更を時代labelとして学ぶ。raw-token adversarial AUC 0.866が典型。
3. **Kyotoをsource欠損と一括解釈:** 2015/2017の欠損と、2021--2022の開催休止は別現象である。
4. **欠損を一種類のNAとする:** winner margin、DNF、scratch、source failureの意味が逆になる。
5. **2013/2014 prior startsをcareer値とする:** raw開始前の履歴を0扱いし、時代差を捏造する。
6. **極端値を機械的に誤記扱いする:** last 3F >60秒の19件（2014--2022）は極端な総時計と対応しており、少なくとも単純なunit errorとは確認できない。
7. **Runner-random adversarial split:** race/date clusterを壊し、fieldの複製contextでAUCを水増しする。

## 9. Leakage / PIT considerations

- strict-prior starts/restは`performance_date < target_date`で作り、同日raceを更新に使用していない。同一horseの同日flat複数startは0件だったが、実装契約は日次emit-before-updateを維持する。
- clock、margin、last 3F、passing order、finish statusは当該raceの結果である。本章ではquality denominatorにだけ用い、`runner_pre_race`へ入れない。
- weather、going、body weightもresult-page由来で、historical availability timestampが保証されない。当日情報としてproduction featureに昇格させない。
- final odds/popularityはmissing auditでもmarket-oracle列として物理隔離し、adversarial classifierへ入れていない。
- direct horse/jockey/trainer IDはkeyとunique population countにだけ使い、classifier featureにはしていない。
- 2023--2025のretained rowは0。2024/2025の値や既知metricをdrift比較に使っていない。

## 10. Modeling implications

| 現表現 | 判断 | Evidence | Confidence | Limitation | Recommended action |
|---|---|---|---|---|---|
| flat population filter | retain | class併用なしでは障害を混入 | high | race-class source依存 | 二重assertを維持 |
| raw race-class token | redesign | 表記だけでera AUCが大きく上昇 | high | grade情報がrawにない | stable tier + restrictionを分離 |
| field size | retain + monitor | 14.35→13.97→13.75、date-block CI分離 | high | scheduleとの交絡 | race-relative化とyear sliceを維持 |
| `career` history | redesign semantics | 2014平均6.39はleft truncationの影響 | high | 真のpre-2013 historyなし | observed starts/span/uncertaintyを併記 |
| generic missing flags | simplify | missing clusterはstatusでほぼ説明 | high | historical status自体はsignalになり得る | not-applicable/source-missingを分離 |
| absolute time/last 3F | defer to content EDA | supportは安定、条件交絡と極端tailあり | medium | condition補正未実施 | robust race-relative検討へ送る |
| random/CV evaluation | retain prohibition | canonicalでもera AUC 0.62--0.64 | high | linear diagnosticのみ | rolling/grouped temporal validation |
| adversarial score | do not add | drift原因の多くが制度/表記 | high | 非線形drift未網羅 | diagnostic専用 |

## 11. Candidate hypotheses

| ID | Question / proposed later test | Evidence and temporal replication | PIT risk | Multiple-comparison risk | Suggested status |
|---|---|---|---|---|---|
| EDA-A-01 | raw classをstable tier、age/sex restriction、旧新label flagへ分解するとera依存を減らせるか | raw/canonical AUC差0.248、0.151；両比較で再現 | low | low | ready-for-experiment候補 |
| EDA-A-02 | observed history span、pre-race starts、left-truncation uncertaintyを分離するとearly-year driftを説明できるか | 2014 prior starts 6.39、2015 8.40、以後約8.7--9.6 | low | medium | proposed |
| EDA-A-03 | historical result missingを`status / not-applicable / source-missing`へ分解すると冗長missing signalを減らせるか | status条件付き欠損とphi 0.81--1.00、全期間で持続 | low | low | ready-for-experiment候補 |
| EDA-A-04 | schedule/venue regimeごとのerrorをrolling OOTで分けると2022 gapを説明できるか | Kyoto 15.8%→6.46%→0%、canonical 2022 AUC 0.644 | low | medium | model-error workstreamへ |
| EDA-A-05 | field-size低下に対してrace-wise weighting/normalizationの効果は安定か | field平均が同方向に0.60頭低下、CI分離 | low | medium | target/model EDAへ |

これらはobservationから登録する仮説であり、本章の記述統計だけで有効なproduction changeとは認定しない。

## 12. What not to conclude

- adversarial AUCが高いから「2022を予測できない」、またはera classifierをfeatureにすべき、とは言えない。
- class raw tokenのdriftは能力分布の変化ではなく、主に制度表記の変更である。raw labelの強いimportanceを競馬signalと解釈しない。
- clock/last 3Fの小さな中央値変化から、馬のspeedやpaceが経年的に変化したとは言えない。
- missing rateが安定していることは、known missing Kyoto racesや取消race selectionが無害であることを意味しない。
- 2014のprior startsが少ないことを、若い馬・経験の少ない馬が多かった証拠にしない。
- rare extreme last 3Fを誤記または正当なperformanceと断定しない。source照合なしでは両方が残る。
- 本章はdata qualityとdistribution driftを確認しただけで、feature有効性、target選択、model採否を決めていない。
