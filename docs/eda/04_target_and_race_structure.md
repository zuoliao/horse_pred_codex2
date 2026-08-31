# Targets and Stochastic Race Structure

## 1. Questions

1. JRA flat raceを独立runnerの集合ではなく、一つの有限choice setとして扱うと、field sizeとbase rateはどの程度target semanticsを変えるか。
2. winner binary、Top3、graded relevance、full rank、rank percentile、time gap、margin、condition-adjusted performance residualは、同じ実現結果から何を保持し、何を捨てるか。
3. pairwise rankingとrace-wise choiceは、どの比較を多く数え、どの確率構造を直接表現するか。
4. 3着と4着の間に、時計差として特別な不連続が観測されるか。それともTop3 cutoffは主に意思決定上の離散化か。
5. dead heat、DNF、DQ、demotion、0.1秒時計丸めを、targetごとにどう扱うべきか。
6. discoveryで見つけた構造はreplicationとconfirmationでも同方向か。

本章はtargetの記述とshallow diagnosticであり、本格model、production feature、target mappingの採択を行わない。

## 2. Data scope

approved local raw `race_results_merged.csv`（SHA-256 `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`）を正規化し、`date <= 2022-12-31`をassertした。保持された最大target dateは`2022-12-28`、2023--2025 target rowは0である。2013はcondition modelのstate warm-upとquality確認にだけ使用し、target claimへ含めていない。

| 区分 | 期間 | 用途 | flat declared runner / race | strict choice runner / race | dates | 不確実性・区分 |
|---|---|---|---:|---:|---:|---|
| warm-up | 2013 | state/qualityのみ | 48,431 / 3,324 | — | — | target claim禁止 |
| discovery | 2014--2019 | 探索 | 285,917 / 19,925 | 271,740 / 18,966 | 658 | exploratory |
| replication | 2020--2021 | 時間再現 | 93,044 / 6,660 | 88,578 / 6,356 | 215 | replicated |
| confirmation | 2022 | 優先順位確認 | 45,810 / 3,331 | 43,537 / 3,176 | 108 | confirmed、単年 |

flatはrace全行のsurfaceが芝/ダートで、`race_class`に`障害`を含まないraceである。strict choice setは、flat、取消・除外なし、starterのみ、公式winnerが少なくとも1頭、を満たす。DNF/DQはstarterとしてchoice setへ残す。raw populationのstatus監査とstrict predictive denominatorを混ぜない。

期間別countは観測rawのcensusである。平均・率・連続値の95% intervalはrace dateを単位にseed `20260901`で1,000回bootstrapした。source欠損race、将来開催、同一horseの反復依存はintervalに含まれない。primary aggregationはrace macroで、pair総数を扱う表だけpair denominatorを明記する。

## 3. Definitions

- **choice set:** 同一`race_id`のstrict starter集合。winner選択確率はrace内で合計1を要求する。
- **hard winner binary:** 公式1着を1、他starterを0。1着同着raceではpositiveが複数になる。
- **coherent winner mass:** 1着同着が`m`頭なら各`1/m`、他0。race内target massは常に1。
- **Top3:** 公式着順`<=3`。同着によりpositive頭数が3を超える場合がある。
- **graded relevance:** 現行LambdaRankと同じ`1着=3, 2着=2, 3着=1, その他=0`。gainは別のmodel設定であり、ここではlabel情報だけを比較する。
- **full rank:** numeric official rankの全順序。DNF/DQはnumeric finisherより後のcensored tieとして扱い、相互の真の順序を捏造しない。
- **rank percentile:** `max(0, 1-(model_rank-1)/(field_size-1))`。field sizeを揃えるが、隣接順位を等間隔と仮定する。
- **time gap:** `runner clock - official winner clock`を秒/1000mへ換算。小さいほど良い。DNF/DQと時計欠損は未定義。
- **margin:** sourceの隣接着差token。時計の0.1秒丸めより細かい順序を含み得るが、tokenを距離や秒へ変換する規則は自明でない。
- **performance residual:** strictly earlier dateだけで推定した条件別expected winner秒/1000mからrunner時計を引く。正が速い。ここでは既存の固定prequential ridgeと`[-5,+5]` clipをshallow target diagnosticとして再構成した。
- **pairwise target:** race内の2頭比較。全pairを数えると1 raceあたり`n(n-1)/2`個になり、大頭数raceと下位同士が支配し得る。
- **race-wise choice target:** winner identity/massをchoice set全体で一度に扱う。coherent probabilityに直結する一方、2着以下の順序は捨てる。
- **entropy/capacity:** 本章のentropyはfield sizeとlabel encodingが作る記述的な不確実性・表現容量であり、競走のirreducible aleatoric noiseの推定ではない。

## 4. Methods

### Choice-set and status audit

各strict raceについてfield size、hard/coherent winner率、Top3頭数、dead heat、DNF/DQ、numeric-rank欠損を集計した。field-size bandは`<=9 / 10--13 / 14--16 / 17+`とし、base rateをrace macroで比較した。raw flat populationでは取消・除外raceを別に数えた。

### Information-retention diagnostics

winner/Top3 binaryとgraded labelについて、race内runner比率からBernoulli/categorical entropyを計算した。winner identity、unordered/ordered Top3、strict full orderについては、それぞれ`log2(n)`、`log2(C(n,3))`、`log2(n!/(n-3)!)`、`log2(n!)`を表現容量の上限として計算した。dead heat/DNFを持つraceではfull-order値は上限であり、実際の識別可能情報を過大評価する。

### Rank, clock, and margin continuity

numeric finisherについてwinnerからの秒/1000m gapを計算し、race内rank percentileとnegative gapのSpearmanをrace macro化した。隣接numeric rankの平均表示時計差を`1-2 / 2-3 / 3-4 / 4-5 / 5+`に分け、date bootstrap intervalを得た。equal displayed clockとnegative inversionも数えた。marginはnumeric nonwinnerをdenominatorにtoken頻度とblankを集計し、数値mappingはfitしていない。

### Fixed shallow performance residual

2013から日次emit-before-updateで51-dimensional reference-coded ridgeを更新し、course×surface、distance、going、class tier、race-age restrictionからexpected winner秒/1000mを推定した。alpha、cold gate、clipは既存SPEED-01定義のままで、target結果に合わせて変更していない。2013 residualは報告せず、2014--2022 strict choice setだけを集計した。これはcondition adjustmentの情報分解を見るdiagnosticで、predictive modelや新feature testではない。

### Pairwise and race-wise baselines

全pair、比較可能pair、winnerを含むpair、Top3を含むpairをexact countした。race-wise choiceのshallow nullはuniform probabilityで、Log Loss=`log(n)`、Brier=`1-1/n`とした。特徴量を使うmodel、permutation、production retrainingは実行していない。

## 5. Descriptive findings

### 5.1 Choice sets and base rates

| 期間 | strict race / runner | mean field [95% CI] | median [IQR] | coherent winner rate [95% CI] | Top3 rate [95% CI] |
|---|---:|---:|---:|---:|---:|
| discovery | 18,966 / 271,740 | 14.328 [14.267, 14.381] | 15 [13,16] | 7.290% [7.255, 7.330] | 21.889% [21.782, 22.011] |
| replication | 6,356 / 88,578 | 13.936 [13.845, 14.015] | 15 [12,16] | 7.556% [7.503, 7.614] | 22.692% [22.532, 22.870] |
| confirmation | 3,176 / 43,537 | 13.708 [13.577, 13.836] | 14 [12,16] | 7.701% [7.611, 7.792] | 23.107% [22.835, 23.386] |

field size低下に伴い、race-macroのuniform winner/Top3 base rateは両方上昇した。これはtarget prevalenceのdriftであり、能力予測の改善ではない。

| Field band | Discovery races | winner base | Top3 base | Replication winner / Top3 | Confirmation winner / Top3 |
|---|---:|---:|---:|---:|---:|
| `<=9` | 1,240 | 12.20% | 36.64% | 12.36% / 37.09% | 12.44% / 37.33% |
| `10--13` | 4,756 | 8.62% | 25.89% | 8.69% / 26.10% | 8.71% / 26.14% |
| `14--16` | 11,003 | 6.46% | 19.40% | 6.49% / 19.50% | 6.51% / 19.53% |
| `17+` | 1,967 | 5.61% | 16.85% | 5.62% / 16.91% | 5.62% / 16.85% |

**Observation.** field-size conditional base rateは3期間でほぼ同じだが、field-size mixが変化した。

**Interpretation.** runner-micro binary lossやaccuracyはfield-size構成に影響される。race-macro評価、race-wise probability normalization、field-size sliceを維持する必要がある。

### 5.2 Dead heat, DNF, DQ, and demotion

| 期間 | raw nonstarter races / flat races | strict any-dead-heat | first dead heat | DNF race | DQ race | missing-rank starters |
|---|---:|---:|---:|---:|---:|---:|
| discovery | 959 / 19,925 (4.81%) | 581 (3.06%) | 25 (0.132%) | 788 (4.16%) | 0 | 846 / 271,740 (0.311%) |
| replication | 304 / 6,660 (4.56%) | 192 (3.02%) | 16 (0.252%) | 252 (3.97%) | 1 | 271 / 88,578 (0.306%) |
| confirmation | 155 / 3,331 (4.65%) | 63 (1.98%) | 4 (0.126%) | 113 (3.56%) | 0 | 124 / 43,537 (0.285%) |

hard winner binaryのrace-macro positive rateは`7.299% / 7.578% / 7.712%`で、coherent massよりわずかに高い。差はfirst dead heatによりrace target massが1を超えるためである。

**Interpretation.** hard binaryはtraining labelとして同着者をwinner扱いできるが、race-wise確率評価では各co-winnerへ`1/m`を割り当てる必要がある。DNF/DQはwinner/Top3では負例、full rank/pairwiseではnumeric finisherより後のcensored outcome、clock/margin targetではmissingとする。demotion/DQ raceを連続時計targetへ無条件に入れると、各期間1件のadjacent clock inversionが残ったため、clean-clock変換では明示的除外が必要である。

### 5.3 What each target retains

| Representation | Retains | Discards / imposes | Exception contract | Initial assessment |
|---|---|---|---|---|
| winner binary | winner identity、hard event | 全loser差、race内sum=1 | first dead heatはmultiple positive | `P(win)`用にretain |
| coherent race choice | exclusive winner mass、choice set | 2着以下の順序 | co-winnerへ`1/m` | probability baselineとして優先 |
| Top3 binary | place inclusion、winnerより高いpositive density | 1/2/3差、4着以下 | 3着同着でpositive>3可 | auxiliary候補、単独主targetにしない |
| graded top3 | 1/2/3の順序とtop-heavy性 | 4着以下を全て0 | dead heat label共有 | 現行controlとしてretain、最適とは未証明 |
| full rank | 全numeric順序とtie | 着差、条件差 | DNF/DQはcensored last | raw rank MSEは禁止 |
| rank percentile | field-size normalized order | 等間隔仮定、gap magnitude | DNF/DQを0へcensor | descriptive/label候補のみ |
| time gap | 連続的なrace内performance差 | race全体の速さ、0.1秒以下 | DNF/DQ missing、demotion監査 | rich target候補 |
| margin token | sub-clock local separation、close finish | token→秒の自明な尺度なし | dead heat/特殊token別扱い | mapping固定前はtarget化しない |
| performance residual | race間condition speed + race内gap | estimator依存、pace/斤量等未補正 | clean clock、PIT fit必須 | auxiliary regression候補 |
| all pairwise | ordinal comparisons、tie | coherent probability、magnitude | censored/tied pairをmask | race weightingなしでは不適 |

## 6. Temporal replication

### 6.1 Label entropy and representation capacity

| Race-macro diagnostic | Discovery [95% CI] | Replication [95% CI] | Confirmation [95% CI] |
|---|---:|---:|---:|
| winner binary entropy, bits | .374 [.372,.375] | .383 [.381,.385] | .388 [.385,.391] |
| Top3 binary entropy, bits | .747 [.746,.749] | .759 [.757,.762] | .766 [.762,.769] |
| graded-label entropy, bits | 1.094 [1.090,1.097] | 1.118 [1.113,1.124] | 1.131 [1.123,1.140] |
| winner identity capacity, bits | 3.812 [3.805,3.818] | 3.767 [3.756,3.776] | 3.741 [3.726,3.756] |
| unordered Top3 capacity, bits | 8.514 [8.491,8.534] | 8.364 [8.329,8.394] | 8.279 [8.229,8.329] |
| ordered Top3 capacity, bits | 11.099 [11.076,11.119] | 10.949 [10.914,10.979] | 10.864 [10.814,10.914] |
| strict full-order upper bound, bits | 37.971 [37.744,38.174] | 36.511 [36.168,36.808] | 35.653 [35.167,36.132] |

binary/graded runner-label entropyはfield縮小でpositive率が上がるため増えた一方、winner identityやfull-orderのchoice capacityはfield縮小で下がった。この逆方向は、runner label entropyとrace outcome complexityを同じ「noise」と呼べないことを示す。

uniform race-wise choiceのLog Lossは`2.642 [2.638,2.647] / 2.611 [2.604,2.617] / 2.593 [2.583,2.604]` nats、Brierは`.9271/.9244/.9230`だった。後期の改善はfield size低下だけで発生するため、year間metric比較には同一baselineまたはpaired差が必要である。

### 6.2 Rank and gap continuity

time gapはnumeric strict startersの`99.689% / 99.694% / 99.715%`を覆った。winner gapを0とした秒/1000mの`q25 / median / q75 / q99`は、discovery `.278/.643/1.167/3.667`、replication `.313/.688/1.250/3.875`、confirmation `.300/.688/1.250/3.889`だった。

| Adjacent rank | Discovery mean [95% CI] | Replication | Confirmation | Denominator |
|---|---:|---:|---:|---|
| 1--2 | .139 [.137,.142] | .164 [.159,.169] | .169 [.163,.175] | 18,941 / 6,340 / 3,172 boundaries |
| 2--3 | .129 [.127,.131] | .144 [.139,.148] | .145 [.139,.151] | 18,904 / 6,329 / 3,166 |
| 3--4 | .112 [.110,.114] | .120 [.117,.123] | .122 [.117,.127] | 18,877 / 6,325 / 3,168 |
| 4--5 | .102 [.100,.104] | .112 [.109,.116] | .113 [.109,.117] | 18,867 / 6,316 / 3,169 |
| 5+ adjacent | .194 [.192,.196] | .207 [.203,.211] | .211 [.205,.216] | 175,163 / 56,252 / 27,435 |

3--4 gapは2--3より小さく4--5より大きいという滑らかな順序で、Top3境界だけの特別な断絶は観測されなかった。効果方向と大小関係はreplication/confirmationでも同じである。ただし`5+`は多数の異なるrank boundaryと大敗tailをまとめており、Top5と直接比較できない。

race-macro Spearman(`rank percentile`, `-time gap`)は`.9949 [.9948,.9949] / .9952 [.9950,.9953] / .9952 [.9950,.9954]`で非常に高い。time gapはrankとほぼ同じ順序を持ちながら、隣接差の大小を追加する。distinct adjacent rankのequal displayed clockは`23.85% / 21.95% / 21.76%`あり、0.1秒時計だけでは順序の細部を表せない。

### 6.3 Margin token replication

numeric nonwinner denominatorは`251,903 / 81,935 / 40,233`、blankは`14 / 4 / 1`だけだった。最多tokenは全期間`クビ`（19.39% / 18.21% / 18.13%）、次いで`1/2`（12.22% / 11.92% / 11.51%）である。`ハナ`は7.94%から6.72%、6.61%へ低下し、token frequencyには小さなdriftがある。

**Observation.** margin coverageは高く、equal-clock gapの補助情報を持つ可能性がある。

**Interpretation.** tokenは隣接局所差で、秒/距離のinterval尺度ではない。frequencyだけから`ハナ=.02秒`等を再推定したり、2022に合わせてmappingを調整してはならない。

### 6.4 Condition-adjusted performance residual

| Diagnostic | Discovery | Replication | Confirmation |
|---|---:|---:|---:|
| runner coverage | 99.612% | 99.620% | 99.685% |
| raw q01 / median / q99, sec/1000m | -4.054 / -.659 / +1.191 | -4.192 / -.659 / +1.283 | -3.979 / -.597 / +1.226 |
| abs(raw)>5 clip share | .407% | .430% | .403% |
| winner condition residual SD | .728 | .784 | .712 |
| within-race time-gap SD | .849 | .869 | .877 |

定義上、runner residualは`winner condition residual - within-race time gap`へexactに分解され、再構成max errorは全期間0だった。winner condition componentのSDはwithin-race gapと同じorderであり、condition adjustmentは単なるrank再符号化ではない。一方、残差は期待時計modelの仕様に依存し、因果的な馬能力targetではない。

## 7. Uncertainty

- count/rateは観測rawのcensusだが、2015/2017の既知missing racesと、それらが後続履歴へ与える影響はintervalに含まれない。
- date bootstrapは開催日clusterを扱うが、horse、jockey、trainerの期間横断依存やsource revisionを扱わない。
- confirmationは2022単年であり、効果量の最終確定ではなく方向確認である。
- entropy/capacityはtarget encodingとfield sizeの記述で、同条件raceを反復した潜在performance分散やirreducible noiseではない。
- full-order capacityは全馬がstrictly orderedという上限で、dead heatとDNF censoringを無視している。
- adjacent clock gapは公式0.1秒表示を使うため、equal clockでも真の走破時間が等しいとは限らない。
- margin tokenはほぼcompleteだが、tokenの物理距離・秒換算を本章では検証していない。
- performance residualのcondition modelは透明でPIT-safeだが、pace、斤量、track maintenance、風等を含まず、expected clockのmisspecificationが残る。
- 多数のtarget記述を比較しているため、sliceの小差を新target採択の証拠にしない。

## 8. Failure cases

1. **Runner-independent binary:** 同一raceの各runnerを独立Bernoulliとして扱い、予測確率和を拘束しない。field size driftでbase rateも変わる。
2. **Dead heat mass > 1:** hard co-winner labelsをrace-wise probability targetへそのまま使う。evaluationは`1/m` coherent massが必要。
3. **Top3を連続性の証拠なく絶対境界化:** 3--4時計差は滑らかで、4着以下が無情報という観測根拠はない。
4. **Raw rank MSE:** 1着差と10着差を等間隔とし、field sizeと着差を混同する。
5. **Unweighted all-pair loss:** 大頭数raceを`O(n^2)`で過重し、top-choiceと関係の薄いbottom-bottom pairを大量に数える。
6. **DNF/DQの架空順位:** numeric finisherの後というcensoringを超えて、DNF同士に任意順序を付ける。
7. **Demotion/DQ clockの無監査利用:** 公式rankと時計順のinversionをcontinuous targetへ混入する。
8. **Clockだけでfine orderを復元:** adjacent distinct ranksの約22--24%は同一表示時計である。
9. **Margin tokenを線形秒として扱う:** ordinal/local tokenを未検証のinterval尺度へ変換する。
10. **Full-period performance normalization:** future winner clocksや同日後続raceを過去residualへ混ぜる。
11. **2013をtarget evidenceへ含める:** 2013はstate warm-up/quality専用であり、discovery countへ入れない。

## 9. Leakage / PIT considerations

- winner、Top3、rank、clock、margin、statusは当該raceのoutcomeであり、本章ではtarget/analysis namespaceにのみ存在する。`runner_pre_race`へjoinしない。
- performance residualのcondition estimatorは、全raceを日次でpredictしてから当日winner clocksをupdateした。同日race順を利用せず、future row追加で過去値が変わらない形に限定した。
- 2013はcondition stateのwarm-upにのみ使用し、2013 residual、target rate、effect claimを報告していない。
- goingは完了済みpast raceのperformance normalizationにだけ使い、current target-race goingをhistorical pre-race featureへ昇格させない。
- final odds、popularity、payoutは一切joinしていない。marketとの対応は別workstreamのexplicit oracleに限る。
- pairwise展開は同一race内outcomeからだけ作り、他raceの未来順位やopponent future resultを使わない。
- 2023--2025 target rowは0。既知の2024 experiment metricをtarget選択根拠へ再利用していない。

## 10. Modeling implications

| Representation / decision | Judgment | Evidence | Confidence | Limitation | Recommended action |
|---|---|---|---|---|---|
| Binary `P(win)` | retain | betting decisionへ直接接続、choice set明確 | high | loser情報を全廃 | coherent race normalizationと併用 |
| Top3 binary | auxiliary only | positive率約22--23%、winnerよりdense | medium | 1/2/3と4+を潰す、3--4断絶なし | separate diagnostic target候補 |
| current graded Top3 LambdaRank | retain as control | top choiceを重視しつつ1/2/3を区別 | medium | 4+情報を捨てる | EDAだけで変更しない |
| raw full-rank regression | reject | rankはgapとfield sizeを表さない | high | ordinal modelなら別 | MSE targetにしない |
| rank percentile | descriptive / possible relevance | field-size normalization | medium | equal-spacing仮定 | 単独production採択なし |
| time gap | retain as rich outcome candidate | 99.7% coverage、adjacent magnitude保持 | high | condition speedを欠く、丸めtie | robust/condition-adjusted targetへ |
| margin token | defer mapping | nonwinner blankは19件だけ、close-order補助 | medium | interval尺度でない、frequency drift | predeclared mapping auditだけ |
| condition residual | prioritize as auxiliary target | cross-race SD .71--.78、coverage >99.6% | medium-high | estimator依存、因果能力でない | regression/Huberを独立実験 |
| all pairwise ranking | redesign weighting | pairの約61--63%はTop3非関与 | high | race-wise確率を直接出さない | race normalization/top weighting必須 |
| race-wise choice | prioritize transparent baseline | probability sum1、uniform nullが明示的 | high | lower-order情報なし | conditional logit/PL baseline候補 |

全pairwise countはdiscovery `1,875,914`、replication `597,595`、confirmation `289,333`で、winnerを含むpairは`13.49% / 13.78% / 13.96%`、Top3を少なくとも1頭含むpairも`37.42% / 38.12% / 38.56%`に留まった。full-order化は情報を増やすが、目的との整合を指定しなければ下位pairを主に増やす。

## 11. Candidate hypotheses

| ID | Question / proposed later test | Evidence and temporal replication | PIT risk | Multiple-comparison risk | Suggested status |
|---|---|---|---|---|---|
| EDA-B-01 | 同一feature/splitで、coherent race-wise conditional-logitまたはtop-choice PLはpost-softmax Binary/Rankerより確率品質を改善するか | choice set base rateとuniform nullが3期間で安定、dead-heat mass契約明確 | low | medium | ready-for-preregistration候補 |
| EDA-B-02 | condition-adjusted residualをHuber/regressionで別model化し、race内rankとwinner probabilityへ後段変換すると補助教師として有効か | coverage >99.6%、condition component SDが3期間でwithin-gapと同order | medium | medium | content workstream後の候補 |
| EDA-B-03 | current top3 relevanceに対し、1つだけ固定したgap-bin relevanceは4着以下の連続情報を保ちつつbottom-pair支配を避けるか | 3--4に断絶なし、clock continuityとpair compositionが3期間で再現 | medium | high | mapping根拠をcontent workstreamで固定後 |

これらはEDA hypothesis registryへ渡す候補であり、本章では実験していない。LightGBMの直接multi-task化を前提にせず、winner、Top3、performance targetを別々のestimandとして評価してから統合方法を判断する。

## 12. What not to conclude

- Top3 label entropyがwinner binaryより高いことから、Top3が「noiseの少ない正解」または必ず良い補助taskとは言えない。
- 3--4時計差に断絶がないことから、Top3評価指標や複勝意思決定が不要とは言えない。これはperformance continuityと意思決定境界が別であることを示す。
- rankとtime gapのSpearmanが約.995であることから、time gapが冗長とは言えない。順位が捨てる隣接差の大小とrace間speedを残す。
- performance residualの分散が大きいことを、潜在能力を正しく推定した証拠にしない。condition-model errorも含む。
- margin token coverageが高いことから、任意の秒換算や後付けmappingを正当化しない。
- all-pairwise targetの情報量が多いことを、winner予測に最適と解釈しない。大部分は下位同士の比較である。
- race-wise choiceが確率的にcoherentであることを、既存GBDTより高性能という結果に読み替えない。まだmodel比較をしていない。
- dead heat、DNF、DQがrareであることを、silent dropの理由にしない。target mass、group size、clock coverageを変える。
- discovery/replication/confirmationの記述的一致をproduction採択、因果効果、2024/2025 performanceの保証にしない。
