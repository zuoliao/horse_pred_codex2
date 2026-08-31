# OOT model errors and final-market gap

## 1. Questions

本workstreamは、既存rolling-origin prediction artifactについて、(a) Binary/LambdaRankの2020–2022 OOT性能とcalibrationは時間方向にどう動くか、(b) winner rank、高信頼誤り、両familyの不一致はどこに残るか、(c) history、connections、opponent/field uncertainty、race conditionで誤りがどう変わるか、(d) final-odds marketとの差を当日情報欠如、modeling、target、calibration、selectionの候補へどこまで分解できるか、を記述する。production model、ensemble、feature採択、2024/2025参照は行わない。

## 2. Data scope

no-odds予測は`artifacts/eval_roll_001_current_best_20260831/predictions_scoring.csv.gz`（SHA-256 `0ee8db968f6a1d6be8165123efd0eeab19c546e76e95b5e1da91d0d608223b1a`）を50,000行ずつstreamした。各chunkから`role=evaluation AND evaluation_year in {2020,2021,2022}`だけを保持し、reject側のmetric集計を行っていない。source 703,270行のうち保持は264,230 runner-prediction行、2023/2024/2025保持行は0である。

| OOT year | races | runners per model | dates | Binary feature state | Rank feature state |
|---|---:|---:|---:|---|---|
| 2020 | 3,184 | 44,542 | 109 | PV-01 254列 | lean 253列 |
| 2021 | 3,172 | 44,036 | 106 | PV-01 254列 | lean 253列 |
| 2022 | 3,176 | 43,537 | 108 | PV-01 254列 | lean 253列 |

このartifactでは2019はroll-2020のcalibration yearであり、適切なevaluation foldではない。したがって2019 OOTを捏造せず、2020–2022の3 foldだけを固定した。また、このartifactは後に採択されたPACE-01 255列candidateではないため、本章のerror sliceを現行255列modelへそのまま同一視しない。

outcome/context/stateはapproved raw（SHA-256 `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`）を2022-12-31で物理filterして別途joinした。final odds/popularityはさらに独立した`market` tableに隔離した。

## 3. Definitions

- **race metric:** graded relevance `1着=3, 2着=2, 3着=1, other=0`のNDCG@3、co-winnerに`1/m`を配るTop-1 winner mass、race Log Loss、race Brier。
- **winner rank:** model確率降順におけるofficial winnerの平均順位。複数winnerは平均する。
- **high-confidence selection/error:** model top probabilityが0.30以上のraceをselection、そのtop horseがofficial winnerでないraceをerrorとする。これは賭けselectionではない。
- **disagreement:** BinaryとLambdaRankのtop horse IDが異なるrace。正解率はこの診断だけhard winner eventを用いる。
- **history:** winnerのstrictly-prior JRA平地startsを`0 / 1 / 2–3 / 4–9 / 10+`へ固定区分する。`0`は観測履歴なしであり、race classの「新馬」と同義ではない。
- **connections:** jockey/trainerのprior starts/winsから各`(wins+1)/(starts+20)`を計算し、その平均を`<.06 / .06–.08 / .08–.10 / >=.10`へ固定区分する。winner-sideのoutcome-conditioned診断である。
- **field uncertainty:** race内`mean(1/sqrt(horse_starts_pre+1))`。quartile cutは2014–2019 discovery raceだけで固定し、2020–2022へ変更せず適用した。
- **market oracle:** complete raceの各`1/final_win_odds`をrace内正規化した事後確率。締切前価格でも実行可能価格でもない。

## 4. Methods

予測artifactのcalibrated probabilityを再fitせず評価した。primary unitはraceで、平均・率の95% intervalはrace dateをblockとしてseed 20260901、1,000回bootstrapした。calibrationは固定bin `[0,.02,.05,.10,.20,.30,.40,1]`を使い、各runnerを`1/field_size`で重み付けしてrace寄与を揃えた。

horse/jockey/trainer stateは2013以降を日付順にemit-before-updateし、同日の結果を同日別raceへ見せていない。race condition sliceはsurface、class group、production定義と一致する`<1400 / 1400–1799 / 1800–2199 / >=2200m`、field-size bandを事前固定した。cell 100 races未満は表から除外した。

market oracleはprimary analysisにmarket列がないことをassertした後、別market tableを`race_id, horse_id`で明示joinした。全runnerでfinite odds >=1が揃うraceだけを対象とした。本対象では9,532 raceすべてcompleteだった。final oddsはsliceの記述以外に使わず、feature、calibration、acceptance、weight selectionには使っていない。

## 5. Descriptive findings

### 5.1 OOT core metrics

| Year | Model | NDCG@3 [95% CI] | Top-1 | Log Loss [95% CI] | Brier |
|---|---|---:|---:|---:|---:|
| 2020 | Binary | .4633 [.4531,.4744] | .2654 | 2.1764 [2.1457,2.2038] | .8473 |
| 2020 | Rank | .4661 [.4551,.4778] | .2649 | 2.1832 [2.1539,2.2135] | .8484 |
| 2021 | Binary | .4656 [.4550,.4763] | .2615 | 2.1576 [2.1319,2.1839] | .8425 |
| 2021 | Rank | .4567 [.4461,.4670] | .2563 | 2.1823 [2.1559,2.2090] | .8469 |
| 2022 | Binary | .4775 [.4668,.4885] | .2690 | 2.1154 [2.0832,2.1470] | .8391 |
| 2022 | Rank | .4745 [.4629,.4859] | .2612 | 2.1256 [2.0938,2.1566] | .8418 |

Binaryは3年すべてLog Loss/Brierで良く、NDCG@3は2020だけRankが上、2021/2022はBinaryが上だった。これはfamily差が存在しても一方向のranking優位ではないことを再確認する記述であり、新たなfamily採択ではない。

### 5.2 Winner rank and confidence

winner rankのmedianは両model・全年度で3。Binaryのwinnerがtop3外となるraceは43.75%→42.56%→41.69%、Rankは43.06%→43.69%→42.32%だった。p75/p90は2020–2021の6/9から2022の5/8へ改善した。

Binaryがtop probability >=.30を出す割合は37.28%/32.63%/33.94%、そのうち外す条件付き割合は65.54%/62.80%/63.17%。Rankは34.05%/26.45%/30.92%、条件付き外れ率65.22%/62.22%/63.54%だった。確率.30前後なら過半数が外れるのは数学的に自然であり、error countだけを過信の証拠にしない。calibration binとの併読が必要である。

### 5.3 Family disagreement

top choice不一致率は23.84%/24.68%/23.55%で安定した。不一致raceのhard top-1はBinary 19.76%/21.97%/21.66%、Rank 19.50%/19.92%/18.32%。両familyが異なる誤差を持つ余地はあるが、過去の固定50:50 ENS-01棄却を覆すensemble証拠ではない。

### 5.4 History, connections, uncertainty, and conditions

- winnerの観測JRA平地starts=0におけるBinary Log Lossは2.342/2.266/2.329で、一貫して難しい。ただしこれはrace classの新馬winnerを意味せず、地方・海外・障害歴も完全には表さない。startsとの関係も単調ではない。
- winner connection prior smoothed rateが`<.06`のBinary Log Lossは2.746/2.628/2.623、`>=.10`は1.647/1.679/1.710。方向はRankも同じ。ただしこれは「勝った馬のconnectionが事前に弱く見えたrace」を後から選ぶoutcome-conditioned sliceで、connection featureの因果効果や追加価値ではない。
- discovery固定のfield uncertainty quartileとBinary Log Lossは非単調だった。2022はlowからhighへ2.174/2.121/2.075/2.087で、平均field uncertaintyだけではcold-start難度を十分表さない。
- classでは新馬とopen/3winが一貫して難しい。Binary Log Lossは新馬2.320/2.249/2.290、open 2.300/2.377/2.385、3win 2.294/2.380/2.250。一方、未勝利は2.096/2.074/2.015だった。class、field size、winner history、connectionは相互に交絡する。
- 2022 surface差は小さく、Binary Log Lossはdirt 2.119、turf 2.112。distanceではsprint 2.194がmiddle 2.031より悪いが、field size/context未調整の記述である。

## 6. Temporal replication

core性能は2022へ向けてLog Loss/Brier/NDCGが概ね改善したが、Top-1は2021に一度低下した。BinaryのRankに対する確率優位は3/3年、NDCG優位は2/3年である。top-choice disagreementは約24%、winner rank medianは3、winner-history cold-startとlow connection sliceの難しさは三年間で同方向だった。

race-weighted fixed-bin ECEはBinary .00373/.00249/.00547、Rank .00299/.00161/.00401で、小さいが2022に悪化した。2020の`.40+` binはBinaryのmean .473にobserved .386、Rankの.479に.404でoverconfidenceが大きかったが、race-weight denominatorは各32.95/31.97と小さい。2021/2022には同程度の差が再現せず、高確率tailの年別変動を恒常的calibration defectと断定しない。

## 7. Uncertainty

- 2020–2022は既に研究参照されたscreening foldsで、sealed holdoutではない。既存metadata上のprior selection use下限も2020=2、2021=2、2022=6である。
- date bootstrapは開催日clusterを扱うが、model refit uncertainty、horse/connectionの年跨ぎ依存、source revisionを含まない。
- 3年しかなく、temporal direction consistencyは独立な有意性検定ではない。
- error sliceは多数あり、多重比較補正をしていない。connectionとwinner-historyはoutcome-conditionedで、selection biasを強く受ける。
- ECEはbin境界依存で、Log Loss/Brierの代替ではない。tail binはeffective race weightが小さい。
- market oracleはfinal oddsを真のlatent probabilityと仮定する比較ではない。pari-mutuel take、丸め、late information、群集誤差を含む。
- 2020–2022でfinal oddsはcompleteだったが、これは締切前取得可能性や実行可能性を意味しない。

## 8. Failure cases

1. **2019をevaluation扱い:** 現artifactではcalibration roleなのでOOT error claimに使えない。
2. **2023行の混入:** sourceには存在するがloader保持0を必須にした。2024/2025も0である。
3. **runner-microで大頭数raceを過重:** primary errorはrace macro、calibrationは1/field-size weightを用いる。
4. **高信頼error数だけを見る:** predicted .30なら期待外れ率は.70前後で、conditional rateとreliabilityを分ける必要がある。
5. **winner-side sliceの因果解釈:** low-history/connection winnerを後から選ぶため、feature ablation効果ではない。
6. **field uncertaintyの単調仮定:** quartile結果は非単調で、new-race、class、field-sizeを混ぜる。
7. **不一致から即ensemble:** diversityだけでは単体best超えを保証せず、ENS-01 negative resultがある。
8. **final oddsをselectionへ逆流:** market gapが大きいsliceを見てmodel/feature/weightを選ぶことは禁止する。

## 9. Leakage / PIT considerations

prediction loaderは各chunkでrole/year filter後の行だけを保持し、保持DataFrameに2023以降がないことをassertした。diagnostic horse/jockey/trainer stateは同日一括更新で、target当日結果をpre-stateへ含めない。direct entity IDはjoin/state keyだけで、modelへ入力していない。

primary analysis表にはfinal odds/popularity列が存在しないことをassertした。market tableは最後にexplicit joinし、oracle metricとmarket-gap sliceだけを生成した。final odds、人気、oracle gapはfeature、recalibration、candidate acceptance、ensemble weightへ使っていない。2024/2025 target outcomeは保持・集計していない。

## 10. Model–market gap decomposition

| Year | Market LL / NDCG@3 / Top-1 | Binary LL gap | Rank LL gap |
|---|---:|---:|---:|
| 2020 | 1.9949 / .5270 / .3208 | +.1814 | +.1883 |
| 2021 | 1.9471 / .5348 / .3272 | +.2105 | +.2353 |
| 2022 | 1.8932 / .5469 / .3445 | +.2222 | +.2325 |

これは加法的・識別済み分解ではなく、次の競合説明を切り分ける診断である。

| Candidate component | Evidence | What remains unidentified |
|---|---|---|
| 当日/追加情報の欠如 | winner starts=0のBinary market gapは.292/.357/.330、6+は.137/.187/.183。cold-startでmarket優位が一貫して大きい | final oddsが使う血統、調教、馬体重、当日馬場、取消等の個別寄与は未分離 |
| modeling / representation | 6+ startsでもgapは正、winner rank top3外が約42%。既知履歴が多くても取り切れていない | feature不足、GBDT表現、label noiseのどれかは未識別 |
| target / objective | Binary–Rank LL差は.0069/.0248/.0103とmarket gapより小さく、NDCG優位方向も年で変わる。不一致は約24% | supervised race-wise choiceやcontinuous targetの改善余地は未実験 |
| calibration | fixed-bin ECEは.0016–.0055で、tail overconfidenceは年依存 | ECEとmarket LL gapは同じ加法尺度ではなく、「残差=ranking」とは言えない |
| selection optimism / temporal drift | 3 foldは既参照、2022のprior useが多い。market自体も2020→2022で改善しgapが拡大 | 真の将来gapにはprospective/untouched期間が必要 |

したがって「market gapの主因はX%」とは定量分解できない。最も再現的な診断はcold-start winnerでgapが大きいこと、既知履歴6+でもgapが消えないこと、calibration誤差だけでは説明しにくいことである。

## 11. Candidate hypotheses

最大3件に限定し、いずれも2024/2025やfinal oddsを採否に使わない。

1. **EDA-H-01: cold-start information audit** — JRA平地history 0とrace class新馬を別々の固定sliceとし、connection分解、正確なrace条件、将来のPIT付き血統・調教・馬体重sourceのどれか1群ずつをrolling-originで検証する。新馬raceのtraining除外とは別仮説にする。
2. **EDA-H-02: supervised race-wise probability objective** — 同一feature/splitでconditional logit/Plackett–Luce baselineを比較し、Binary/Rank disagreementと確率差がobjective由来か診断する。marketやblend weightを使わない。
3. **EDA-H-03: preregistered error-slice guardrails** — new/open/3win、winner-history 0、field-size bandを次候補の監視sliceとして固定する。slice改善だけで採択せず、全体Log Loss/Brier/NDCG/Top-1をprimaryにする。

## 12. What not to conclude

- final marketが強いことから、final oddsをprimary prediction featureやcandidate selectionへ入れてよいとは言えない。
- market gapの拡大からno-odds modelが悪化したとは言えない。model core metricも同時に改善しており、market側の改善が大きい。
- JRA平地history 0のwinnerや新馬classが難しいことから、新馬戦をtrainingから除外すべきとは言えない。両者も同義ではない。
- connection sliceの大差からconnection列を増やすべきとは言えない。outcome-conditionedで既存130列との重複もある。
- ECEが小さいことからprobability問題が解決済み、またはgapが全てranking問題とは言えない。
- Binary/Rank不一致からensembleを再採択すべきとは言えない。
- 2020–2022の3 foldをuntouched final holdout、2019を追加OOT fold、final oddsを実行可能価格として扱わない。
