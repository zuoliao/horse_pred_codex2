# Opponent and field structure

## 1. Questions

本workstreamは、(a) 対象馬を含むfield集約と除くopponent-only集約が何を表すか、(b) 過去raceの相手水準をcareer/recentでどう保持できるか、(c) current fieldのability・spread・経験構成・surface switcher・front候補が独立した構造を持つか、(d) 過去走内容と相手水準を1数値へ潰さず2軸で扱う根拠があるか、を記述する。結論はEDA上の仮説までであり、本格model、production feature、2024/2025 outcomeによる選択を行わない。

## 2. Data scope

approved rawは`/Users/zuoliao/dev/horse_ai/oumasan/raceinfo/race_results_merged.csv`、SHA-256は`270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`である。chunkごとに`date <= 2022-12-31`を適用してから正規化し、629,967 raw行中488,715行を保持した。保持範囲は2013-01-05～2022-12-28、2023以降の保持行は0である。2013はstate warm-upだけに使いtarget claimをしない。

primary denominatorはflat、取消・除外なし、starterのみ、公式winnerありのstrict choice setである。

| period | dates | races | runners |
|---|---:|---:|---:|
| discovery 2014–2019 | 658 | 18,966 | 271,740 |
| replication 2020–2021 | 215 | 6,356 | 88,578 |
| confirmation 2022 | 108 | 3,176 | 43,537 |

1 raceを1 choice setとし、race-level量はraceを分母、horse-level履歴量はrunnerを分母にする。intervalは明記しない限りrace dateを1 blockとして1,000回bootstrapした95%区間である。

## 3. Definitions

pre-race abilityはordinal pairwise Elo（初期値1,500、K=24、scale=400）を同日一括更新で再構成した。current raceのrating列を`r_i`、頭数を`n`とすると、self-inclusive field meanは`mean(r)`、対象馬を除くopponent meanは`(sum(r)-r_i)/(n-1)`である。opponent top-k/max/spreadも各対象馬を除いた集合から計算する。field competitivenessはrating-softmaxの正規化entropy、effective contenders `1/sum(p_i^2)`、rating spreadで記述する。uncertainty proxyは`1/sqrt(starts_pre+1)`、coldはstarts=0、experiencedはstarts>=3とした。

historical opponent observationは各完了済み過去raceにおける対象馬以外のpre-race rating平均であり、そのraceの対象馬の着順、時計、将来の相手成績を使わない。career、last-1/3/5、90日半減指数平均、`decay90-career` trendをemit-before-updateで作る。surface switcherは直前の既知surfaceとの不一致で、初出走はmissingである。front candidateは凍結済みPACE-01履歴が0.5を超える馬、front pressureは`sum(max(PACE01-0.5,0))`とした。

performance axisはSPEED-01と同じ51列のcondition design、ridge alpha 1.0、510 clean-race cold start、同日一括更新、±5秒/1,000m clipによる当該runnerの`expected winner clock - runner clock`である。これはoutcome-sideの分析量で、field-quality axisのpre-race field meanとは結合しない。

## 4. Methods

全pre-race stateは日付順にemitし、その日の全raceをemitした後にだけ更新した。current opponent表現は代数的同値性、within-race Spearman、inclusiveとの差の分布を確認した。historical表現はcoverage、career/recent/window/decay間のrank相関、career opponent-onlyとcareer inclusiveとの差を期間別に比較した。current fieldはclass groupとfield-size bandでも集約し、performance residualとの関係はdateごとのSpearmanを平均し、同じdate-level estimandをdate-bootstrapした。これは因果効果でもモデル増分価値でもない。

探索を広げないためmapping/HPO/interaction searchは行わず、mean/top-k/max/spreadは意味論の比較に限定した。旧field-relative 15列の再投入、future opponent result、current outcomeから作るfield-quality、同日後続race情報は使用していない。

## 5. Descriptive findings

### Observation

- current opponent meanは定義上`(field_sum-self)/(n-1)`である。ratingにtieがなく相関を定義できたraceでは、self ratingとのwithin-race Spearmanは全期間で平均-1.000だった。inclusive meanとの差は全runner平均がほぼ0だがSDはdiscovery 1.912、replication 1.880、confirmation 1.896 Eloである。global相関は正（0.554、0.624、0.607）でも、race内では自己能力を符号反転している。
- opponent top-3とinclusive top-3の差は全期間でmedian 0、平均は-0.990、-1.037、-0.972 Eloだった。opponent maxは対象馬が唯一の最大ratingのときだけsecond maxへ落ち、opponent spreadも対象馬が端点かで変わる。これらはpure race-level strengthではなく、対象馬のfield内位置を機械的に混ぜる。
- historical opponent履歴のany-history coverageは90.10%、89.90%、89.91%、3走以上は72.36%、71.45%、71.47%、5走以上は58.39%、57.33%、57.54%だった。約10%のcold-startは構造的欠損である。
- historical career opponent-onlyとcareer inclusiveの差は平均-0.129/-0.108/-0.103 Elo、SD約1.00で小さい。career opponentと90日decayのSpearmanは0.977/0.980/0.980、last-3とlast-5は0.985/0.987/0.987で、窓を増やしても強い冗長性がある。`decay90-career`のmedianは1.895/1.996/1.996 Eloだが、rating levelの時間変化も含むため純粋な「最近相手が強化」の証拠ではない。
- field rating meanは1,518.36/1,519.89/1,518.77、field spread平均は79.14/74.11/72.46 Eloだった。正規化entropyは0.99625 [0.99620, 0.99629]、0.99656 [0.99648, 0.99664]、0.99666 [0.99656, 0.99676]で、raw Elo softmaxはかなり平坦である。一方、top-rated horseのtie補正winner率は15.95% [15.48, 16.42]、14.96% [14.07, 15.88]、15.59% [14.26, 16.82]だった。
- classで構造差が大きい。新馬は全期間でfield mean 1,500、spread 0、cold share 100%であり、ordinal horse ratingだけではfieldを区別できない。2022のfield mean/spreadは未勝利1,500.34/51.76、1勝1,515.87/87.75、2勝1,538.55/108.01、3勝1,560.92/110.04、open 1,560.81/95.01だった。field meanはclass情報と強く重なるので、それ自体を新規signalと見なせない。
- cold shareは10.12%/10.23%/10.36%、experienced shareは72.21%/71.25%/70.82%。surface switcher shareはprior surface既知raceを分母に13.31% [13.13, 13.48]、12.79% [12.50, 13.10]、12.88% [12.50, 13.25]だった。
- PACE履歴既知率は89.85%/89.76%/89.63%。front candidate数は平均6.67/6.50/6.38、front pressureは1.368/1.345/1.322で、方向は緩やかに低下した。ただしこれはPACE-01から決定論的に得るfield compositionの記述で、追加signalの証明ではない。
- performance residual coverageは99.61%/99.62%/99.69%（270,686/88,241/43,400 runners）。pre-race field meanとのdate-level Spearman平均は0.221、0.156、0.181で、date-block 95%区間は[0.209, 0.233]、[0.136, 0.176]、[0.151, 0.212]だった。strong-fieldとgood-performanceの同時割合は12.63% [12.17, 13.10]、12.44% [11.71, 13.23]、13.14% [12.00, 14.38]である。

### Interpretation

current opponent-only meanは「相手だけ」を名乗れても、race内ではself ratingの逆変換を含む。race-constantなfield meanとself abilityを別列として扱う方が意味が明確である。historical opponent qualityには過去race価値という意味があるが、career/recent窓は互いにもinclusive履歴にも非常に近く、同時投入の根拠は弱い。performance residualとfield qualityの正相関は三期間で同方向なので2軸を保持する価値はあるが、class/context残差、selection、rating scale driftを含み、予測増分とは読めない。

## 6. Temporal replication

主要な方向は三期間で一致した。(1) current selfとopponent meanのwithin-race関係は-1、(2) historical careerとdecay90は0.97超、(3) cold shareは約10%、experienced shareは約71–72%、switcherは約13%、(4) field meanとperformance residualの相関は正、である。confirmationだけを根拠に定義を調整していない。

量的な非定常性もある。field spreadは79.14から74.11、72.46へ低下し、front pressureも1.368から1.345、1.322へ低下した。top-rated hitはreplicationでやや低い。class構成、頭数、COVID期、rating stateの累積期間が混ざるため、絶対levelを固定閾値として使うべきではない。field-size band別entropyは各期間で概ね0.9957～0.9972で、頭数増加に伴う機械的変化もある。

## 7. Uncertainty

race-macro率の95%区間はrace date block bootstrap 1,000回、seed 20260901で算出した。performance相関のpointと区間はともに各date内Spearmanの平均であり、dateを再標本化する。同一horse/connectionの長期依存は残るため、区間は方向の頑健性診断として読む。多数のclass/window/composition記述を同時に見ており、多重比較補正をしていない。

ordinal ratingは真のlatent abilityではなく、初期値1,500、既往数、class移動、履歴期間に依存する。特に新馬は全馬tieで、within-race correlationを定義できないraceがある。top-rated hitのdenominatorは全strict races、tieはwinner該当数/max-tie頭数で按分した。opponent-only履歴は相手のpre-race stateの推定誤差を継承する。

## 8. Failure cases

- 新馬はrating mean/spread/competitivenessが全て初期値由来で区別不能である。除外理由ではなく、connection・pedigree等の別情報源が必要という欠損構造である。
- 対象馬を除くmean/top-k/max/spreadは、対象馬がfieldの中央か端かで変わる。大量のrelative変形へ展開すると同じself-vs-field関係を重複表現する。
- PACE履歴、opponent履歴、prior surfaceはいずれも約10% missingで、主に初出走で共起する。0埋めは「中立」と「未知」を混同する。
- performance residualが欠けるのは0.31～0.39%で、condition不明、clean-clock条件外等のoutcome-side欠損である。これをpre-race field-quality欠損と混同しない。
- entropyは現scaleでほぼ1に集中し、単独の競争均衡featureとしては分解能が弱い。scaleを結果に合わせて調整する探索は本EDAの範囲外である。

## 9. Leakage / PIT considerations

rating、historical opponent、PACE、surface switchはすべて`history_date < target_date`でemitし、同日raceは一括更新した。opponent observationは相手の当該race前ratingだけを使い、相手の当該race結果・後続成績を使わない。対象馬の当該race結果もfield-qualityへ入れていない。performance residualは明示的にanalysis-only outcome namespaceであり、同じraceの予測入力にしてはならない。過去走履歴へ採用する場合も、完了済み過去走として後日のtargetにだけemitする必要がある。

classやcurrent field memberはpre-raceで既知でも、取消・変更時刻のhistorical PIT完全性はraw日付だけでは保証できない。本分析は既存PIT-C契約を超えて厳密なpublication-time claimをしない。IDはstate/join keyだけで、model input候補ではない。

## 10. Modeling implications

1. current raceでは`self ability`、race-constantな`field quality`、`field uncertainty/composition`を意味の異なる軸として保つ。target別opponent meanをself abilityの代用品として再導入しない。
2. past performanceの価値は、対象馬自身のperformance residualと、その時点のfield qualityを別々に履歴化する設計が自然である。差・比・大量interactionへ先に潰さない。
3. career/last-3/last-5/decay90 opponent qualityは高相関なので、候補化するなら1表現を事前固定する。OPP-RECENTの既知negative/inconclusive resultと合わせ、窓の算術違いを反復探索しない。
4. current field compositionでは、rating spread、experienced mix、switcher share、front pressureは異なる意味を持つが、1 experimentで全部入れない。PACE-derived量はPACE-02系の既存結果と重複確認が必要である。

## 11. Candidate hypotheses

優先候補は最大3件に限定する。

1. **EDA-E-01: ability / field-quality separation** — self pre-race ratingを維持し、race-constantなfield mean 1列を追加する限定ablation。ただしclassとの重複を先に診断し、新馬ではmissing/initial-state indicatorを分離する。target別opponent-only mean、top-k/max/spreadの同時追加は禁止する。
2. **EDA-E-02: two-axis past-race value** — 完了済み過去走についてcondition-adjusted performance residualと、そのraceのpre-race field qualityを別々の90日履歴として保持する。最初の試験では差、積、trend、複数windowを混ぜない。
3. **EDA-E-03: one current-field composition component** — rating spread、experienced share、switcher share、front pressureのうち1つだけを意味と既存feature重複から事前選択し、rolling-originで検証する。旧field-relative 15列は復活させない。

いずれも本EDAでは採否を決めず、2024/2025を開かない。既存OPP-RECENTのnegative/inconclusive resultを新しいwindow探索で上書きしない。

## 12. What not to conclude

- opponent-onlyという名前だけでinclusiveより優れているとは言えない。current meanはself ratingを機械的に含む変換である。
- field meanとperformance residualの相関から、強い相手と走った馬が将来強い、または新featureがmodel metricを改善するとは言えない。
- top-rated hit約15%からratingを再調整すべきとは言えない。本workstreamはrating HPOではない。
- 新馬のrating情報が空であることから、新馬戦をtraining populationから除外すべきとは言えない。
- front pressureの期間差からPACE interactionを量産すべきとは言えない。
- 旧field-relative 15列の単純復活、2024単年でのfeature selection、2025参照、購入戦略・ROI最適化は本結論に含まれない。
