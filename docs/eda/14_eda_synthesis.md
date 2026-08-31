# Phase 5A EDA synthesis

## Executive conclusion

現データには時間を越えて再現するsignalがある。特に、減衰付きhorse history、connections、同条件でのperformance persistenceは強い。一方、最大のボトルネックは単一のLightGBM parameterではなく、過去走を「自馬の走行内容」と「そのraceの相手水準」に分け切れていないこと、winner/top-heavy labelが走行内容を捨てること、cold-historyと条件替わりで情報の信頼度を表現できていないことである。

EDAはproduction改善を実証していない。Binary PV-01 254列とLambdaRank lean 253列を凍結controlとして維持し、次の実験は人間選択後に一件ずつ行う。

## 1. 現データの内部妥当性と外的妥当性

- **Evidence:** 2022年末まで488,715 raw rows / 34,504 races。flat historical performanceは471,557 rows / 33,240 races、predictive viewは450,340 rows / 31,689 races。duplicate key、主要parse failure、post-2022保持、predictive viewへのmarket/current outcome混入は0。last 3Fは完走馬で3件だけ欠損。
- **Confidence:** 内部妥当性は中～高。PIT-C stateとsame-date emit-before-updateはtest/review済み。
- **Limitation:** 2014は2013だけのwarm-upでcareer historyが左打切り。raw class tokenに制度変更があり、publication timestamp・地方/海外履歴・当日情報がない。
- **Recommended action:** retrospective PIT-Cとして使用を継続し、prospective PIT-Aや外部母集団へ一般化しない。

## 2. 最も強い予測signal

- **Evidence:** 90日減衰horse-history race associationは`.457/.430/.459`。jockey career EBは`.276/.262/.258`、trainerは`.189/.180/.190`。直近走signalはlagとともに`.40`前後から`.10`前後へ減衰する。
- **Confidence:** 高い方向再現。ただし単変量associationであり現modelへの増分ではない。
- **Limitation:** horse ability、assignment、race placementが混在する。
- **Recommended action:** horse formとconnectionsをretainし、追加より縮約・意味分解を優先する。

## 3. signalの時間安定性

- **Evidence:** horse decay、connection、same-condition persistence、field-quality/performanceの正方向はD/R/Cで一致。field sizeは14.35→13.97→13.75へdriftし、class token driftはadversarial validationの主因になった。
- **Confidence:** 大signalの方向は中～高、小sliceのmagnitudeは低～中。
- **Limitation:** 2022は1年、期間は既に研究で参照済みでuntouchedではない。
- **Recommended action:** 今後もrolling foldsの方向一致とdate-block intervalを採択条件にする。

## 4. 現featureが捨てている情報

- **Evidence:** winner binaryは全loserを同一化し、top-heavy rankは4着以下を同一化する。race-relative clock、last 3F、passing trajectory、condition-adjusted residual、過去raceのfield qualityは現表現で限定的。history0ではratingが全馬1,500となり区別不能。
- **Confidence:** 高い情報論的観察、増分性能の確信は低い。
- **Limitation:** 情報が存在することとGBDTで有効なことは別。
- **Recommended action:** race-value二軸とperformance targetを独立実験にする。

## 5. 着順・時計・着差・上がり・通過順位の役割

- **Evidence:** finish percentileとnegative time gapのrace-macro Spearmanは約`.995`。3着/4着に特別な不連続はなく、同時計隣接は約22–24%。margin token orderとgapは約`.957`だが秒mapping未同定。winner最速last3Fは37–41%のみ。prior last3Fと次走は約`.31`、passing positionは約`.11–.18`。
- **Confidence:** 構造観察は高い。production mappingは低い。
- **Limitation:** absolute timeは距離・馬場・venue・seasonと交絡し、passing orderの記録点構造はcourse依存。D章のtaxonomyは内容監査用、G章のdistance/rest taxonomyはtransition監査用で境界が異なり、両者を直接replicationとは呼ばない。次実験では一定義へ固定する。
- **Recommended action:** 着順はcontrol、condition residualはtarget候補、last3F/passingはrace-relative historyとして限定検証、margin mappingは保留。

## 6. 相手水準の表現

- **Evidence:** current opponent-only meanは同一race内でself ratingと数学的にSpearman `-1`となり、純粋なfield qualityではない。historical careerと90日opponent historyは`.977–.980`で冗長。field qualityとperformance residualの日別Spearmanは`.221/.156/.181`。
- **Confidence:** current leave-one-outの問題は高、二軸仮説は中。
- **Limitation:** positive cross-race associationはclass/placement/scale driftを含む。
- **Recommended action:** self ability、race-constant field quality、uncertaintyを分け、past performance residualとpast field qualityを別々に履歴化する。

## 7. jockey/trainer情報の縮約

- **Evidence:** signalは安定するが現130列は`2 entities × 13 blocks × 5 stats`で、64診断列に`|rho|>=.90`が27 pair。raw entity rateは極端decileで平均回帰する。
- **Confidence:** 冗長性は高、最適縮約形は中。
- **Limitation:** 130列全体のrolling retrained ablationは未実施。trainer keyはraw name。
- **Recommended action:** long-term EB level、short-term deviation、effective n、uncertaintyへ縮約する仮説を登録する。

## 8. current fieldの表現

- **Evidence:** cold share約10%、experienced share約71–72%。leave-one-out opponent meanはself rankを反転する。large-field/high-spreadは両familyでR/Cのwinner lossが高いが、旧field-relative 15列は削除で改善済み。
- **Confidence:** composition gapは中、具体featureは低～中。
- **Limitation:** field-level outcome main effectはrace内で定数、runner勝率相関では評価不能。
- **Recommended action:** race-constant mean/spread/uncertaintyを少数・別意味で扱い、旧15列は復活させない。

## 9. modelが弱いrace条件と原因候補

- **Evidence:** history0、新馬class、open/3win、surface switch、大距離変更、large-field/high-spreadでOOT lossが高い。winner rank medianは3、top3外は約42%。
- **Confidence:** slice方向は中。因果原因は低。
- **Limitation:** winner-conditioned selection、class/field-size/history交絡、PACE-01以前のfrozen predictions。
- **Recommended action:** sliceをguardrailとして固定し、slice専用modelは作らない。

## 10. 市場差のうち現データで改善できそうな部分

- **Evidence:** JRA平地history 10走以上のwinnerでもmodel–market Log Loss gapは正。winner top3外が約42%、Binary/Rank disagreement約24%。
- **Confidence:** modeling/representation余地が残る点は中。
- **Limitation:** market情報を成分分解できず、gapは因果寄与でない。
- **Recommended action:** race-value、target、race-wise objectiveをno-odds rolling評価する。

## 11. 新しい当日データが必要そうな部分

- **Evidence:** JRA平地history 0でmarket gapが特に大きい。rawは血統、調教、正確なgrade/斤量、馬体重のPIT snapshot、直前変更・馬場・天候を欠く。
- **Confidence:** 追加情報が市場差の一部を説明する可能性は中。
- **Limitation:** final marketから個別情報の寄与は識別不能。MacでJV-Link prospective collectionは保留中。
- **Recommended action:** 今回は収集を開始せず、将来はsource/PIT付きで一群ずつ診断する。

## 12. BinaryとLambdaRankの違い

- **Evidence:** 2020–2022でBinaryはLog Loss/Brierが3/3良い。NDCGはBinaryが2/3、Rankが1/3。top-choiceは約24%異なるが固定ensembleは既に棄却。
- **Confidence:** 一方の恒常優位がない点は高。
- **Limitation:** OOT artifactはPACE-01以前の254/253列で、objectiveとfeature interactionを完全分離しない。
- **Recommended action:** 両者をcontrolとして保持し、family比較よりrace-wise objectiveの透明baselineを先に置く。

## 13. target formulationを変える根拠

- **Evidence:** 3着/4着境界は連続、full pairwiseのwinner関与は約14%だけで下位pairが支配。time/performance residualは高coverage。top3 multitaskに固有の境界根拠はない。
- **Confidence:** 別targetを試す根拠は中～高、特定targetが勝つ確信は低い。
- **Limitation:** diagnosticのみで本格model未比較。
- **Recommended action:** coherent race choiceとcondition-adjusted continuous targetを別実験で比較。multi-taskはauxiliary targetのstandalone価値確認後。

## 14. LightGBMをさらに育てる価値

- **Evidence:** rolling OOTで安定したsignalと改善余地があり、Binary/Rankとも条件替わりに共通誤差を持つ。
- **Confidence:** controlとしての価値は高、feature追加だけでmarket gapを閉じる確信は低い。
- **Limitation:** HPOは既に変更なし、局所算術変形はnegativeが多い。
- **Recommended action:** LightGBMはrepresentation/target仮説の非線形controlとして維持し、無制限feature searchはしない。

## 15. alternative modelの優先度

- **Evidence:** objective mismatchの可能性はあるが、Binary/Rank差はmarket gapより小さい。CatBoost/XGBoost固有の必要性はEDAで示されない。
- **Confidence:** conditional logit/PLを先にする判断は中。
- **Limitation:** 非線形race-wise PLは未実装。
- **Recommended action:** transparent choice model → performance target →必要ならalternative GBDT。DNNはdefer。

## 16. 追加データ取得の優先度

- **Evidence:** cold-history、新馬/openでgapが大きく、現rawに正確なrace condition、斤量/handicap、血統、調教、当日PIT情報がない。
- **Confidence:** data gapの存在は高、費用対効果は未確定。
- **Limitation:** final odds sensitivityだけでは採択不能。
- **Recommended action:** user方針どおり今回収集は保留。再開時は正確なgrade/斤量、prospective当日snapshotの順で一群ずつ。

## 17. 次に実施すべき実験

1. `EDA-S01-RACE-VALUE-2AXIS`
2. `EDA-S02-RACEWISE-CHOICE`
3. `EDA-S03-PERFORMANCE-TARGET`
4. `EDA-A01-TRANSITION-RELIABILITY`
5. `EDA-A02-CONNECTION-COMPRESSION`
6. `EDA-A03-LAST3F-RELATIVE`（既存SEC-3F重複確認後）
7. `EDA-B01-WORKLOAD-INTERACTION`

最優先は上位3件だけであり、EDA完了時点では未実行である。

## 18. 当面実施すべきでない案

- PV-06 margin seconds mappingの調整
- margin-aware ratingの追加算術変形
- 旧field-relative 15列の復活
- 新馬戦のtraining除外
- 2024/2025を使うfeature/parameter selection
- calibration parameter、ensemble weight、ROI/EV threshold探索
- condition別model、大規模DNN、Web UI、新規scraping

## What changed from the prior working assumptions

Top3を別taskとして足せば自動的にnoiseが減る、という根拠は得られなかった。3着/4着は連続であり、multi-task化より先に、race-wise winner probabilityとcontinuous performanceをそれぞれ単独で評価すべきである。また「opponent-only current mean」はfield qualityではなくrace内self positionの逆変換になりうる。相手水準はrace-constantな軸、自馬performanceは別軸として設計し直す必要がある。

本EDAからまだ主張できないのは、どのS候補がmetricを改善するか、market gapの何割がdata/model/targetに由来するか、profitabilityが改善するか、である。
