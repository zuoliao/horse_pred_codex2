# S1 Two-axis past-race value preregistration

**Phase:** `Phase 5B: EDA-guided representation / objective research`  
**Experiment ID:** `s1_two_axis_race_value_20260901`  
**Registered:** 2026-09-01 JST, before S1 model metrics  
**Raw fingerprint:** `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`

## 1. Research question

過去raceの価値を、(a)その馬が条件対比でどれだけ良く走ったかを表すhorse performanceと、(b)そのraceの全starterがrace前にどれだけ強かったかを表すrace-constant field qualityに分けることで、現在の履歴特徴に追加予測情報が得られるかを検証する。

field qualityは同じraceの全runnerで完全に同じ値とする。対象馬を除くleave-one-out opponent meanはrunner-relativeでありfield qualityとは呼ばず、S1では使用しない。

## 2. Scope and firewall

- 2013: history / condition-model warm-up only
- 2014–2017/18/19: foldごとのexpanding model train
- 2018/19/20: early stopping
- 2019/20/21: calibration
- 2020/21/22: evaluation
- 2023: S1では読込・fit・calibration・評価に使わない
- 2024/2025: loaderで拒否し、設計・parameter・採否に使わない
- final odds、人気、市場情報、ROI: 全て不使用

rawはchunk単位で`2022-12-31`以下だけをretainしてからnormalizeする。全arm、両family、全foldでchoice-set populationとrole規則を共通にする。

## 3. Discovery-only definition audit

監査はcutoff-safeな2014–2019 flat discovery populationだけで行った。field候補の監査母集団は19,925 races、二軸と既存履歴の監査母集団はstrict 271,740 runners / 18,966 racesである。2020以降のtarget metric、2024/2025、marketは定義選択に使用していない。

### 3.1 Field-quality candidates

| Candidate | Missing races | Spearman vs mean | Spearman vs field size | Variance explained by class-group diagnostic |
|---|---:|---:|---:|---:|
| full-field pre-race Elo mean | 0 | 1.0000 | −.1173 | .7967 |
| full-field pre-race Elo median | 0 | .9635 | −.1313 | .7898 |
| top-3 pre-race Elo mean | 0 | .9067 | .0010 | .8371 |

top-3はclass proxyがより強く、medianはmeanよりfield-size依存が小さくなかった。全starterを等しく表し、既存PIT stateと一致し、説明と実装が最も単純な**full-field arithmetic mean**を固定する。class-group内SDは10.39あり、class tokenの決定論的コピーではないが、classとの高い重複リスクは保持する。

### 3.2 Independence and redundancy

90日履歴同士または既存履歴とのdiscovery Spearmanは次のとおり。

| Pair | Spearman |
|---|---:|
| performance – field quality | .3216 |
| performance – PV-01 signed time gap | .7101 |
| performance – 90d mean finish | −.6000 |
| performance – current Elo | .5545 |
| field quality – PV-01 | .3676 |
| field quality – 90d mean finish | −.1200 |
| field quality – current Elo | .5902 |
| field quality – existing career field mean | .9792 |
| field quality – current class tier | .8269 |

二軸は相互に決定論的重複ではない。performanceもPV-01やmean finishのコピーではない。一方、field-quality 90日履歴はcareer field meanと非常に高相関であり、C2/C3がincrementalに効かない可能性を事前に認める。

### 3.3 Prior exposure

performance axisは既存SPEED-01と意味的に近い。ただしSPEED-01はPACE-01入りcontrol、2020–2023 folds、全期間prequential condition updateを用いた。S1は正式254/253 controls、2020–2022、fold-train frozen normalizerを用いるため別実験である。既存SPEED-01結果は先行証拠であり、S1採否値として再利用しない。

## 4. Frozen performance axis

### 4.1 Raw observation and sign

各clean past raceのrunnerについて、公式時計を距離補正したseconds per 1000mへ変換する。

```text
raw performance residual
  = expected winner seconds per 1000m
    - runner official seconds per 1000m
```

大きいほど速く良いperformanceを意味する。observationは`[-5,+5]` seconds/1000mへclipする。

### 4.2 Condition adjustment

期待winner clockはridge main-effects modelで推定する。

- ridge alpha: `1.0`
- intercept: penaltyなし
- response: official winner seconds per 1000m
- course × surface: JRA 10 venue × turf/dirt
- exact distance: frozen 21 levels
- going: 良 / 稍重 / 重 / 不良
- class tier: new / maiden / 1win / 2win / 3win / open
- age restriction: 2yo / 3yo / 3yo+ / 4yo+
- design dimension: 51 including intercept
- no scaling of dummy columns
- season/month: 含めない
- field size: 含めない
- direction: venue/course表現以外では追加しない
- day/course track variant: 含めない

season、field size、interactionを加えないのはS1内の自由度を増やさず、EDA/SPEED-01で監査済みの透明な定義を固定するためである。

### 4.3 Fold fitting and temporal semantics

各foldのtrain期間内では日付順prequentialにcondition stateをfitし、各train targetのfeatureをその日より前のcondition stateだけでemitする。train終了時の係数をfreezeし、early stopping、calibration、evaluation期間ではcondition modelを更新しない。これら後続期間で完了したpast raceのrunner residualはfreeze済み係数で算出し、horse history stateだけを更新する。

この規則により、train rowをfull-train係数でbackfillすること、evaluation outcomeでnormalizerを更新すること、target race outcomeが自身のpre-race featureへ入ることを禁止する。foldごとに係数、fit race count、最大fit date、hashを保存する。

### 4.4 Status semantics

- clean race: JRA平地、既知条件、正距離、finite winner clock、demotion/DQなし、nonwinner clockがwinnerより速くない
- DNF: starterではあるがperformance observationを追加しない
- scratch/exclusion: nonstarterでありfield/value/history更新対象外
- demotion/disqualification: race全体のperformance observationを追加しない
- dead heat: official co-winner clocksが一致する場合だけcleanとして扱い、同じexpected clock基準を用いる
- inconsistent co-winner clocks、unknown condition、invalid clock/distance: race observationなし
- missing history: `NaN`であり0補完しない

## 5. Frozen field-quality axis

sourceは現feature pipelineと同じordinal pairwise global Eloとする。

- initial rating: 1500
- K: 24
- scale: 400
- updates: starter official finishによるpairwise average
- same-date: 全raceのpre-race ratingをemit後にbatch update
- field-quality observation: **全starterのpre-race global Elo算術平均**
- missing: starterが存在するraceではなし。全cold fieldは1500でありmissingにしない
- future opponent results: 使用しない
- leave-one-out/self-excluded mean: 使用しない

field-quality observationはpast race全runnerへ同じ値を記録し、race結果確定後に各starterのhorse historyへ追加する。current target raceのfield qualityはS1 historical featureへ直接入れない。

## 6. Frozen history aggregation

追加列は次の2列だけである。

1. `race_value__decay_90d__performance_residual`
2. `race_value__decay_90d__field_quality`

各horseについてtarget dateより前に確定したpast race observationだけを、`weight = 2 ** (-elapsed_days / 90)`で加重平均する。同日の全target featureをemitしてから同日結果をupdateする。career、last1/3、30/180日、trend、max、variance、uncertainty、condition-specific、difference、interactionは追加しない。

## 7. Four arms and controls

| Arm | Binary | LambdaRank |
|---|---|---|
| C0 | PV-01 254-feature control | lean 253-feature control |
| C1 | C0 + performance | C0 + performance |
| C2 | C0 + field quality | C0 + field quality |
| C3 | C0 + performance + field quality | C0 + performance + field quality |

Binary/LambdaRankのmodel parameter、seed、population、fold、calibrationはarm間で固定する。feature interaction列はない。LambdaRank PV-01 254版はprospective-onlyでS1 controlに使わない。

## 8. Rolling folds

| Fold | Train | Early stopping | Calibration | Evaluation |
|---|---|---|---|---|
| `roll_2020` | 2014–2017 | 2018 | 2019 | 2020 |
| `roll_2021` | 2014–2018 | 2019 | 2020 | 2021 |
| `roll_2022` | 2014–2019 | 2020 | 2021 | 2022 |

3foldを固定する。2019 evaluationを加えて短いtrainを作らず、2023を開かない。

## 9. Metrics and diagnostics

Primary race-macro metrics:

- NDCG@3
- Top-1 official winner mass
- winner reciprocal rank（score tieはaverage rank、co-winnerはequal target mass）
- race Log Loss
- race Brier

Calibration diagnostics:

- calibration yearでfitしたtemperature
- slope equivalent `1 / temperature`
- race-softmax identificationによるintercept `0`
- fixed-bin race-balanced reliability summary
- ECE（補助のみ）

Stability:

- evaluation-year別
- year macro
- improved years / 3
- year-stratified four-date moving-block paired 95% interval、10,000 resamples、seed `20240830`

Fixed guardrail slicesはwhole raceを選ぶだけで、runnerをchoice setから除かない。sliceは採択に使わず定義変更もしない。

- official winnerの観測prior JRA-flat historyが0
- race class new
- race class open/graded
- starter count 15以上
- 少なくとも一頭のofficial winnerがknown previous surfaceからswitch
- 少なくとも一頭のofficial winnerがknown previous distanceから400m以上変更

## 10. Comparisons and selection accounting

各familyで次の5比較、合計10 comparisonsを登録する。

- C1 − C0
- C2 − C0
- C3 − C0
- C3 − C2: field存在下のperformance増分
- C3 − C1: performance存在下のfield増分

performance axisはC1−C0とC3−C2、field axisはC2−C0とC3−C1、joint representationはC3−C0と両conditional incrementを併読する。

Probability path:

- primary: year-macro race Log Loss improvement `>= .002`
- improved years: `>= 2/3`
- paired 95% interval lower `> 0`
- Brier improvement `>= 0`
- NDCG improvement `>= -.002`
- Top-1 improvement `>= -.005`

Ranking path:

- primary: year-macro NDCG@3 improvement `> 0`
- improved years: `>= 2/3`
- paired 95% interval lower `> 0`
- Log Loss improvement `>= -.002`
- Brier improvement `>= -.001`
- Top-1 improvement `>= -.005`

Comparison classification:

- `supported`: probabilityまたはranking pathを全て通過
- `weakly_supported`: primary pointが正、`>=2/3`年で正、guardrail通過、intervalが0を跨ぐためsupported未達
- `rejected`: primary interval全体が悪化側、またはguardrail違反
- `inconclusive`: 上記以外。effectがcomplexity minimum未満の場合も自動acceptしない

Axis classificationは単独差とconditional incrementを別々に記録する。一方だけの結果で「独立した二軸」を支持しない。C3だけがsupportedでもinteractionを作らず`joint-only`として人間へ報告する。

## 11. Feature diagnostics

新2列についてfold/year別coverage、missingness、distribution、drift、相互相関、PV-01、mean finish、Elo、current class、既存career field meanとの相関を保存する。各armでgain/split importance、新列のTreeSHAP dependence、race-aware permutation dependenceを保存するが、importanceを採択根拠にしない。

必須auditはfield-quality race-constant violations、fold coefficient fit scope、future opponent使用0、class-only決定論的proxyでないこと、PV-01/mean finishとの完全重複0である。

## 12. PIT and implementation gates

unit/integration testは少なくとも次を固定する。

1. future append invariance
2. target outcome mutation invariance
3. same-date order/result invariance
4. field quality race-constant
5. full-field pre-race rating meanとの一致
6. performanceはpast result確定後だけupdate
7. condition normalizer fit rowsはfold train内だけ
8. 2023–2025、とくに2024/2025をS1 loaderが拒否
9. market/odds feature 0
10. direct horse/jockey/trainer ID feature 0

## 13. Stop rule

S1終了後にperformance、field、jointを個別分類し、正式controlを変えるべきかを判断する。S2/S3は実行しない。performance支持ならS3、全arm不支持ならS2を次優先として提案する。fieldまたは両軸支持、joint-onlyの場合は人間判断を求める。2024/2025を開かず停止する。
