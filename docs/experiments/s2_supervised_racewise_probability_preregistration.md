# S2 Supervised race-wise probability preregistration

**Phase:** `Phase 5B: EDA-guided representation / objective research`  
**Experiment ID:** `s2_supervised_racewise_probability_20260901`  
**Registered:** 2026-09-01 JST, before S2 model metrics  
**Raw fingerprint:** `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`

## 1. Research question

1 raceを1 choice setとしてwinner probabilityを直接最適化すると、runnerごとのBinary probabilityを学習してからrace内softmaxする現行方式より、coherent win probabilityとrankingが改善するか。また、S1で支持されたhistorical performance residualの増分価値はobjectiveを変更しても再現するか。

S3のcontinuous targetは使用しない。S1 field quality、interaction、追加window、追加feature、odds、人気、marketは投入しない。

## 2. Scope and firewall

- train: evaluation yearの3年前までの2014年以降
- model validation / early stopping: evaluation yearの2年前
- probability calibration: evaluation yearの前年
- evaluation: 2020、2021、2022
- 2013:既存PIT stateのwarm-upだけ
- 2023/2024/2025: source isolation後のloaderへ渡さない
- odds、人気、final market、ROI: 不使用

rawは2022年末以前だけを物理retainしてからnormalizeする。全armで同じeligible choice set、fold role、seed、Binary PV-01 feature scopeを使う。2024/2025やevaluation metricを見てobjective、capacity gate、feature、fold、parameter、calibrationを変更しない。

## 3. Frozen four-arm comparison

| Arm | Model | Features | Role |
|---|---|---:|---|
| `B0` | LightGBM Binary control | PV-01 accepted 254 | formal matched control |
| `B1` | LightGBM Binary | B0 + S1 performance residual, 255 | S1 replication |
| `R0` | supervised race-wise probability | B0と同じ254 | objective comparison |
| `R1` | supervised race-wise probability | B1と同じ255 | objective + S1 comparison |

S1 field-quality列は使用しない。`R0−B0`と`R1−B1`がobjective比較、`B1−B0`と`R1−R0`がperformance feature比較である。LambdaRank lean 253とS1のLambdaRank + performance結果は既存immutable evidenceからdescriptive contextとして引用し、S2のprimary armにはしない。

探索的に`(R1−R0)−(B1−B0)`をdifference-in-differencesとして保存するが、採否基準にはしない。

## 4. Winner target and probability

各raceのofficial winner massを教師とする。winnerが1頭なら1、同着k頭なら各`1/k`、その他は0で、race内合計は必ず1とする。winnerなし、choice set分割、重複runner、非contiguous groupはfail closedする。

race-wise lossはrunner micro平均ではなくrace macro平均である。

```text
utility: u_ri
probability: p_ri = exp(u_ri) / sum_j exp(u_rj)
loss: -(1 / number_of_races) * sum_r sum_i y_ri log(p_ri)
```

race-wise modelの`T=1` native probabilityを必ず保存する。主比較では全armを公平に扱うため、別calibration年だけで1-parameter temperatureをfitしたcoherent probabilityを使う。rankingはtemperature前utilityで評価する。calibration slope equivalentは`1/T`、interceptはrace-commonで同定不能なため0とする。

## 5. Stage L: transparent linear Conditional Logit

最初のrace-wise modelはinterceptなしの線形utility `u=Xβ` とする。race-common interceptやrace-constant main effectはchoice set内で相殺される。これはモデル仮定であり、使用不能列とcondition × runner interaction不足を診断へ記録する。

前処理はfold trainだけでfitする。

- non-finiteをtrain medianで補完
- train mean / standard deviationで標準化
- train標準偏差`<1e-12`はscale 1として係数0に固定
- z-scoreを`[-10,10]`へclip
- missing indicatorや新しいderived featureは追加しない
- transformerをvalidation/calibration/evaluationへfreeze

optimizationはSciPy L-BFGS-B、zero initialization、最大250 iteration、`ftol=1e-10`、`gtol=1e-6`とする。L2候補`[1e-4, 1e-3, 1e-2]`はmodel-validation native race Log Lossだけで選択し、同点は大きいL2を選ぶ。evaluationは選択に使わない。各fitでconvergence、iteration、gradient norm、coefficient norm、loss、transform hashを保存する。

線形capacity-matched診断として、同じtransform・L2 gridのrunner-level線形Binary `LB0/LB1`も保存する。これはobjective attributionの補助でありprimary 4-armや採否には含めない。

## 6. Metric-blind capacity gate and conditional Stage N

Stage Lのevaluation指標を計算する前に、各foldのmodel-validation年だけでcapacity gateを判定する。

```text
uniform LL = mean_r log(field_size_r)
recovery = (uniform LL - R0 native LL) / (uniform LL - B0 native LL)
```

solver/objective invariantが失敗した場合は修正して再実行し、非線形stageの根拠にはしない。B0がuniformを上回らず分母が正でないfoldはcapacity indeterminateとしてunder-capacity側に数える。

- `recovery < .75`が3fold中2fold以上: Stage Lはunder-capacity、事前登録済みStage Nを実行
- それ以外: Stage LをS2 candidateとしStage Nは実行しない

Stage NはLightGBM nonlinear utilityにgrouped softmax cross-entropyを与える。B0/B1と同じ254/255列、同じtree parameter、seed、round cap、validation年early stoppingを使い、`N0/N1`として独立記録する。

```text
gradient = p - y
diagonal Hessian = max(p * (1 - p), 1e-6)
```

LightGBM APIはfull Hessianのoff-diagonalを受け取らないため、Stage Nは厳密なNewton PLではなく`nonlinear utility + diagonal-Hessian grouped softmax cross-entropy`と呼ぶ。race loss自体が1 race 1 unitなので追加runner weightは使わない。Stage Nが発動した場合は`N0/N1`をS2のprimary race-wise armsとし、Stage Lは透明baselineとして併記する。

## 7. Rolling folds

| Fold | Train | Model validation | Calibration | Evaluation |
|---|---|---|---|---|
| `roll_2020` | 2014–2017 | 2018 | 2019 | 2020 |
| `roll_2021` | 2014–2018 | 2019 | 2020 | 2021 |
| `roll_2022` | 2014–2019 | 2020 | 2021 | 2022 |

S1/S3と同じfoldを固定する。S1 performance historyはfold-specific train normalizer、prequential train history、frozen post-train normalizer、same-date emit-before-updateをそのまま再利用する。

## 8. Metrics, uncertainty, and slices

Primary race-macro metrics:

- race Log Loss
- race Brier
- NDCG@3
- Top-1 official winner mass
- winner reciprocal rank

Calibration diagnosticsはnative/calibrated Log LossとBrier、temperature、slope equivalent、fixed-bin race-balanced reliability summary、ECEを保存する。

Stabilityはevaluation-year別、3年macro、改善方向年数、year-stratified four-date moving-block paired 95% interval（10,000 resamples、seed `20240830`）を保存する。

固定sliceはS1/S3と同じで、定義変更には使わない。

- history-0 winner
- new race
- open / graded
- starter 15頭以上
- official winnerのsurface switch
- official winnerの距離変更400m以上

## 9. Acceptance

全deltaはcandidate改善を正とする。

Probability path:

- macro Log Loss改善 `>= .002`
- 改善年 `>= 2/3`
- paired 95% interval lower `> 0`
- Brier非悪化
- NDCG `>= -.002`
- Top-1 `>= -.005`
- native `T=1` Log Loss deltaがmatched Binary native mapping比`>= -.002`

Ranking path:

- macro NDCG@3改善 `> 0`
- 改善年 `>= 2/3`
- paired 95% interval lower `> 0`
- Log Loss `>= -.002`
- Brier `>= -.001`
- Top-1 `>= -.005`

各比較を`supported / weakly_supported / inconclusive / rejected`へ分類する。interval以外を満たしprimary pointと2/3年が正ならweak、primary interval全体が悪化側またはguardrail違反ならreject、それ以外はinconclusiveとする。S2はprobability-objective仮説なのでranking-only成功は`ranking_only / weak`でありS2 supportedとは数えない。

objectiveは`R0−B0`と`R1−B1`（Stage N発動時は`N0−B0`と`N1−B1`）のprobability pathがともにsupportedなら`supported`、一方supportedかつ他方non-rejectedなら`weakly_supported`、support/rejectが衝突すれば`inconclusive`、両方rejectなら`rejected`、それ以外は`inconclusive`とする。

S1 performance featureのobjective横断再現は`B1−B0`とrace-wise `R1−R0`（Stage N時は`N1−N0`）を分けて分類する。両方でprobability path supportedなら`objective_robust_supported`、一方だけなら`objective_dependent / inconclusive`とする。

## 10. PIT, leakage, and implementation gates

1. future rows appendで過去S1 featureとmodel transformが変わらない
2. target-race outcome変更でpre-race S1 featureが変わらない
3. same-date race order変更でS1 featureが変わらない
4. B0/R0 feature checksum一致、B1/R1はB0 + performance 1列だけ
5. 全arm・全roleでchoice-set key/population一致
6. train-only imputation/scaling/L2 selection; later-role outcome変更で不変
7. winner massがrace内1、同着はfractional mass
8. hand calculationと有限差分でlinear loss/gradient一致
9. race/runner order、race-common utility shiftでloss/probability不変
10. native/calibrated probabilityのrace内和が1
11. calibration yearだけでtemperature fit
12. loaderは2023/2024/2025、とくに2024/2025を拒否
13. market/odds/popularity feature数0
14. direct horse/jockey/trainer ID feature数0

## 11. Diagnostics and interpretation limits

coverage、missingness、year drift、feature checksum、coefficient/importance、permutation dependence、native/calibrated probability、choice-set sum error、linear race-constant列、nonlinear gainを保存する。importanceは採択根拠にしない。

Linear Conditional LogitとLightGBM Binaryの差にはobjectiveとfunction classの差が残る。capacity gateまたはStage N、capacity-matched linear Binary診断を用いず、線形の失敗だけからrace-wise objectiveを棄却しない。Stage Nも対角Hessian近似であり、full-set interaction/attention modelではない。

## 12. Stop rule

S2完了後、objective価値、S1 feature再現、Binary/LambdaRank/race-wiseの役割、Inter-horse/race-set modelへ進む根拠、次候補を報告する。正式control変更の要否を記録した後、人間レビュー待ちで停止する。次モデル、追加feature、2024/2025 milestone、market評価は自動実行しない。
