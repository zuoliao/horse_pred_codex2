# S3 Condition-adjusted performance target preregistration

**Phase:** `Phase 5B: EDA-guided representation / objective research`  
**Experiment ID:** `s3_condition_adjusted_performance_target_20260901`  
**Registered:** 2026-09-01 JST, before S3 model metrics  
**Raw fingerprint:** `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`

## 1. Research question

各runnerのcondition-adjusted continuous performanceを教師にしたLightGBM Huber回帰は、winnerだけを正例にするBinaryまたはtop-heavy relevanceを用いるLambdaRankより、race-specific latent performanceを安定して学習できるか。

S1で支持されたperformance residualの意味を教師へ移すが、S1の90日performance feature、field-quality feature、interaction、追加windowは投入しない。今回の変更はtarget formulationだけである。

## 2. Scope and firewall

- model train: foldごとの2014年からevaluation yearの3年前まで
- early stopping: evaluation yearの2年前
- probability calibration: evaluation yearの前年
- evaluation: 2020、2021、2022
- 2013: existing PIT feature stateのwarm-upに限る。target normalizerのfitには使わない
- 2023/2024/2025: raw chunk filter後のS3 loaderへ渡さず、設計・fit・calibration・評価・採否に使用しない
- odds、人気、final market、ROI: 不使用

rawは2022年末以前だけを物理retainしてからnormalizeする。全methodで同じeligible choice sets、fold role、seedを用いる。regressionのtrain/early-stopping rowsだけはfinite targetを必要とするが、calibration/evaluation choice setはtarget欠損runnerも含む全eligible runnerをscoreする。

## 3. Frozen target

### 3.1 Raw value and sign

clean raceの各完走runnerについて公式時計をseconds per 1000mへ距離正規化し、次を教師とする。

```text
condition-adjusted performance
  = expected winner seconds per 1000m
    - runner official seconds per 1000m
```

値が大きいほど良いperformanceで、`[-5,+5]` seconds/1000mへclipする。生着順MSEは使用しない。

### 3.2 Condition normalizer

S1と同じ51次元ridge main-effects定義を用いる。

- response: official winner seconds per 1000m
- ridge alpha: `1.0`、interceptはpenaltyなし
- course × surface: JRA 10 venue × turf/dirt
- exact distance: frozen 21 levels
- going: 良 / 稍重 / 重 / 不良
- class tier: new / maiden / 1win / 2win / 3win / open
- age restriction: 2yo / 3yo / 3yo+ / 4yo+
- season、field size、direction追加、track/day variant、interaction: なし

各foldでnormalizerは**2014年からfold train終了年までのclean raceだけ**をpoolしてfitする。early stopping、calibration、evaluationの結果は係数へ一切入れず、train終了時の係数をfreezeして全roleのtargetを変換する。training labelの定義器をtraining outcomesにfitする通常の教師変換であり、normalizer自体はmodel featureではない。係数、fit date、fit year count、hashを保存する。

### 3.3 Status semantics

- clean race: JRA平地、既知条件、正距離、finite winner clock、demotion/DQなし、nonwinner clockがwinnerより速くない
- DNFまたはclock欠損runner: target `NaN`
- scratch/exclusion: nonstarterでありmodel population外
- demotion/disqualification: race全体をtarget欠損
- dead heat: official co-winner clocksが一致する場合だけclean
- unknown condition、invalid distance/clock、inconsistent co-winner clocks: race全体をtarget欠損

欠損targetを0に置換しない。training/early-stoppingではfinite-target runnerだけをlossへ入れる。

## 4. Frozen methods and feature scopes

| Method | Objective | Features | Paired reference |
|---|---|---:|---|
| `binary_control` | Binary winner | PV-01 accepted control, 254 | — |
| `lambdarank_control` | top-heavy LambdaRank | lean conservative control, 253 | — |
| `huber_binary_scope` | Huber continuous performance | Binaryと同じ254 | `binary_control` |
| `huber_lambdarank_scope` | Huber continuous performance | LambdaRankと同じ253 | `lambdarank_control` |

LambdaRank PV-01 254版はprospective-onlyのためcontrolにしない。Huberは`objective=huber`、`alpha=.9`、learning rate `.05`、31 leaves、minimum child samples 100、最大500 trees、feature/bagging fraction `.9`、L2 `1.0`、seed 42、early stopping 50を固定する。HPOを行わない。race-balanced sample weightを用いる。

Huberの予測値は大きいほど良いutilityとしてrace内順位を作る。その値を勝率とは解釈せず、calibration年だけで1-parameter race-softmax temperatureをfitし、evaluation年にcoherent win probabilityを出す。controlも同じcalibration手順を用いる。

## 5. Rolling folds

| Fold | Train | Early stopping | Calibration | Evaluation |
|---|---|---|---|---|
| `roll_2020` | 2014–2017 | 2018 | 2019 | 2020 |
| `roll_2021` | 2014–2018 | 2019 | 2020 | 2021 |
| `roll_2022` | 2014–2019 | 2020 | 2021 | 2022 |

S1と同じ3foldを固定する。evaluation結果を見てfold、target、clip、Huber alpha、feature scope、temperature mappingを変えない。

## 6. Metrics and diagnostics

Primary race-macro metrics:

- NDCG@3
- Top-1 official winner mass
- winner reciprocal rank
- race Log Loss
- race Brier

Calibration diagnostics:

- calibration-year temperatureとslope equivalent `1 / temperature`
- identified intercept `0`
- fixed-bin race-balanced reliability summaryと補助ECE

Target diagnosticsはcoverage、missingness、year distribution、clip rate、runner MAE/Huber loss、race-wise Spearmanを保存するが、採否はchoice/probability metricsで行う。

Stabilityはevaluation-year別、3年macro、改善方向年数、year-stratified four-date moving-block paired 95% interval（10,000 resamples、seed `20240830`）を保存する。

固定guardrail sliceはS1と同じで、feature/target再設計には使わない。

- history-0 winner
- new race
- open / graded
- starter 15頭以上
- official winnerのsurface switch
- official winnerの距離変更400m以上

## 7. Comparisons and acceptance

主比較は二つだけである。

1. `huber_binary_scope − binary_control`
2. `huber_lambdarank_scope − lambdarank_control`

全deltaはcandidate改善を正とする。

Probability path:

- year-macro Log Loss改善 `>= .002`
- 改善年 `>= 2/3`
- paired 95% interval lower `> 0`
- Brier非悪化
- NDCG `>= -.002`
- Top-1 `>= -.005`

Ranking path:

- year-macro NDCG@3改善 `> 0`
- 改善年 `>= 2/3`
- paired 95% interval lower `> 0`
- Log Loss `>= -.002`
- Brier `>= -.001`
- Top-1 `>= -.005`

各pathは` supported / weakly_supported / inconclusive / rejected`へ分類する。interval以外を満たしprimary pointと2/3年が正ならweak、primary interval全体が悪化側またはguardrail違反ならreject、それ以外はinconclusiveとする。

S3全体は両paired comparisonで同じpathがsupportedなら`supported`、一方supportedかつ他方weak/inconclusiveなら`weakly_supported`、family間で支持とrejectが衝突すれば`inconclusive`、両方rejectなら`rejected`とする。単一familyだけの改善を一般的なlatent-performance支持とは呼ばない。

## 8. PIT and leakage gates

1. post-train future rowsをappendしてもfrozen normalizerと既存targetが変わらない
2. early-stopping/calibration/evaluation outcome変更がnormalizerを変えない
3. same-date row/race orderでtargetが変わらない
4. normalizer fit year/dateがfold train内だけ
5. target outcome列はfeature scopeに入らない
6. missing targetはfitから除外し0教師にしない
7. calibrationはcalibration yearだけでfitする
8. loaderは2023/2024/2025、とくに2024/2025を拒否する
9. market/odds/popularity feature countは0
10. direct horse/jockey/trainer ID feature countは0

## 9. Stop rule

S3完了後に正式controlを変更すべきか、S2を次にするかperformance modelingを深掘りするかを報告する。ただしS2、S1 featureとの組合せ、target variant、追加feature、2024/2025 milestoneは実行せず、人間レビュー待ちで停止する。
