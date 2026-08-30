# Rating module R0–R6 protocol

## Objective and boundary

RatingをLightGBM内部の派生特徴としてだけ扱わず、race直前の相対強度とcoherent win probabilityを出す独立moduleとして先に評価する。algorithm family、parameter、確率写像を固定した後も、horse stateは各race結果の確定後にforward updateする。評価期間中の過去raceでstateを更新するprequential運用は許容するが、評価結果を見たparameter再選択はしない。

2025 retrospectiveは読込・fit・評価・選択に使わない。odds、人気、馬名、horse IDの数値表現はrating計算へ入れない。horse IDはstate keyにだけ使う。更新対象はJRA平地raceのstarterで、取消・除外は更新せず、障害raceは更新しない。同日の全race predictionをemitしてから結果をbatch updateする。

## Time protocol

| Period | Use |
|---|---|
| 2013 | state warm-up only |
| 2014–2017 | parameter候補の追加warm-up / state形成 |
| 2018–2021 | fixed annual rolling-origin parameter selection |
| 2022 | module family / condition extension validation |
| 2023 | winner Log Lossによるtemperature calibrationのみ |
| 2024 | frozen moduleのone-shot development評価とR6 LightGBM統合 |
| 2025 | unused |
| 2026+ | prospective final |

## Stages

### R0: exact current-Elo control

現feature builderと同じinitial 1500、K 24、scale 400、pairwise average updateを独立stream runnerで再現する。scored runnerのpre-race rating、same-date visibility、starter/jump inclusion、runner identityが完全一致しなければ後続へ進まない。

### R1: coherent standalone output

pairwise Elo scoreをrace softmaxへ変換し、raceごとに確率和1を保証する。ranking、race Log Loss/Brier、race-balanced reliability、surface/distance/class/field-size条件別指標を共通評価器で保存する。R1はparameterを変更しない。

### R2: training-period Elo selection

K=`8/16/24/32/48`、scale=`200/400/600`の15候補を2018–2021 annual rolling-originだけで比較する。各候補は2013から連続更新し、各選択年のrace直前予測をscoreする。race Log Loss最小をprimaryとし、Brier、NDCG@3、Top-1、複雑性の順でtie breakする。2022以降はparameter選択へ使わない。

### R3: probabilistic rating family

winner massをtargetにrace softmax likelihoodのonline gradientで更新する`online_top1_pl`を、learning rate=`.05/.1/.2/.4`で同じ2018–2021 protocolにより選ぶ。これはPlackett–Luceのtop-choice stageに相当し、full finishing-order PLとは呼ばない。選択済みEloと2022で比較し、Log Lossが改善し、Brier `>=−.001`、NDCG@3 `>=−.002`、Top-1 `>=−.005`の改善guardrailを満たす場合だけfamilyを置換する。

### R4: condition-specific state

R3の固定familyについて芝/ダート別stateをglobalと並行更新し、`global + w * (surface - global)`をrace scoreとする。`w=.25/.5/.75/1`を2018–2021だけで選び、2022でglobal-onlyと比較する。R3と同じacceptance ruleを満たす場合だけ採用する。distance、venue、classは同じ実験へ混ぜない。

### R5: frozen PIT artifact

R4までに選ばれたspecを固定し、2013から2024まで一回のchronological passでrunner-level artifactを生成する。raw probabilityは全期間PIT-safeである。2023だけでtemperatureをfitし、そのcalibrated probabilityは2024 standalone評価専用とする。LightGBM train rowへ未来の2023 calibratorを適用しないため、R6 featureにはraw probabilityだけを渡す。

### R6: LightGBM integration

current best `abl_006_drop_field_relative`（253列）をcontrolとし、R5で凍結した5列groupだけを追加する。Binary/LambdaRank、seed、split、calibration、runner populationを揃え、2024の4-date paired block bootstrap 10,000回で比較する。ranking/probabilityのいずれかが支持され、他主要metricが既存guardrail内の場合だけ統合候補としてacceptする。改善しなくてもR0–R6 module実装自体は完了とし、negative resultを保存する。

## Interpretation constraints

- standalone ratingの改善がLightGBM改善を保証するとは仮定しない。
- gain importanceだけで採用しない。
- 2024条件別sliceでparameterやsurface weightを事後調整しない。
- final oddsは使用しない。
- R6後も2026+ prospective確認なしに収益性やfinal汎化を主張しない。
