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

## R5 frozen result and R6 registration

Clean implementation commit `ce6d5cc`からR0–R5を実行した。R0は489,674 scored runnersで既存global Eloの保存済みfloat32値と完全一致し、mismatch `0`、最大差`0`だった。

- R2 selected: pairwise Elo、K=`48`、scale=`200`
- R3 top-choice PL: 2022 Log Loss改善`−.10930`、NDCG差`−.14032`でreject
- R4 surface blend `w=.25`: 2022 NDCG差`+.00351`だがLog Loss改善`−.00381`でreject
- R5 frozen spec: global pairwise Elo、K=`48`、scale=`200`、surface blend=`0`
- 2023 temperature: `0.5187022312`
- 2024 standalone raw: NDCG@3 `.35333`、Top-1 `.16929`、Log Loss `2.44403`、Brier `.89910`
- 2024 standalone calibrated: ranking不変、Log Loss `2.40153`、Brier `.89459`、ECE `.00774`

R6では、R5 artifact SHA-256=`0967d04a30b75812015e9d00ed68a71ec344424c4f4099a3c91f21ba6cdfb738`、frozen spec SHA-256=`40ae7a9c179abf12ef4c4376ce887b72cbc5304cdf2c7de2878c97621114ea6d`を固定する。LightGBMへは未来の2023 calibratorを適用せず、raw PIT 5列だけを渡す。

| Arm | Features | Expected count |
|---|---|---:|
| Control | `abl_006`と同じsemantic 6 groups | 253 |
| Candidate | control + frozen `rating_module` 5 columns | 258 |

Candidate追加列はscore、raw coherent win probability、global starts、surface-condition starts、`1/sqrt(global starts+1)` uncertainty proxyである。新273列cacheから旧268列の値・順序・NaN位置が全行一致し、同cacheの253列controlが`abl_006`を予測・metric差`<=1e-12`で再現することを実行gateとする。

R6採否はIMP-004/005と同じ二経路を使う。Probability pathはLog Loss改善point `>=.002`かつpaired 95% interval下限`>0`、Brier非悪化、NDCG `>=−.002`、Top-1 `>=−.005`。Ranking pathはNDCG改善interval下限`>0`、Log Loss `>=−.002`、Brier `>=−.001`、Top-1 `>=−.005`。2025 rows used=`0`、odds used=`false`を必須とする。

## R6 result

273-column cacheは533,853 rowsで、旧268列の順序・値・NaN位置が完全一致した（mismatch `0`、最大差`0`）。2025の新5列は全て欠損。253-feature controlも`abl_006_drop_field_relative`をrunner、予測、metric、best iteration、temperatureまで完全再現した。

| Model | Candidate NDCG@3 | NDCG improvement [95% CI] | Candidate Log Loss | LL improvement [95% CI] | Brier improvement | Top-1 improvement | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Binary | .49333 | +.00207 `[-.00352,+.00757]` | 2.08510 | +.00042 `[-.00603,+.00707]` | −.00121 | −.00164 | reject |
| LambdaRank | .48814 | −.00430 `[-.00808,−.00047]` | 2.08910 | −.00444 `[-.00919,+.00039]` | −.00085 | −.00328 | reject |

Binaryはranking pointが改善したが区間下限は0未満で、Log Loss改善も`.002`未満、Brierはranking-path guardrail `−.001`を外れた。LambdaRankはNDCG interval全体が悪化側である。両familyとも採用しない。

新raw win probabilityはgain importanceがBinaryで2位・12.6%、LambdaRankで4位・4.46%と強く利用された。しかし新scoreは旧global EloとSpearman `.975`、新probabilityは旧horse-minus-field Eloと`.800`で、既存rating/race-relative情報との重複が大きい。standalone moduleが予測力を持つことと、既存LightGBMへ追加情報を与えることは別だった。

## Final conclusion

R0–R6のsoftware、PIT artifact、standalone評価、LightGBM integrationは完了した。固定moduleは透明な独立benchmarkおよび将来の他model入力候補として保持する。現LightGBM bestは253-feature `abl_006_drop_field_relative`のままで、module 5列はdefault featureへ昇格させない。

R2のK48/scale200はgrid境界だったため、より高速な更新が最適という可能性は残る。ただし2024 R6結果を見てgridを拡張しない。追加rating研究を行う場合は、新しい事前登録のもと2018–2021 selectionと2022 validationだけでdynamic decayまたはuncertainty-aware方式を検証し、2024へ反復適合させない。

Tracked source of truth: [`experiments/rating_module_20260830/summary.json`](../../experiments/rating_module_20260830/summary.json)。full predictions、models、cache、bootstrap artifactは`artifacts/`と`data/`にlocal保存しGit対象外とする。
