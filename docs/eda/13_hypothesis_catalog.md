# Phase 5A hypothesis catalog

## Purpose and status

本章はEDAから生じた仮説を登録する。採択・production実装ではない。canonical machine-readable sourceは`experiments/eda_20260901/hypothesis_registry.csv`であり、2024/2025、final odds、ROIを優先順位付けに使っていない。優先度は時間再現、support、情報新規性、PIT安全性、既存表現との重複、実装費用、現modelの弱点との対応で決めた。

## Priority S: human reviewを待つ最大3候補

| ID | Question | Replicated evidence | Proposed experiment | Main risk |
|---|---|---|---|---|
| `EDA-S01-RACE-VALUE-2AXIS` | 過去走の走行内容と当時の相手水準を別軸で履歴化すると改善するか | field qualityとcondition residualの日別Spearmanが`.221/.156/.181`、全期間正 | residualとrace-constant field qualityを別々の90日履歴各1列としてrolling ablation | residual補正器のfold内fit、既存ratingとの重複 |
| `EDA-S02-RACEWISE-CHOICE` | race-wise probability objectiveは現Binary/Rankよりcoherent probabilityを改善するか | choice-set構造が安定、family top-choice不一致約24%、objective差はmarket gapより小さい | 同一features/splitsの線形conditional logitまたはtop-choice PL baseline | 線形baselineのunderfitをobjective否定と誤読しない |
| `EDA-S03-PERFORMANCE-TARGET` | continuous performance targetはwinner/top3 labelより走行内容を保持するか | 3着/4着境界は連続、rankとtime gapは約`.995`、residual coverage >99.6% | fold内condition residualをHuber回帰しrace内順位化・確率化 | track/day/condition補正のleakageとdrift |

3件は互いに混ぜない。最初の実験を人間が選び、その一仮説だけを固定設計で実施する。

## Priority A

| ID | Observation | Interpretation | Status |
|---|---|---|---|
| `EDA-A01-TRANSITION-RELIABILITY` | same surface/venue/direction、距離差200m以内で過去performance持続性が高い | switchは過去stateを無効化せず信頼度を弱める | proposed |
| `EDA-A02-CONNECTION-COMPRESSION` | jockey/trainer signalは安定する一方、130列に決定論的関係と高相関pair | long-term EB、deviation、support、uncertaintyへ縮約できる可能性 | proposed |
| `EDA-A03-LAST3F-RELATIVE` | winner最速last3Fは37–41%だけ、過去last3Fと次走は約`.31` | 着順以外のperformance contentを持つ | proposed。既存SEC-3Fとの重複監査が先 |

## Priority B/C and negative evidence

| ID | Finding | Disposition |
|---|---|---|
| `EDA-B01-WORKLOAD-INTERACTION` | 14–29日かつ30日2走以上は1走より3期間低い | confoundingが強い。低自由度1案だけの候補 |
| `EDA-B02-MARGIN-TOKEN` | token orderは安定するが秒mappingは同定できない | PV-06を再開せずofficial semantics追加監査まで保留 |
| `EDA-C01-NEW-HORSE-EXCLUSION` | history 0は約10%、connectionsは利用可能、既存fit除外は悪化 | rejected-by-EDA。新馬戦は除外しない |

旧field-relative 15列の復活、arithmetic rating変形、PACE-02 interaction、calibration parameter search、ensemble再探索は登録しない。既存negative/inconclusive evidenceに対する事後適合になるためである。

## Target and model implications

| Question | Current decision |
|---|---|
| Binary winner target | frozen controlとしてretain |
| LambdaRank `3/2/1/0` | frozen controlとしてretain。最適labelとは未確定 |
| full-order relevance | unweighted all-pairは下位pairに支配されるため採用しない |
| top-3 multitask | 3着/4着に特別な不連続を確認できず、直ちに採用しない |
| multi-task learning | auxiliary performance targetのstandalone OOT価値を確認するまでdefer |
| race-wise probability | transparent baselineをS候補にする |
| LightGBM | nonlinear controlとして継続 |
| CatBoost/XGBoost/DNN | representation/target検証後へdefer |
| ensemble | ENS-01 negativeを新証拠なしに再開しない |

## What not to conclude

- EDA associationはretrained incremental performanceではない。
- three-period direction consistencyはuntouched confirmationではない。
- market gapが大きいsliceは、final oddsをfeature選択へ使う根拠ではない。
- history 0とrace class新馬は同義ではない。
- candidate registryの`ready-for-experiment`は自動実行許可ではない。Phase 5A完了後は人間選択を待つ。
