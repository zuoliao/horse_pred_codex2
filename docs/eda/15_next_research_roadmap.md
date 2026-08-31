# Next research roadmap after Phase 5A

## Decision gate

Phase 5Aは候補の優先順位付けで停止する。以下の最優先3件をまだ実行しない。人間が一件を選んだ後、`one experiment = one hypothesis`でpreregisterし、2024/2025とfinal marketを選択に使わずrolling-originだけでscreenする。

## Prioritized roadmap

| Priority | Hypothesis | EDA evidence | Proposed experiment | Required data | PIT risk | Expected knowledge | Cost |
|---|---|---|---|---|---|---|---|
| S1 | Two-axis past-race value | condition residualとfield qualityの日別Spearman `.221/.156/.181`、相手windowは高冗長 | residualとrace-constant pre-race field qualityを別々の90日履歴各1列として現controlへ一群追加 | existing raw + pre-race rating | low～medium | 「どう走ったか」と「誰と走ったか」の独立増分 | medium |
| S2 | Supervised race-wise probability | one-winner choice set、family disagreement約24%、Binary proper score優位だがranking優位不定 | 同一feature/splitの線形conditional logitまたはtop-choice PL。fold内calibration込み | existing PIT view | low | post-softmaxでなくobjective自体をrace-wiseにする価値 | medium |
| S3 | Condition-adjusted performance target | 3/4着境界は連続、rank/time-gap約`.995`、residual coverage >99.6% | fold-trainだけでcondition residualを作りHuber回帰、race内順位・確率へ変換 | existing result/time/context | medium | winner以外の走行内容を教師にする価値 | high |
| A1 | Transition-aware state reliability | same surface/venue/direction、距離差<=200mで持続性が高い | surface switchまたはdistance changeのどちらか1つをstate reliability interaction化 | existing context/history | low | condition替わりで履歴をどう減衰すべきか | low |
| A2 | Hierarchical connection compression | career signal安定、130列に決定論的/高相関関係、entity rateは平均回帰 | long-term EB + short deviation + effective n + uncertaintyと現130列をrolling比較 | existing connection state | low | signal維持とfingerprint/redundancy低減の両立 | medium |
| A3 | Race-relative last 3F history | winner最速37–41%、次走関連約`.31`、欠損ほぼ0 | 既存SEC-3Fとの定義重複を監査し、非重複なら一表現だけ検証 | existing last3F | low | finish/timeと異なるperformance content | low |
| B1 | Low-degree rest × workload | focal cellの方向がD/R/C一致 | 事前固定1 interaction、age/class/historyをguardrail | existing history | low | nonlinear workloadが現main effectsを補うか | low |
| B2 | Connection-conditioned cold-start audit | history0でもconnectionは通常50+ starts、market gapはcold-historyで大 | feature追加前に新馬classとhistory0を分離したerror audit | existing + future one-source data | conditional | training除外でなく不足情報の場所 | low～medium |
| B3 | Margin token semantics audit | token order安定、同時計多い、秒mapping未同定 | official semantics/sourceだけ追加監査。mappingはまだ作らない | authoritative specification | low | PV-06を安全に再開可能か | medium |

## Acceptance rules for the selected next experiment

- discovery foldで変換・bin・priorを固定し、replication/confirmationで変更しない。
- Binary/Rankまたは新objectiveを同一data contractで比較する。
- primaryはrace-macro NDCG@3、Top-1、Log Loss、Brier。ranking改善でもproper-score guardrail、probability改善でもranking guardrailを置く。
- date block uncertainty、年別方向、多重比較回数、history0/new/open等の固定sliceを保存する。
- 2024は人間が明示的にmilestone確認を承認した重要候補だけ。2025は使わない。
- negative/inconclusiveもmachine-readable artifactとREADMEへ残す。

## Stop / defer list

| Item | Reason |
|---|---|
| Top3 auxiliary task / multitaskを直ちに追加 | 3着と4着に特別な境界を確認できず、auxiliary targetのstandalone価値が未確認 |
| full-order unweighted pairwise | winner関与pairは約14%だけで下位pairに支配される |
| new-horse race fit exclusion | 既存ablationが悪化し、EDAも除外を支持しない |
| old field-relative 15 columns | 削除で改善、current opponent meanはself rank逆変換を含む |
| margin token seconds mapping | orderは分かるが尺度根拠が不足 |
| connection列追加 | 既に130列あり、まず縮約が必要 |
| ensemble/calibration search | ENS-01/field-size temperatureはnegative、新predictive signalではない |
| CatBoost/XGBoost/DNN | representation/target gapの検証を先に行う |
| betting/UI/scraping | 現phaseとuser方針の範囲外 |

## Human decision requested

次の実行候補はS1、S2、S3の3件だけである。EDAはどれが勝つかを決めていない。情報表現を先に問うならS1、確率objectiveを先に問うならS2、教師情報の捨て方を先に問うならS3となる。選択されるまでproduction modelと局所feature queueは停止状態を維持する。
