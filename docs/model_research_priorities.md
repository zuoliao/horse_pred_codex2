# No-odds予測モデル研究の優先順位

**記録日:** 2026-08-31 (JST)  
**状態:** S0/S1完了。Aランク移行判断のsource of truth
**適用期間:** ユーザーによる明示的な改訂まで

## 1. 現状認識

現在は購入戦略やWeb UIを高度化する段階ではなく、no-odds予測モデルを
改善する段階である。

| Model / oracle | Features | NDCG@3 | Top-1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|
| Binary `pv_001_candidate_signed_time_gap` | 254 | .4976 | .2965 | 2.0787 | .82767 |
| LambdaRank `abl_006_drop_field_relative` | 253 | .4924 | .2881 | 2.0847 | .82995 |
| final-odds market oracle | — | .5539 | .3468 | 1.8828 | .7815 |

LambdaRankのPV-01追加版は全point metricが改善したが区間上inconclusiveで
ある。final oddsは締切後の市場情報であり、PIT-C no-oddsモデルとは情報集合
が異なる。「市場にまだ劣る」ことだけを理由に購入戦略へ移行しない。

2024 developmentは多数の実験で既に参照している。以後、2024単年のpoint
estimateだけで小さな改善を採用しない。2025は反復選択に使用しない。

## 2. 優先順位と最小実験

### S0: 先に閉じる基盤

| ID | テーマ | 最小実験・成果物 | 判断基準・制約 |
|---|---|---|---|
| DOC-SYNC | 現在状態の文書同期 | README、AGENTS、handoff、development planの完了・採用・棄却・次タスクを同期 | 古い「PV-02が次」等を残さない |
| EVAL-ROLL | rolling-origin評価基盤 | expanding-window複数fold。各fold内でfit、early stopping、calibrationを完結し、2020～2023等を年別評価。macro平均、年別方向一致率、block CIを保存 | 新仮説は原則rollingでscreen。2024は重要候補だけのmilestone。2025は反復選択に使わない |
| LIVE-DATA | prospective snapshot収集開始 | 出馬表、取消・変更、馬体重、天候、馬場、締切前oddsを`published_at`/`ingested_at`付きで低頻度・cache-first収集する設計と安全な開始状態 | モデル研究と並行。購入判断やROI最適化へ進まない。追加source gateと利用条件を守る |

### S1: rolling基盤上で完了する研究

| ID | テーマ | 最小実験・成果物 | 判断基準・制約 |
|---|---|---|---|
| PV-06 | raw着差tokenによる同時計補完 | 2014～2021でtoken順序・異常を監査し1変換を固定、2022を最初のgate | mappingを2024で調整しない。根拠が弱ければ終了 |
| OPP-RECENT | recent opponent-only field strength | 各過去raceで自身を除いた他馬のpre-race rating平均または上位平均を算出し、90日半減履歴平均1列だけ追加 | 対象馬自身の結果、相手の将来成績を混ぜない。career差分等を同時投入しない |
| SEC-3F | race-relative上がり3F履歴 | 過去race内上がり順位percentileまたはfield median差から1表現を固定し、90日減衰履歴1列だけ追加 | 通過順位・脚質interactionを混ぜない。絶対秒を異条件で直接比較しない |
| HPO-01 | LightGBM時系列hyperparameter探索 | rolling foldsだけでBinary/Rankerを別々に限定探索。leaves、min samples、depth、L1/L2、feature/bagging fraction、max bin等 | 2024で選ばない。feature追加とparameter探索を混ぜない。Ranker truncationは別仮説 |
| ENS-01 | Binary/LambdaRank単純ensemble | coherent probabilityの固定50:50またはlog-space平均を最初に評価。weightを選ぶならrollingだけ。独立temperatureをfit | 2024でweight探索しない。単体bestを明確に上回らなければ採用しない |

PV-06は2026-08-31に完了済みで、全point改善だがprimary Log Loss区間が
0を跨ぎinconclusiveだった。再調整しない。S1の残りもEVAL-ROLL固定後に
一仮説ずつ完了した。

### A: S完了後の有力候補

| Priority | ID | テーマ | 最初の限定scope |
|---|---|---|---|
| A1 | INFOSET-01 | T-close non-odds情報の価値診断 | 馬体重、増減、馬場、天候のoracle sensitivity。historical PITとは呼ばない |
| A1 | PACE-01 | 通過順位から脚質・位置取り | 序盤percentile、最終corner、position gain、先行率のhorse履歴。直線1000m等は別扱い |
| A1 | PACE-02 | field pace pressure | PACE-01支持後にfront-runner数等を少数追加 |
| A1 | SPEED-01 | 条件補正済みspeed figure | course/surface/distance/going/class/age条件の期待時計残差。未来track variant禁止 |
| A2 | COND-01 | condition transition適性 | 距離差、surface変更、休養と過去performanceの縮約interaction |
| A2 | CONN-01 | jockey/trainer分解・縮約 | jockey/trainer、短期/長期、条件別/全体を別ablation。直接ID禁止 |
| A2 | DATA-CORE | 基本情報の不足補完 | 正確なgrade/race条件、斤量・handicapを独立table・別ablationで優先 |

### B/C: 後続候補

| Priority | ID | テーマ | 制約 |
|---|---|---|---|
| B1 | TARGET-01 | continuous performance target | signed time-gapまたはspeed residualを固定後、regression/Huber。生着順MSEは禁止 |
| B1 | LTR-02 | LambdaRank relevance再検討 | full-order、着差bin、top-k gainを別実験。評価NDCGを都合よく変更しない |
| B2 | MODEL-ALT | CatBoost/XGBoost同条件比較 | 同じfeatures、rolling split、probability mapping。直接entity IDは禁止 |
| B2 | PROB-RANK | conditional logit / supervised PL | 透明な線形top-choice baselineから。online rating PL失敗と区別 |
| C | CAL-02 | 高度calibration | signal追加後。Log Loss/Brier primary、ECE単独改善では採用しない |
| C | DNN-01 | Intra-/Inter-horse DNN | 構造化signalを一巡後、小型DeepSetsから。同一PIT/splitでGBDT比較 |

## 3. 承認済み実行順

1. DOC-SYNC
2. PV-06の完了証拠確認（再実験・再調整はしない）
3. EVAL-ROLL
4. OPP-RECENT
5. SEC-3F
6. HPO-01
7. ENS-01
8. PACE-01、支持時のみPACE-02
9. SPEED-01
10. COND-01 / CONN-01
11. DATA-CORE
12. TARGET-01 / MODEL-ALT / PROB-RANK

LIVE-DATAは上記と並行して開始する。時間付きデータは後から取り戻せない
ため、予測研究中も安全な収集開始を遅らせない。

## 4. 当面優先しないもの

- margin-aware Eloのabsolute/delta等の追加算術変形
- 削除済みfield-relative 15列の単純復活
- calibration parameterだけの細かな探索
- 2024だけを見た大量feature search
- 2025を繰り返し開くmodel選択
- 複雑なEV閾値、Kelly、券種拡張
- Web UIの作り込み
- 大規模DNN

## 5. 共通実験ルール

- `1 experiment = 1 interpretable hypothesis`を維持する。
- 2025、final odds、人気をfeature/parameter選択やaccept/rejectに使わない。
- rolling-origin複数年で方向安定性を先に確認する。
- 2024は重要候補のmilestone確認に限定する。
- 単一metricで採用しない。
- ranking改善にもLog Loss/Brier guardrail、probability改善にもNDCG/Top-1
  guardrailを置く。
- 多重比較数と同じdevelopmentを見た回数をmetadataへ残す。
- negative resultもREADMEとmachine-readable summaryへ残す。
- final-odds marketはoracle診断だけに使う。
- 予測モデルと購入判断アルゴリズムを分離する。

## 6. Sランク完了条件

SランクはS0とS1を含む。次をすべて満たした時だけ完了とする。

1. DOC-SYNCが完了し、この文書がcurrent work queueとして参照される。
2. EVAL-ROLLがコード、config、test、baseline artifactを持つ。
3. LIVE-DATAが公式根拠、source gate、PIT-A schema、cache/diff/rate方針、
   executable collectorまたはsource非許諾時の明示的blocked stateを持つ。
4. PV-06の既存結果がcurrent decisionへ統合される。
5. OPP-RECENT、SEC-3F、HPO-01、ENS-01がそれぞれ独立preregistration、
   rolling evaluation、machine-readable result、accept/reject判断を持つ。
6. README、`docs/research/research_conclusions.md`、`docs/handoff.md`が
   Sランクの結果と次のAランクtaskを示す。

## 7. 2026-08-31完了判定

上記6条件を満たし、Sランクは完了した。結果は次のとおり。

| ID | 状態・判断 |
|---|---|
| DOC-SYNC | 完了 |
| EVAL-ROLL | 完了。2020～2023の4 fold baselineを固定 |
| LIVE-DATA | 公式JV-Link source gate、schema、append-only archive、test完了。実収集はWindowsのみの公式transportとMac環境の不一致によりユーザー保留 |
| PV-06 | inconclusive、2024未開封 |
| OPP-RECENT | Binary inconclusive、LambdaRank reject、未採用 |
| SEC-3F | Binary inconclusive、LambdaRank rolling acceptance |
| HPO-01 | Binary no-change、LambdaRank candidate reject。parameter維持 |
| ENS-01 | 固定50:50はscreen reject。weight探索なし、2023 confirmation未開封 |

追加で、新馬戦をBinary fitからだけ除外する`SHIMBA-FILTER-001`もrolling
評価しrejectした。新馬戦は難しいが学習からは除外しない。

統合結論は
`docs/experiments/s_rank_model_research_conclusions_20260831.md`、機械可読集約は
`experiments/s_rank_model_research_20260831/summary.json`を参照する。次のmodeling
S完了時点の次taskは`PACE-01`だった。現在の結果と次taskは§8を参照する。
LIVE-DATA実収集はユーザーが明示的に再開するまで保留する。

## 8. PACE-01完了と次task

`PACE-01`は2014～2021だけのmapping監査後、2区間以上ある過去走の最初の
通過位置をrace内percentile化し、90日半減履歴1列として固定した。1区間のみの
198 raceは全て新潟芝直線1000 mであり、corner位置と解釈せず観測更新から
除外した。

2020～2023 rollingでBinaryはLog Loss `+.00659 [+.00333,+.00978]`、
LambdaRankはNDCG `+.00440 [+.00197,+.00683]`。Binaryはprobability path、
LambdaRankはranking/probability両pathを通過し、両familyで採択した。2024/2025
は未使用。次taskは`PACE-02`とし、固定済みPACE-01からcurrent-fieldの先行集中を
一表現だけ作る。最終corner、position gain、複数thresholdは混ぜない。

LIVE-DATA実収集は、公式JV-LinkがWindowsのみで現環境がMacであるため、
ユーザー判断で後日に延期した。非公式Mac transportやJRA Web scrapingへは
切り替えない。
