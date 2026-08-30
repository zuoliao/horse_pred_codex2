# MVP実装仕様

**確定日:** 2026-08-30 (JST)  
**対象:** 開発タスクGOV-01～EVAL-01（タスク1～19。タスク16の実行に加え、後続比較の下準備を含む）

## 1. 予測プロダクト

初期baselineは`local_raw_pit_c_prevday`とする。承認済み結果rawをイベント日順に再構築するPIT-Cであり、当時配信版の完全再現ではない。rawに公表時刻がないため、target raceと同じ開催日の結果は、実際の発走順にかかわらず履歴へ利用しない。

初期モデルで利用するcurrent-race情報は、日付、競馬場、芝/ダート、距離、回り、raw classから作る粗いclass/条件、枠、馬番、性、年齢、field size、およびtarget日より前だけで作った履歴特徴とする。

天候、馬場状態、当日馬体重・増減、単勝、人気、当該着順・時計・着差・通過・上がりは初期primary modelへ入れない。これらは結果側、final-market側、または将来の`T_close`専用候補に分離する。

## 2. データとGit

- canonical raw filename: `race_results_merged.csv`
- raw pathは`--raw-path`または`HORSE_PRED_RAW_CSV`で注入し、workstation固有の絶対pathを仕様へ固定しない。
- raw fingerprint: `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`
- data pathはCLI引数または環境変数で注入し、コードへ固定しない。
- raw、中間、加工dataset、model、runner-level predictionは`data/`、`datasets/`、`artifacts/`に置きGit管理しない。
- Gitで管理するのはmanifest、config、schema、aggregate metrics、コード、テスト、要約文書である。

## 3. 対象race・結果規則

- JRA平地の芝・ダートのみ。障害は除外する。
- race全体を一つのsplitとし、runner単位で分割しない。
- 通常および降着表記の数値着順は数値部分をofficial finishとして保持する。
- `中`と`失`はstarterとしてfield sizeとstart countへ含め、winner=0、ranking relevance=0とする。数値finishを要する平均には入れない。
- `取`と`除`は非starterとしてstart/history更新へ入れない。発表時刻を復元できないため、これらを含むraceは初期scored datasetからrace単位で除外するが、監査表から削除しない。
- 同着1着は各runnerのbinary targetを1とし、coherent probability評価のtarget winner massは1着`m`頭へ`1/m`ずつ配る。
- official winnerを確定できないraceはscoring対象外とし、件数を報告する。
- final oddsと人気はmodel selectionに使わず、事後のmarket oracle診断だけに使用する。

## 4. PIT更新規則

日付`d`の全runnerについて、次の順序を必須とする。

1. `date < d`までに確定したstateだけから日付`d`の全race特徴をemitする。
2. 同日全raceの特徴emit完了後に、starterの結果をhorse/jockey/trainer/rating stateへ一括反映する。
3. 非starter、障害、target以後の結果は更新へ入れない。

最低限、同日別race非参照、同一race内非参照、future rows追加時の過去特徴不変、split境界、forbidden featureを自動テストする。

## 5. 時間分割

metric確認前に次を固定する。

| split | 期間 | 用途 |
|---|---|---|
| warm-up | 2013-01-01～2013-12-31 | 履歴state初期化のみ |
| train | 2014-01-01～2021-12-31 | model fit |
| model validation | 2022-01-01～2022-12-31 | LightGBM early stoppingだけに使用 |
| calibration | 2023-01-01～2023-12-31 | coherence/temperature calibration fitだけに使用 |
| development | 2024-01-01～2024-12-31 | 主なMVP比較。公式比108 race不足を必ず付記 |
| retrospective test | 2025-01-01～2025-12-31 | 追加の時系列頑健性確認。既に別repoで参照済みのためsealed finalではない |
| prospective final | 2026-01-01以降 | 将来観測が必要な真のfinal候補 |

2024/2025の結果を見て、この研究roundのfeature、model、calibrator、splitを変更しない。変更する場合は新しいexperiment IDと将来期間を必要とする。

## 6. feature contract

feature groupは分離し、各列にavailability semanticsを付ける。

- `race_context`: venue、surface、distance、around、class tier、age/sex、gate/post、field size
- `horse_history_basic`: starts、win/top3 rate、相対finish、直近1/3/5/10走、surface/distance帯/venue条件別集計
- `form_workload`: days since start、14/30/60/90/180/365日出走数・距離、指数減衰集計、過去馬体重
- `connections_pit`: jockey/trainerの過去starts、win/top3、performance
- `field_relative`: 今回field内の履歴/rating差、順位、z-score。target outcomeは使わない
- `rating_strength`: forward-only horse rating、field平均、平均との差、過去race strength

horse/jockey ID、馬名、騎手名、調教師名はstate結合にだけ使い、model featureへ直接入れない。

## 7. model contract

### Binary

- LightGBM `objective=binary`
- winner=1、その他starter=0。初期はclass weightを使わない。
- raw runner probability、race内合計、race内正規化版を別variantとして保存する。

### LambdaRank

- 1 race = 1 queryで、race runnerが連続していることを検証する。
- relevanceは1着=3、2着=2、3着=1、その他=0。同着は同じrelevance。
- `label_gain=[0,1,3,7]`、NDCG@1/3/5を固定する。
- scoreは確率と呼ばず、calibration前はranking専用とする。

## 8. coherent probabilityと校正

- Binaryはraw確率`p`を`epsilon=1e-6`でclipしてlogitへ変換し、
  `softmax(logit(clip(p))/T)`をrace-wiseに適用する。`T=1`版と2023年でfitしたtemperature版を区別して保存する。これは`p/sum(p)`ではない。
- LambdaRankはraw scoreへrace-wise softmaxを適用し、`T=1`版と2023年temperature版を保存する。
- temperatureは2023 calibration splitだけでfitし、winner massに対するrace-level Log Lossを最小化する。
- calibratorへodds・人気を入れない。
- calibration前後のrace内確率和、Log Loss、multiclass Brier、reliabilityを保存する。

## 9. MVP統合到達条件（タスク16～19）

- 一つのcommandでmanifest検証、feature生成、split、baseline、Binary、LambdaRank、temperature fit、2024評価を再現できる。2025 retrospective testは`--include-retrospective-test`を明示した場合だけ予測・評価・保存する。
- uniform baseline、Binary、LambdaRankについて、race macroのNDCG@1/3/5、top-1、Log Loss、Brier、reliability、主要subgroupを同一race集合で比較できる。
- final odds帯とnormalized market probabilityは別market artifactを使う事後診断として明記し、primary prediction artifactから分離する。bet selectionや実行可能ROIを主張しない。
- feature importance、aggregate metrics、run metadata、data/config/git fingerprintを保存する。
- rawデータ、runner-level prediction、model binaryはGit管理しない。
