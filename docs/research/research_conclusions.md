# 調査結論と実装前仕様

**対象:** JRA中央競馬の予測・購入判断支援<br>
**調査統合日:** 2026-08-30 (JST)<br>
**状態:** 調査・仕様化と既存ローカルデータの採用判断まで完了。本格実装、追加取得、学習、バックテストは未着手。

## 1. 結論

本プロジェクトの既決事項は、研究によって概ね支持された。特に、次は保持する。

- JRA中央競馬を対象とし、初期MVPはデータ生成機序が異なる障害競走を除いた**平地競走（芝・ダート）**に絞る。
- 予測モデルと購入判断を分離する。
- 主予測モデルと主校正器にはオッズを入れず、オッズは市場比較と購入判断にのみ使う。
- 同一のpoint-in-time特徴と時間分割でLightGBM BinaryとLambdaRankを比較する。
- 馬・騎手・調教師等のIDは結合と逐次状態管理に使うが、初期モデルへ生カテゴリとして投入しない。
- 履歴は保持し、複数走数窓、暦日窓、指数減衰、条件別集約を並列に作る。
- 単勝、固定額、固定閾値の単純な購入ルールから始める。
- 初期改善の採否をROIだけで決めず、ranking、確率、校正、条件別、市場比較、バックテストを併記する。
- DNN、複雑な確率順位モデル、資金配分最適化は、構造化データのベースライン後へ延期する。

ただし、次の重要な修正が必要である。

1. **承認済み既存rawによる無料予測trackを先に置く。** ユーザーが本非公開・私的プロジェクト内での利用を承認した、netkeiba由来のローカルraw（2013～2025年、629,967出走、44,761レース）をMVP主データとする。JRA公式結果はcoverage・grade・lap・払戻等の照合・補完候補とし、新規の体系的取得は別のlicense gateに従う。
2. **長期成績データがあることと、当時見えていた状態・オッズを長期再現できることは別である。** 無料PDFには締切前odds履歴がなく、JRA-VANの時系列oddsも公式保証が1年である。長期の実行可能ROI検証を当然には作れない。
3. **Binaryのraw確率とLambdaRank scoreを、そのままEVへ渡さない。** race内合計1のcoherentな最終勝率ベクトルを明示的に作り、確率評価を通過したものだけを購入判断へ渡す。
4. **`P(win) × 暫定オッズ`は真のEVではない。** JRAはpari-mutuelで表示価格を固定できないため、これは `snapshot EV proxy` と呼ぶ。
5. **過去レース価値に二つの時点意味を持たせる。** 過去レース当時に凍結したex-ante評価に加え、ターゲット予測時点までに確定・公表された相手の後続成績によるas-of再評価も概念上はPIT整合にできる。ただし、ターゲットごとの再構築、版管理、transaction-time品質の制約を満たす場合に限る。
6. **前日モデルと当日モデルを分ける。** 馬体重、天候・馬場、当日変更は、発表前の予測へ遡及投入しない。

## 2. 統合した調査成果

| Workstream | 文書 | 主な役割 |
|---|---|---|
| A | [data_sources.md](data_sources.md) | データ源、料金、取得方式、規約、履歴 |
| B | [available_features.md](available_features.md) | 生項目、派生特徴、欠損・有用性 |
| B | [point_in_time.md](point_in_time.md) | 予測時点、版管理、as-of join、PIT品質 |
| C | [horse_racing_prediction.md](horse_racing_prediction.md) | GBDT、Binary、LTR、競馬予測研究 |
| D | [ranking_rating_and_strength.md](ranking_rating_and_strength.md) | rating、相手強度、PL/BT/Thurstone、過去走価値 |
| E | [form_fatigue_and_interactions.md](form_fatigue_and_interactions.md) | フォーム、負荷、適性、枠、脚質、ペース |
| F | [probability_and_calibration.md](probability_and_calibration.md) | proper score、校正、coherent確率 |
| G | [betting_market.md](betting_market.md) | JRA払戻、市場暗黙確率、暫定オッズ |
| H | [backtesting_and_leakage.md](backtesting_and_leakage.md) | 時間分割、リーク、ROI不確実性、試行管理 |
| A追加 | [free_data_mvp.md](free_data_mvp.md) | 無料優先MVPの統合判断、7問への回答、decision gate |
| A追加 | [free_data_netkeiba.md](free_data_netkeiba.md) | netkeiba無料層、URL/ID、規約、通信制限、PIT |
| A追加 | [free_data_japan_sources.md](free_data_japan_sources.md) | JRA/JBIS/競馬ラボ/NAR/JMA等の国内比較 |
| A追加 | [free_data_public_datasets.md](free_data_public_datasets.md) | Kaggle/GitHub、公開API、海外公式dataのprovenance |
| 横断review | [free_data_synthesis_review.md](free_data_synthesis_review.md) | JRA PDF主候補案の批判レビュー、A/B/C判定 |
| 追加監査 | [local_existing_dataset.md](local_existing_dataset.md) | 既存ローカルrawのprovenance、coverage、リーク監査、利用承認、採用判断 |

各workstreamの完了後、モデル→確率→市場→精算の接続、データPIT、疲労・適性の外的妥当性を別担当者が横断レビューした。レビューで発見した、過去レース価値の過剰な凍結、無情報なcalibration-in-the-large、snapshot EVの用語、同着・取消・ROI bootstrapの曖昧さは担当文書と本結論へ反映した。

## 3. 証拠強度

| 結論 | 信頼度 | 根拠と限界 |
|---|---:|---|
| 承認済みローカルraw 2013～2025を無料MVP主データにする | 高 | ユーザーが本非公開・私的プロジェクト内利用を明示承認。機械可読でID・主要結果項目を持つ。coverage欠損、PIT-C、締切前odds不在は残る |
| JRA成績PDF 2002+を照合・長期補完候補にする | 中～高 | 公式一次文書、全成績・主要field・まとまったPDF。構造化難度、安定ID、PIT、新規取得の利用条件が弱い |
| JRA-VAN/JV-Linkを厳密PIT・市場trackの有料主系統にする | 高 | 公式SDK、仕様、料金、2026-08-07に質問された特定条件下の個人研究を可とする公式Staff回答。ただし派生成果公開等は未確認 |
| JRA Web/netkeiba/JBISを無許諾スクレイピングしない | 高 | 閲覧可能性は自動取得・保存・ML許諾を意味せず、包括的な公式許諾/APIを確認できなかった |
| LightGBM BinaryとLambdaRankを併設する | 中～高 | LightGBM公式仕様と表形式一般の根拠は強い。競馬文献はwinner分類優位とLTR優位で衝突し、現代JRAの直接比較はない |
| online Elo/BTを最初の相手調整ratingにする | 中 | 理論と他競技の動的rating根拠はあるが、現代JRAの方式比較はない |
| race-wideな馬場・course・距離・年齢contextを特徴にする | 中～高 | JRA直接研究と公式情報がある。ただし個体ごとの適性や因果効果までは直接示さない |
| 個体別の適性・form・負荷・馬体重変化を柔軟な特徴にする | 低～中 | 機序と一部JRA/海外観察研究はあるが、選択交絡と外的妥当性の制約が強い。単調な因果則や固定閾値は支持されない |
| 校正は時間外sliceで行い、Log Loss/Brier/reliabilityで選ぶ | 高 | proper scoring ruleと校正文献。JRA固有の最良校正器は未確定 |
| final oddsによる選択はlook-ahead | 高 | JRA公式のpari-mutuel仕様と、最終オッズが購入時に未確定であることから直接導かれる |
| JRAのfavorite–longshot biasを収益源と仮定しない | 高 | 日本を含む証拠はあるが、期間・券種差が大きい。2025年の直前変動研究は4頁の予備的報告 |

## 4. 推奨データ源と利用条件

### 4.1 主ソース

無料優先MVPでは、**利用承認済みの既存ローカルraw（2013～2025年）**を機械可読なno-odds予測baselineの主ソースとする。詳細な監査結果と利用範囲は[既存ローカルデータセットの監査](local_existing_dataset.md)に従う。

- 629,967出走、44,761レースを持ち、race/horse/jockey ID、条件、着順、時計、通過、上がり3F、最終単勝オッズ、馬体重を利用できる。
- 本リポジトリは非公開・私的利用であり、ユーザーは対象データの本プロジェクト内利用を明示的に承認した。したがって既存データに対するfree license gateは閉じた。
- 元データはnetkeiba由来としてprovenanceを保持する。この承認を新規scraping、再配布、外部公開、第三者利用へ一般化しない。
- 既存`features.csv`には全期間統計と同一race内集計によるリークがあるため使用せず、rawからPIT pipelineで再生成する。
- 2015年2レース、2017年36レース、2024年108レースの不足が確認されており、coverage gateは未完了である。
- これは `PIT-C event reconstruction` であり、当時のpre-race版、変更時刻、締切前oddsを再現しない。

**JRA公式全レース成績PDF（2002年以降）**は、主催者一次情報としてcoverage、grade、lap、票数、払戻、例外の照合・補完候補とする。

- 主催者公式の確定結果で、競走条件、出走馬、斤量、馬体重、時計・着差、通過・上がり、race-level lap、最終単勝odds・人気、払戻、票数等を一つの系統から得られる。
- raceごとのHTML巡回より少ない文書数で全開催を確認できる。一方、年度によりPDFの文書単位・命名・text抽出順が異なり、安定した外部ID/schemaもない。
- JRAはopen-data/ML license、公開API、機械取得rateを明示していない。全履歴の体系的取得、長期保存、数値抽出、個人ML、派生成果公開の用途を具体化して書面照会し、回答前は少数手動PoCに留める。

**JRA-VAN Data Lab./JV-Link**は、無料coreの限界を越える最優先の有料upgradeとする。

- 2026-08-30時点で月額2,090円（税込）。40年超のJRA公式データを掲げる。[公式製品ページ](https://jra-van.jp/dlb/)
- JV-LinkはWindows 10/11日本語版のActiveX/COMで、正規インターフェース経由の取得が必要。[公式システム概要](https://jra-van.jp/dlb/sdv/about.html)
- 公式Staffは、正規SDK/JV-Link、個人のローカル研究、質問対象の調教データを使い、生データを外部AIへ送らないという具体的な質問条件を規約上問題ないと回答した。この回答範囲を他の利用形態へ一般化しない。[公式開発者コミュニティ回答](https://developer.jra-van.jp/t/topic/964)
- 一方、利用規約第7条は私的利用外の複製・改変・公表・第三者利用を強く制限する。[JRA-VAN利用規約](https://jra-van.jp/info/rule.html)

JRA-VANを導入した場合、公開可否を書面確認するまでは、保守策として生データ、復元可能な派生データ、学習済みモデル、個別予測を公開リポジトリや外部AI/SaaSへ置かない。規約がこれらをすべて同じ法的扱いで明示禁止した、という意味ではない。コード公開とデータ・artifact公開を分ける。サービス利用地域を日本国内に限定する条項もあるため、取得hostと保存・処理先の所在を確認し、許諾なしに海外cloud regionへraw dataを転送しない。複数人共有、cloud保存、第三者向けWeb表示、model配布、有償利用へ進む前に、具体的用途を記載してJRA-VAN/JRADBへ書面確認する。

### 4.2 補助ソース

- **JRA現行出馬表/当日Web:** 許諾後、prospective PIT-Aの候補。出馬表PDFは原則8週間で、後日archiveではない。
- **気象庁CSV:** 公共データ利用規約、出典・加工表示、品質flagを守り、降水・気温・風等の独立補完候補とする。JRA公表weather/goingを置換しない。
- **JBIS:** 血統・生産・セリの照合候補。大量取得・加工MLは事前許諾が必要。
- **netkeiba:** 承認済み既存rawはMVP主データとして使用する。コメント、独自指数、調教表示等を得るための新規自動取得は、公開バルクAPI、PITスナップショット、包括的な自動取得許諾を確認できないため別途採用しない。
- **JRDB:** 独自指数・調教・パドックの増分を後で一特徴群ずつ検証する候補。履歴、PIT、オッズ依存、ML利用許諾を契約前に確認する。
- **NAR公式DL:** JRA馬の地方歴補完候補。JRA主対象のベースライン後へ延期し、利用条件とID対応を確認する。

可視ページ、規則的URL、robots許可/不在を機械取得・保存・MLの許諾と解釈しない。無許諾のnetkeiba/JBIS/競馬ラボscraping、競馬ブックのprogram取込、upstream権利不明のKaggle/GitHub JRA datasetは不採用とする。

## 5. 期待できる履歴と、期待してはいけない履歴

| データ | 公式仕様・確認結果 | プロジェクト上の意味 |
|---|---|---|
| 承認済み既存raw | 2013～2025。629,967出走、44,761レース。146レース以上のcoverage不足あり | 無料MVP主データ。PIT-C、final oddsのみ、既存featuresは再生成 |
| JRA Web結果 | 1986+。2000年以前は一部情報が不完全または表示差 | 無料照合。単一schemaの主取得には向かない |
| JRA全レース成績PDF | 2002+。結果、馬体重、通過・上がり、lap、最終odds・払戻等 | 公式照合・長期補完候補。新規の体系的取得・保存・MLはlicense gate |
| JRA現行出馬表PDF | 原則8週間掲載 | 許諾後に今後のPIT-Aを蓄積。過去へ遡れない |
| JRA-VANの主なレース・馬・払戻等 | 原則1986+。1994-07より前は一部項目未整備 | 有料upgradeの長期結果・履歴基盤 |
| JRA-VAN最終単複枠・馬連odds | 1993-06+ | 事後市場比較には使えるが、実行可能選択には使えない |
| JRA-VAN木曜時点SNAP/SNPN | 2004+、online保持保証1年 | 関係者・馬情報の時点版。ただし長期任意時点を保証しない |
| JRA-VAN坂路/wood調教 | 坂路2003+。woodは美浦2021-07-27+、栗東2021-12-07+ | 無料では困難な独立feature群 |
| JRA-VAN時系列odds | 単複枠・馬連、約5～10分間隔、公式保証1年 | 実行可能ROI評価の最大制約。導入初日から継続保存 |

「2002年以降の公式PDF」も「JRA-VANの1986年以降」も、完全なtransaction-time履歴を意味しない。PDFは結果版、JRA-VAN初回setupは後日訂正を反映したlatest-known版でありうる。項目×年×競馬場×レース種別の欠損率、PDF layout、訂正、コード変遷を測るまで、学習開始年と具体split年を確定しない。

## 6. 予測時点とPIT品質

### 6.1 二つの予測プロダクト

承認済みrawと無料の長期結果PDFから再現できるのは、target以前の確定結果から作る `PIT-C` のno-odds予測である。下表の `T_prevday` / `T_close` を厳密に再現するには、prospective snapshotまたはJRA-VANが必要である。結果ページ/PDFの当日馬体重・天候・馬場を、発表時刻の証拠なしに過去の `T_close` へ投入しない。

| 版 | 暫定cutoff | 目的 | 主な利用情報 |
|---|---|---|---|
| `T_prevday` | 馬番確定出馬表取得後、開催前日18:00 JST候補 | 長期で比較しやすい早期予測 | 出馬表、枠、斤量、騎手、過去情報、cutoffまでの変更 |
| `T_close` | その時点で公表済みの予定発走10分前候補 | 実運用・購入判断 | 上記＋発表済み馬体重、天候・馬場、変更、同cutoff以前のオッズ |

18:00とT-10は研究上の暫定値であり、公式締切保証ではない。JV-Linkの実配信、`published_at`、受信遅延、特徴生成時間、購入導線を計測後、T-15またはT-10を一つ事前登録する。実発走時刻から逆算して都合の良いスナップショットを選ばない。cutoff時点の`scheduled_start_version`で一度閉じ、閉鎖後の発走延期で再判断する場合は別`decision_snapshot_id`として事前規則化する。

主な将来運用対象は`T_close`、比較用・fallbackは`T_prevday`とする。ただし長期の`T_close`相当履歴は不完全なので、次の二つを別manifest・別coverage・別finalとして扱う。

- **Free prediction track:** 承認済み既存raw 2013～2025を中心とし、JRA公式結果でcoverage・項目を照合するPIT-Cのno-odds予測研究。予測仮説比較には使えるが、実行可能ROIの証拠にしない。
- **Executable-market track:** PIT-Bでのsource-time replayと、PIT-Aでのend-to-end operational/shadow評価。PIT-Bは意思決定情報時点には整合しうるが、当方の当時受信遅延・処理・購入可能性までは証明しない。実運用可能性の主張はPIT-Aに限定する。

### 6.2 時刻と品質

最低限、`event_time`, `published_at`, `source_effective_at`, `ingested_at`, `processing_completed_at`, `scheduled_start_version`, `actual_start`, `finalized_at`, `source_revision`を分ける。

PIT品質は次の4段階で保存する。

- `PIT-A observed`: 当方が実時間受信し全版を保持。
- `PIT-B source snapshot`: 公式時系列/スナップショットを後日取得し発表時刻がある。
- `PIT-C event reconstruction`: 最新既知履歴を過去イベント順に再構築。当時版保証なし。
- `PIT-D current-master inferred`: 現在値から過去を推測。原則モデル入力禁止。

情報時点整合性の主張はPIT-A/Bに限定する。実際の受信遅延、処理完了、購入可能性まで含むend-to-end実行可能性の主張は、投票導線を別途検証したPIT-Aに限定する。PIT-Cは長期学習と仮説比較には使えるが、当時版完全再現や長期収益の証拠としない。

## 7. MVPデータと特徴量仕様

### 7.1 分離するテーブル層

```text
append-only source records
    -> PIT-filtered event/history state
    -> no-odds model features
    -> raw model outputs
    -> coherent race probability

market snapshots ---------------------> market comparison / decision only
results and official payouts ---------> labels / settlement only
```

市場列や結果列を「便利な一枚表」に混在させない。データセットmanifestには、ソース版、取得期間、PIT品質比率、cutoff、欠損率、訂正件数、データfingerprintを保存する。

### 7.2 MVP特徴量の優先群

| Feature group | 内容 | 初期位置づけ |
|---|---|---|
| `race_context` | 場、芝/ダート、距離、コース、クラス/条件、頭数、枠/馬番、斤量、年齢、性 | core baseline |
| `horse_history_basic` | 複数窓の相対着順、着差、条件調整time、上がり、通過順位、過去馬体重、観測数・欠損 | core baseline |
| `connections_pit` | 騎手・調教師・馬×騎手の過去のみの縮約統計、乗替り | core baseline |
| `field_relative` | 自馬特徴のfield内順位/差、pre-race能力分布、cold-start数 | core後の最初の追加 |
| `rating_strength` | PIT-safe forward-only Elo/BT、相手強度、expected-vs-actual surprise | 独立仮説として追加 |
| `form_workload` | 日数、14/30/60/90/180/365日出走数・距離、campaign、複数decay | 独立仮説として追加 |
| `suitability` | 縮約した芝ダート・広い距離帯・馬場・コース適性 | 条件を一つずつ追加 |
| `pedigree_history` | 時点以前の父/母父/牝系産駒統計 | legal/coverage監査後 |
| `training` | 坂路、後にウッド。時計・本数・最終追切間隔・coverage | 独立ablation、未収録≠ゼロ |
| `race_day_state` | 当日馬体重の個体内偏差/比率、発表済み天候・馬場・変更 | `T_close`専用 |
| `market_snapshot` | 時点単勝オッズ、取得できる範囲の票数、鮮度 | 主モデル外、比較・判断専用。票数はschema/coverage監査後 |

生の馬ID、騎手ID、調教師ID、父ID等はモデルへ入れない。これらは時点正しい履歴結合とstate更新のキーにのみ使い、数値・縮約統計・supportへ変換する。未経験馬はpopulation prior、観測数、uncertainty、missing indicatorで扱う。

無料JRA PDF trackの初期対象は、`race_context`、`horse_history_basic`、名称監査付き`connections_pit`、`field_relative`、`rating_strength`、`form_workload`、粗い`suitability`、`weather_jma`とする。`training`、長期`market_snapshot`、深い`pedigree_history`、時刻付き`race_day_state`は無料coreに含めず、許諾済みprospective dataまたは有料sourceの独立ablationとする。

### 7.3 フォーム・疲労・適性の解釈

- `days_since_last_start`は観測可能だが、休養中の訓練量ではない。
- 出走数・距離は負荷の代理であり、適応と損傷の両方を含む。単一「疲労スコア」へ圧縮しない。
- 馬体重は固定±kg閾値を使わず、個体内偏差、変化率、年齢・性・季節、測定間隔を併記する。`T_close`だけで使う。
- 距離・surface・馬場・コース適性は、raw勝率ではなく相手・クラス等を調整した成績を縮約する。
- 枠はJRAのコンピュータ抽選により比較的監査しやすいが、コース、距離、頭数、脚質との限定的な交互作用として扱う。
- 「逃げ馬が多ければ差し有利」は機序的に妥当でもJRA直接証拠が不足する。field pace構成は独立実験であり、ハードルールにしない。

## 8. Ratingと過去レース価値

### 8.1 最初のrating

最初は透明な**PIT-safe forward-only Elo/Bradley–Terry-style rating**を特徴生成器として用いる。これは未来を使わない逐次更新を意味し、ratingが因果効果を推定するという意味ではない。

- race前snapshotだけを予測行へ結合する。
- 同一raceの全期待値をpre-race状態で計算し、結果後にbatch更新する。
- pairwise actual/expectedの集約をfield sizeに対して正規化し、runner順序によらない更新にする。
- 同着はtie、cutoff後取消はno updateとする。
- `algorithm_version`, `pre_state`, `post_state`, `rating`, `starts_count`, `days_since_update`, `coverage`を保存する。`starts/recency/coverage proxy`をBayesian uncertaintyと呼ばない。
- global ratingから始め、surface、広い距離帯、surface×距離deviationを一つずつ追加する。
- K、初期値、margin利用、decay、context blendはdevelopmentだけで選び、versionを保存する。

TrueSkill/Glicko等は、単純ratingのcold-start・uncertainty問題を実測した後へ延期する。TrueSkill Through Timeのようにfutureを使うsmoothing stateは予測特徴にしない。

### 8.2 過去走価値の二つの意味

過去レース`h`に対して次を別列・別feature groupにする。

1. `value_ex_ante_h`: `h`直前のratingとfieldでexpected performanceを作り、実着順/相手を上回った割合との差を保存する。
2. `value_reassessed_asof_target`: ターゲットcutoffまでに判明した相手の後続成績で`h`のfieldを再評価する。

後者もターゲットcutoff以下で確定・公表された情報だけなら概念上利用可能である。ただし各targetごとに相手stateを`available/finalized_at < target_cutoff`だけで再構築し、`h`のimmutableなex-ante stateを書き換えずtarget-specific derived featureとして別保存する。同じtarget race出走馬のcurrent stateもcutoff直前までに限定する。計算コストとtransaction-time品質制約が大きいため、前者をMVP基本、後者を独立ablationとする。PIT-Cでは当時版欠如の品質限界を明示し、ターゲット後の相手成績、生涯最終成績、full-history fit/smoothed ratingは使わない。

過去走価値を一つの人手scoreに固定せず、expected-vs-actual、class、margin、標準化time、lap/pace、斤量、枠、馬場、coverageを別列としてLightGBMへ渡す。

## 9. MVPモデル・出力契約

### 9.1 共通

- 1 raceを分割不能なgroupとし、全runnerを同じsplitへ置く。
- 比較モデルは同一データfingerprint、特徴、cutoff、race集合を使う。
- 比較不能な欠損raceは共通集合の主指標と、各モデルcoverageを分けて報告する。
- flat win targetを主対象とし、row/query集合は予測cutoff時点のeligibleな`as_of_field`で固定する。

### 9.2 LightGBM Binary

| 項目 | 初期仕様 |
|---|---|
| row | runner |
| target | official first place = 1、他のas-of runner = 0。cutoff後取消も0。1着同着は各馬1のmarginal target |
| objective | `binary` |
| race weight | 各runner `1 / as_of_field_size`、race総weight 1。unweighted runner学習は別感度実験 |
| class weight | `is_unbalance=false`, `scale_pos_weight=1`, `class_weight=None` |
| output | raw runner probability `p_i` と race内`sum(p_i)` |

race weightingはclass balancingではない。大頭数raceを学習上自動的に重くせず、評価単位をraceへ寄せる工学的初期選択であり、JRAで最適という証拠はない。重み付き/無重みの選択自体に結果差が出る可能性があるため、主仕様を固定し、変更は独立実験とする。raw Binary targetは同着時に正例質量が`m`となるmarginal eventで、後述するcoherent winner-mass scoreとは意味が異なるため、target conventionをartifactへ保存する。

### 9.3 LightGBM LambdaRank

| 項目 | 初期仕様 |
|---|---|
| query | 1 race = 1 group |
| relevance | 1着=3、2着=2、3着=1、他=0。同着は同じrelevance |
| `label_gain` | `[0, 1, 3, 7]` |
| `eval_at` | `[1, 3, 5]` |
| primary ranking metric | NDCG@3。NDCG@1/@5とexact Top-1は診断 |
| truncation | `lambdarank_truncation_level=6`を初期固定 |
| query normalization | `lambdarank_norm=true` |
| extra weight | primaryでは追加row/query weightなし |
| output | raw race ranking score `s_i`。確率と呼ばない |

full-order linear gainは「下位着順も情報として使う」別仮説であり、初期top-3 relevanceと混ぜない。`truncation_level=6`はtop-3目的の工学的初期値で、JRA最適の証拠はない。NDCGはrace内計算後にrace macro平均する。NDCG@1は2・3着にも部分利得を与えるためexact Top-1と同義ではない。Spearman/Kendallは完走馬の補助指標とする。

### 9.4 三層の出力

```text
Binary raw p_i                  LambdaRank raw s_i
        \                              /
         \-- explicit calibration/coherence mapping --/
                              ↓
                 coherent q_i, sum_i(q_i)=1
                              ↓
                 market comparison / EV proxy
```

raw Binary `p_i`、raw LambdaRank `s_i`、最終coherent `q_i`を別artifactとして保存する。`p_i`や`s_i`を直接EV proxyへ渡さない。

初期の明示的候補は次である。

- Binary: `q_i=p_i/sum_j(p_j)`をidentity baselineとする。
- Binary calibration challengers: Plattは保存したraw marginまたは`logit(clip(p, eps))`へ、betaは`log(clip(p, eps))`と`log(1-clip(p, eps))`へ、isotonicはraw `p`へ適用し、その後race正規化する。`eps`はconfigへ保存する。
- Binary coherence challenger: `softmax(logit(p_i)/T)`。
- LambdaRank: `softmax(s_i/T)`をfirst-place mappingとして使う。

runner calibratorはbase modelが未学習の後続calibration sliceだけで、各row `1/n_asof` weightのbinary Log Lossを最小化してfitする。isotonicにも同じweightを使う。`T>0`は各race一票のfractional winner targetによるrace winner NLLを最小化する。Lambda softmaxはPlackett–Luce-likeな**一着mapping**に過ぎず、full-rank PL modelや順位別校正済み分布とは呼ばない。どのbinary calibratorも正規化後に校正が変わりうるため、最終`q`で評価する。中間の未正規化出力は診断専用で、EV proxyにはcoherent `q`だけを渡す。

mappingはrolling developmentの平均race winner NLLをprimaryとして一つ選び、Brierとreliabilityをguardrail・報告指標にする。`sum(p)=0`、NaN/inf、候補1頭以下はfail/no-scoreとし、0/1のclip、fallback、coverageをdata contractで固定する。

## 10. 確率校正計画

```text
warm-up/history
    -> base-model train
    -> out-of-time calibration slice (base model未学習)
    -> development evaluation and rule selection
    -> untouched final
```

- rolling-origin development foldごとに上記順序を守る。
- calibratorはそのraceを学習していないbase modelのscoreだけでfitする。
- mapping familyの最終選択はcoherent `q`の平均race winner NLLをprimaryとし、race multiclass Brierとreliabilityをguardrail・報告指標にする。複数指標のうち都合のよいものを事後選択せず、ROIで校正器を選ばない。
- 最初はglobal calibrator。field size、surface、class等は診断し、安定した誤差と十分な勝者数がある場合だけ別実験にする。
- 主校正器にオッズやオッズ帯を入れない。それはodds-aware modelとして別比較にする。
- ECEはbin数・境界・件数を保存した副指標。単独採否に使わない。
- reliabilityはrunner-weightedとrace-weightedを明示し、uncertaintyはrace cluster、時間安定性はrace-day/week blockで扱う。
- 同着は`m`頭の1着にwinner mass `1/m`を割り、`-sum(y_i log q_i)`とBrierを計算する。これは擬似target規約であり、同着joint eventモデルではない。除外感度も報告する。

final用recipeは次で固定する。rolling foldのout-of-time予測だけでbase hyperparameter、mapping、閾値を選ぶ。final直前にfresh calibration windowを確保し、baseはそのwindow開始前まででfit、そのbaseのout-of-time scoreでcalibratorをfitする。baseが学習した行のscoreでcalibratorをfitしない。final中はbaseとcalibratorを固定する。developmentをbase再学習へ加える場合も、base未学習のfresh calibration windowを残す。

## 11. 市場確率と購入判断

### 11.1 市場baseline

時点付き表示単勝オッズ`o_i,t`から

\[
q^{mkt}_{i,t}=\frac{1/o_{i,t}}{\sum_j 1/o_{j,t}}
\]

を作る。これは控除と共通scaleを除いたpool share近似であり、「客観的な真の確率」ではない。時点付き票数から正確にpool shareを再構築できる場合はそちらを優先する。JRAの通常単勝払戻率は80%である。[JRA公式払戻式](https://www.jra.go.jp/faq/pop03/1_17.html)

同一decision snapshot、同一as-of field、同一race集合で、uniform `1/n`、市場、no-odds modelを比較する。final oddsは`final-market oracle diagnostic`だけに使う。

### 11.2 固定初期ルール

このルールは、許諾済みのcutoff付きodds snapshotを持つprospective/executable-market trackだけに適用する。無料の長期PDFにあるfinal oddsで過去の購入馬を選ばない。

```text
券種:                 単勝
購入額:               選択馬1頭につき1 unit（運用時は¥100を想定）
選択score:            coherent q_i × decision snapshot表示オッズ
scoreの名称:          snapshot EV proxy
primary threshold:    > 1.10（証拠で最適とされた値ではなく、事前固定する工学的初期値）
diagnostic thresholds: > 1.00, > 1.20
精算:                 JRA公式の実払戻・返還
資金配分:             最適化しない、Kelly不使用
```

1.10は価格変動を正確に補正する値ではない。閾値をdevelopmentで別値へ変更する場合、それ自体を記録された戦略実験とし、final前に一つ固定する。複数馬が通過すれば各馬1 unitとし、race当たり購入数と総exposureを報告する。

snapshot表示オッズはlockされない。snapshot→最終払戻倍率の変化、選択後にproxyが閾値を下回る割合、オッズ鮮度をslippage診断として保存する。実現損益は表示最終オッズでなく公式払戻で計算する。

cutoff後取消の確率を別途予測しない初期モデルでは、snapshot EV proxyは出走継続確率と返還を明示モデル化していない。この限界も報告し、取消ticketは実際の公式返還で精算する。

## 12. 分割と評価仕様

### 12.1 期間

実データのcoverageを見ずに絶対年を断定しない。取得後、結果を見ずにtrackごとに期間を固定する。

**Prediction track（長期no-odds予測）**

- warm-up: scoring開始以前の利用可能履歴。rating・集約初期化に使う。
- train: developmentより前の全適格履歴をexpanding windowで使用。
- development: 原則としてfinal直前の3完全暦年を、少なくとも3つのrolling annual originで評価する。
- calibration: 各development評価窓の直前6か月を暫定初期幅とし、base trainと評価から分離する。幅変更はdevelopment内だけで検証する。
- final: 検証可能なcore historyが十分なら最新の2完全暦年、短ければ1完全暦年。dataset freeze時に年を記録し、全research roundで一度だけ開封する。

**Executable-market track（時点オッズを使う意思決定）**

- 上記の「development 3年＋final 1～2年」を機械的に適用しない。時系列オッズの公式保持保証が1年だからである。
- 取得できたPIT-B期間はsource-time replayとして別manifestで評価し、PIT-Aは蓄積開始日からshadow/prospective評価する。
- final期間を確保できない短い履歴では、収益性の確証を主張せず、coverage・slippage・精算正確性を中心に報告する。

2026年に最新完全年が2025年でも、`2024–2025をfinal`と今ここで確定はしない。項目coverageとPIT品質監査を終えた後、outcomeを集計・モデル評価する前にmanifestへ固定する。

初期finalは、final開始前に§10の手順でfreezeした一つのbase model＋calibratorで最後まで通す。年次・四半期等の逐次refitを評価する場合は、別実験としてcadence、各training/calibration cutoff、model fingerprintを事前固定する。

### 12.2 指標

**Ranking**

- primary: NDCG@3
- exact Top-1、NDCG@1/@5、winner MRR
- Spearman/Kendallは完走馬のみの補助

**Probability**

- primary: coherent `q`のrace winner NLL
- race multiclass Brier（1/2係数なし）、uniform skill、field-size別
- runner-micro binary Log Loss/Brier
- race-macro binary Log Loss/Brier
- raw Binary `sum(p_i)`分布

`runner-micro`は全row等重み、`race-macro binary`は各race内runner平均後のrace平均とする。raw Binaryでは同着各馬を`y=1`、coherent winner scoreでは各馬`y=1/m`とし、異なるtarget semanticsをartifact metadataへ保存する。

**Calibration**

- fixed-bin reliability、bin件数・勝者数・uncertainty
- raw Binaryについてrunner logistic calibration intercept/slope
- coherent `q`はslopeとreliabilityを中心に診断する。各raceの予測・観測winner massがともに1なので、pooled intercept/calibration-in-the-largeは構造的に無情報となることを明記する
- ECEは定義・bin仕様付き副指標
- probability、field size、surface、距離、class、時点別

**Market**

- 同じ時点・同じraceのmarket NLL/Brier、Top-1
- model minus marketのpaired score差
- odds/favorite-longshot帯は診断。主校正器には使わない

**Betting**

- purchased stake、effective non-refunded stake、official return、net profit
- primary gross ROI=`official return including refunds / purchased stake`（100% break-even）、primary net profit=`official return including refunds - purchased stake`
- diagnostic exposure ROI=`(official return - refunds) / effective non-refunded stake`。effective stakeが0のblock/raceは単独ratioを作らずaggregate分子・分母へ寄与させる
- bets、races bet、複数bet率、hit rate、odds/proxy帯
- purchased cash-flow基準のcumulative P/L・最大drawdownをprimaryとし、effective-exposure基準は診断。連敗、月/四半期/年別も報告する
- snapshot-to-final movementとthreshold reversal
- primary閾値と事前登録した診断閾値を区別

不確実性はticket IIDで計算しない。同raceの全ticketを保持し、連続するrace-dayまたはweek blockを再標本化して、各resampleでprimaryは`sum(official return including refunds)/sum(purchased stake)`、診断はeffective exposureの対応ratioを再計算する。分母0のresampleは区間計算から除外し件数を報告する。race IID bootstrapは副診断に限定する。普遍的な「必要bet数」は設定せず、実際のpayoff分布、時系列幅、区間推定、prospective確認で判断する。

## 13. リーク防止の必須規則

1. splitはrace単位の時間順。ランダムsplit禁止。
2. 全特徴について`published_at <= prediction_cutoff`。end-to-end実行可能性を主張するPIT-A評価では`ingested_at`と`processing_completed_at`もcutoff以下。PIT-Bはsource-time replayと呼ぶ。
3. as-of joinはcutoff以下の最大時刻。nearest joinで未来値を拾わない。
4. target raceの結果、払戻、確定人気、最終オッズを特徴へ入れない。
5. 騎手・調教師・種牡馬等の年末通算や全期間frequencyを過去へbackfillしない。
6. aggregate、imputer、encoding、feature selection、calibratorをfuture splitでfitしない。
7. ratingはrace前stateを入力し、結果後にbatch updateする。future smoothing禁止。
8. 過去レースの相手再評価はtarget cutoff以下だけ。ex-ante値とas-of-target再評価を分離する。
9. 同日先行raceの結果は、PIT-A/Bで確定配信がcutoff前と分かる場合だけ利用する。長期PIT-Cではevent timeだけで確定済みと推測せず、保守的に開催日開始前までとする。固定availability lagを使う代替はcoverage/timestamp gateで一つに固定する。
10. source訂正を上書きせず版管理する。PIT-Cの訂正前版欠如は品質限界として表示する。
11. primary no-odds feature schemaとcalibratorにodds、人気、票数、odds由来指数を含めない。
12. final holdout結果を見て特徴、校正、rating、閾値、除外規則を変えない。

将来実装では「future rawを追加しても同じsource snapshot/manifestから過去特徴が完全一致する」テストを必須にする。PIT-Cは後日訂正でprovider snapshot自体が変わりうるため、同じmanifest/fingerprint内の再現性と、当時版保証の限界を分ける。

## 14. 同着・取消・異常結果

- **同着:** Binary raw targetは各公式1着を1とする。coherent確率scoreは1着`m`頭へunit winner mass `1/m`を配るが、これはticketの同着込みmarginal probabilityやjoint dead-heat probabilityではない。LambdaRankは同順位へ同relevance（例: 1,1,3着なら3,3,1）。Top-1は予測馬がいずれかの1着ならhit。件数と除外感度を報告する。
- **cutoff前取消:** as-of fieldから除外し、field-relative特徴と`q`を再計算する。
- **cutoff後取消:** decision row/queryには残し、Binary targetとLambda relevanceは0、ratingはno updateとする。選択済みticketは公式返還で精算し、後からfieldを再正規化しない。実走馬だけの再正規化はpost-event診断。将来full-order modelでは通常最下位扱いせず、censor/exclusion policyを別途定義する。
- **競走中止・失格・降着:** official winnerが確定していればBinaryでは勝者以外0。LambdaRankの比較不能runnerは初期top-3 relevanceでは0とし、原因flagを履歴へ残す。race全体に確定winner/順位がなければ予測score対象外だが、settlement ledgerから削除しない。
- **払戻:** dead heat、返還、JRAプラス10等を公式払戻のまま使う。モデル上の擬似targetと精算を混同しない。

## 15. 既決事項のretain / modify / reject

| 既決事項・仮説 | 判断 | 具体化 |
|---|---|---|
| JRA中央競馬 | **retain + narrow** | MVPは中央平地。障害は別研究 |
| 予測と購入判断の分離 | **retain strongly** | no-odds model → coherent probability → market/decision |
| 初期オッズ非入力 | **retain strongly** | calibratorも非入力。odds-awareは別比較 |
| LightGBM Binary + LambdaRank | **retain strongly** | 同一PITデータで相補的baseline |
| raw Binary probabilityをEVへ直結 | **modify** | race coherenceと校正を通した`q`だけを渡す |
| LambdaRank scoreを勝率扱い | **reject** | scoreのままranking評価。確率mappingは別fit |
| ID非入力 | **retain** | IDは履歴/ratingのstate keyに限定 |
| 履歴を少数走へ限定しない | **retain strongly** | 複数窓・decay・条件別・support |
| 過去相手強度を当時版だけに固定 | **modify** | ex-ante版を基本、targetまでのas-of再評価を別特徴で許容 |
| 単一の疲労/フォーム/過去走価値score | **reject for MVP** | componentを分離しGBDTで組合せを学習 |
| PLを後続候補 | **retain with caution** | first full-rank baseline。position別biasを検証 |
| DNN/end-to-end | **defer** | GBDT、rating、校正、PIT基盤の限界確認後 |
| 固定額単勝EV閾値 | **retain + rename** | snapshot EV proxy、主閾値1.10、実払戻精算 |
| early ROI >100%不要 | **retain strongly** | probability/ranking改善を先に評価 |
| 時系列split | **retain and strengthen** | calibration slice、rolling dev、research-round final |
| 1 experiment = 1 hypothesis/state | **retain strongly** | 全試行・閾値・final accessを台帳化 |

## 16. 延期するもの

- race video、paddock image/video、ニュース/SNS/テキスト、LLM handicapping
- 自動投票
- Kelly、portfolio optimization、複雑な券種・組合せ戦略
- raw entity ID、entity embedding、大規模Transformer
- learned Race-value encoder
- TrueSkill/Glicko/dynamic PLの同時導入
- full finishing distributionを前提とした複勝・連系馬券
- 細かいcourse×distance×going×style rating、希少交互作用の大量探索
- 無許諾のWeb scraping、データ・model artifactの公開

Plackett–Luceは最初の高度な確率順位baseline候補だが、日本を含む古典研究でlater-position biasが報告されている。[ranking調査](ranking_rating_and_strength.md) したがってPLを最終モデルと決めず、position calibrationを確認し、必要ならheteroscedastic Thurstone/normal・gamma order-statisticへ進む。

## 17. 実装前のdecision gates

本調査は実装開始の無条件承認ではない。Phase 2で次を解消し、ユーザーが仕様を受け入れてからデータ基盤へ進む。

1. **Existing-data license gate（完了）:** ユーザーが既存ローカルrawの本非公開・私的プロジェクト内利用を承認した。新規取得、第三者共有、外部公開は承認範囲外とする。
2. **Additional-source gate:** JRA PDF/Webや民間サイトから新規に体系的取得する場合は、対象、件数、頻度、raw保存、数値抽出、ML、公開範囲を別途確認する。不許可・不明確なら承認済みrawの範囲に留めるか、正規JRA-VANへ切り替える。
3. **Environment gate:** 無料source・有料sourceのいずれでも、許諾された取得host、保存・処理境界、cloud/国外転送、backupを文書化する。JRA-VAN利用時は日本国内のWindows JV-Link hostを含める。
4. **Coverage gate:** 少数sampleで年×場×項目の欠損、PDF layout、同着・取消・非完走、ID解決率、訂正、コード変遷を監査する。有料化時は時系列odds最古日・鮮度も監査する。
5. **Timestamp gate:** 無料長期PDFはPIT-Cと明示する。prospective trackでは`T_prevday`と`T_close`、発走変更時再判断、max odds stalenessを固定し、後のoddsで補完しない。
6. **Outcome gate:** 同着、取消、競走中止、失格、降着、返還の実頻度を測り、上記規約を最終確定する。
7. **Split gate:** coverage監査後、train/calibration/development/finalの絶対日付を、metric集計前にmanifestへ固定する。
8. **Publication gate:** 非公開に固定しない成果について、データ、派生特徴、予測、学習済みmodel、aggregate reportの各区分を具体的に示し、公開前に書面許諾または明確な公式根拠を得る。

## 18. 推奨する最初の実験順

各段階を一つの解釈可能な変更として進める。

1. environment / coverage / timestamp / outcome / split gateを閉じる。既存rawのlicense gateは完了済みである。
2. `local_raw_pit_c`のcore contractとinvariant test仕様を確定する。
3. 同一features・splitでBinary rawとLambdaRank rawを比較する。
4. base modelを固定してcoherence/calibration mappingだけを比較する。
5. probability指標で`q`をfreezeし、final-odds oracle市場baselineと比較する。これはbet選択ではない。
6. 許諾後のprospective snapshotが蓄積してから、固定1.10 snapshot-proxy ruleを公式払戻で精算する。
7. `rating_strength`、`weather_jma`等の無料feature groupを一群ずつablationする。
8. 無料coreを固定後、調教・時点odds・deep pedigree等の有料増分を一群ずつ比較する。

model、calibrator、thresholdを同じ実験で同時変更しない。

## 19. 最大の実装リスク

1. **過去snapshot不足:** 長期予測性能と長期実行可能ROIを同じ証拠として扱えない。
2. **既存rawのcoverage:** 少なくとも146レースが不足し、race class、lap、払戻等も欠ける。欠損を無視した評価は期間・条件別指標を歪めうる。
3. **既存featuresのリーク:** 全期間統計、同一race内の行単位更新、結果・市場列混在があり、rawからの再生成が必須である。
4. **有料fallbackのWindows依存:** JV-Link ActiveX/COMにより取得と学習環境の分離が必要。
5. **PIT-Cの訂正:** 長期historyは当時配信版を完全再現しない可能性がある。
6. **score/probability混同:** LambdaRank scoreやbinary raw probabilityを無言でEV利用すると、確率品質の根拠が崩れる。
7. **late odds movement:** snapshot proxyで選んだ馬の最終払戻倍率は変わる。
8. **rare-event variance:** 高オッズ・高proxyの少数betはROI区間が非常に広くなりうる。
9. **interaction sparsity:** 距離・馬場・course・脚質を細分すると支持例が急減する。
10. **trial multiplicity / concept drift:** 多数試行と制度・course・市場変化がholdout解釈を歪める。

## 20. 最終勧告

次フェーズで行うべきことは、大規模モデル実装ではなく、承認済みrawのcoverage・例外・欠損監査、`local_raw_pit_c`とprospective trackの**data contract・PIT・dataset・experiment specificationの確定**である。JRA公式結果は不足raceと不足fieldの照合・補完に使う。その後に初めて、最小の共通datasetでBinaryとLambdaRankの二つのbaselineを作る。

利益可能性について現時点で肯定も否定もできない。承認済み無料履歴には締切前odds系列がなく、有料JRA-VANでも時系列oddsの公式保証は1年である。長期の実行可能収益性は、今後蓄積するPIT-A snapshotによるshadow/prospective評価が不可欠である。承認済みrawのPIT-Cで予測性能を先に検証し、市場後収益の証拠を別trackで積み上げる方針が、今回の一次情報、利用承認、無料優先条件に最も整合する。

## 21. 2026-08-31 実験研究の追補

初期実装後のPV-01～PV-06およびGR-001は、上記の「一仮説ずつ、
時間分離、確率とrankingを別評価」という方針を維持して実行した。
この追補時点の具体的な実装判断は次のとおりである。

- Binaryは、過去90日減衰signed時計差を1列追加したPV-01を現開発
  baselineとして維持する。2026年以降のprospective確認前に確定的な
  final改善とは扱わない。
- raw `着差` tokenで0.1秒同時計を補うPV-06は、2014～2021 auditでは
  写像可能だったが、2022 primary Log Lossの区間が0を跨いだため
  inconclusiveである。写像を2022/2024に合わせて再調整しない。
- LambdaRankは従来の`1着=3, 2着=2, 3着=1, その他=0`を維持する。
  2・3着を同値にして上位半数へbucket教師を広げたGR-001は、2022の
  Log LossとBrierを有意に悪化させたためrejectする。
- GR-001は「4着以下の情報に価値がない」とは示していない。今回の
  結果が否定したのは、2着対3着の教師を捨てる代わりに上位半数の
  coarse orderを加える特定の交換である。別設計を行う場合も独立仮説
  とし、この2022結果の頭数別sliceへ適合させない。
- 次のwork queueは`docs/model_research_priorities.md`をsource of truthとする。
  EVAL-ROLLを先に固定し、OPP-RECENT、SEC-3F、HPO-01、ENS-01をrolling
  foldsで一仮説ずつ評価する。LIVE-DATAはsource gateの下で並行開始する。

機械可読な根拠は`experiments/race_content_20260831/pv_006_summary.json`
および`experiments/graded_rank_20260831/summary.json`、詳細protocol/resultは
`docs/experiments/pv_006_margin_token_refinement_20260831.md`と
`docs/experiments/gr_001_graded_lambdarank_20260831.md`をsource of truthとする。

## 22. 2026-08-31 Sランク完了追補

EVAL-ROLLを実装して2020～2023の4 expanding foldを固定し、S0/S1を全て
完了した。Sランク選択に2024/2025、odds、人気は使用していない。

- OPP-RECENTの相手のみpre-race Elo履歴は既存career相手水準と高相関で、
  Binaryはinconclusive、LambdaRankはproper scoreを有意に悪化させreject。
- SEC-3Fのrace内上がりpercentile 90日履歴はLambdaRank NDCGを
  `+.00256 [+.00045,+.00465]`改善し、ranking pathで採択。Binaryは
  inconclusive。絶対上がり秒やpace interactionは混ぜていない。
- HPO-01はBinaryでeligible候補なし。Rankerは2020～2022で選んだ
  `feature_fraction=.75`が2023で全point悪化しreject。parameterは維持。
- ENS-01固定50:50はLambdaRankより強かったが、Binary比Log Loss改善
  `+.00157`が事前minimum `+.002`未達でreject。weight探索せず、2023未開封。
- 新馬戦はhorse履歴がなく難しいが、fitからだけ除くと全race Log Loss
  `-.00097`、NDCG `-.00225`。通常race sliceにも改善がなく、学習に維持。
- LIVE-DATAはJRA Web scrapingではなく公式JRA-VAN/JV-Linkだけを許す。
  timestamp付きappend-only archiveまで実装したが、actual collectionは
  private Windows host・契約/key・transportがないためactivation blocked。

現Binary development incumbentはPV-01 254特徴。2024の保守的LambdaRank
referenceはlean 253特徴のまま、pre-2024 rolling candidateはlean+SEC-3F
254特徴とする。次はPACE-01を一仮説としてrolling評価し、支持時だけ
PACE-02へ進む。統合根拠は
`docs/experiments/s_rank_model_research_conclusions_20260831.md`および
`experiments/s_rank_model_research_20260831/summary.json`を参照する。

## 23. PACE-01追補

通過順位の最初の2区間以上あるtokenをrace内序盤位置percentileに変換し、
90日半減のhorse履歴1列として評価した。2014～2021 auditでは1区間のみの
198 raceが全て新潟芝直線1000 mだったため、これらはcorner位置と解釈せず
観測更新から除外した。

2020～2023 rollingでBinary Log Lossは`+.00659 [+.00333,+.00978]`、
LambdaRank NDCGは`+.00440 [+.00197,+.00683]`。BinaryはLog Loss/Brier、
LambdaRankは全4 primary/guardrail metricが4年全て改善し、両familyで採択した。
2024/2025は未使用である。次はこの固定済みstyle履歴だけからfield pace pressure
を作るPACE-02を一仮説として評価する。根拠は
`docs/experiments/pace_01_early_position_20260831.md`と
`experiments/pace_01_20260831/summary.json`を参照する。
