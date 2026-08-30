# 国内の無料公開データ源：取得可能性・PIT・利用条件の比較

## 0. 調査範囲と結論の読み方

- 対象は、JRA 中央競馬モデルの学習・評価を念頭に置いた国内の無料公開情報である。地方競馬、気象、血統・セールは補完源として扱う。
- 調査日は **2026-08-30 (JST)**。大量取得、認証回避、非公開 API の探索、実装は行っていない。公開ページ、公式仕様、利用規約、robots.txt と少数の表示例だけを確認した。
- 本書でいう「無料」は料金が 0 円または一般閲覧できるという意味に限る。**無料閲覧、robots.txt 上の許可、自動取得許諾、複製・保存許諾、加工・ML 利用許諾、再配布許諾はすべて別問題**である。
- 法的判断ではない。規約が曖昧なソースについては、安全側に「要書面確認」と分類した。
- `確認できず` は不存在の断定ではなく、今回確認した公式公開資料では確認できなかったという negative result である。

### 判定語

| 判定 | 意味 |
|---|---|
| 採用候補 | 公開方法と利用条件が比較的明確。ただし設計・許諾条件は別途満たす必要がある |
| 照合用 | 人手による一次情報確認には有用だが、主取得源にはしない |
| 要書面確認 | 自動取得、蓄積、加工 ML の少なくとも一つが明示されない、または制限が強い |
| 不採用 | 現行規約または提供方式が予定する自動 ML パイプラインと明確に衝突する |

## 1. Executive findings

### 1.1 事実

1. 今回確認した範囲では、**JRA 中央競馬について、長期履歴・十分な項目・予測時点別スナップショット・公式バルク/API・自動取得と保存加工 ML の明示許諾をすべて無料で満たす単一ソースはなかった**。
2. JRA 公式 Web/JRADB は主催者による結果の一次情報で、1986 年以降の結果を閲覧できるが、公開一括 API、自動取得上限、長期 PIT スナップショット、加工 ML の包括許諾は確認できなかった。
3. JBIS-Search は公式機関を情報源として血統・繁殖・セールの長期情報を提供する。しかし一般 robots.txt は `Crawl-delay: 600`、`/csv/` 等を除外し、利用条件も私的範囲外の複製・改変・第三者利用等を制限する。
4. 競馬ラボは無料登録後を含む人間向け情報が豊富で、結果、最終オッズ、馬体重、通過順、ラップ、コメント等を表示する。一方、公式 API、履歴スナップショット、取得頻度、加工 ML 許諾は確認できず、FAQ は記事・写真・データ等の複写転載を禁じる。
5. NAR は今回の候補中で唯一、一般向けに ZIP/CSV の公式ダウンロード仕様を公開する。地方競馬のレース情報は 1998-01 以降、オッズは 2026-03 以降、当日ファイルは約 2 分更新で中間オッズを含む。ただし JRA 本体ではなく、robots.txt は `/KeibaWeb/DataDownload/` を一般クローラから除外し、規約は事前許諾のない転載・複製を禁じる。
6. 気象庁の観測データは競馬そのものではないが、公共データ利用規約に基づく再利用条件と CSV 提供が明示される。
7. netkeiba と競馬ブックの無料層は人間向け検証には有力だが、双方とも私的閲覧を超える利用に強い制限がある。競馬ブックは「プログラムでのデータ取り込み」を明示的に禁止する。
8. JRA-VAN TRY、公式 LINE、無料記事・分析は公式データ由来の便利な表示・派生情報だが、履歴バルク/API ではない。「JRA-VAN Data Lab. の無料ソフト」はデータが無料という意味ではない。

### 1.2 解釈

- 無料 Web の寄せ集めは、取得不能よりも **権利、PIT、訂正履歴、ID 接続、仕様変更**がボトルネックになる。
- 気象庁は、権利条件と機械可読な提供方式がともに明示された点で、この調査内では最も採用しやすい無料補完源である。
- 表示上の最終オッズを保存できても、意思決定時点の表示オッズを再現できなければ snapshot EV のバックテストには使えない。
- JRA/NAR/JBIS/民間サイトの同名フィールドは、出典時点、訂正反映時刻、表記、ID が同じとは限らない。多数決で統合するより、公式主系統と照合系統を分離すべきである。
- 無料のみを制約にすると、JRA-VAN/JV-Link を正規主ソースにする既存結論より、再現性・法務・PIT の品質が大きく低下する。

### 1.3 推奨

1. **無料 Web スクレイピングを MVP の主データ契約にしない。** 無料長期予測trackでは、JRA公式成績PDFをJRAへの書面確認付き主候補とし、JRA-VAN/JV-Linkは厳密PIT・市場trackの有料fallbackとする。
2. JRA公式成績PDFは、回答前は少数手動PoC・一次情報照合・仕様確認に限定する。体系的取得、長期保存、数値抽出、加工MLは書面確認後に判断する。
3. 気象庁 CSV は、出典表示・加工表示・品質フラグ保持を条件に無料補完源として採用候補にする。
4. NAR CSV は JRA 馬の地方歴補完として価値があるが、自動ダウンロード、長期保存、加工学習、派生物の扱いを NAR に書面確認してから採用判断する。
5. JBIS、競馬ラボ、netkeiba は照合用に留める。競馬ブックのプログラム取込は行わない。
6. 許諾が得られない間は、Web 表示を手作業で数件確認することと、データセットを構築することを混同しない。

## 2. 主要候補の横断比較

### 2.1 項目・期間・機械可読性

`○` は公式資料または表示例で確認、`△` は部分的・最終値のみ・期間不明、`—` は今回確認できず。

| ソース | 主対象 | 公称/確認期間 | 結果・着順 | 最終/現行オッズ | 時系列オッズ | 調教 | ラップ/通過 | 馬体重 | 血統/個体 | 公式バルク/API |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JRA 公式 Web/PDF | JRA | 結果 1986+、PDF 2002+ | ○ | ○ | — | △（一部動画・G1 関連公開物） | ○ | ○ | ○ | — |
| JBIS-Search | 中央・地方・生産 | 馬は原則 1984 年生+、JRA 結果 1986+ | ○ | △（人気・払戻中心） | — | — | ○ | ○ | ◎ | — |
| 競馬ラボ | 主に JRA | 公式保証不明。表示例は 1980 年代末/2008 年以降にも存在 | ○ | ○ | — | △（選定記事等） | ○ | ○ | ○ | — |
| NAR 公式 DL | 地方 | レース 1998-01+、オッズ 2026-03+ | ○ | ○ | △（当日約2分スナップショット） | — | ○ | ○ | ○ | **ZIP/CSV** |
| netkeiba 無料層 | 中央・地方 | DB は 1956 年以降デビュー馬/中央結果を案内 | ○ | ○ | — | —（主要調教 DB は有料） | ○ | ○ | ○ | — |
| 競馬ブック無料層 | 中央・地方 | 無料レース検索は本年、馬成績は直近5走等の制限 | △ | △ | — | △（無料表示制限） | △ | △ | △ | —、取込禁止 |
| JRA-VAN TRY/LINE/無料記事 | JRA | 当日/特定日または記事単位 | △ | ○（直近等） | — | — | △（派生分析） | ○（LINE 等） | △ | — |
| 気象庁 | 気象 | 地点・項目ごと | — | — | — | — | — | — | — | CSV DL |
| JRHA セール結果 | 生産・セール | 年次ページ単位 | — | — | — | — | — | — | 血統・落札価格 | — |

### 2.2 PIT・利用条件・運用適性

| ソース | 予測前公開時刻 | 過去版/訂正履歴 | robots.txt の確認結果 | 自動取得・保存・ML | 暫定判定 |
|---|---|---|---|---|---|
| JRA 公式 | 出馬表、馬体重、馬場、オッズ等を順次公開 | 任意 Web snapshot 保証なし。結果 PDF は文書 archive | 全 UA に path disallow なし | 包括許諾・rate/API を確認できず | 照合用・要書面確認 |
| JBIS | 中央出馬表 木16:30頃/金土11:00頃、結果 月12:00以降 | 過去 vintage 保証なし | `/csv/` 等 disallow、一般 600 秒 delay | 規約制限が強い | 照合用 |
| 競馬ラボ | 出馬表/オッズ/結果を順次更新 | historical vintage 保証なし | 一部 path disallow、一般 delay 記載なし | FAQ/規約上高リスク | 照合用 |
| NAR DL | 当日約2分、月次 毎日2時頃 | ZIP名に Unix time。ただし任意時点 archive 保証なし | DL path disallow、一般 10秒 delay | 公式DLだが複製・自動化範囲は要確認 | 条件付き補完候補 |
| netkeiba | オッズ/結果等を随時 | snapshot/API 保証なし | 確認した `www`/`db` robots URL は404 | 私的利用を超える利用を制限 | 照合用 |
| 競馬ブック | 項目別公開時刻を掲示 | snapshot/API 保証なし | 旧 host では robots 内容を確認できず | **プログラム取込を禁止** | 自動取得は不採用 |
| JRA-VAN 無料表示 | 当日/直近 | 履歴 dataset ではない | path disallow なし | 規約は個人的私的利用に限定 | 照合・UI参考 |
| 気象庁 | 過去DLは原則前日分まで | 遡及訂正あり、CSVに取得時刻/品質情報 | `data.jma.go.jp/robots.txt` は404 | 公共データ規約。過度の自動アクセスを控える | 採用候補 |
| JRHA | セール後年次公開 | 年次 HTML/PDF。PIT snapshot 保証なし | path disallow なし | 二次利用条件の十分な記載を確認できず | 手動照合・要確認 |

**robots.txt の注意:** 上表は 2026-08-30 に観測したクローラ向け指示であり、契約・著作権・データベース権・サーバ負荷・ML 利用の許諾ではない。robots が空または 404 でも取得許可とは解釈しない。

## 3. JRA 公式 Web、JRADB、公式公開物

### 3.1 証拠・事実

- JRA FAQ は過去レース成績を **1986 年以降**掲載すると案内する。2000 年以前は一部情報が掲載されない、または表示が異なる場合がある。成績 PDF は 2002 年以降である。
- 出馬表・結果ページと成績 PDF の表示例から、開催・競走条件、天候、馬場、枠/馬番、馬名、性齢、斤量、騎手、着順、タイム、着差、通過順、推定上がり、馬体重と増減、調教師、人気、最終単勝オッズ、払戻、票数、レースラップ等を確認できる。
- 馬体重は発表後速やかに出馬表や単複オッズ画面へ反映される。G1 等の一部については木曜頃の「調教後の馬体重」も別ニュースとして公表されるが、これは競走日馬体重とは異なる。
- 馬場状態は含水率だけで機械判定せず総合判断される。含水率は 2018-07-27 以降、クッション値は 2020-09-11 以降の archive があり、原則として開催後の木曜に archive 更新される。開催前/当日の公表時刻と archive 更新時刻は同一ではない。
- 公式 YouTube 等に調教映像があるが、全馬・全追切の構造化時計 dataset ではない。無料 JRA Web だけで全期間の構造化調教データを得られるという証拠は確認できなかった。
- `https://www.jra.go.jp/robots.txt` は全 UA に対して path disallow を記載していない。一方、「ご利用に際して」は著作権等を掲げ、特に画像・映像等の二次利用について許諾手続きを案内する。
- 今回、JRA Web の一般向け公式 REST API、全件 CSV/Parquet download、自動取得 rate、HTTP cache/delta contract、恒久保存・加工 ML の包括許諾は確認できなかった。
- JRADB の画面 URL は `CNAME` 等の opaque query を含むものがあり、人間向け HTML/PDF である。公式の versioned schema や SLA は確認できなかった。

### 3.2 PIT 解釈

- 現在の出馬表ページは取消、騎手変更、馬体重、オッズ等で更新される。後日表示された値を、そのまま前日・発走60分前に既知だった値とみなせない。
- 結果ページの最終オッズは市場ベースラインには使えても、購入判断時に固定された価格ではない。snapshot EV 検証には決定時刻付き表示オッズが必要である。
- 成績 PDF は結果の検証に強いが、予測前 snapshot の再現源ではない。英語版 race programme も前夜公開で、枠順後の取消・騎手変更が反映されない旨を明示するため、版の意味が異なる。
- 馬場 archive を過去開催の分析に使う場合も、`measurement_time`、当時の `published_at`、後日 archive の `retrieved_at` を分ける必要がある。

### 3.3 不確実性

- テキスト数値の個人的ローカル ML が JRA の許諾なしにどこまで可能か、公開条件からは確定できない。
- 過去ページの URL 恒久性、訂正時の旧値保持、オッズ表示の更新間隔と過去 snapshot 保持は不明。
- JRA 内部の競走馬/騎手/調教師 ID が Web URL に現れても、外部 DB との公式 crosswalk または API 主キーとして保証されるとは確認できない。

### 3.4 示唆・提案

- JRA 公式は `source_of_truth_for_adjudication` とし、主データ取得は正規 JRA-VAN/JV-Link 系統に置く。
- JRA Web 自動取得を検討する前に、対象 URL、日次件数、取得間隔、raw HTML/PDF 保持期間、数値の加工学習、予測値/モデル公開を列挙して書面照会する。
- 許諾なしに画面を巡回して retrospective dataset を作らない。JRA 公式からは、人手で仕様・例外・結果を少数照合する。

## 4. JBIS-Search

### 4.1 証拠・事実

- JBIS は JRA、NAR、公益財団法人ジャパン・スタッドブック・インターナショナル、日本軽種馬協会等を情報源とする。
- 馬情報は原則 1984 年生以降、1972～1983 年生は一部、それ以前は不完全。中央競馬結果は 1986 年以降である。海外競走等は完全収録ではない。
- 公式 help は平日更新、JRA 結果は月曜12時以降、中央出馬表は木曜16:30頃および金・土曜11:00頃と案内する。
- 馬詳細はプロフィール、5代血統、全競走成績、父/母父/産駒成績、市場取引等を含む。馬検索は100万頭超と案内する。
- 2025 年の公式結果 PDF 表示例には、競走条件、天候・馬場、着順、枠/馬番、血統、性齢、騎手・斤量、タイム・着差、通過順、上がり3F、独自指数、人気、馬体重増減、調教師、馬主、生産者、ラップ、払戻がある。表示例では単勝オッズそのものより人気・払戻が中心だった。
- 「業務利用・二次利用について」および利用規約は、個人的用途等、原則「そのまま」の利用を示し、私的範囲外の複製・改変、営利利用、第三者利用・公開等を制限する。
- robots.txt は一般 UA に `Crawl-delay: 600` を指定し、`/premium/`、`/simulation/`、`/csv/` 等を除外する。GPTBot、ClaudeBot 等は全 path disallow である。
- 公式 public bulk/API、PIT snapshot、オッズ時系列、delta feed、schema version/SLA は確認できなかった。

### 4.2 解釈

- 600 秒 delay と `/csv/` 除外は、少なくとも広範なクローリングを想定した公開経路ではないことを強く示す。ただし robots は利用許諾の代わりではない。
- 「そのまま」の利用という条件は、正規化、特徴量生成、モデル学習と緊張する。個人的研究だから加工 ML も当然許されるとは解釈できない。

### 4.3 不確実性

- ページ上の内部馬 ID は便利だが、JRA/NAR/JRA-VAN と共有される公式 ID である保証は確認できない。

### 4.4 示唆・提案

- 血統、繁殖、セール、同名馬の識別を人手確認する照合源として使う。
- JBIS 由来特徴を本番 dataset に入れる前に、取得方法、件数、保持、加工、学習済みモデル、予測画面での表示、共同利用を具体化して許諾を得る。
- 許諾後も更新時刻が race start 後になる項目を当該レース予測へ混入させず、`published_at` を別管理する。

## 5. 競馬ラボ

### 5.1 証拠・事実

- サービス案内は、無料会員登録後を含めコンテンツを無料提供するとする広告型サービスである。
- レース結果表示例には、日付・開催、天候・馬場、クラス、芝/ダート・距離、頭数・発走、賞金、着順、枠/馬番、馬名、性齢、斤量、騎手、人気、最終単勝オッズ、タイム・着差、通過順、上がり、調教師、馬体重、払戻、コーナー順、200m ラップ、前後半3F/4F、ペースラベル、レース後コメント等がある。
- 馬ページ表示例にはプロフィール、5代血統、獲得賞金、通算成績、過去走の条件・人気・着順・騎手・頭数・枠・時計・着差・ペース・上がり・馬体重・通過順等がある。
- 1980年代末の馬/競走ページや 2008 年の結果ページが検索可能な例はあるが、全期間・全項目の公式 coverage statement は確認できなかった。
- 調教記事や選定馬の追切情報はあるが、無料で全出走馬を網羅する構造化調教 DB の仕様は確認できなかった。
- 利用規約は知的財産侵害、商用利用、サービス妨害等を禁じる。FAQ はサイトの記事、写真、データ、動画等の複写・転載・貸与を禁じる。
- URL は `/db/race/YYYYMMDD.../` のように規則的に見えるが、公式 API、permalink 保証、versioned schema、rate、cache/delta、訂正履歴は公表されていない。
- robots.txt は一部認証/検索/広告 path を除外し、一般 UA の crawl-delay は記載しない。Bing に10秒 delay を指定する。

### 5.2 解釈

- 規則的 URL は API 契約ではない。HTML layout 変更、会員 wall、広告配信、bot 対策で壊れうる。
- 無料閲覧と dataset 生成の間の許諾が欠けるため、自動収集して ML に使うリスクは高い。

### 5.3 不確実性

- 民間のペースラベル、指数、コメントは有用かもしれないが、定義・オッズ依存・PIT・将来継続性は確認できず、公式結果との関係も不明である。

### 5.4 示唆・提案

- 表示比較と特徴アイデア探索に限定し、取得元にしない。
- 独自コメント/指数を将来検証したい場合は、一特徴群のアブレーションとして、利用許諾、計算定義、公開時刻、オッズ利用有無を確認してから契約する。

## 6. NAR 地方競馬情報サイト・公式データダウンロード

### 6.1 証拠・事実

- 2026-05-21 版の公式説明書は、PC 版画面のボタンから ZIP 内 CSV をダウンロードする機能を定義する。
- 当日ファイルはレース情報・オッズ情報を約2分ごとに更新し、中間オッズを含む。月次ファイルは毎日午前2時頃更新し、オッズは確定値だけを含む。
- レース情報は 1998-01 以降、オッズ情報は 2026-03 以降。古いレース情報に欠損がありうる。仕様は予告なく変更される場合がある。
- ZIP 名には Unix timestamp が付く。CSV の行項目には一般的な `published_at` や revision ID は見当たらない。
- レース一覧 CSV は競馬場、日付、レース番号、発走、種類・名称、芝/ダート、回り、距離、天候、馬場、頭数、条件、賞金、上がり4F/3F、最大15ハロン、最大8コーナーの通過順を持つ。
- 出馬表 CSV は枠、馬番、馬名、性齢・毛色・生年月日、父・母・母父、騎手/所属/斤量/集計成績、調教師、馬主、生産、馬体重増減、全成績、左右・競馬場・距離別成績、最高時計、結果、上がり、人気等を持つ。
- オッズ CSV は全賭式の組番、オッズ/最大、人気を持つ。払戻 CSV は各式別の組番・払戻・人気を持ち、同着時は1レース2行以上になる。
- 調教情報は CSV 仕様にない。
- 利用規約は画像・テキスト等の権利を原則主催者等に帰属させ、事前許諾のない転載・複製を禁じ、完全性・最新性を保証しない。異常件数のアクセスは遮断しうる。
- robots.txt は一般 UA に10秒 delay を指定し、`/KeibaWeb/DataDownload/`、当日動的ページ、DataRoom 等を除外する。コメントは machine/API endpoint をクロール対象外とする。

### 6.2 PIT 解釈

- ZIP filename timestamp は `observed_at` の候補にはなるが、配信元の正式な項目別 `valid_from` ではない。
- 当日ファイルを毎版合法的に保存できれば地方オッズ snapshot を構成できる可能性はある。しかし公式 archive が過去の全2分版を提供するとは書かれていない。後日の月次ファイルからは時系列を復元できない。
- 出馬表 CSV に結果列もあるため、同じ schema の post-race 版から pre-race dataset を作る際は、着順、タイム、上がり、人気等の post-event 列を明示的に遮断する必要がある。
- 月次ファイルは「現在の最新既知版」であり、過去時点の vintage と同義ではない。

### 6.3 事実（対象範囲・ID）

- NAR は地方競馬を対象とする。JRA とは芝・クラス体系・賞金・競走環境が異なる。
- 公開 CSV 仕様には永続 horse ID を確認できなかった。

### 6.4 解釈

- 地方走を JRA 走と同じ学習行として無条件結合すると domain shift を生む。
- JRA 馬の地方遠征歴、転入歴、ダート適性については、過去歴の補完価値がありうる。

### 6.5 提案

- NAR に、自動ボタン/API利用の可否、推奨取得間隔、過去snapshot保存、CSV加工ML、派生特徴/学習済みモデル、非公開個人利用、再配布禁止境界を文書で確認する。
- 許可されても更新周期以上に poll せず、full snapshot として content hash、ZIP timestamp、retrieved_at、manual version を保存する。これは将来実装時の提案であり、本調査では取得していない。
- 馬名だけで join せず、生年月日、性、父、母、母父を用いた確率的/ルール型照合と例外監査を行う。
- 最初のアブレーションは「JRA-VAN が既に持つ JRA 関連地方歴」との増分比較にする。

## 7. その他の中央競馬向け無料サイト

### 7.1 netkeiba 無料層

#### 事実

- 公式 DB 案内は 1956 年以降にデビューした競走馬、および中央レース、騎手、調教師、馬主、生産者等を検索できるとする。
- 人間向けページにはレースカード、オッズ、結果・払戻、馬体重、通過順、ラップ等がある。調教タイム DB、コメント、独自指数等の主要部分は有料プランである。
- 更新時刻 help は出馬表、オッズ、結果、調教、コメント等の概略公開時刻を示すが、過去 snapshot download ではない。
- 利用規約（2022-04-01 改定）は承認なく私的利用を超えて取得データを利用・公開することを制限し、商用利用とその準備等も禁じる。公開 API、一括 DL、自動取得許諾、PIT archive は確認できなかった。
- 2026-08-30 に確認した `https://www.netkeiba.com/robots.txt` と `https://db.netkeiba.com/robots.txt` は 404 だった。これは取得許可を意味しない。

#### 解釈

- 歴史の深さと UI は優秀だが、ML dataset の主系統に必要な権利・schema・PIT contract がない。

#### 提案

- JRA 公式との人手照合だけに使う。
- 将来必要なら、用途、日次量、raw保持、加工、共同利用、モデル/予測公開を明記して運営会社へ照会する。認証画面の自動化は行わない。

### 7.2 競馬ブック無料層

#### 事実

- 公式提供内容表は、無料層に中央出馬関連、直近5走等の馬情報、本年レース検索など制限付き機能があることを示す。完全履歴・全調教・全独自情報は無料ではない。
- 利用規約はサービスを閲覧目的とし、**プログラムでのデータ取り込みを禁止**する。また個人的利用を超える転載・複写・蓄積・転送を禁じる。

#### 解釈

- robots の有無や無料会員登録にかかわらず、プログラム取込を禁じる明示規約が優先する。

#### 提案

- 自動 ML 取得源としては不採用とする。
- 人が新聞情報の意味を調べる参考にはできるが、数値を dataset へ転記・蓄積する運用に拡張しない。

### 7.3 非公式 API、GitHub/Kaggle 配布物

#### negative result と不確実性

- 今回、出所、取得許諾、再配布許諾、PIT、訂正履歴を公式一次資料で検証できる無料 JRA API/第三者 dataset は確認できなかった。
- `API` を名乗る wrapper や scraping library は、上流サイトの許諾とデータ利用権を自動的に付与しない。Kaggle/GitHub のライセンスがコードだけを対象とし、同梱データの権利を覆わない場合もある。

#### 提案

- 非公式配布物は、URLが動く、CSVを読める、ライセンス欄があるという理由だけで採用しない。上流ソース、取得日、規約、再配布権、欠損、ID、PIT を追跡できないものは benchmark にも使わない。

## 8. JRA 公式関連の無料公開物

### 8.1 JRA-VAN TRY、公式 LINE、無料分析

#### 事実

- JRA-VAN は JRA 公式データの許諾を受けたサービスである。TRY の無料会員は基本的に特定日・当日レースの機能に限定され、有料版でも human-facing service である。
- TRY は走破時計、平均ラップ、上がり速度、脚質、直接対戦、レースレベル、枠/騎手傾向等から派生した AI 指数を案内する。
- JRA-VAN 公式 LINE は無料で直近レースの単勝・複勝オッズ、馬体重、好調騎手等を表示する。
- 無料の「データde出～た」等は解説・派生分析であり、生レコードの versioned bulk dataset ではない。
- JRA-VAN 利用規約は、本人が個人的・私的に使う範囲を超える情報の複製・改変・公表等を制限する。Data Lab. 通常契約での第三者配信・商用提供には JRADB 契約が必要との公式案内がある。

#### 解釈

- 無料表示は UI/特徴量仮説/市場チェックには有用だが、学習データ取得経路ではない。
- 「Data Lab. 対応の無料ソフト」はソフト料金を指し、JVData 契約が無料という意味ではない。

#### 提案

- 無料範囲を技術的に回収するより、正規 JV-Link で時刻付き原データを取得する方が、このプロジェクトの再現性に合う。

### 8.2 Horse Racing in Japan、統計・ratings、公式番組

#### 事実

- JAIRS/JRA 関連の英語サイトは race card/result、JRA race programme PDF、格付競走一覧、騎手・調教師ランキング、年次 JRA/NAR 統計、JPN Thoroughbred Rankings 等を公開する。
- race programme は原則前日夕方公開で、枠順後の取消・騎手変更を反映しないと明記する。非JRA馬の過去成績が完全でない場合もある。
- 年次統計・ratings は公式集計だが retrospective aggregate で、各出走馬の race-time PIT raw record ではない。

#### 解釈

- 国際馬、格付、年間集計の照合には有用だが、学習 row の主取得源に必要な粒度とPITを持たない。

#### 提案

- 公開時点と対象期間を保持し、後日 rating を過去レースの事前特徴へ逆流させない。

## 9. 無料の補完データ

### 9.1 気象庁の過去気象 CSV

#### 証拠・事実

- 気象庁「過去の気象データ・ダウンロード」は全国の気象台・アメダスについて、原則昨日までの地点・期間・項目を選択し CSV で取得できる。
- CSV は観測値に加えて品質情報、現象なし情報、均質番号を含められ、ダウンロード時刻をヘッダに持つ。
- 観測場所移転、環境・方法変更による不均質、資料不足、疑問値、欠測を区別する。掲載値は遡及修正される場合がある。
- 一回の request 量に上限があり、アクセス集中や大容量時は遅延・失敗しうる。自動化ツールによる過度のアクセスを控えるよう明記する。
- 権利表記がない気象庁コンテンツは公共データ利用規約第1.0版に従って利用でき、出典表示と、加工した場合の加工表示を求める。一部法令・個別権利の例外がある。

#### PIT・品質解釈

- JRA の公式 `weather`/`going` は競走主催者の公表値であり、最寄り AMeDAS 値の代替ではない。気象庁は補助的な降水、気温、風等に限定する。
- 競馬場から観測点までの距離、標高、観測周期、局地雨、散水を記録しなければ、誤った精密さになる。
- 遡及修正があるため、後日の同一期間 CSV と当時取得版が一致する保証はない。

#### 提案

- 権利条件が明確な無料補完源として採用候補にする。まず競馬場ごとの候補観測所・距離を仕様化し、JRA 公表天候/馬場とは別 feature group で評価する。
- strict vintage が必要なら合法的に取得した CSV、download time、品質 flag、station metadata を保存し、最新版の再取得と当時版を区別する。
- quality flag を欠落させず、疑問値・欠測・不均質を通常値へ無理に補間しない。

### 9.2 JRHA セレクトセール結果

#### 事実

- 日本競走馬協会の年次セール結果は、上場番号、上場馬名、性、毛色、父、母、販売者、購買価格、購買者を公開し、写真・ブラックタイプ等へリンクする。2026 年ページは日別・性別・種牡馬別集計も公開する。
- 年次 HTML/PDF は人手検索できるが、公式 bulk/API、PIT/revision contract、二次利用・自動 ML の包括条件は今回確認できなかった。

#### 解釈

- セール価格は能力の真値ではなく、血統、外見、期待、取引環境、選抜を含む市場信号である。欠測は未売却、他市場、非上場と混ざる。

#### 提案

- JBIS/JRA-VAN で足りない場合の手動照合候補にし、自動収集前に JRHA へ許諾を確認する。上場時名と登録競走馬名の join は母名・出生年・性・父で監査する。

## 10. URL/API安定性、rate、cache、delta の結論

### 10.1 事実

| ソース | URL/schema 安定性 | 公称 rate/update | cache/delta | 訂正 |
|---|---|---|---|---|
| JRA Web | 人間向け opaque URL/PDF。schema version なし | 自動 rate なし | API delta/ETag contract 未確認 | Web 更新、PDF/FAQ。旧版保証なし |
| JBIS | 内部 ID URL はあるが API contract なし | UI 更新時刻、robots 600秒 | delta 未確認 | 最新表示。旧版保証なし |
| 競馬ラボ | 規則的 race URL だが非契約 | 公称自動 rate なし | delta/cache 未確認 | latest page、revision log 未確認 |
| NAR DL | CSV 項目説明書あり、予告なし変更あり | 当日約2分、月次日次 | full ZIP。delta なし。file name timestamp | 月次は最新化、旧snapshot archive保証なし |
| netkeiba | human HTML、API/SLA なし | 項目別概略更新時刻 | 未確認 | historical vintage 未確認 |
| 競馬ブック | human HTML、取込禁止 | 項目別公開時刻 | 自動取得対象外 | JRA照合を要求 |
| JRA-VAN 無料表示 | human service | 当日/直近 | bulk/delta なし | archive保証なし |
| 気象庁 | CSV仕様/helpあり | 1 request上限、過度な自動化回避 | bulk条件指定、deltaなし | 遡及修正を告知 |

### 10.2 解釈

- full snapshot 配信しかない場合、前版との差分は利用者側で計算できるが、それは公式 delta API ではない。取得許諾と保存許諾が前提である。
- HTTP `ETag`/`Last-Modified` を一度観測しても SLA とはみなさない。HTML selector や URL pattern に依存する基盤は versioned API より高い保守費を持つ。

### 10.3 提案

- 公称 rate がないときに「礼儀として1秒」等を発明しない。許諾を先に得て、提供者が示す更新周期・上限・取得方法に従う。
- 将来の合法的取得では、`source_url`, `source_document_version`, `retrieved_at`, `content_hash`, `event_time`, `source_published_at`, `valid_from`, `revision_seq` を分離する。

## 11. ID・品質・補完関係

### 11.1 事実

- source ごとに race/horse/person の ID と表記が異なる。NAR 公開 CSV 仕様には永続 horse ID が見当たらず、JRA/JBIS/民間サイトの URL ID の公式 crosswalk も確認できない。
- 旧字体、全半角、空白、記号、外国馬表記、馬名変更、騎手/調教師の略称、地方の2001年前後の年齢表記変更が join を壊す。
- NAR は同着で払戻 row が複数になりうる。1 race = 1 payoff row と仮定してはいけない。

### 11.2 解釈・不確実性

- race key を単なる日付+レース番号にすると、競馬場・主催区分をまたいで衝突する。
- 表記正規化だけで個体同一性を確定できない例が残りうる。公式 crosswalk がなければ match confidence と例外監査が必要になる。

### 11.3 推奨する論理層（実装は将来）

1. `source_native_id`: 各ソースの値を改変せず保持する。
2. `canonical_race_key`: 主催区分 + 競馬場 + 競走年月日 + レース番号を核とし、JRA 開催回・日次等を補助する。
3. `horse_identity_evidence`: 馬名だけでなく生年月日、性、父、母、母父、生産者を使い、match confidence と根拠を保持する。
4. `field_provenance`: 同名列を上書きせず、source、published_at、retrieved_at、revision を持つ。
5. `adjudication`: 結果・取消・払戻は主催者公式を優先し、民間差異は source error と即断せず公表時点差を調べる。

### 11.4 補完関係

| 主目的 | 第一候補 | 補完/照合 | 使わない代替 |
|---|---|---|---|
| JRA 競走・PIT | 正規 JRA-VAN/JV-Link | JRA Web/PDF | 無料サイト横断 scraping |
| 公式結果判定 | JRA Web/PDF | JRA-VAN | 民間結果だけで確定 |
| 血統・繁殖・セール | JRA-VAN | JBIS/JRHA の人手照合 | 無許諾 bulk JBIS |
| 地方遠征歴 | JRA-VAN収録分 | 許諾後の NAR CSV | JRA/地方を同分布結合 |
| 天候補助 | 気象庁 CSV | JRA 公表 weather/going | AMeDAS で公式馬場を置換 |
| 調教 | JRA-VAN有償正規データ | 公式動画/ニュースの人手確認 | 無料記事 scraping |
| 市場比較 | 決定時刻付き正規 odds snapshot | JRA 表示の人手確認 | 最終オッズを購入時オッズ扱い |

## 12. 推奨データ契約と no-go

### 12.1 推奨

1. 無料限定の長期予測案は、JRA公式成績PDFを条件付き主候補とする。ただしJRA回答前は少数手動PoCに限定し、PIT-C以上を主張しない。
2. 気象庁を明示条件の良い無料補完候補とする。厳密PIT・安定ID・調教・時点oddsが必要になればJRA-VAN/JV-Linkへupgradeする。
3. NAR は許諾回答後の独立 feature group とし、地方歴の増分だけを検証する。
4. JBIS/JRHA は血統・セールの欠落監査に使い、bulk化しない。
5. 競馬ラボ/netkeiba は UI 上のクロスチェックに限定する。競馬ブックは自動取得しない。
6. どのソースも、本番採用前に terms/robots/manual を日付付きで保存し、問い合わせ回答を data contract に添付する。

### 12.2 明確な no-go

- robots.txt が空だからクロールする。
- 規則的な URL を undocumented API と呼ぶ。
- 無料会員資格を使って全履歴を自動回収する。
- 最終結果ページから pre-race field を再構成し、当時既知だったと仮定する。
- final odds を意思決定時 snapshot として ROI を算出する。
- HTML/CSV の raw data や復元可能な派生物を公開 repository に入れる。
- source ごとの ID を根拠なく同一視する。
- 非公式 Kaggle/GitHub dataset の upstream license を検証せず利用する。

### 12.3 提供者へ確認すべき共通質問

1. 個人・非商用・ローカル環境で、公開情報をプログラム取得してよいか。
2. 対象 endpoint、想定日次件数、推奨間隔、並列数、取得可能時間帯。
3. raw response/CSV/PDF を長期保存し、訂正前 snapshot を保持してよいか。
4. 正規化、集計特徴、モデル学習、評価に利用してよいか。
5. 学習済みモデル、feature importance、個別予測値を本人だけが見る場合と第三者へ示す場合の扱い。
6. 生データを含まないコード/schema、非復元集計の公開可否。
7. cloud/国外 region、バックアップ、共同研究者、外部 SaaS/AI への送信可否。
8. 将来の有償提供・法人化で必要な契約。

## 13. 未解決事項

- JRA Web の数値テキストについて、個人ローカル ML、定期取得、raw snapshot 保持を明示する公式回答。
- NAR 公式 DL の UI 操作以外の自動取得、2分版の長期保存、加工 ML に関する公式回答。
- JBIS の「そのまま」条件と、非公開の集計・モデル学習の境界。
- 競馬ラボ/netkeiba の個人的研究における自動取得・蓄積・加工の境界。ただし主ソースにする必要性は低い。
- JRA/NAR/JBIS 間の公式 horse ID crosswalk の有無。
- 無料 JRA Web における項目別 coverage、欠損、訂正履歴、URL 恒久性の体系的仕様。
- 気象庁観測点と10競馬場の妥当な対応、およびJRA公式馬場との増分価値。
- JRHA 以外の国内市場を含む無料セール情報の完全性と利用許諾。

## 14. 一次・公式資料

すべて **accessed 2026-08-30**。日付がページにないものは `date not stated` とした。robots.txt は同日の観測 snapshot であり、将来の状態を保証しない。

### JRA

- 日本中央競馬会, 「過去のレース成績は、どこに掲載されていますか？」, date not stated: https://www.jra.go.jp/faq/pop02/1_6.html
- 日本中央競馬会, 「レース成績データ」, date not stated: https://www.jra.go.jp/datafile/seiseki/index.html
- 日本中央競馬会, 「馬体重を知りたいのですが？」, date not stated: https://www.jra.go.jp/faq/pop02/2_13.html
- 日本中央競馬会, 「調教後の馬体重とは何ですか？」, date not stated: https://www.jra.go.jp/faq/pop02/2_4.html
- 日本中央競馬会, 「馬場状態およびクッション値に関する情報」, date not stated: https://www.jra.go.jp/keiba/baba/kaisetsu/index.html
- 日本中央競馬会, 「2026年の含水率・クッション値（馬場情報）」, 2026 archive: https://www.jra.go.jp/keiba/baba/archive/
- 日本中央競馬会, 「ご利用に際して」, date not stated: https://www.jra.go.jp/use/
- 日本中央競馬会, `robots.txt`, date not stated: https://www.jra.go.jp/robots.txt
- Japan Association for International Racing and Stud Book, “JRA Race Programme”, annual pages, date not stated: https://japanracing.jp/en/racing/result/programme/2025.html
- Japan Association for International Racing and Stud Book, “Racing”, date not stated: https://japanracing.jp/en/racing/

### JBIS

- 公益社団法人日本軽種馬協会, 「更新/提供データについて」, date not stated: https://www.jbis.or.jp/help/data/
- 公益社団法人日本軽種馬協会, 「馬詳細情報について」, date not stated: https://www.jbis.or.jp/help/horse/details/
- 公益社団法人日本軽種馬協会, 「業務利用・二次利用について」, date not stated: https://www.jbis.or.jp/help/use/
- 公益社団法人日本軽種馬協会, 「利用規約」, date not stated: https://www.jbis.or.jp/terms/
- 公益社団法人日本軽種馬協会, 「競走結果 PDF（表示例）」, 2025-09-20: https://www.jbis.or.jp/race/result/2025/09/20/106/r2025092010610.pdf
- 公益社団法人日本軽種馬協会, `robots.txt`, date not stated: https://www.jbis.or.jp/robots.txt

### 競馬ラボ

- 株式会社Do Innovation, 「利用規約」, revision history includes 2024-09-11: https://www.keibalab.jp/info/agreement.html
- 株式会社Do Innovation, 「よくある質問」, date not stated: https://pc.keibalab.jp/info/faq.html
- 株式会社Do Innovation, 「特定商取引法に基づく表記」, date not stated: https://www.keibalab.jp/info/law.html
- 株式会社Do Innovation, 「レース結果・払戻（表示例）」, 2024-08-11: https://www.keibalab.jp/db/race/202408110411/raceresult.html
- 株式会社Do Innovation, `robots.txt`, date not stated: https://www.keibalab.jp/robots.txt

### NAR

- 地方競馬全国協会・地方競馬主催者, 『データダウンロード機能説明書（一般ユーザー向け）』, updated 2026-05-21: https://www.keiba.go.jp/pdf/manual/data_pdf_manual.pdf
- 地方競馬全国協会・地方競馬主催者, 「利用規約」, date not stated: https://www.keiba.go.jp/terms.html
- 地方競馬全国協会, `robots.txt`, date not stated: https://www.keiba.go.jp/robots.txt

### netkeiba・競馬ブック

- 株式会社ネットドリーマーズ, 「競馬データベース」, date not stated: https://db.netkeiba.com/
- netkeiba ヘルプ, 「レース情報（タイム指数等）の更新時間は？」, date not stated: https://support.keiba.netkeiba.com/hc/ja/articles/18930182696985-%E3%83%AC%E3%83%BC%E3%82%B9%E6%83%85%E5%A0%B1-%E3%82%BF%E3%82%A4%E3%83%A0%E6%8C%87%E6%95%B0%E7%AD%89-%E3%81%AE%E6%9B%B4%E6%96%B0%E6%99%82%E9%96%93%E3%81%AF
- 株式会社ネットドリーマーズ, 「netkeiba.com 利用規約」, revised 2022-04-01: https://www.netkeiba.com/info/kiyaku.html?rf=footer
- 株式会社ケイバブック, 「提供内容（スマホサイト）」, current service table, date not stated: https://p.keibabook.co.jp/entry/teikyo/s
- 株式会社ケイバブック, 「無料会員登録・ご利用規約」, date not stated: https://p.keibabook.co.jp/entry/top

### JRA-VAN・関連公開物

- JRAシステムサービス株式会社, 「JRA-VAN TRY」, current service page, date not stated: https://jra-van.jp/try/
- JRAシステムサービス株式会社, 「LINEで競馬」, date not stated: https://jra-van.jp/pr/line/
- JRAシステムサービス株式会社, 「JRA-VAN利用規約」, current terms, date not stated: https://jra-van.jp/info/rule.html
- JRA-VAN Data Lab. 開発者コミュニティ, 「競馬データの商用利用のお問い合わせについて」, official staff post 2026-07-06: https://developer.jra-van.jp/t/topic/899
- JRAシステムサービス株式会社, `robots.txt`, date not stated: https://jra-van.jp/robots.txt

### 気象・セール

- 気象庁, 「気象庁ホームページについて」, current terms, date not stated: https://www.jma.go.jp/jma/kishou/info/coment.html
- 気象庁, 「気象データ高度利用ポータルサイト」, date not stated: https://www.data.jma.go.jp/developer/
- 気象庁, 「過去の気象データ・ダウンロード」, current system, date not stated: https://www.data.jma.go.jp/risk/obsdl/index.php
- 気象庁, 「このページでできること」, date not stated: https://www.data.jma.go.jp/risk/obsdl/top/help2.html
- 気象庁, 「ダウンロードファイル（CSVファイル）の形式」, date not stated: https://www.data.jma.go.jp/risk/obsdl/top/help3.html
- 気象庁, 「過去の気象データ・ダウンロード FAQ」, date not stated: https://www.data.jma.go.jp/risk/obsdl/top/faq.html
- 一般社団法人日本競走馬協会, 「セレクトセール2026 セール結果」, sale dates 2026-07-13/14: https://www.jrha.or.jp/selectsale/2026_list2.html
- 一般社団法人日本競走馬協会, 「セレクトセール2026 集計・種牡馬別取引成績」, 2026-07-13/14: https://www.jrha.or.jp/selectsale/2026_list4.html
- 一般社団法人日本競走馬協会, `robots.txt`, date not stated: https://www.jrha.or.jp/robots.txt
