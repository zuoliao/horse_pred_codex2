# 無料データ・公開データセット・公開 API の追加調査

## 1. 調査目的と結論

本書は、JRA 中央競馬向け MVP の主データ源を決めるために、Kaggle、GitHub、政府・競馬当局の公開データ、研究データリポジトリ、海外競馬 API を追加探索した結果である。調査日は **2026-08-30**。データ本体の大量取得や実装は行わず、検索結果、dataset card、メタデータ、仕様書、利用条件だけを確認した。

### 高信頼の結論

- **無料で、JRA の長期 runner-level データを、取得元の許諾が追跡可能な形で提供し、PIT 特徴と現実的な締切前オッズを再現できるデータセットは見つからなかった。**
- Kaggle の JRA データは、(a) netkeiba スクレイピングと明記、(b) 出所・ライセンス不明、(c) uploader が CC0 等を付しただけで取得元の再配布許諾が確認できない、のいずれかだった。Kaggle 上の license badge は、元データの権利処理を証明しない。
- GitHub で有用なのは、正規契約した JRA-VAN データを DB 化する **コード**である。Apache-2.0 等のソフトウェアライセンスは JRA-VAN データを無料化・再許諾しない。
- 韓国 KRA と Racing Queensland には、当局自身が提供する無料かつ機械可読の公開データがある。由来と利用許諾は第三者再配布データより明瞭だが、競走制度、馬場、クラス、出走集団、賭式・市場が JRA と異なるため、JRA の精度や ROI を検証するデータにはならない。
- 気象庁の過去観測 CSV は、出典表示・加工表示を条件に再利用できる、数少ない国内の無料補完候補である。ただし最寄り観測点の降水・気温・風は JRA 公式の天候・馬場状態を代替しない。
- 無料公開データの妥当な役割は、外部スキーマの研究、小規模な ETL・grouped evaluation の試作、方法論の再現に限られる。**JRA MVP の本番学習データを置き換えない。**

### 推奨判断

1. 公開再配布dataset/APIをJRA MVPの主データにしない。統合結論では、無料予測trackはJRA公式成績PDFを条件付き主候補、厳密PIT・市場trackはJRA-VAN/JV-Linkを有料fallbackとする。
2. Kaggle/GitHub の JRA 再配布 CSV は、学習、ベースライン数値の提示、単体テスト fixture、派生物公開のいずれにも採用しない。権利と PIT の二つを同時に解消できないためである。
3. 公開 API の追加候補は「JRA データ源」ではなく「外部 sandbox」として別管理する。採用優先度は KRA、Racing Queensland の順。ただし取得前に小規模な schema/欠損/履歴範囲の確認を行う。
4. 無料補完源としては気象庁 CSV を採用候補にする。観測所、観測時刻、品質 flag、取得時刻、後日訂正を保持し、JRA 公式 weather/going との差分特徴としてだけ検証する。
5. データ処理の unit test には、第三者 CSV の一部をコピーせず、明示的に合成した架空レースを使う。
6. 将来、JRA-VAN 取込コードを比較するなら `miyamamoto/jrvltsql` は有力な技術参照先である。ただし契約、地域、保存、第三者提供の境界は JRA-VAN 規約が支配する。

## 2. 判定方法

### 2.1 評価軸

各候補を次の順で評価した。

1. **由来**: 競馬当局・研究者自身の一次公開か、第三者 scrape/転載か。
2. **権利**: データに適用されるライセンスが明示され、公開者がその権利を付与できると追跡できるか。コードのライセンスと入力データのライセンスは分ける。
3. **JRA 適合性**: JRA 中央競馬を直接含むか。海外データから移せるのは実装パターンか、競馬固有の予測関係か。
4. **PIT**: 予測時点で既知だった出馬表、取消、馬体重、馬場、オッズ等の版と発表時刻を再現できるか。結果行の最終オッズは PIT オッズではない。
5. **内容と品質**: 期間、runner/race/entity ID、結果、条件、血統、ラップ、オッズ、欠損、訂正、dead heat/non-finisher 表現。
6. **運用性**: 更新性、API/ファイル形式、認証、無料枠、rate limit、SLA・安定性。
7. **許される利用**: ローカル保存、加工、ML、派生物、商用利用、再配布。

### 2.2 判定ラベル

| 判定 | 意味 |
|---|---|
| 採用候補 | JRA MVP の主学習データに使える可能性が高い |
| 外部 sandbox | 権利は比較的明瞭だが非 JRA。処理や方法論の試験だけ |
| 研究参照のみ | 限定的な論文再現・スキーマ研究には使えるが、MVP データにはしない |
| 保留 | 公式な許諾、由来、履歴等の追加確認が必要 |
| 不採用 | 現状の証拠では取得・保存・ML・再配布、または PIT の要件を満たさない |

本書は法的助言ではない。特に、uploader が付けたオープンライセンスが第三者の権利を消滅させるとは解釈しない。

## 3. 横断比較

### 3.1 JRA に近い無料候補

| 候補 | 由来・表示ライセンス | 期間・主な内容 | PIT / odds | 更新・技術 | 保存・ML・再配布 | JRA MVP 判定 |
|---|---|---|---|---|---|---|
| Kaggle `takamotoki/jra-horse-racing-dataset` | card が netkeiba scrape と明記。Kaggle 表示は CC BY 4.0 | 1986-01-05～2021-07-31。結果、払戻系 odds、lap、corner。約100 MB compressed | 結果と最終情報。取得時刻・事前版なし | 2021-08-16 更新で停止 | uploader が元サイトの再配布権を有する証拠なし | **不採用** |
| Kaggle `ayuser/horse-racing-in-japan` | source は `Japan` のみ、license `unknown` | 2010～2021、単一 CSV 約935 MB uncompressed | card から PIT 版を確認できず | 2022-04-17 更新 | 許諾範囲不明 | **不採用** |
| Kaggle `noriyukifurufuru/japan-horse-racing-2010-2025` | CC0 表示、source/説明なし | 2010～2025、races/results/payouts の3ファイル、約110 MB compressed | 事前版・時点情報なし | 2025-12-28 更新 | CC0 を付与できる chain of title 不明 | **不採用** |
| Kaggle Dataset Labs 2024 sample | source は JRA public site と自己申告、CC BY-NC-SA 4.0 | 2024年の3か月 sample。race/runner/result/final odds/pedigree 等、約3.1 MB | final odds。履歴 snapshot なし | 2026-07-14 更新。full dataset は外部で $49 と案内 | 非商用制約。公式取得許諾の証拠なし | **不採用** |
| NAR 公式データ DL | NAR 公式機能。サイト規約は無断転載・複製を制限 | 地方 race は 1998-01～、odds は 2026-03～。出馬表、結果、払戻、ラップ等 | 当日 ZIP は timestamp 付き中間 odds、月次は結果・確定 odds。ただし過去 PIT 履歴は短い | 日次・月次。CSV/ZIP。異常アクセス遮断あり | DL 機能は公式だが恒久保存・加工 ML・派生物共有の明示許諾を確認できず | **保留、地方補助のみ** |
| 気象庁 過去気象 CSV | 気象庁公式。権利表記のない content は公共データ利用規約1.0、出典・加工表示が必要 | 全国の気象台・AMeDAS。降水、気温、風等。観測所・項目ごとに期間差 | 原則前日まで。CSV は download time と品質・不均質情報を持つ。遡及訂正あり | UI 条件指定 CSV。1 request data limit、過度な自動 access を控えるよう明記 | 条件を守れば保存・加工 ML の補完候補。個別権利・法令の例外あり | **採用候補（補完のみ）** |
| GitHub `miyamamoto/jrvltsql` | コード Apache-2.0。データは正規 JRA-VAN/JV-Link | JRA-VAN の出馬表、結果、払戻、確定 odds、公式時系列 odds 等を SQLite/DuckDB/PostgreSQL へ保存 | 公式時系列は単複枠・馬連約1年、速報は全賭式約1週。収集継続が必要 | 483 commits を確認。Windows/JV-Link中心、x86_64 Linux/Wine 経路も記載 | コードの利用権と JVData の権利は別。契約キー必須 | **技術参照候補、無料データではない** |

### 3.2 海外・研究用候補

| 候補 | 公式性・ライセンス | 期間 / fields / ID | odds・PIT | 更新・rate limit | JRA への転用 | 判定 |
|---|---|---|---|---|---|---|
| Korea Racing Authority Open API | KRA が韓国政府 Public Data Portal で直接公開。「use permission range limitless」、無料 | 詳細結果は race date/no、finish、horse/jockey/trainer/owner IDs、負担重量、着差、馬体重、final win/place dividend、time、rating、equipment 等。期間欄は空 | 結果 API。予測前 snapshot/odds 時系列の保証なし。current roster は PIT 履歴でない | REST/XML、key 必須。dev 自動承認、production 審査。traffic は API ごとに異なる（roster 100,000、rating 3,000、詳細結果は表示値空欄） | schema、stable ID、外部 rating の比較には有用。韓国・済州馬、クラス、競馬場、市場は JRA と異なる | **外部 sandbox** |
| Racing Queensland Open Data | Racing Queensland 公式。「Open Licence Yes」「free for use without copyright restriction」。ただし商用情報に料金を課す権利を留保 | thoroughbred meeting/race、nominations、acceptances、weights、results、horse/race details。target publication Nov 2013 | odds 時系列は仕様で確認できず。API の現在値を過去 PIT とみなせない | daily/as needed。SOAP 1.1/1.2、HTTP GET/POST、ASMX/XML。数値 rate limit/SLA 不明 | ETL、race grouping、結果例外の試験には有用。豪州 handicap・surface・市場差が大きい | **外部 sandbox** |
| Betfair Historical Data Basic | Betfair 公式。無料 Basic | Stream API 導入後の2016～。exchange market JSON/TAR、1分間隔 last traded price、Basic は volume なし | timestamped odds は強み。ただし exchange であり JRA pari-mutuel ではない。JRA market coverage は確認できず | Historic Data API。account/jurisdiction 条件。正確な API limit は今回確認できず | odds replay の一般技術には有用。JRA の displayed odds/払戻には代替不可 | **研究参照のみ** |
| NYRA Big Data Derby 2022 | NYRA/NYTHA 等が公式 competition として提供 | 2019年 NYRA、4 files、in-race tracking と race data | 主に競走中位置。JRA の予測前 odds/PIT履歴ではない | static competition data | pace/trajectory の将来研究には有用。1 US circuit・1年で MVP form/history 不足 | **研究参照のみ** |
| Dryad `Racehorses are getting faster` | 著者・論文対応データ、Dryad DOI | GB turf: elite winners 1850～2012、全馬 speed 1997～2012。元資料は Ruff's/Raceform | 市場 odds や予測前版なし | 2015 static、42.83 MB | time/going/distance の方法研究には有用。JRA runner-level MVP/ROI には不足 | **研究参照のみ** |
| The Racing API | 民間 aggregator。自身で「official provider ではない」と明記 | core は UK/IRE/HK、group-level global。racecards/results/form/entity IDs/odds | Pro は UK/IRE odds history。JRA 完全 coverage なし | 5 req/s default。Standard £59.99/月、Pro £99.99/月。無料常設 plan は確認できず | 有料かつ JRA 不足 | **不採用** |
| Podium Racing API | PA Media/Podium の仕様書。country list に Japan | global card/result/form、UUID、odds shows、revisions/history endpoint。Japan の実 coverage 深度は未確認 | entity update history と betting timestamp は設計上優秀 | REST + optional SNS push、6 MB response limit、資格情報必須。価格・無料枠・rate limit・利用権は公開資料で未確認 | 条件が明確なら将来照会価値あり。ただし無料候補ではなく、JRA authoritative source とも未確認 | **保留（商用照会候補）** |

## 4. Kaggle 調査

### 4.1 調査手順とプラットフォーム上の注意

#### 事実

2026-08-30 に Kaggle 公式 CLI で `horse racing`、`japan horse racing`、`thoroughbred racing`、`hong kong horse racing` を検索し、候補の `metadata` と `files` 一覧だけを取得した。データ本体は取得していない。

Kaggle 公式 CLI の metadata 仕様では、dataset publisher が `licenses` を **exactly one entry** 指定し、`userSpecifiedSources` に出所説明を入力する。すなわち、license と source は uploader が登録する metadata である。Kaggle 利用規約も user submission の責任を投稿者側に置く。Kaggle の badge は、プラットフォームが元サイトとの chain of title を監査・保証したことを意味しない。

#### 解釈

採否は次の二段階にする必要がある。

1. Kaggle dataset 自体の表示ライセンスは何か。
2. uploader は、そのライセンスで元データを再配布できる権利を持つか。

第2段階が確認できなければ、CC0/CC BY 表示だけで採用しない。特に scrape 元の規約が転載・複製・自動収集を制限する場合、uploader のライセンス表示との矛盾を解消できない。

### 4.2 日本競馬候補

#### A. `takamotoki/jra-horse-racing-dataset`

**事実**

- card は「データは netkeiba.com からスクレイピング」と明記する。
- 1986-01-05～2021-07-31 の race results、betting odds、lap times、corner passing orders を含む。corner order は card 上 2002-06-15 以降。
- race result には race/runner composite ID、条件、着順、枠馬番、性齢、斤量、騎手、time、margin、corner、上がり、単勝、人気、馬体重、調教師等がある。別 odds file は払戻・確定情報で、同着用の複数列も持つ。
- Kaggle metadata は CC BY 4.0、更新 2021-08-16、compressed 約100 MB、検索時 4,534 downloads、usability 1.0 を表示した。

**不確実性・矛盾**

- netkeiba の取得・再配布許諾を示す文書は card にない。
- race page をいつ取得したか、後日の訂正を含むか、各列が予測時点に存在したかを復元できない。
- 「odds」は締切前 snapshot ではなく、結果・払戻と結び付いた最終値である。

**示唆・判定**

内容だけ見れば魅力的だが、権利と PIT の両方が失格である。**不採用**。この CSV で高い ROI を報告しても、現実の予測時点に再現できない。

#### B. `ayuser/horse-racing-in-japan`

**事実**

- 2010～2021 の単一 CSV、約935 MB uncompressed。
- Kaggle metadata の source は単に `Japan`、license は `unknown`、usability は約0.47。2022-04-17 更新。

**判定**

由来、利用権、列定義、PIT が不足するため **不採用**。欠けた情報をファイルの存在や download 数で補ってはならない。

#### C. `noriyukifurufuru/japan-horse-racing-2010-2025`

**事実**

- results、races、payouts の3ファイル、約110 MB compressed。
- Kaggle metadata は CC0、2010～2025 と題し、2025-12-28 更新。
- metadata に dataset description と user-specified source がない。usability は約0.29。

**判定**

CC0 は出所を代替しない。公開者が何を根拠に権利放棄できるか確認不能なので **不採用**。

#### D. `datasetlabs/japan-horse-racing-ml-dataset-2024-sample`

**事実**

- card は 2024 年の3か月 sample とし、race/date/course/track/distance/going/weather、horse ID/name/sex/age/weight、trainer/jockey、finish/time/margin/running position、final win odds/popularity、pedigree 等を掲げる。
- metadata は CC BY-NC-SA 4.0、source を JRA public site とし、約3.1 MB。full dataset は 2024～2026-07、$49 の外部販売へ誘導する。
- card は scraping の手間を省く旨を販売上の利点としているが、JRA からの自動取得、保存、再販売の許諾文書は示さない。

**解釈・判定**

非商用・ShareAlike 制約だけでなく、元データの取得・再販売権が不明である。少量 sample でも本プロジェクトの fixture として取り込まない。**不採用**。

### 4.3 海外 Kaggle 候補

#### 事実

- `gdaley/hkracing`: HK races/runs の2 CSV、CC0、検索時 9,971 downloads。card は HKJC race card を参照するが、user-specified source と再配布許諾はない。
- `deltaromeo/horse-racing-results-ukireland-2015-2025`: 現在の title は 1988～2026、source は Racing Post results、CDLA Sharing 1.0、週次更新、大容量。Racing Post 公式規約は、personal/non-commercial の限定を超える reproduction、modification、distribution、republication を事前書面許可なく禁じる。
- `hwaitt/horse-racing`: 1990～2020、CC BY-NC 4.0、source は vague な `Open sources`。card 自身が SP は race start 前に利用できず profit prediction に正確でないと注意する。
- `hassankamran011/historic-australian-horse-racing-dataset`: CC0 表示だが、harness.org.au と betfair.com.au を scrape したと明記する。
- `jpmiller/race-data`: Big Data Derby mirror、`copyright-authors`。公式 competition のデータと見られるが、mirror の metadata は competition rules を代替しない。

#### 解釈・判定

いずれも JRA 学習用に転用しない。外国競馬でモデルの code path を動かせても、JRA の効果量・calibration・market edge の証拠にはならない。また、元サイトの制約と uploader license が衝突する dataset は、地域にかかわらず採用しない。

## 5. GitHub 調査

### 5.1 正規データの取込コード

#### `miyamamoto/jrvltsql`

**事実**

- JRA-VAN Data Lab./JV-Link の JRA データを SQLite/DuckDB/PostgreSQL に取り込む Python tool。repository は Apache-2.0、2026-08-30 時点で 483 commits。
- README は JRA-VAN 契約と service key を要件とし、未登録時は fail closed と説明する。
- 通常データ、確定 odds、単複枠・馬連の公式時系列 odds、開催週の全賭式速報 odds を区別する。JRA-VAN 側の保持を公式時系列約1年、速報約1週と記載する。
- Windows/JV-Link が公式想定環境。x86_64 Linux + Docker/Wine 経路も同梱するが、ARM/Apple Silicon は非対応、64-bit SDK 経路も未検証と明記する。

**解釈・示唆**

- 「無料 JRA dataset」ではないが、公式取得と DB schema を接続する技術参照として価値がある。
- repository の Apache-2.0 は source code に適用される。JVData の保存・加工・公開は JRA-VAN 契約に従う。
- 本プロジェクトで将来比較する場合も、コードを採用する判断とデータ利用条件の判断を別々にレビューする。

### 5.2 Web scraper repositories

#### 事実

- `new-village/KeibaScraper` は netkeiba の entry/result/odds/horse を parse する Apache-2.0 library。README は、利用によって大きな負荷を与えうること、scraping が terms of service に違反しうること、personal/educational use であることを明記する。
- `zliu43/netkeiba-scraper` は、全馬の HTML を S3 に archive し、processed data を Kaggle へ upload する大規模構想を公開している。これは技術設計の存在であり、取得・保存・再配布の許諾証拠ではない。
- ほかにも HTML selector を直接使う `ebi44323/keiba_ai`、`tanakatsu/netkeiba_crawler` 等が検索で見つかる。保守性は対象画面の非公開仕様に依存する。

#### 解釈・判定

- scraper の OSS license は、対象サイトのデータやアクセス方法を許諾しない。
- rate limit、cache、User-Agent が「丁寧」であることも、利用規約上の許可とは別問題である。
- netkeiba/JRA/JBIS 等からの取得を行う scraper は、本プロジェクトでは **不採用**。コードをそのまま vendor したり、出力 dataset を利用したりしない。

### 5.3 GitHub license の一般則

GitHub 公式文書は、license がなければ default copyright が適用され、他者は原則として reproduction、distribution、derivative work を行えないと説明する。また、GitHub の license detector は repository の LICENSE file を識別するもので、dependencies や第三者入力データの権利まで評価しない。したがって次を守る。

- repository が public であることを利用許諾とみなさない。
- code license と `data/` 配下のデータ license を別々に確認する。
- scraped source、competition input、商用 rating 等が混在する場合、repository 全体の MIT/Apache 表示をデータに広げない。

## 6. 競馬当局・政府の公式公開データ

### 6.1 Korea Racing Authority / data.go.kr

#### 事実

韓国政府 Public Data Portal には KRA 自身が提供する複数の REST API がある。

- `KRA_race_detail_result`: 2021-09-23 登録、2026-08-16 更新。Seoul、Jeju、Busan-Gyeongnam、将来の Yeongcheon を対象に、race date/no、finish、entry no、horse no/name、origin、sex/age、carried weight、jockey/trainer/owner no/name、margin、body weight/change、final win/place dividend、record、rating、equipment、participation を XML で返す。無料、「use permission range limitless」、development は自動承認、production は審査。temporal range と development traffic の数値は page 上空欄。
- `KRA_race_horse_list`: current roster、horse no、grade、origin、sex/age、career、prize、recent race、ratings、active flag 等。無料、同じ許諾表示。development traffic 100,000。current state の一覧であり、過去 snapshot ではない。
- `raceHorseRating`: horse no/name と rating 1～4。無料、real-time、development traffic 3,000、production は use-case 審査。
- `RC race information`: result、all bet-type final dividends、distance、record、going、weather、body weight 等。最終配当であり締切前 odds snapshot ではない。

#### 不確実性

- 詳細結果 API の temporal coverage が metadata にない。登録日以前をどこまで query できるかは小規模 call での確認が必要。
- rating 1～4 の時点定義、訂正履歴、欠損率、racehorse ID の永続性を今回の metadata だけでは確定できない。
- 「real-time」は更新周期を表すが、過去版を保存することを意味しない。

#### JRA への示唆

- 利用権と provenance が明瞭な外部 schema の最有力候補。race/runner/entity ID、non-starter、equipment、rating の型設計を比較できる。
- ただし Jeju native horse、韓国のクラス/handicap、競走場構成、投票市場等が JRA と異なる。KRA で優れた feature/model が JRA に一般化するとは主張しない。
- 大量取得はせず、必要になった場合に数日・数レースだけ query して、schema、文字コード、欠損、ID、dead heat/non-finisher 表現を監査する。

### 6.2 Racing Queensland Open Data Web Services

#### 事実

- Racing Queensland は Government Open Data Strategy の一部として public web services を提供する。
- official page は Thoroughbred Racing Data を「meeting/race、horse nominations、acceptances、weights、race results」、target publication November 2013、daily/as needed、Open Licence `Yes` とする。
- copyright section は、web services の情報を Australia law と international treaty の下で copyright restriction なく free for use とする。一方、アクセスを随時制限でき、commercial purpose の情報利用には料金を課しうると留保する。
- API は legacy ASMX/XML で、SOAP 1.1/1.2、HTTP GET/POST を案内する。meeting/date range、race details、horses、horse details、race reports、gear、stewards report 等の operation がある。
- 数値 rate limit、uptime SLA、revision retention、odds history は公開 page/spec から確認できなかった。

#### 解釈・示唆

- private research の外部 sandbox としては、第三者 scrape よりはるかに安全である。
- commercial use へ移る場合は、料金適用、保存、派生物、再配布を確認する。
- 2014 年代の ASMX 技術と current service の動作差、過去 meeting range の実取得範囲を少量 request で検証する必要がある。
- JRA の turf/dirt、クラス体系、斤量、枠効果、市場へ数値を移植しない。

### 6.3 NAR 公式ダウンロード

#### 事実

- 2026-03-27 の機能更新で、地方競馬情報サイトに race/odds CSV download が追加された。2026-05-18 版説明書は、当日 `YYYYMMDD_(timestamp)_race.zip` / `_odds.zip`、月次 `YYYYMM_(timestamp)_race.zip` / `_odds.zip` を定義する。
- race list は date、race no、post time、type/class、distance、weather/going、field size、prize、上がり・lap 等を持つ。既存調査では race data は 1998-01～、odds は 2026-03～。
- 公式サイト利用規約は、掲載情報の権利が原則主催者等に帰属し、事前許諾のない転載・複製を禁じる。異常アクセス時の IP block も明記する。

#### 解釈・提案

- 公式 DL が存在するため「画面 scrape」より正規であるが、ML 用長期保存、加工、再配布まで許諾されたとは断定しない。
- JRA 転入馬の地方歴補完に価値があり得るが、JRA-VAN に既収録の関連地方歴との差分、ID mapping、2026-03 以前の odds 不在を考えると MVP 主データにはならない。
- 採用前に NAR へ、個人ローカル ML、長期保存、特徴量・予測の派生、共同利用の可否を書面照会する。

### 6.4 気象庁の過去気象データ

#### 事実

- 気象庁「過去の気象データ・ダウンロード」は、全国の気象台・AMeDAS の地点、期間、項目を選び CSV で取得できる。
- download page は一回の data volume に上限があること、混雑の原因となる automated tool 等による過度な access を控えることを明記する。
- CSV は download time、data heading、data rows を持ち、観測値に品質情報、現象有無、不均質情報を付けられる。
- 気象庁は、権利表記のない website content を公共データ利用規約第1.0版の下で利用できるとし、出典表示と、編集・加工した場合の加工表示を求める。個別法令・第三者権利等の例外は別途確認が必要。
- 気象データは過去に遡って修正される場合があり、公式 page は主な修正を告知する。

#### 解釈・提案

- 本調査で確認した国内無料源の中では、provenance、機械可読性、再利用条件が最も明瞭な補完候補である。
- 競馬場に最寄りの観測点を対応させても、局地的な競馬場内 weather、散水、芝・砂含水、JRA の総合的 going 判定とは一致しない。JRA 公式 weather/going を置き換えず、降水履歴、気温、風等の追加 feature group として ablation する。
- 将来取得する場合は station metadata、observation timestamp、download timestamp、quality/homogeneity flag、source URL、加工表示を保持する。最新版再取得で過去版を黙って上書きしない。

### 6.5 公的ポータルの負の結果

#### 事実

- 日本の e-Gov/e-Stat 系検索では、JRA runner-level race result/odds の open dataset は確認できなかった。JRA-55 は気象庁の再解析名称であり競馬 JRA ではない。
- DATA.GOV.HK では、HKJC の runner-level racing results dataset/API を確認できなかった。HKJC public racing pages が閲覧可能でも、open-data license や bulk API の根拠にはならない。
- France data.gouv の `L'institution des courses` は 2018 年の監査関連小規模データで、race/runner results ではない。
- The Jockey Club Fact Book/State Fact Books は breeding、racing、auction の集計統計を公開するが、runner-level 学習データではない。定義・件数の sanity check には使える。
- BHA の public results page はあるが、open bulk API/license を確認できない。Racecourse Data Company は fields、owner、trainer、jockey、weight、draw、rating 等の pre-race data を third parties に license する主体だと明記する。

#### 解釈

政府・当局サイトに表や検索画面があることと、ML 用の一括取得・保存・再配布が許されることは別である。集計統計は母集団件数や field size 分布の検算には使えるが、runner row を復元してはならない。

## 7. 市場データと海外民間 API

### 7.1 Betfair Historical Data

#### 事実

Betfair 公式説明は次を示す。

- Stream API 導入後の 2016 年以降、ほぼ全 exchange markets の historical data。
- JSON/TAR、event/market filters、timestamped odds と volume。
- Basic は無料、1分間隔 last traded price、volume なし。Advanced は1秒・volume、Pro は50ms・full depth で有料。
- 別の free historical racing CSV は、国/日/win-or-place ごとに pre-play/in-play min/max、weighted average、BSP、winner 等を持つ。

#### 不確実性・示唆

- JRA 国内 pari-mutuel market の coverage は確認できず、Betfair exchange price は JRA の表示単勝 odds と生成機構が異なる。
- account、居住地・jurisdiction、commercial reuse、redistribution、Historical API rate を利用時に確認する必要がある。
- 一般的な odds replay、cutoff、in-play exclusion の試験には使えるが、JRA EV backtest の代替にはしない。

### 7.2 The Racing API

#### 事実

- core complete coverage は UK、Ireland、Hong Kong。global は group races と selected handicaps。JRA 全レース coverage の記載はない。
- Standard £59.99/month、Pro £99.99/month。default rate limit 5 requests/sec。
- provider は「official provider ではない」「manual input と freely available public records を aggregate/transform」と明記し、update frequency を保証しない。
- data resale は permission なしで禁止。apps/websites/data analysis/ML への利用を掲げるが、betting operators/sportsbooks は禁止。

#### 判定

無料候補でなく、JRA の coverage と authority も不足するため **不採用**。

### 7.3 Podium Racing API

#### 事実

- 2025-05 版 user guide の country list は Japan を含む。
- racecards、results、last six form、participant UUID、live odds shows、SP、non-runner、status、revisions/history endpoint、optional AWS SNS push を記載する。
- response limit 6 MB、Basic Auth で30分 token。価格、無料枠、request rate、Japan の期間・全レース coverage、保存/ML/再配布条件を公開資料から確認できなかった。

#### 解釈・判定

PIT revision model は本プロジェクトの要件に近いが、契約条件が空白なので無料データ調達案には入らない。将来、JRA-VAN 以外の法人 feed を比較するときだけ、Japan coverage、authority chain、historical depth、price、private ML、model outputs を書面照会する。

## 8. 研究機関・原著論文付随データ

### 8.1 Dryad: `Data from: Racehorses are getting faster`

#### 事実

- Sharman and Wilson、Dryad DOI `10.5061/dryad.qn82p`、2015-08-06 publication、42.83 MB。
- `1850_2012_elite_winners.xlsx` は GB turf elite winners、`1997_2012.xlsx` は GB turf runners の speed。usage notes は Ruff's Guide、Raceform annual/interactive を source とする。
- 対応原著は *Biology Letters* 2015、ground softness 等を調整して speed trend を分析した。

#### 解釈・判定

原著に対応し DOI と由来が明瞭な点は Kaggle mirror より強い。ただし、JRA ID、現行 form、予測前 fields、odds、PIT revisions を満たさない。time/going/distance normalization の論文再現に限定する。

### 8.2 NYRA Big Data Derby 2022

#### 事実

- NYRA 公式発表によれば、competition は 2019 NYRA racing data と in-race horse tracking の4ファイルを提供し、9,349 potential competitors が access した。
- health、path efficiency、tactics 等の分析が目的で、JRA の通常 racecard/history dataset とは粒度が違う。
- Kaggle notebook の Apache-2.0 は notebook code の license であり、competition input data の一般再配布 license ではない。Kaggle 自身も competition data の用途は各 competition rules に従うとしている。

#### 判定

将来の trajectory/pace encoder 研究の参照。MVP の構造化過去成績や JRA market evaluation には使わない。competition 外利用条件を再確認せず mirror を repository に置かない。

### 8.3 OpenML/UCI/Zenodo 等の検索結果

#### 事実

- OpenML は dataset license と metadata を要求し API 検索を提供するが、今回の検索で、JRA または信頼できる runner-level thoroughbred race dataset を確認できなかった。
- UCI の `Horse Colic` は veterinary outcome dataset で、horse-racing prediction dataset ではない。
- Zenodo/Dryad には equine biomechanics、genomics、injury 等の研究データがあるが、JRA race outcome MVP の予測 row に接続できる ID・時点・coverage を持たない。

#### 解釈

「horse」という keyword hit を競馬予測データと誤認しない。健康・ゲノムデータは将来の別研究課題であり、個体 link と利用可能時点がない現状では out of scope。

## 9. JRA へ一般化できない証拠と境界

### 9.1 直接一般化できない事項

- **競走制度**: KRA、Hong Kong、UK、Australia、US は class/handicap、斤量決定、開催・調教体系が異なる。
- **出走集団**: 韓国 Jeju horse、香港の限られた horse population、US dirt-heavy circuits 等は JRA と共変量分布が異なる。
- **馬場・course**: surface composition、turn direction、straight length、going definitions、timing method が異なる。
- **市場**: Betfair は exchange、UK SP は bookmaker starting price、JRA は pari-mutuel displayed odds と final payout。price の確率解釈と takeout が一致しない。
- **PIT**: static results CSV の odds/popularity/body weight は、値自体が pre-race に存在し得ても、その CSV がどの時点版を保存したかを証明しない。
- **entity ID**: provider-local horse/jockey/trainer ID は jurisdiction 間で共通ではない。name join は同名・表記・移籍で壊れる。

### 9.2 一般化してよい範囲

- race/group/runner の relational schema。
- grouped train/evaluation と一 race 内 probability coherence のテスト方法。
- scratch/non-runner/dead heat、missing odds、revision の data contract 設計。
- API ingestion の idempotency、raw payload hash、`observed_at`/`effective_at`/`ingested_at` の分離。
- 原著論文の統計手法を再現する practice。ただし JRA 効果量は別検証する。

## 10. 具体的な提案

### 10.1 データ源の decision gate

無料候補を JRA 学習データへ昇格させるには、少なくとも次をすべて満たす必要がある。

1. JRA/JRA-VAN/正規 licensee からの chain of title を文書で示せる。
2. automated retrieval、ローカル保存、加工、private ML の許諾が明記される。
3. race/runner/horse/jockey/trainer の ID と列定義がある。
4. historical coverage、訂正、欠損、non-finisher/dead heat が監査できる。
5. target prediction timestamp に対応する snapshot または event timestamp がある。
6. backtest に使う odds が cutoff 時点で観測可能で、final odds/払戻との区別がある。
7. model、aggregate features、predictions、evaluation artifact の共有可否が分離して確認できる。

今回の無料 JRA 候補はこの gate を通らない。

### 10.2 外部 sandbox を使う場合の最小調査

実装フェーズで必要性が生じた場合のみ、KRA または Racing Queensland に対して大量取得前の小規模 probe を行う。

- 3～5 race days、数十 race に限定する。
- raw response と HTTP metadata、query、取得時刻を保存する。
- race/runner/entity ID の uniqueness と persistence を確認する。
- scratch、DQ、DNF、dead heat、取消 race、欠損 carried weight/body weight を探す。
- 同一 query の後日再取得で revision が起きるか比較する。
- pre-race card と result を別 table/event にし、result で pre-race row を上書きしない。
- API-specific traffic と production approval を確認する。

この probe の目的は data contract の検証であり、JRA model score の事前推定ではない。

### 10.3 repository での provenance 記録案

将来データを採用するときは source ごとに最低限次を machine-readable に残す。

```text
source_name
source_owner
source_url
acquisition_method
license_name
license_url_or_contract_id
license_checked_at
allowed_storage
allowed_ml
allowed_redistribution
temporal_coverage_claimed
observed_at_semantics
effective_at_semantics
revision_policy
rate_limit
raw_payload_hash
```

`license_name` が `CC0` でも、`source_owner` と `source_url` が説明できなければ gate を fail させる。

## 11. 不確実性・追加確認事項

- KRA 詳細結果 API の実 temporal depth、欠損率、訂正履歴、詳細結果ページで空欄だった development traffic。
- Racing Queensland ASMX endpoints の現在の安定性、全履歴への range query、rate limit、commercial-use fee の適用境界。
- NAR 公式 download の private ML・長期保存・派生 feature/model 利用許諾。
- Betfair Basic の日本 racing market coverage、account/jurisdiction、API download limit、commercial model と redistribution 条件。
- NYRA Big Data Derby の current competition-specific data license と competition 外利用範囲。
- Podium の Japan coverage（JRA/NAR、全 race/group races の別）、履歴年、source authority、価格、rate、保存、ML、派生物利用。
- Kaggle dataset が将来、元権利者の明示許諾や公式 dataset card を追加した場合は再審査できる。ただし現行版を先に取得・利用しない。

## 12. 主要な一次・公式資料

すべて **accessed 2026-08-30**。日付は資料に表示された publication/update date。日付のないものは `date not stated`。

### Kaggle

- Kaggle, `kaggle-cli/docs/datasets_metadata.md`, current official CLI documentation, date not stated: https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md
- Kaggle, `Terms of Use`, current terms, date not stated: https://www.kaggle.com/terms
- takamotoki, `JRA日本中央競馬会 Horse Racing Dataset`, updated 2021-08-16: https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset
- ayuser, `Horse Racing in Japan`, updated 2022-04-17: https://www.kaggle.com/datasets/ayuser/horse-racing-in-japan
- noriyukifurufuru, `Japan Horse Racing Data 2010-2025`, updated 2025-12-28: https://www.kaggle.com/datasets/noriyukifurufuru/japan-horse-racing-2010-2025
- Dataset Labs, `Japan Horse Racing ML Dataset (2024 sample)`, updated 2026-07-14: https://www.kaggle.com/datasets/datasetlabs/japan-horse-racing-ml-dataset-2024-sample
- gdaley, `Horse Racing in HK`, update year displayed as 2019 in CLI search, exact date not confirmed: https://www.kaggle.com/datasets/gdaley/hkracing
- deltaromeo, `Horse Racing results - UK/Ireland 1988-2026`, updated 2026-06-04: https://www.kaggle.com/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025
- jpmiller, `Horse Racing - Big Data Derby`, date not stated: https://www.kaggle.com/datasets/jpmiller/race-data
- hwaitt, `Horse Racing`, dataset coverage through 2020, exact update date not confirmed: https://www.kaggle.com/datasets/hwaitt/horse-racing
- hassankamran011, `Historic Australian Horse Racing Dataset`, date not stated: https://www.kaggle.com/datasets/hassankamran011/historic-australian-horse-racing-dataset

### GitHub / code licensing

- Miyamoto, `miyamamoto/jrvltsql`, current repository, Apache-2.0, date not stated: https://github.com/miyamamoto/jrvltsql
- Miyamoto, `jrvltsql/docs/architecture.md`, date not stated: https://github.com/miyamamoto/jrvltsql/blob/master/docs/architecture.md
- new-village, `KeibaScraper`, current repository, Apache-2.0, date not stated: https://github.com/new-village/KeibaScraper
- zliu43, `netkeiba-scraper/project-design.md`, date not stated: https://github.com/zliu43/netkeiba-scraper/blob/main/project-design.md
- GitHub Docs, `Licensing a repository`, current documentation, date not stated: https://docs.github.com/en/enterprise-cloud@latest/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository
- GitHub Docs, `REST API endpoints for licenses`, current documentation, date not stated: https://docs.github.com/en/rest/licenses/licenses

### 日本の公式資料

- 日本中央競馬会, `過去のレース成績は、どこに掲載されていますか？`, date not stated: https://www.jra.go.jp/faq/pop02/1_6.html
- 日本中央競馬会, `ご利用に際して`, date not stated: https://jra.jp/use/
- 地方競馬全国協会, `データダウンロード機能説明書（一般ユーザー向け）`, updated 2026-05-18: https://www.keiba.go.jp/pdf/manual/data_pdf_manual.pdf
- 地方競馬全国協会, `利用規約`, current terms, date not stated: https://www.keiba.go.jp/terms.html
- 地方競馬全国協会, `システムメンテナンスの実施について (3/26)`, 2026-03-23: https://www.keiba.go.jp/jranet/topics/2026/topics_20260323.pdf
- 気象庁, `気象庁ホームページについて`, current terms, date not stated: https://www.jma.go.jp/jma/kishou/info/coment.html
- 気象庁, `過去の気象データ・ダウンロード`, current system, date not stated: https://www.data.jma.go.jp/risk/obsdl/index.php
- 気象庁, `ダウンロードファイル（CSVファイル）の形式`, date not stated: https://www.data.jma.go.jp/risk/obsdl/top/help3.html
- 気象庁, `過去の気象データ・ダウンロード FAQ`, date not stated: https://www.data.jma.go.jp/risk/obsdl/top/faq.html
- デジタル庁, `e-Govデータポータル データセット検索`, current portal, date not stated: https://data.e-gov.go.jp/data/dataset/

### 韓国 KRA

- Korea Racing Authority / Public Data Portal, `KRA_race_detail_result`, registered 2021-09-23, edited 2026-08-16: https://www.data.go.kr/en/data/15089492/openapi.do
- Korea Racing Authority / Public Data Portal, `KRA_race_detail_result` schema.org catalog record, modified 2026-08-16: https://www.data.go.kr/catalog/15089492/openapi.json
- Korea Racing Authority / Public Data Portal, `KRA_race_horse_list`, registered 2021-09-23, edited 2025-05-31: https://www.data.go.kr/en/data/15089503/openapi.do
- Korea Racing Authority / Public Data Portal, `한국마사회 경주마 레이팅 정보`, registered 2014-06-07, edited 2026-04-19: https://www.data.go.kr/data/15057323/openapi.do
- Korea Racing Authority / Public Data Portal, `한국마사회_RC경마경주정보`, registered 2020-09-23, modified 2026-04-19: https://www.data.go.kr/catalog/15063950/openapi.json

### Australia / UK / Hong Kong / US

- Racing Queensland, `Open Data`, current official page, date not stated: https://www.racingqueensland.com.au/about/reports-and-data/open-data
- Racing Queensland, `Open Data Web Services`, current service index, date not stated: https://www.racingqueensland.com.au/OpenDataWebServices/
- Racing Queensland, `Open Data Web Services Protocol & Interface Specification`, current PDF, date not stated: https://www.racingqueensland.com.au/rq-open-data-web-services.pdf
- Racing Australia, `Terms of Use`, current terms, date not stated: https://www.racingaustralia.horse/AboutUs/TermsOfUse.aspx
- Racing Post, `Terms and conditions`, updated 2025-01-31: https://help.racingpost.com/hc/en-us/articles/208996085-Terms-and-conditions
- Racecourse Data Company, `About Us`, date not stated: https://www.racecoursedatacompany.com/about-us/
- British Horseracing Authority, `Results`, current public results page, date not stated: https://www.britishhorseracing.com/racing/results/
- New York Racing Association, `Overwhelming number of submissions for the inaugural Big Data Derby`, 2022-11-15: https://www.nyra.com/aqueduct/news/overwhelming-number-of-submissions-for-the-inaugural-big-data-derby/
- Kaggle / NYRA-NYTHA, `Big Data Derby 2022`, 2022 competition: https://www.kaggle.com/competitions/big-data-derby-2022
- The Jockey Club, `Resources / State Fact Books`, current page, date not stated: https://home.jockeyclub.com/Default.asp?area=12&section=Resources
- DATA.GOV.HK, `Developer Center`, current page, date not stated: https://data.gov.hk/en/dev-center
- data.gouv.fr, `L'institution des courses`, dataset concerning a 2018 audit, exact publication date not confirmed: https://www.data.gouv.fr/datasets/linstitution-des-courses

### Market/API/research repositories

- Betfair, `Historical Data Sources`, published 2023-08-02: https://www.betfair.com.au/hub/education/how-to-model/historical-data-sources/
- Betfair, `Betfair Historical Data Feed Specification`, date not stated: https://historicdata.betfair.com/Betfair-Historical-Data-Feed-Specification.pdf
- The Racing API, `Horse Racing API & Database`, current product/pricing page, date not stated: https://www.theracingapi.com/
- The Racing API, `Terms of Service`, current terms, date not stated: https://www.theracingapi.com/terms-of-service
- Podium Sports / PA Media, `Podium Racing API User Guide`, May 2025: https://podiumsports.com/wp-content/uploads/2025/05/Podium-Racing-API-User-Guide.pdf
- Sharman, Patrick and Wilson, Alastair J., `Data from: Racehorses are getting faster`, Dryad, published 2015-08-06, DOI 10.5061/dryad.qn82p: https://datadryad.org/dataset/doi:10.5061/dryad.qn82p
- Sharman, Patrick and Wilson, Alastair J., `Racehorses are getting faster`, *Biology Letters* 11:20150310, 2015, DOI 10.1098/rsbl.2015.0310: https://doi.org/10.1098/rsbl.2015.0310
- OpenML, `Using datasets`, current official documentation, date not stated: https://docs.openml.org/data/use/
- UCI Machine Learning Repository, `Horse Colic`, donated 1989-08-06: https://archive.ics.uci.edu/dataset/47/horse+colic
