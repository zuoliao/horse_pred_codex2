# 無料データ優先MVP：データ源の比較と採用判断

**調査日:** 2026-08-30 (JST)  
**対象:** JRA中央競馬の平地競走  
**状態:** 調査・比較・仕様提案のみ。スクレイパー実装、大量取得、学習は未着手。

## 1. 結論

### 事実

- 無料で、JRAの長期runner-level履歴、十分な項目、予測時点別snapshot、機械可読な公式bulk/API、自動取得・長期保存・加工MLの明示許諾をすべて満たす単一ソースは確認できなかった。
- JRA公式の全レース成績PDFは2002年以降を年別に公開し、結果、競走条件、斤量、馬体重、着差、時計、通過順、上がり、race-level lap、最終単勝オッズ・人気、払戻、票数等を含む。これは今回確認した無料JRA一次情報の中で、長期学習用データとして最も内容が広い。
- JRAの現行競馬場別出馬表PDFは原則8週間掲載される。長期archiveではないため、将来時点の入力を残すには今後の継続保存が必要である。
- JRAの公開一括API、機械取得rate、恒久保存・加工MLの包括的な公開許諾は確認できなかった。`robots.txt` は全pathを許す形だが、これは再利用ライセンスではない。
- netkeibaは1956年以降の中央レースを検索でき、画面項目も豊富である。しかし無料で安定確認できるのは主に結果情報で、リアルタイム最新オッズ、調教、指数、厩舎コメント等は無料対象外である。公開bulk API、PIT archive、差分feed、数値rate limit、包括的な自動取得・ML許諾は確認できなかった。
- 気象庁の過去気象CSVは、公共データ利用規約、出典・加工表示、品質flag、取得時刻が明確であり、無料補完源として採用しやすい。
- Kaggle/GitHub上のJRA CSVは、netkeiba scrape、出所不明、またはupstreamの再配布権を確認できないものだった。uploaderのCC0/CC BY表示だけでは権利の連鎖を証明しない。

### 統合判断

ユーザーの「まず無料を最大限活用する」という優先順位を反映し、次の二段階案を採用する。

1. **無料MVPの主データ候補はJRA公式の全レース成績PDF（2002年以降）とする。** これは結果・ラベル・履歴特徴・最終市場診断用の `PIT-C event reconstruction` を作る候補であり、完全な当時版や実行可能な市場backtestを意味しない。
2. **実際の体系的な取得開始はlicense gate後とする。** JRAへ、個人・非商用・ローカル研究におけるPDFの定期取得、長期保存、数値抽出、特徴量化、ML、成果物の扱いを具体的に照会する。明示許諾が得られない場合、無料案は「権利が明確な主データ源」には昇格できず、正規JRA-VAN/JV-Linkへ切り替える。

この判断は「JRA Webの自動取得が許諾済み」という結論ではない。**費用ゼロで技術的・内容的に最も適した候補**と、**利用条件まで解決した採用済みソース**を区別する。

## 2. 候補比較

| ソース | 主な項目・期間 | odds・調教・lap・馬体重 | PIT / 技術 | 利用条件・リスク | 無料MVP判定 |
|---|---|---|---|---|---|
| JRA全レース成績PDF | 2002+。全出走結果、条件、関係者、時計、着差、通過、払戻等 | final win odds、票数、race lap、上がり、馬体重あり。構造化調教なし | 年・開催別PDFは比較的安定。結果なのでPIT-C | open-data/ML許諾、bulk rateなし。robotsはpath制限なし | **条件付き主候補** |
| JRA現行出馬表・当日Web | 現在の枠、馬番、斤量、騎手、属性、変更、馬体重、馬場、odds等 | 当日情報あり。履歴snapshotなし | 出馬表PDFは8週間。HTML URLはopaque、値は更新される | 定期取得・保存・MLは要照会 | **許諾後のprospective補完** |
| netkeiba無料 | 中央検索1956+。近年結果、馬・血統・関係者 | final odds、lap、馬体重あり。最新odds・調教等は有料 | stable-looking IDはあるが外部API仕様なし。旧年field差、PITなし | 多数requestの通信制限FAQ。自動取得・保存・ML許諾なし | **目視照合** |
| JBIS | 馬は原則1984年生+、JRA結果1986+。血統・繁殖・セールが強い | 結果、通過、lap、馬体重、人気・払戻。調教/時系列oddsなし | API/PITなし。一般crawl-delay 600秒、`/csv/`等を除外 | 「そのまま」の個人利用等。加工MLは要照会 | **血統等の目視照合** |
| 競馬ラボ | 結果・馬・血統・コメント。正式coverage保証なし | final odds、lap、馬体重。全馬構造化調教なし | 規則的HTMLだがAPI/schema/PITなし | 複写転載制限、自動ML許諾なし | **目視照合** |
| 競馬ブック無料 | 本年検索、直近走等の限定表示 | 一部情報。無料履歴は限定 | 人間向け画面 | プログラム取込を明示禁止 | **自動取得不採用** |
| NAR公式ZIP/CSV | 地方race 1998-01+、odds 2026-03+ | 結果、約2分当日odds、lap、馬体重、血統。調教なし | 公式CSVだがJRAではない。過去全snapshot保証なし | DL pathをrobots除外、保存・加工MLは要書面確認 | **許諾後の地方歴補完** |
| 気象庁CSV | 観測所別の気温、降水、風等 | 競馬固有情報なし | 機械可読、品質flag、取得時刻。遡及訂正あり | 公共データ利用規約。過度な自動accessを避ける | **採用候補** |
| Kaggle/GitHub JRA data | 例: 1986–2021、2010–2025等 | final odds、lap、馬体重を含む例あり | 静的でPITなし、更新停止/定義不足 | upstream取得・再配布権が未証明 | **不採用** |
| KRA / Racing Queensland | 韓国・豪州の公式API/open data | sourceごと。JRA市場ではない | 公式machine-readableでschema研究向き | 比較的明確だが非JRA | **外部sandboxのみ** |
| JRA-VAN/JV-Link（有料） | 原則1986+、公式schema・ID・速報 | final/time-series odds、調教、馬体重、変更等 | 正規interface、時刻・更新・cache。Windows依存 | 月額、私的利用等の規約、公開範囲は要管理 | **無料案失敗時/高度化時の主系統** |

詳しい根拠は、[netkeiba重点調査](free_data_netkeiba.md)、[国内無料ソース比較](free_data_japan_sources.md)、[公開dataset/API調査](free_data_public_datasets.md)を参照する。

## 3. 推奨する無料データ構成

### A. 条件付き主系統

`JRA公式全レース成績PDF 2002+`

- official result・払戻をsource of truthとする。
- 長期の過去走履歴、ラベル、最終市場診断を構成する。
- PDF単位でsource URL、取得日時、content hash、HTTP metadata、対象開催を記録する。
- 体系的取得はJRAへの照会回答後に開始する。本調査では少数の仕様確認しか行っていない。

### B. prospective系統

`JRA公式の現行出馬表/当日公表値`

- 許諾後、出馬表、取消・変更、馬体重、天候・馬場、同時点oddsを今後保存する。
- Web画面の後日最新版ではなく、当方が実際に観測した `ingested_at` 付きsnapshotを残す。
- 長期結果PDFと混ぜず、`PIT-A observed` として別trackにする。

### C. 明示条件が比較的良い無料補完

`気象庁の過去気象CSV`

- 競馬場近傍観測所の降水、気温、風、湿度等を候補にする。
- JRA公表の天候・馬場を置換しない。局地雨、散水、観測地点差があるため独立feature groupとする。
- station metadata、observation time、download time、quality/homogeneity flag、加工表示を保持する。

### D. 許諾後だけ追加する補完

- NAR公式CSV: JRA出走馬の地方遠征・転入歴。馬名だけで結合せず、生年月日、性、父、母、母父等でmatch evidenceを持つ。
- JBIS/JRHA: 血統・生産・セールの欠損照合。bulk特徴化は別途許諾が必要。
- netkeiba/競馬ラボ: 公式値の表示差や例外の少数目視確認。主取得経路にはしない。

## 4. 無料構成で作れるfeature

JRA公式PDFの取得・加工がlicense gateを通ることを条件に、次を構築できる。

### 今回レースの静的・事前情報

- 競馬場、開催、race number、芝/ダート、距離、コース区分
- race class/grade、年齢・性・賞金・斤量条件、field size
- 枠、馬番、性齢、負担重量、騎手、調教師
- current racecardで公表される範囲の生年月日、父母系、馬主、生産者、賞金・戦績概要

### 過去走から再構成できる情報

- 着順、相対着順、着差、走破時計、上がり、通過順位
- race-level lap、前後半・pace proxy、field size
- 当時の距離、surface、course、class、weather、going、枠、斤量、騎手、馬体重・増減
- 最終単勝オッズ・人気、払戻、票数。ただし一次予測modelから除外し、市場診断・結果側に分離
- horse/jockey/trainerの過去のみの複数窓・条件別・指数減衰集計
- days since last start、14/30/60/90/180/365日出走数、累積距離等のworkload proxy
- course・surface・距離帯・going適性、field内相対値
- forward-only Elo/Bradley–Terry系rating、相手強度、race strength
- JMA由来の過去降水、気温、風等と、その品質・観測所距離

### 当日情報

当日情報は過去PDFから事前snapshotとして再現できない。許諾済みprospective収集後だけ、次を `T_close` 専用で使う。

- 当日馬体重と前走比
- 公表済み天候・馬場、取消・除外、騎手・course・発走時刻変更
- cutoff以前に観測した単勝oddsと鮮度

## 5. 無料構成では取得困難な重要feature

| feature | 無料構成の限界 | JRA-VAN導入での改善 |
|---|---|---|
| 構造化調教時計 | JRA無料Webの動画・一部ニュースは全馬・全期間の表形式ではない。netkeiba主要調教は有料かつ再利用未許諾 | 坂路2003+、woodは美浦2021-07-27+・栗東2021-12-07+の正規record |
| 歴史的な締切前odds系列 | 長期の公式無料archiveを確認できない | 単複枠・馬連の約5–10分系列。ただし公式online保証は1年 |
| pre-race版・変更履歴 | 結果PDFはlatest-known result。旧出馬表・取消・訂正の完全archiveなし | SNAP/速報record、発表時刻、変更種別。ただし古いtransaction historyは完全ではない |
| persistent ID / schema | Web IDの外部仕様・crosswalk・版管理なし | race/horse/person等の公式schemaとkey、コード表 |
| current odds/票の安定取得 | 人間向けWebで更新し、API/rate/保存許諾なし | 正規JV-Linkの現行・速報取得 |
| 全期間の深い血統・entity master | PDFだけでは限定的。current masterを過去へ結合するとleak | 基本情報・SNAP・血統record。ただし時点意味の監査は必要 |
| paddock、厩舎comment、馬具・脚元、独自指数 | 無料の体系的・再利用可能な構造dataなし | JRA-VANでも全て解決するわけではなく、JRDB等の別契約候補 |

## 6. historical oddsと市場評価

### 再現できるもの

- 2002年以降のJRA成績PDFにある最終単勝odds・人気・公式払戻を使った、事後のmarket baseline、favorite/longshot帯別診断、calibration比較。
- 払戻を使った決済ロジックの検証。
- `final-odds oracle diagnostic`。これは、もし最終価格を事前に知っていた場合の診断にすぎない。

### 再現できないもの

- 過去全期間について、T-10等で実際に表示されていたoddsと発表時刻。
- 取得遅延、特徴生成時間、発注余裕、発売締切を含むend-to-endの実行可能性。
- 結果ページのfinal oddsを選択時oddsに置き換えたROI。

したがって、無料MVPの評価を二つに分ける。

1. **Long-horizon prediction track:** JRA結果PDFによるPIT-C。ranking、Log Loss、Brier、calibration、final-odds oracle診断を行う。長期実行可能ROIは主張しない。
2. **Prospective executable-market track:** 許諾後、今後の `PIT-A` snapshotを蓄積し、同一cutoffの予測・odds・変更を結ぶ。十分な期間まではshadow evaluationとする。

## 7. スクレイピング・取得運用のリスク

### 規約・法務

- Webで無料閲覧できること、URLが規則的であること、robotsでpathが許可されることは、保存・加工・ML・再配布の許諾ではない。
- 著作権法30条の4は、情報解析等の「思想又は感情の享受を目的としない利用」を一定範囲で認めるが、権利者の利益を不当に害する場合を除く。文化庁資料は、情報解析市場、技術的措置、販売database等との関係を個別に検討している。これをサイト規約、契約、アクセス制御、サーバ負荷を無効化する包括的許可と解釈しない。
- netkeiba規約は私的利用外の複製・公開等と営業利用を制限し、公式FAQは多数requestがサービスへ支障を与えると判断した場合の予告なき通信制限を明記する。
- JBISは一般crawlerへ600秒delayと一部path除外、NARは10秒delayに加えてdownload/dynamic pathを除外する。競馬ブックはprogram取込を明示禁止する。

### 技術・品質

- HTML selector、JavaScript endpoint、opaque query、会員列は予告なく変わる。
- 訂正feedやold versionがなければ、差分取得に成功してもPITを再現できない。
- source間ID crosswalkがなく、馬名・人名の曖昧joinは誤対応を生む。
- `403`/`429`を回避する工夫、IP rotation、認証・paywall回避、内部endpointの逆解析は行わない。

### 許諾を得た場合の運用原則

- 提供者が指定する方法・頻度を優先し、公開されていないrateを負荷試験で探らない。
- 並列取得を避け、更新周期より短くpollせず、指数backoffし、`403`/`429`で停止する。
- immutableなraw snapshotとmanifestを保存し、同一URLを重複取得しない。`ETag`/`Last-Modified`があればconditional requestに使うが、観察したheaderをSLAとみなさない。
- full snapshotからの差分はローカルで算出し、sourceのdelta APIがあるかのように扱わない。
- terms/robots/manualの確認日、問い合わせ回答、許可された目的・host・保存期間・公開範囲をdata contractへ記録する。

本節は法的助言ではない。曖昧な利用条件を技術設計で埋めず、提供者への具体的照会で閉じる。

## 8. JRA-VANを有料導入すると解決すること

### 解決・大幅改善する点

- JRA公式データを正規JV-Linkで取得する経路と、versioned record仕様。
- 原則1986年以降の長期結果、entity、払戻、血統等と、Web PDFより深い項目・安定ID。
- 馬体重、天候・馬場、取消、騎手・course・発走時刻変更等の速報record。
- 木曜時点SNAP/SNPN、調教record、時系列odds、現行odds、訂正・更新処理。
- HTML/PDF layout変更への依存低下と、ローカルDBへの再現可能な取込。

### 解決しない点

- 1986年からの完全なtransaction-time history。初期setupは後日訂正済みのlatest-known版を含みうる。
- 長期の締切前odds系列。時系列oddsのonline保持保証は1年であり、今後の継続取得が必要である。
- cloud、共同利用、第三者表示、派生feature/model/予測公開の自動的許諾。
- Windows 10/11日本語版ActiveX/COM依存、月額費用、取得環境と学習環境の境界。
- JRDB等が持つ独自paddock・comment・馬具情報の全て。

有料化の価値は、単にfeature数が増えることより、**公式schema・ID・速報時刻・正規取得経路によって、PITと運用品質を上げること**にある。

## 9. 採用decision gate

本格取得・実装前に、次を閉じる。

1. **JRA license gate:** 対象PDF/Web、件数、頻度、raw保存、数値抽出、個人ローカルML、予測・model・aggregate公開を示して回答を保存する。
2. **Free-core gate:** 許諾が得られればJRA PDF 2002+を主系統とする。得られなければ、無料JRA再配布datasetへ迂回せず、JRA-VANへ切り替える。
3. **Coverage gate:** 小規模sampleで年×場×fieldの欠損、PDF layout、例外、同着、取消、非完走、ID解決率を監査する。
4. **PIT gate:** 長期PDFはPIT-C、今後の観測snapshotはPIT-Aとしてmanifestと評価を分離する。
5. **Odds gate:** final oddsはoracle診断専用。実行可能market評価はcutoff付きprospective dataだけとする。
6. **Supplement gate:** JMAは出典・加工表示と品質flagを保持する。NAR/JBIS等は別許諾・別feature groupとする。
7. **Publication gate:** raw data、復元可能派生物、予測、model、集計報告を分け、公開可否をsourceごとに確認する。

## 10. 具体的な回答

1. **無料MVPに最も適した主データ源:** 技術・項目面ではJRA公式全レース成績PDF（2002+）。ただし利用許諾確認前は「条件付き候補」であり、明確なopen datasetではない。netkeibaより一次性・結果確定性・PDFのまとまりで優先する。
2. **推奨構成:** JRA結果PDF + 許諾後のJRA prospective出馬表/当日snapshot + JMA気象CSV。NAR地方歴、JBIS/JRHA血統・セールは許諾後の補完。民間サイトは目視照合。
3. **無料feature:** race context、全過去走成績、時計・着差・上がり・通過・lap、過去馬体重、関係者PIT集計、form/workload、適性、相手強度、最終市場診断、JMA気象。
4. **困難feature:** 構造化調教、長期の締切前odds、歴史的pre-race版・変更履歴、安定ID/schema、深い時点血統、paddock/comment/馬具等。
5. **historical odds:** final oddsによる事後診断までは可能。歴史的T-10 oddsによる実行可能backtestは不可。今後の許諾済みsnapshot蓄積が必要。
6. **scraping risk:** 規約・権利・アクセス制御・負荷と、HTML変更・訂正欠落・ID誤結合・PIT欠落の両方が大きい。robots不在/許可を利用許諾としない。
7. **JRA-VANの増分:** 1986+の正規schema/ID、速報・変更・調教・時系列/現行odds、正規interface、更新処理。長期transaction-time/oddsと公開許諾はなお別課題。

## 11. 一次・公式資料

すべて **accessed 2026-08-30**。

- 日本中央競馬会, 「レース成績データ」: https://www.jra.go.jp/datafile/seiseki/report/index.html
- 日本中央競馬会, 「レース成績の見方」: https://www.jra.go.jp/datafile/seiseki/report/mikata.html
- 日本中央競馬会, 「過去のレース成績は、どこに掲載されていますか？」: https://www.jra.go.jp/faq/pop02/1_6.html
- 日本中央競馬会, 「競馬場別出馬表・8週間掲載」: https://www.jra.go.jp/keiba/rpdf/
- 日本中央競馬会, 「ご利用に際して」: https://www.jra.go.jp/use/
- 日本中央競馬会, `robots.txt`: https://www.jra.go.jp/robots.txt
- 文化庁, 「デジタル化・ネットワーク化の進展に対応した柔軟な権利制限規定（著作権法第30条の4等）」: https://www.bunka.go.jp/seisaku/chosakuken/hokaisei/h30_hokaisei/
- 文化庁, 『AIと著作権に関する考え方について』, 2024-03-15: https://www.bunka.go.jp/seisaku/bunkashingikai/chosakuken/hoseido/r05_07/pdf/94021801_03.pdf
- 株式会社ネットドリーマーズ, 「netkeiba.com 利用規約」, revised 2022-04-01: https://www.netkeiba.com/info/kiyaku.html?rf=footer
- netkeibaヘルプ, 「データベースの閲覧ができない・通信制限がかかった（スクレイピングについて）」: https://support.keiba.netkeiba.com/hc/ja/articles/39720493823129-%E3%83%87%E3%83%BC%E3%82%BF%E3%83%99%E3%83%BC%E3%82%B9%E3%81%AE%E9%96%B2%E8%A6%A7%E3%81%8C%E3%81%A7%E3%81%8D%E3%81%AA%E3%81%84-%E9%80%9A%E4%BF%A1%E5%88%B6%E9%99%90%E3%81%8C%E3%81%8B%E3%81%8B%E3%81%A3%E3%81%9F-%E3%82%B9%E3%82%AF%E3%83%AC%E3%82%A4%E3%83%94%E3%83%B3%E3%82%B0%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6
- 気象庁, 「気象庁ホームページについて」: https://www.jma.go.jp/jma/kishou/info/coment.html
- 気象庁, 「過去の気象データ・ダウンロード」: https://www.data.jma.go.jp/risk/obsdl/
- JRAシステムサービス株式会社, 「データの詳細仕様」: https://jra-van.jp/dlb/ddata.html
- JRAシステムサービス株式会社, 『JVData仕様書』Ver.4.9.0.1, 2024-08-07: https://jra-van.jp/dlb/sdv/sdk/JV-Data4901.pdf
- JRAシステムサービス株式会社, 「JRA-VAN利用規約」: https://jra-van.jp/info/rule.html
- JRA-VAN Data Lab.開発者コミュニティ, 「AIからJV-LINKへのダイレクトアクセスの可否」, official answer 2026-08-07: https://developer.jra-van.jp/t/topic/964
