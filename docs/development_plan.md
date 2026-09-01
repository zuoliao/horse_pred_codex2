# 開発タスクリスト

**作成日:** 2026-08-30 (JST)  
**更新日:** 2026-09-01 (JST)
**前提:** Phase 5A EDAとS1 Two-axis past-race valueまで完了。人間選択によりS3 Condition-adjusted performance targetをcurrentとし、S2はqueuedのまま維持する。
**対象:** JRA平地競走、no-odds予測を先行し、購入判断を別層にする。

## 進行原則

- raw・中間・加工済みdatasetはGit管理せず、manifest、fingerprint、schema、設定、品質集計、コード、テストを管理する。
- canonical sourceは承認済みの`race_results_merged.csv`とし、既存`features.csv`は学習入力に使わない。
- 特徴量はrace単位で履歴stateを更新し、target raceおよびfuture dataを参照しない。
- オッズ、人気、当該レース結果はprimary prediction modelから分離する。
- 時間分割をmetric確認前に固定し、final holdoutで反復調整しない。
- モデル実験は原則として、1 experiment = 1 hypothesis = 1 reproducible commitとする。

## タスクリスト

| 順序 | ID | タスク | 依存 | 成果物・完了条件 |
|---:|---|---|---|---|
| 1 | GOV-01 | データ配置・manifest契約を確定 | なし | 外部pathを設定で注入し、SHA-256、size、row count、schema、source、利用範囲をmanifestへ保存。データ実体がGit対象外であることをtest/checkで確認 |
| 2 | DATA-01 | raw schemaと型・コードを固定 | GOV-01 | 26 raw列の型、nullable、単位、列意味、race/horse/jockey key、surface/venue codeを文書化。未知値を黙ってdropしない |
| 3 | DATA-02 | race coverageを公式結果と監査 | DATA-01 | 既知の不足146レースをrace ID単位で特定。年×場×月×surfaceのcoverage表を生成し、補完・除外・flagの方針を確定 |
| 4 | DATA-03 | outcome・例外規則を確定 | DATA-01 | 同着、降着、失格、競走中止、取消、除外、返還のlabel・field size・学習対象・精算規則をfixture付きで決定 |
| 5 | PIT-01 | 予測時点とcolumn availabilityを固定 | DATA-01 | 各raw/derived列を`history_only`、`T_prevday`、`T_close`、`outcome`、`final_market`へ分類。過去rawはPIT-Cと明示 |
| 6 | SPLIT-01 | 時間分割とholdout方針を事前登録 | DATA-02, PIT-01 | warm-up/train/calibration/development/retrospective testの絶対日付をmanifestへ固定。既利用の2025年を完全未使用finalと呼ばず、2026年以降のprospective final方針を記録 |
| 7 | EXP-01 | experiment artifact仕様を確定 | GOV-01, SPLIT-01 | experiment ID、config、commit hash、data fingerprint、seed、metrics、predictions、importanceを保存するschemaと単一実行commandの仕様を作成 |
| 8 | PIPE-01 | raw loader・正規化層を実装 | DATA-01, DATA-03 | 決定論的にrace/runner/result/final_marketへ分離。文字コード、数値parse、非完走、同着をtestで保証 |
| 9 | PIPE-02 | PIT-safe履歴state engineを実装 | PIPE-01, PIT-01 | race開始前stateから特徴を出し、race全体の結果を一括更新。同一race内リークとfuture-row追加不変性のtestを通過 |
| 10 | FEAT-01 | 最小race/runner contextを実装 | PIPE-02 | venue、surface、distance、class、sex/age、枠・馬番、field size等をfeature group化。利用時点をschemaに記録 |
| 11 | FEAT-02 | horse history・form/workloadを実装 | PIPE-02 | 全履歴を保持し、複数走数窓、14～365日窓、指数減衰、休養日数、累積距離、条件別成績を作成 |
| 12 | FEAT-03 | connection・field-relative特徴を実装 | PIPE-02 | jockey/trainerの過去のみ集計とfield内相対値をrace batch更新で作成。生entity IDはmodel featureから除外 |
| 13 | FEAT-04 | forward-only rating・race strengthを実装 | PIPE-02 | 単純EloまたはBT-style ratingを一方式だけ導入し、相手強度と過去走価値をtarget以前だけで生成 |
| 14 | QA-01 | canonical dataset検証suiteを完成 | FEAT-01～04, SPLIT-01 | forbidden列、timestamp、同一race更新、future append invariance、split境界、duplicate、欠損率、feature coverageのtestを全通過 |
| 15 | BASE-01 | 非学習baselineを評価 | QA-01, EXP-01 | 一様勝率、簡単な履歴勝率等を同一splitで評価し、NDCG、Log Loss、Brier、校正、条件別指標をartifact化 |
| 16 | EXP-001 | LightGBM Binary baseline | BASE-01 | `P(win)`二値分類を固定features・splitで実行。race内確率和、校正、ranking、市場帯別診断を記録 |
| 17 | EXP-002 | LightGBM LambdaRank baseline | BASE-01 | Binaryと同一features・splitでranking objectiveだけを変更し、NDCG/top-k/rank diagnosticsを比較 |
| 18 | PROB-01 | coherent勝率化・calibration比較 | EXP-001, EXP-002 | raw scoreをrace内合計1の勝率へ写像。時間外calibration sliceだけで候補を比較し、Log Loss/Brierで一方式をfreeze |
| 19 | EVAL-01 | 統合評価reportを生成 | PROB-01 | ranking、確率、校正、race class、距離、surface、field size、final odds帯の診断を同じpredictionsから再現可能に生成 |
| 20 | BET-01 | final-odds oracle診断を実装 | EVAL-01 | final oddsを選択には使わず、市場比較・oracle診断だけを検証。selection、精算、ROIは時点付き市場データの別タスクへ分離 |
| 21 | DEC-01 | 最初のbaseline採否レビュー | EXP-001～BET-01 | Binary/LambdaRankの改善・悪化・不確実性を比較し、次の1仮説を人間が選べるdecision reportを作成 |
| 22 | DATA-04 | JRA公式による不足項目補完 | DATA-02, 追加source gate | 欠落race、詳細grade、lap、票数、払戻を独立table/feature groupとして補完。raw上書きはしない |
| 23 | FEAT-05 | JMA気象feature ablation | DEC-01 | 観測所対応・品質flag・時刻意味を保持し、気象feature groupだけを追加した独立experimentを実行 |
| 24 | LIVE-01 | prospective snapshot仕様・収集 | PIT-01, 追加source gate | 出馬表、変更、馬体重、馬場、oddsについて`published_at/ingested_at`付きPIT-Aを保存。rate/cache/diff方針を固定 |
| 25 | LIVE-02 | shadow evaluation | LIVE-01, PROB-01 | 固定cutoff・固定model・固定EV proxyで購入せずに予測を記録し、十分な将来期間で実行可能性を評価 |
| 26 | PAID-01 | JRA-VAN導入判断 | DEC-01, LIVE-01 | 無料構成の欠損と追加価値を定量化し、調教、正規ID、速報、時系列oddsのどれを検証するか一特徴群ずつ決定 |
| 27 | EXP-003 | lean config × surface-conditioned rating | DEC-01 | 253-feature field-relative-dropをcontrolに、芝/ダート別Elo familyだけを追加。2024 paired blockで独立評価し、2025不使用、2026+ prospective方針を維持 |
| 28 | EVAL-ROLL | rolling-origin評価基盤 | PROB-01 | fold内でfit/early stopping/calibrationを完結し、2020～2023等を年別評価。macro、方向一致、block CI、多重比較metadataを保存 |
| 29 | OPP-RECENT | recent opponent-only strength | EVAL-ROLL | 自身を除くpre-race相手ratingの90日半減履歴1列だけをrolling評価 |
| 30 | SEC-3F | race-relative上がり3F履歴 | EVAL-ROLL | race内relative表現を1案固定し、90日半減履歴1列だけをrolling評価 |
| 31 | HPO-01 | LightGBM限定時系列探索 | EVAL-ROLL | Binary/Rankerを別々にrolling foldsだけで限定探索。feature実験と混ぜない |
| 32 | ENS-01 | Binary/Ranker単純ensemble | EVAL-ROLL | 固定50:50 coherent probability ensembleと独立temperatureをrolling評価 |
| 33 | SHIMBA-FILTER-001 | 新馬戦fit-population ablation | EVAL-ROLL | Binaryのgradient fitからだけ新馬戦を除外し、全race評価でnoise仮説を検証 |
| 34 | PACE-01 | horse-level位置取り・脚質履歴 | EVAL-ROLL | 通過順位の少数表現を固定し、直線1000 m等を別扱いしてrolling評価 |
| 35 | PACE-02 | rival-only field pace pressure | PACE-01 | 固定PACE-01から他馬だけの先行圧を1列表現しrolling評価 |
| 36 | PACE-03 | final recorded position history | PACE-01 | 最終通過位置percentileの90日履歴1列をrolling評価 |
| 37 | PACE-04 | normalized position-gain history | PACE-01 | 序盤から最終位置への変化を遷移数補正した90日履歴1列でrolling評価 |
| 38 | SPEED-01 | condition-adjusted speed figure | PACE-01～04 | 期待時計を過去情報だけで推定し、条件補正済み残差の履歴1列をrolling評価 |
| 39 | COND-01 | condition-transition suitability | SPEED-01 | `superseded_by_eda`。広い旧案を閉じ、狭いA1 transition reliabilityへ再定義 |
| 40 | EDA-5A | systematic EDA and problem reformulation | 既存実験の安全なcheckpoint | **完了**。2014～2022の時間再現、共通view、全workstream、三者PASS、仮説registry、最大3候補のroadmapを1 commandで再現。production変更なし |
| 41 | S1-RACE-VALUE-2AXIS | horse performanceとrace-constant field qualityの二軸履歴 | EDA-5A | **completed**。performance supported、field-quality単独 inconclusive、joint supported。production controlは自動変更せず停止 |
| 42 | S2-RACEWISE-CHOICE | supervised race-wise probability objective | S1 decision | `priority_S2 / queued`。S1後も有効だが、S3より後を推奨。人間選択まで実行しない |
| 43 | S3-PERFORMANCE-TARGET | condition-adjusted continuous performance target | S1 decision | `priority_S3 / current`。target formulationだけを変更し、2020～2022 rollingで実行 |
| 44 | A1-TRANSITION-RELIABILITY | condition switch時のstate reliability | S1～S3 review | `still_valid_future_candidate / deferred_by_eda` |
| 45 | A2-CONNECTION-COMPRESSION | 130 connection列の階層的縮約 | S1～S3 review | `still_valid_future_candidate / deferred_by_eda` |
| 46 | A3-LAST3F-RELATIVE | race-relative last3F履歴 | SEC-3F重複監査、S1～S3 review | `still_valid_future_candidate / deferred_by_eda` |

## 2026-08-31実行状態

> 2026-09-01 phase decision: Phase 5Aは完了し、Phase 5BのS1を人間が選択した。比較基準はBinary PV-01 254特徴、LambdaRank lean 253特徴。LambdaRank PV-01 254特徴版はpoint改善・interval inconclusiveのprospective候補に限定する。2024/2025とmarketはS1判断に使わない。

| 範囲 | 状態 | 証拠 |
|---|---|---|
| GOV-01～QA-01（1～14） | 完了 | data/feature/spec実装、PIT・split・例外fixture。障害raceがflat stateを更新しない回帰testを追加 |
| BASE-01～BET-01（15～20） | 完了・corrected | 障害混入修正後、Uniform/History/Binary/LambdaRank、2023校正、2024評価、final-odds oracleを再生成 |
| DEC-01（21） | 完了 | data health、10,000回block bootstrap、7 group ablation、SHAP/permutation、条件別errorを統合 |
| IMP-001～005 | 完了 | compact relative reject、surface EloはRanker rankingのみ支持、field-size calibration reject、lean surface family reject、expected-actual race value未採用 |
| EXP-003相当 / IMP-004 | 完了・reject | lean 253-feature controlへのsurface rating追加は両familyで棄却 |
| Rating R0～R6 | 完了 | K48/scale200 ordinalをfreeze。margin-aware actualは校正後standalone支持、LightGBM統合は未採用 |
| PV-00～PV-06 | 完了 | PV-01 Binary採用。PV-02 raw reject、PV-03 standalone支持、PV-04/05未採用、PV-06 2022 inconclusive |
| GR-001 | 完了・reject | upper-half graded relevanceは2022 proper scoresを有意悪化。従来top-three教師を維持 |
| EVAL-ROLL | 完了 | 2020～2023の4 expanding fold、year macro、方向一致、paired date-block CI、2024/2025 firewall |
| OPP-RECENT | 完了・未採用 | Binary inconclusive、LambdaRank reject |
| SEC-3F | 完了・Rank rolling採択 | Binary inconclusive。LambdaRank NDCG `+.00256 [+.00045,+.00465]` |
| HPO-01 | 完了・変更なし | Binary no-change、Ranker候補は2023 reject |
| ENS-01 | 完了・reject | 固定50:50はBinary比minimum未達、2023 confirmation未開封 |
| LIVE-DATA | groundwork完了・ユーザー保留 | 公式JV-Link source gate、schema、append-only archive実装。JV-LinkはWindowsのみで現環境がMacのため後日に延期 |
| S0/S1 goal | 完了 | 統合判断は`docs/experiments/s_rank_model_research_conclusions_20260831.md` |
| SHIMBA-FILTER-001 | 完了・reject | 新馬戦fit除外はLL/NDCG悪化。新馬戦を学習に維持 |
| PACE-01 | 完了・両family採択 | 序盤位置percentile 90日履歴1列。Binary probability pathとRank ranking/probability pathが通過。2024未開封 |
| PACE-02 | 完了・未採用 | rival-only先行圧1列。両family inconclusive |
| PACE-03 | 完了・未採用 | 最終位置履歴1列。Binary inconclusive、LambdaRank reject |
| PACE-04 | 完了・未採用 | 遷移数補正position gain履歴1列。両family inconclusive |
| SPEED-01 | 完了・両family rolling採択 | prequential条件補正speed履歴1列。2024 Binary supported、Rank directionally consistent |
| COND-01 | superseded_by_eda | broad interaction案を閉じ、A1 transition reliabilityへ狭く再定義 |
| EDA-5A | completed | 全workstream、共通contract、再現CLI、三者review、registry、synthesis、roadmapを完了 |
| S1-RACE-VALUE-2AXIS | completed | 24 fit完了。performance supported、field-quality単独 inconclusive、joint supported。2023～2025/market未使用 |
| S2-RACEWISE-CHOICE | queued / priority_S2 | objective mismatchを調べる独立候補。S3後を推奨し、今回未実行 |
| S3-PERFORMANCE-TARGET | current / priority_S3 | S1で安定したperformance residualを教師へ拡張。preregister後に実行し、S2へ自動移行しない |
| A1 / A2 / A3 | deferred_by_eda | valid future candidates。S1～S3より後 |

## Phase 5B decision tree

| ID | Hypothesis | Status | Dependency | Result | Next action |
|---|---|---|---|---|---|
| S1 | Past-race valueをhorse performanceとrace-constant field qualityへ分離する | `completed` | Phase 5A | Performance `supported`; field quality `inconclusive`; joint `supported`; C3−C1はBinary weak / LambdaRank reject | production controlを自動変更せず、人間レビュー |
| S3 | condition-adjusted performanceをcontinuous targetとして直接教師にする | `current` | S1 performance support | 人間が選択。target-only比較をpreregisterして2020～2022で評価 | 完了後にS2またはperformance深掘りを人間判断 |
| S2 | supervised race-wise probability objectiveを直接最適化する | `queued` | S3 review | 未実行。S1とは異なるobjective仮説 | S3完了後も自動実行せず人間判断 |
| A1 | condition transition時のstate reliabilityを表す | `deferred_by_eda` | S1～S3 review | 未実行 | S1～S3後 |
| A2 | connection 130列を階層的に縮約する | `deferred_by_eda` | S1～S3 review | 未実行 | S1～S3後 |
| A3 | race-relative last3F履歴を非冗長化する | `deferred_by_eda` | SEC-3F重複監査 | 未実行 | S1～S3後 |

## EDA後のlegacy queue再分類

| Task | Lifecycle status | Result / reason | Next action |
|---|---|---|---|
| PV-06 margin seconds mapping | `inconclusive` + `deferred_by_eda` | 2022全point改善だがprimary LL CI跨ぎ。秒scale未同定 | official semantics新証拠まで再開しない |
| margin-aware rating arithmetic | `superseded_by_eda` / branch closed | PV-02～05完了、追加算術はincremental価値未確立 | artifactを保持し、二軸S1へ置換 |
| OPP-RECENT旧runner-relative案 | `superseded_by_eda` | runner-relative opponent meanはrace qualityではなくself rank逆変換を含む | race-constant field qualityのS1へ置換 |
| old field-relative 15列 | `rejected` | retrained dropで両family全point改善 | 復活しない |
| Top3 multitask直行 | `deferred_by_eda` | 3着/4着に自然な断絶なし | S3 standalone evidence後だけ再検討 |
| new-horse fit exclusion | `rejected` | SHIMBA-FILTER-001悪化、EDAも除外を支持せず | 学習母集団に維持 |
| SEC-3F追加branch | `completed`、A3は`still_valid_future_candidate` | Rank rolling支持済み。EDA候補は定義重複の可能性 | S1～S3後に重複監査 |
| transition interaction旧案 | `superseded_by_eda` | broad COND-01を解体 | A1 one-transition reliabilityとして将来検証 |
| connection追加案 | `still_valid_future_candidate` / `deferred_by_eda` | signalは強いが130列が冗長 | A2 compressionとして将来検証 |
| PACE-02 / PACE-03 / PACE-04 | `inconclusive` / `rejected` / `inconclusive`、completed | 保存済みnegative evidence | 再探索しない |
| HPO-01 | `completed` | parameter変更なし | new representationと混ぜない |
| ENS-01 | `rejected` | fixed 50:50 minimum未達 | new component evidenceまで再開しない |

現在のwork queueは[model research priorities](model_research_priorities.md)、実験結果のsource of truthは`experiments/`、統合判断は`docs/experiments/`と`docs/handoff.md`である。完全なmodel・prediction・bootstrap artifactはGit対象外の`artifacts/`に置く。

## 直近のマイルストーン

| マイルストーン | 対象タスク | 到達条件 |
|---|---|---|
| M1: データ契約確定 | GOV-01～SPLIT-01 | データ版、coverage、例外、PIT、splitがmetricを見る前に固定される |
| M2: PIT dataset完成 | EXP-01、PIPE-01～QA-01 | 同一race・future leakage testを含む検証suiteが通る |
| M3: 初期モデル比較 | BASE-01～PROB-01 | BinaryとLambdaRankを同一条件で比較し、coherent probabilityを得る |
| M4: MVP評価完了 | EVAL-01～DEC-01 | 多面的評価とoracle市場診断を再現可能なartifactとして提示する |
| M4.1: baseline validation | DEC-01、IMP-001～003 | data/evaluation health、uncertainty、ablation、model/error診断と限定改善を2024だけで完了する |
| M4.2: rolling selection | EVAL-ROLL、OPP-RECENT、SEC-3F、HPO-01、ENS-01 | 2024/2025をparameter選択に使わず、複数年rolling evidenceでS1を判断する |
| M4.3: structured A-rank signals | PACE-01、支持時のみPACE-02、SPEED-01 | 一仮説ずつrollingでscreenし、重要候補だけ2024 milestoneへ送る |
| M4.4: systematic EDA | EDA-5A | 2022末cutoff、全workstream、PIT/statistics/domain review、machine-readable registry、最大3候補を再現しproduction変更なしで停止 |
| M5: 実行可能市場評価 | LIVE-01～LIVE-02 | 締切前snapshotを用いたprospective shadow期間が蓄積される |

GOV-01～QA-01のdecision gateはmetric確認前に固定済みである。baseline結果を見てこれらを変更する場合は、新しいexperiment IDと将来評価期間を必要とする。
