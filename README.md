# horse_pred_codex2

JRA中央競馬を主対象とする、競馬予測・馬券購入判断支援システムの研究開発リポジトリです。

> **現在の段階:** 時計・着差路線のPV-01～PV-06とgraded LambdaRank GR-001を完了。PV-06のraw着差token補間は2022でinconclusive、GR-001のupper-half教師はreject。PV-01 Binary 254特徴と従来top-three LambdaRank教師を維持する。
> **最終更新:** 2026-08-31 (JST)

## 1. プロジェクトの目的

最終目的は、単なる的中率ではなく、**長期的な馬券回収率および利益を高める意思決定支援ツール**を構築することです。

- 初期対象はJRA中央競馬とする。
- 地方競馬は、将来の対象拡張または補助データとして利用する可能性を残す。
- 最終的な馬券購入判断と購入操作は人間が行う。
- 予測モデル、購入判断アルゴリズム、自動予測処理、結果確認用Webアプリを段階的に開発する。
- 「安定して利益を得られる状態」は達成を保証できる前提ではなく、再現可能なバックテストと将来データによって検証する研究目標とする。

## 2. 基本設計

システムを次の層に分離します。

```text
競馬データ収集
    ↓
Point-in-time特徴量生成
    ↓
競走結果予測モデル
    ↓
勝率・順位スコア等の推定
    ↓
当該時点のオッズとの比較
    ↓
馬券購入判断
    ↓
予測・根拠・期待値等をWeb UIで提示
    ↓
人間が最終判断
```

特に、**競走結果の予測**と**馬券購入判断**を別の問題として扱います。

### 2.1 最終目的と予測タスクの関係

高い的中率だけでは利益は保証されません。収益化の中心課題は、モデルが推定した確率と市場価格であるオッズの間にある乖離を検出することです。

単勝について最小構成では、

\[
\mathrm{snapshot\ EV\ proxy}_i = \hat{P}_i(\mathrm{win}) \times \mathrm{displayed\ odds}_{i,t}
\]

を初期の購入判断指標とします。JRAの単勝はpari-mutuelで、時点表示オッズは購入時に固定されないため、これは真の確定EVではありません。選択は時点付き表示オッズ、実現損益は公式払戻で評価します。

ただし、初期段階では回収率の直接最適化を急がず、まず予測モデルそのものを改善します。

## 3. オッズの扱い

### 初期方針

- **オッズは予測モデルの入力に使用しない。**
- オッズは、予測後の市場比較、購入判断、バックテストに使用する。
- これにより、モデルが人気情報を模倣しているだけなのか、競走情報から独立した予測能力を得ているのかを切り分ける。

### 将来の比較実験

市場オッズには、取得できていない情報が集約されている可能性があります。このため、オッズを入力するモデルを永久に除外するわけではありません。

将来は少なくとも次を比較可能にします。

1. オッズ非依存の競走結果予測モデル
2. オッズを特徴量に含む予測モデル
3. 市場オッズのみから得るベースライン

予測時点は、オッズだけでなく馬体重、馬場状態、天候、出走取消、騎手変更等の利用可否にも影響します。具体的な予測時点はデータ調査後に確定します。

## 4. 競馬結果に対するモデル仮説

競馬を単純な「勝つ・負ける」の独立二値分類としてだけ捉えません。

各馬には固定的で絶対的な序列があるのではなく、馬の基礎能力、今回条件との適合、状態、対戦相手との関係等によって、レースごとの潜在的なパフォーマンスが決まると仮定します。

馬 \(i\)、レース \(r\) について概念的には、

\[
s_{i,r} = f(\text{能力, 適性, 状態, レース条件, 他馬との関係, ...})
\]

\[
z_{i,r} = s_{i,r} + \epsilon_{i,r}
\]

とし、確率的に変動する実現パフォーマンス \(z\) の順序から実際の着順が生成されると考えます。

この考え方から、観測された一度の着順は「絶対的な正解」ではなく、潜在強度から確率的に生成された一観測として扱います。

### 二値分類だけでは不足する理由

- 2着と18着が同じ負例になる。
- 3着と4着の間だけに本質的な不連続があるとは限らない。
- 低格付けレースの1着より、高水準レースの下位着順の方が能力を示す場合がある。
- 着順の数値をそのまま回帰すると、着順差を等間隔として扱う問題がある。

したがって、勝率推定とレース内ランキングを併用し、将来的には順位生成を確率モデルとして扱う方向を取ります。

## 5. モデル開発方針

### 5.1 初期フェーズ: 非DNNベースライン

初期はTransformer等のDNNを主軸にしません。データ規模が十分にスケールするか未確認であり、構造化データではLightGBMが強いベースラインになる可能性が高いためです。

同一のデータ分割・特徴量を用いて、少なくとも次の2モデルを実装・比較します。

| モデル | 主目的 | 長所 | 注意点 |
|---|---|---|---|
| LightGBM Binary Classification | \(P(\mathrm{win})\) の直接推定 | 確率評価と期待値計算に直結する | 1着以外の順位情報を捨てる |
| LightGBM LambdaRank | レース内の相対順位・強さスコアの推定 | 全体の順位関係を学習しやすい | 出力スコアは勝率ではない |

両者を競合案ではなく、異なる性質を確認するための初期ベースラインとして扱います。

### 5.2 次段階: 確率的ランキング

LightGBMベースラインの後に、Plackett–Luce、Bradley–Terry、Thurstone系等を含む確率的ランキングモデルを調査・検討します。

目標は、潜在強度から次のような確率を一貫して導出できる状態です。

```text
race-specific latent strength
    ↓
probabilistic ranking
    ↓
P(rank = 1), P(rank = 2), ...
    ↓
単勝・複勝・将来の組合せ馬券の確率
```

具体的な手法は既存研究調査後に決定します。

### 5.3 将来仮説: モジュール型end-to-endモデル

将来的なDNNとして、次の意味的な分解を持つ構造を候補とします。

```text
各過去レースの入力
    ↓
Race-value encoder
    ↓
過去各走の表現
    ↓
Intra-horse encoder / aggregator
    ↓
各馬の能力・適性・状態表現
    ↓
Inter-horse interaction
(Cross Attention等)
    ↓
今回の出走構成を反映した各馬表現
    ↓
Probabilistic ranking head
```

| モジュール | 想定する役割 |
|---|---|
| Race-value encoder | その過去レースでどの程度価値のある走りをしたかを表現する |
| Intra-horse encoder | 一頭の履歴から能力、適性、状態、成長・衰退等を集約する |
| Inter-horse module | 今回出走する馬同士の比較と相互作用を扱う |
| Ranking head | レース固有の潜在強度と順位確率を出力する |

これは**将来のモデル仮説**であり、MVPの実装対象ではありません。

また、モジュールに意味のある名称を付けても、中間表現が人間の想定どおりの意味を自動的に持つとは限りません。必要に応じて補助損失や中間評価を設計します。

## 6. 特徴量設計の原則

### 6.1 個体IDを直接使用しない

初期モデルでは、馬ID、騎手ID等を直接カテゴリ特徴量として入力しません。

理由は次の2点です。

1. 新規個体に対応できないcold-start問題を避けるため
2. 個体ごとのデータ量が少なく、ID記憶に依存した学習になる危険を減らすため

ただし個体の履歴情報自体は使用します。

例:

```text
騎手IDそのもの: 使用しない

予測時点までの
- 直近期間の勝率・複勝率
- 芝・ダート別成績
- 距離帯別成績
- 競馬場別成績
- 最近N騎乗の状態量
等: 使用候補
```

同様に、馬についてもIDではなく、予測時点までの履歴から算出した能力・適性・状態量として表現します。

### 6.2 過去履歴を早期に少数走へ限定しない

「直近3走だけ」等に最初から固定せず、取得可能な履歴を保持します。LightGBMには可変長系列を直接入力しにくいため、複数の集約特徴量を生成します。

候補:

- 直近1、3、5、10走
- 全期間
- 経過時間による減衰集約
- 芝・ダート別
- 距離帯別
- 競馬場・コース別
- 馬場状態別
- 最近の傾向、最大値、分散、安定性

どの期間が有効かを単一の人手規則に固定せず、複数表現からモデルに選択させます。

### 6.3 理想的な入力情報

データ取得可能性は調査フェーズで確認します。現時点で予測に利用したい情報は次のとおりです。

#### 馬・過去レース

- 過去レースの格: G1、G2、G3、Listed、Open、条件戦等
- 着順、着差、走破タイム、タイム指数候補
- ラップ、上がり、上がり順位、通過順位
- 距離、芝・ダート、競馬場、コース形状
- 馬場状態、天候
- 枠、斤量、頭数
- 馬体重、馬体重増減
- 休養期間、連戦状況、直近期間の累積負荷
- 脚質・展開に関係する情報
- 年齢、性別、血統等の属性

#### 対戦相手・レース水準

単純な着順よりも、**どの水準のレースで、どの程度の相手と走ったか**を重視します。

候補:

- 出走馬の事前能力評価の平均・上位値・分布
- 上位馬・近接馬との着差
- レース格と出走メンバー水準の併用
- Elo、TrueSkill等を参考にしたpoint-in-time rating
- 芝・ダート、距離帯等に分けた複数rating
- 同レース馬間の相対特徴

予測時点より後の成績を過去特徴として参照しないよう、未来情報リークを厳密に防止します。「同走馬が後にG1を勝った」等は、当時の予測では利用できません。

#### 疲労・状態

- 前走からの日数
- 直近30、60、90日等の出走回数
- 直近期間の走行距離合計
- 連闘・中N週等の間隔表現
- 長距離戦後等の負荷候補
- 輸送等、取得できれば有用な情報

#### 今回のレース条件

- 競馬場、コース、距離
- 芝・ダート
- 馬場状態、天候
- 枠順、斤量
- 出走頭数
- 騎手、調教師の履歴統計
- 馬体重・調教等の直前情報
- 他馬の能力・適性・脚質構成

### 6.4 過去1走の価値を表現するモジュール

過去レースの価値を、人手で一つの固定スコアに決め切らない方針です。

初期は次の複数特徴を明示的に作り、LightGBMに組み合わせを学習させます。

- レース格
- 対戦相手の事前評価
- 自身の相対順位
- 着差、タイム、ラップ等
- 当時の距離、馬場、斤量、枠等

その後、過去1走の情報からperformance valueまたはlatent featureを生成する小規模学習モジュールを追加する案を検討します。

```text
過去1走の情報
    ↓
Performance encoder
    ↓
value / latent features
    ↓
履歴集約
    ↓
主予測モデル
```

教師信号の候補は、次走または将来数走のパフォーマンス予測等です。ただし自己参照やリークを避けられる定義が必要であり、MVP必須機能とはしません。

## 7. 評価設計

モデル単体と、購入判断を含むシステム全体を分けて評価します。

### 7.1 データ分割

- ランダム分割ではなく、原則として時系列分割を使用する。
- 同一レースが複数splitに混在しないようrace単位で分割する。
- 特徴量は各予測時点で利用可能だった情報のみから生成する。
- `train → development backtest → untouched final backtest` を分ける。
- 2025年をhold-outにする案は例示であり、実際の期間は取得可能なデータ範囲を確認後に決定する。
- final期間の結果を見て特徴量、モデル、購入閾値を調整しない。

### 7.2 モデル単体評価

| 評価軸 | 指標候補 |
|---|---|
| Ranking | NDCG、Spearman相関、Top-1 accuracy、Top-k recall等 |
| Probability | Log Loss、Brier Score |
| Calibration | Reliability diagram、ECE、予測確率帯ごとの実勝率 |
| 条件別 | オッズ帯、人気帯、レース格、距離、芝・ダート、頭数等 |
| Market comparison | モデル確率と市場暗黙確率の予測性能比較 |

NDCGのrelevance定義等、競馬に適した具体的な算出方法は調査後に確定します。

### 7.3 購入判断込み評価

- 回収率
- 総購入額、総払戻額、総利益
- 的中率
- 購入レース数、購入馬券数
- 最大ドローダウン
- 利益の分散・時系列安定性
- オッズ帯、期待値帯、レース条件別の成績

単一期間の回収率だけを見てモデルを採用せず、予測指標、校正、購入件数、リスク等を併記します。

## 8. 初期の購入判断アルゴリズム

初期フェーズでは購入判断を最小限に固定し、予測モデル改善の影響を観測しやすくします。

| 項目 | 初期方針 |
|---|---|
| 券種 | 単勝を主対象とする |
| 購入条件 | 予測勝率 × オッズが固定閾値を超えること |
| 購入額 | 一律固定額 |
| 複数閾値 | `EV > 1.0`, `1.1`, `1.2` 等を診断用に固定評価可能とする |
| 資金配分最適化 | 初期は行わない |
| Kelly基準 | 後回し |
| 複勝・組合せ馬券 | 後続フェーズで検討する |

初期に回収率100%以上を必須条件としません。固定ルール下で、モデル改善に伴って予測性能と回収率がどう変化するかを追跡します。

購入ルールを頻繁に最適化すると、予測モデルの改善と戦略の過適合を区別できなくなるため、初期は固定します。

## 9. 実験運用

### 9.1 原則

- **1 experiment = 1 commit** を原則とする。
- 各commitは、その実験を再現できるコード・設定状態を固定する。
- 原則として1実験で1つの仮説、または解釈可能な1つのfeature groupを変更する。
- 同時に大量の変更を入れない。
- データsplit、評価指標、購入ルールを都合よく変更しない。
- feature ablationを標準的に実施できる構造にする。

### 9.2 実行方式

MVPでは、1コマンドで次を実行できます。

```text
特徴量生成
    ↓
学習
    ↓
予測
    ↓
モデル単体評価
    ↓
final-odds oracle診断（購入選択・ROIなし）
    ↓
結果・metadata保存
    ↓
README実験一覧更新
```

```bash
uv sync
uv run horse-pred run-mvp \
  --raw-path /path/to/race_results_merged.csv \
  --output artifacts/mvp_baseline

# sealed finalではない2025 retrospectiveを今回だけ明示的に含める場合
uv run horse-pred run-mvp \
  --raw-path /path/to/race_results_merged.csv \
  --output artifacts/mvp_baseline_with_2025 \
  --include-retrospective-test
```

### 9.3 保存対象

現在の構成:

```text
configs/
  exp_001_binary.json
  exp_002_lambdarank.json

experiments/
  baseline_validation_20260830/
    data_health_summary.json
    uncertainty_summary.json
    ablation_summary.json
    diagnostics_summary.json
    improvement_summary.json

artifacts/                 # Git対象外
  mvp_baseline.../
    metrics.json
    run_meta_*.json
    predictions.csv.gz
    final_market_oracle.csv.gz
    feature_importance.csv
    models/

README.md
```

`run_meta.json` 等には最低限次を保存します。

- experiment ID
- git commit hash
- 実行日時
- config
- 使用データversion
- seed
- モデル種別
- feature set
- コード・依存関係の再現に必要な情報

大容量のモデル・runner予測・market artifactは`artifacts/`へ保存してGit対象外とし、集約metricとレポートだけを追跡します。

### 9.4 実験一覧

各experiment directory内の機械可読ファイルをsource of truthとし、READMEの実験一覧表を自動更新します。READMEを手作業の記録台帳にはしません。

| Exp | Model | Features | 2024 Log Loss ↓ | 2024 Brier ↓ | 2024 NDCG@3 ↑ | 2024 Top-1 ↑ | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| corrected baseline | Uniform | — | 2.5949 | .9222 | .1859 | .0793 | non-learning baseline |
| corrected baseline | History rate | — | 2.5453 | .9111 | .2546 | .1242 | PIT smoothed history baseline |
| corrected baseline | LGBM Binary + T | 268 | 2.0898 | .8312 | .4873 | .2814 | History-rateへの改善はblock bootstrapで安定 |
| corrected baseline | LGBM LambdaRank + T | 268 | 2.0892 | .8301 | .4864 | .2804 | Binaryとの差は未解決 |
| `abl_006_drop_field_relative` | LGBM Binary + T | 253 | 2.0855 | **.82985** | .4913 | **.2912** | lean control; Binary incumbentはPV-01 |
| `abl_006_drop_field_relative` | LGBM LambdaRank + T | 253 | **2.0847** | .82995 | **.4924** | .2881 | conservative Rank incumbent、family差は未解決 |
| `imp_002_surface_conditioned_elo` | LGBM LambdaRank + T | 271 | 2.0870 | .82963 | .4900 | .2832 | corrected baseline比NDCG `+.00364 [+.00005,+.00714]` |
| `imp_004_lean_surface_conditioned_rating` | LGBM Binary + T | 256 | 2.0874 | .83106 | .4942 | .2902 | NDCG pointは上昇したがBrier guardrail違反、reject |
| `imp_005_expected_actual_race_value` | LGBM Binary + T | 254 | 2.0872 | .83082 | .4897 | .2879 | 全point悪化、区間跨ぎでinconclusive。採用せず |
| `imp_005_expected_actual_race_value` | LGBM LambdaRank + T | 254 | 2.0876 | .83044 | .4885 | .2852 | NDCG/Log Loss guardrail違反、reject |
| `r6_frozen_rating_module` | LGBM Binary + T | 258 | 2.0851 | .83105 | .4933 | .2896 | NDCG point改善もBrier guardrail違反、reject |
| `r6_frozen_rating_module` | LGBM LambdaRank + T | 258 | 2.0891 | .83080 | .4881 | .2848 | NDCG差 `−.00430 [-.00808,-.00047]`、reject |
| `pv_001_candidate_signed_time_gap` | LGBM Binary + T | 254 | 2.0787 | **.82767** | **.4976** | **.2965** | LL `+.00685 [+.00050,+.01312]`、NDCG `+.00631 [+.00145,+.01117]`、Binary accept |
| `pv_001_candidate_signed_time_gap` | LGBM LambdaRank + T | 254 | **2.0778** | .82810 | .4941 | .2906 | 全point改善だがprimary区間は0跨ぎ、inconclusive |
| `pv_004_margin_rating_score` | LGBM Binary + T | 255 | 2.0788 | .82811 | .4976 | .2940 | PV-01比ほぼ不変、追加価値inconclusive |
| `pv_004_margin_rating_score` | LGBM LambdaRank + T | 255 | 2.0773 | .82848 | .4962 | .2942 | PV-01比NDCG +.00208だが区間跨ぎ、未採用 |
| `pv_005_margin_rating_delta` | LGBM Binary + T | 255 | 2.0781 | .82724 | .4963 | .2924 | probability point改善もprimary区間跨ぎ、未採用 |
| `pv_005_margin_rating_delta` | LGBM LambdaRank + T | 255 | **2.0761** | **.82641** | .4931 | .2879 | probability point bestだがPV-01比区間跨ぎ、未採用 |

[baseline機械可読summary](experiments/baseline_validation_20260830/improvement_summary.json)、[rating/race-value追加実験](experiments/rating_race_value_20260830/)、[PV-01～PV-06 summaries](experiments/race_content_20260831/)、[GR-001 summary](experiments/graded_rank_20260831/summary.json)、[統合結論](docs/experiments/baseline_validation_conclusions_20260830.md)を参照してください。GR-001は2014～2019 fit、2020 early stopping、2021 calibration、2022 gateのため、上表の2024実験とは直接比較しません。旧Task 16 reportは障害混入修正前のためsupersededである。final oddsは事後oracle専用で、実行可能ROIは評価していません。

詳細な実験レポートを毎回作ることは必須としません。混合した結果を単純に「改善」と要約せず、改善・悪化した指標を事実として記録します。

## 10. Codexとの作業分担

Codexには、実装だけでなく実験設計支援も担当させます。

| モード | 人間から与える内容 | Codexの役割 |
|---|---|---|
| 具体実験モード | 特徴量、モデル変更、比較条件等 | 実装、実験、評価、記録 |
| 仮説検証モード | 「疲労特徴は有効か」等の抽象仮説 | 検証可能なfeature group・実験条件への具体化、実装、評価 |

基本運用:

1. 人間とCodexで仮説と実験範囲を相談する。
2. Codexが解釈可能な実験単位に具体化する。
3. 実装、学習、評価、backtest、記録を行う。
4. 各指標の変化を事実ベースで記録する。
5. 最終的な採否判断は人間が行う。

Codexは、結果が混在している場合に独断で「成功」「改善」と断定しません。

## 11. Webアプリの位置付け

Webアプリは最終システムに必要ですが、初期の主要課題ではありません。

将来の表示候補:

- 当日レースの予測順位
- 各馬の予測勝率・校正情報
- 当該時点のオッズ
- 期待値と固定ルール上の購入候補
- 予測根拠・主要特徴
- 過去の予測成績、回収率、ドローダウン
- 条件別性能

実際の馬券購入操作は自動化せず、人間が最終判断します。

## 12. 調査フェーズ

実装前に、少なくとも次を調査します。

### 12.1 データソース

JRA、JRA-VAN、netkeiba、JBIS、その他候補について、以下を確認します。

- 利用可能なデータ項目
- 過去データ期間
- 更新頻度と直前情報の取得時点
- オッズ履歴の有無
- レース、馬、騎手、調教師、血統、ラップ、調教、馬体重等の範囲
- API、ダウンロード、その他の取得方法
- 費用
- 利用規約、再配布条件
- 自動取得・スクレイピングの可否
- 欠損、訂正、識別子、データ品質
- point-in-time再現性

取得可能性を確認してから、最終的なデータ仕様を確定します。

### 12.2 既存研究・実装

- horse racing prediction
- learning-to-rank
- probabilistic ranking
- LightGBMによる競馬予測
- opponent-adjusted rating
- Elo / TrueSkill系手法
- probability calibration
- market efficiency
- expected-value betting
- bankroll management
- 時系列評価とバックテストのリーク防止

### 12.3 実装前に確定する事項

- 使用データソースと取得方式
- 利用可能な期間、train/dev/finalの具体的範囲
- 予測時点
- 学習ターゲットとランキングloss
- 市場暗黙確率の計算方法
- probability calibrationの方式
- 過去レース価値モジュールの教師信号
- ratingの種類と条件別分割
- 複勝確率および組合せ馬券への拡張方法
- artifact保存方式
- Webアプリの技術構成

### 12.4 調査成果

2026-08-30に `docs/research_plan.md` のworkstream A～Hと、無料データ優先MVPの追加調査を完了し、個別結果と横断的な結論を `docs/research/` に保存しました。

- [調査結論と実装前仕様](docs/research/research_conclusions.md)
- [既存ローカルデータセットの監査と採用判断](docs/research/local_existing_dataset.md)
- [開発タスクリスト](docs/development_plan.md)
- [無料データ優先MVPの統合判断](docs/research/free_data_mvp.md)
- [無料データ源の横断的批判レビュー](docs/research/free_data_synthesis_review.md)
- [netkeiba重点調査](docs/research/free_data_netkeiba.md)
- [国内無料データ源比較](docs/research/free_data_japan_sources.md)
- [公開dataset・API調査](docs/research/free_data_public_datasets.md)
- [データソース](docs/research/data_sources.md)
- [利用可能な特徴量](docs/research/available_features.md)
- [Point-in-Time設計](docs/research/point_in_time.md)
- [競馬予測手法](docs/research/horse_racing_prediction.md)
- [ranking・rating・race strength](docs/research/ranking_rating_and_strength.md)
- [form・fatigue・適性・interaction](docs/research/form_fatigue_and_interactions.md)
- [確率推定・calibration](docs/research/probability_and_calibration.md)
- [JRA市場・market probability](docs/research/betting_market.md)
- [backtest・leakage・評価](docs/research/backtesting_and_leakage.md)

無料の長期no-odds予測trackでは、ユーザーが本非公開・私的プロジェクト内での利用を承認した既存ローカルraw（2013～2025年、629,967出走、44,761レース）を主データとして採用します。既存`features.csv`にはリークがあるため流用せず、rawからPIT準拠で再生成します。JRA公式結果はcoverage・grade・lap・払戻等の照合・補完に使い、締切前oddsを含む実行可能市場trackはprospective snapshotまたは有料JRA-VAN/JV-Linkで分離します。この承認は新規scrapingや外部公開には一般化しません。

raw・中間・加工済みデータと取得cacheはGit管理対象外です。Gitにはsource manifest、fingerprint、schema、構築設定、品質集計、コード、テストのみを保存します。

## 13. 開発フェーズ案

```text
Phase 0: 要求・設計方針のすり合わせ             完了
Phase 1: データソース・既存手法の調査           完了
Phase 2: 既存rawのcoverage/PIT gateと仕様確定    完了
Phase 3: point-in-time特徴量基盤                 実装・検証済み
Phase 4: LightGBM Binary / LambdaRank baseline   完了
Phase 5: 特徴量・rating・calibration改善         進行中（PV-01～PV-06、GR-001完了）
Phase 6: 確率的ランキングおよび高度なモデル
Phase 7: 購入戦略・リスク管理の改善
Phase 8: 自動予測処理・Web UI
```

## 14. 現時点の決定事項と未決事項

| 論点 | 状態 | 内容 |
|---|---|---|
| 最終目的 | 決定 | 的中率ではなく長期的な回収率・利益を重視 |
| 初期対象 | 決定 | JRA中央競馬 |
| 最終購入 | 決定 | 人間が判断・実行 |
| 予測と購入判断 | 決定 | 分離する |
| 初期のオッズ利用 | 決定 | 予測入力から除外し、後段比較に使用 |
| 初期モデル | 決定 | LightGBM BinaryとLambdaRankを比較 |
| DNN | 決定 | 初期は使わず、将来仮説として保持 |
| 個体ID | 決定 | 馬ID・騎手ID等を直接入力しない |
| 履歴 | 決定 | 少数走に早期固定せず複数窓・条件別に集約 |
| 初期券種 | 決定 | 単勝中心 |
| 初期購入戦略 | 決定 | 固定額・固定snapshot EV proxy閾値。公式払戻で精算 |
| 初期ROI目標 | 決定 | 100%超を必須としない |
| 評価 | 決定 | ranking、確率、校正、条件別、市場比較、backtestを併用 |
| 実験運用 | 決定 | 原則1 experiment = 1 commit |
| 実験台帳 | 決定 | 機械可読結果からREADME表を自動更新 |
| データソース | 決定 | 承認済み既存raw 2013～2025を主系統とし、JRA公式結果でcoverage・不足項目を照合・補完。新規取得は別gate |
| 具体的な予測時点 | MVP決定 | 過去結果rawによる保守的PIT-C前日相当。同日の全raceは一括emit後に更新。当日締切前版は別track |
| split期間 | 開発更新 | 既存baselineは2014～21 train、2022 validation、2023 calibration、2024 development。新仮説はrolling-origin複数年でscreenし、2024はmilestoneのみ、2025は反復選択に使わず、2026+ prospective finalを維持 |
| rating方式 | 開発更新 | K48/scale200のmargin-aware actual (`tau=.125`) は前年温度校正で2019～22全改善、2024校正LL 2.3957（ordinal 2.4015）。standaloneは置換候補。LightGBMへのabsolute/delta 1列追加はPV-01比inconclusiveで未採用 |
| 過去走race-content | 開発採用候補 | 90日減衰signed時計差1列はBinaryで採用基準通過、LambdaRankはinconclusive。2026+ prospective確認が必要 |
| LambdaRank教師 | 開発維持 | 従来の`1着=3, 2着=2, 3着=1, その他=0`を維持。2・3着を統合して上位半数へ教師を広げたGR-001は2022でLog Loss/Brierを有意に悪化させreject |
| probabilistic ranking | 調査推奨・延期 | Plackett–Luceを最初の高度baseline候補とするが、順位別biasを検証してから採否判断 |
| artifact管理 | MVP決定 | config、git/data fingerprint、aggregate metricsを追跡し、raw・model・runner予測はGit対象外 |
| Web技術 | 未決 | 後続フェーズで決定 |

## 15. 次の作業

PV-06とGR-001まで完了しました。PV-06は2014～2021のraw着差tokenから0.1秒同時計だけを決定的に補間し、2022で全point指標が改善しましたが、primary Log Loss差 `+.00003 [-.00022,+.00028]` のためinconclusiveで、2024は開いていません。GR-001は2着と3着を統合し、4着から上位半数へcoarse教師を加えましたが、2022 Log Loss差 `-.01539 [-.02068,-.01023]`、Brier差 `-.00201 [-.00349,-.00058]` でrejectです。現bestはBinary PV-01 254特徴、保守的LambdaRankはlean 253特徴・従来top-three教師のままです。

現在のliving work queueは[no-odds予測モデル研究の優先順位](docs/model_research_priorities.md)です。S0/S1を一つのGoalとし、DOC-SYNC、EVAL-ROLL、LIVE-DATA、完了済みPV-06、OPP-RECENT、SEC-3F、HPO-01、ENS-01を閉じます。新規model仮説はEVAL-ROLLを先に固定してからrolling foldsでscreenし、LIVE-DATAはsource gateの下で並行開始します。
