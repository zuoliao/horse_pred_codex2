# モデル・確率・評価仕様

## 1. 対象と非対象

本仕様は `EXP-01`、`BASE-01`、`EXP-001/002`、`PROB-01`、`EVAL-01`、`BET-01` の共通契約を定める。対象はJRA平地競走のno-odds予測である。モデルへのオッズ入力、final oddsを使った過去のbet選択、Kelly sizing、portfolio最適化、自動投票は対象外とする。

実装は次の二層に分ける。

- `src/horse_pred/modeling.py`: group/split検証、非学習baseline、LightGBM Binary/LambdaRank、確率写像、校正。
- `src/horse_pred/evaluation.py`: ranking、確率、校正、条件別、市場oracle診断。

独立rating研究では`src/horse_pred/rating.py`がforward-only stateとrace確率、`src/horse_pred/rating_study.py`が2018–2021 parameter選択、2022方式検証、2023校正、2024固定評価を担う。2025は処理しない。詳細は[`rating_module_r0_r6.md`](experiments/rating_module_r0_r6.md)。

PV-01は2018–2021のstandalone ranking診断で90日表現を固定し、2022で方向を確認、2023で各LightGBM armをtemperature校正してから2024を一回比較する。既存のBinary/LambdaRank target、評価relevance、calibration、4-date paired bootstrapを変更しない。詳細は[`race_content_time_m0_m4_20260831.md`](experiments/race_content_time_m0_m4_20260831.md)。

## 2. 固定時間分割

| split | 絶対期間 | 用途 | 禁止事項 |
|---|---:|---|---|
| `train` | 2014–2021 | LightGBM parameter fit | 校正・最終評価を混ぜない |
| `model_validation` | 2022 | LightGBM early stopping専用 | calibrator fit、モデル比較の最終値に使わない |
| `calibration` | 2023 | temperature/Platt等のfitと方式選択 | base model fitへ戻さない |
| `development` | 2024 | 比較、診断、仕様選択 | retrospective testの結果を見て再調整しない |
| `retrospective_test` | 2025 | 凍結後の一回評価 | threshold、校正器、featureを再選択しない |

`validate_standard_split_partition` は年とsplitラベル、`validate_race_splits` は一つのraceが複数splitへ分断されていないことを検証する。2013年以前の履歴はwarm-up state専用であり、上表の学習・評価行には含めない。

## 3. 入力・group契約

- rowはrunner、queryはraceである。
- 同一raceの全runnerは連続し、一つのsplitに所属する。
- BinaryとLambdaRankは同じrow集合、feature列、splitを使う。
- feature列に対象raceの結果、人気、final odds、払戻を入れない。
- horse/jockey/trainer IDは結合に使えても初期model featureには入れない。
- 非完走、取消、除外の行採否は上流のoutcome契約で解決してから本APIへ渡す。
- `finish_position=1`が勝者である。同着1着が複数いる場合を保持する。

LambdaRankのgroupは行順に依存するため、非連続raceを暗黙sortせず例外にする。各raceのgroup size合計は入力row数と一致しなければならない。

## 4. BASE-01

### Uniform baseline

各raceのfield sizeを `n_r` として全runnerに `1/n_r` を割り当てる。field sizeが違うraceを同じ定数確率で扱わない。

### History-rate baseline

target以前に計算済みの過去勝数 `w_i` と過去出走数 `s_i` だけを受け取り、race内一様確率を平均とするBeta型の単純平滑化を行い、最後にrace内合計1へ正規化する。履歴値のPIT正しさは呼出側が保証する。

## 5. EXP-001/002

### LightGBM Binary

- objective: `binary`
- target: 1着=1、その他=0。同着勝者は各runnerを1とするmarginal target。
- runner weight: `1 / as_of_field_size`。各raceの学習weight合計を1にする。
- class balancingは使わない。
- 2022 `model_validation` のbinary Log Lossだけをearly stoppingに使う。
- raw出力はrunner binary probabilityであり、race内合計1を保証しない。

### LightGBM LambdaRank

- objective: `lambdarank`
- relevance: 1着=3、2着=2、3着=1、その他=0。同着は同relevance。
- `label_gain=[0,1,3,7]`
- `eval_at=[1,3,5]`
- `lambdarank_truncation_level=6`
- `lambdarank_norm=true`
- 2022 `model_validation` のNDCGだけをearly stoppingに使う。
- raw scoreは確率ではない。

## 6. PROB-01

完成した勝率vectorは各raceで `0 <= p_i <= 1` かつ `sum_i p_i = 1` を満たす。

このMVPで固定して比較する写像は次である。

1. Binary raw probability（runner二値確率。coherentではないため診断専用）。
2. Binary raw probabilityを`epsilon=1e-6`でclipしたlogitに対するrace softmax（`T=1`）。
3. 同じBinary logit scoreへ、2023だけでfitしたtemperatureを適用するrace softmax。
4. LambdaRank scoreのrace softmax（`T=1`）。
5. 2023だけでwinner Log Lossを最小化したtemperatureによるLambdaRank scoreのrace softmax。

Binaryの2と3は`p_i/sum_j(p_j)`ではなく、`softmax(logit(p_i)/T)`である。raw probabilityの単純race内正規化とPlattは低水準APIに残す未採用比較候補とし、このMVPのprimary mappingとは呼ばない。

temperatureは正数とし、`q_i(T)=exp(s_i/T)/sum_j exp(s_j/T)` を用いる。これは1着確率への実用的なPlackett–Luce-like写像候補であり、完全な順位生成モデルが正しいという主張ではない。

同着時の確率評価targetはwinner mass方式とする。m頭の同着勝者へ各 `1/m`、他馬へ0を割り当てる。これは共同同着eventのモデルではなく、raceごとの評価質量を1に保つ規約である。

## 7. EVAL-01

同じ予測artifactから次を再計算できるようにする。

### Ranking

- NDCG@1/@3/@5。主指標はNDCG@3。
- Top-1/Top-3 winner mass。
- 同点scoreは元row順で決定論的に処理する。

### Probability

- race winner Log Loss。winner mass cross-entropyをrace macro平均。
- multiclass Brier。half factorなしでrace macro平均。
- runner binary Log Loss/Brierのrunner microとrace macro。
- Binary raw出力のrace内確率和分布と最大誤差。
- uniform/history baselineとの差を同一race集合で比較する。

### Calibration

- 主表はrace-balanced equal-frequency reliability。
- bin count、bin方式、runner count、weight、予測平均、実測勝率、gapを保存する。
- ECEは記述診断であり、単独の採否指標にしない。
- confidence interval/bootstrapは別タスクでraceまたは日/week block単位に追加する。

### Conditional diagnostics

初期条件はsurface、race class、distance band、field-size bandである。条件はrace内で一定でなければならない。最小race数未満のcellは出力しない。favorite/longshotのrunner帯はfinal-odds oracle側に分離する。

## 8. BET-01: final-odds oracleの境界

`final_odds_oracle_diagnostic` は各raceの `1/final_odds` をrace内正規化し、次だけを返す。

- market Log Loss/Brierとmodelとの差。
- inverse-odds sumのrace平均。
- final-odds帯別のmodel probability、market probability、実測勝率。

このAPIにはselection、EV threshold、stake、profit、ROIの引数・出力を置かない。final oddsは購入判断時に未知であり、これでrunnerを選ぶとlook-aheadになる。固定済みの別selectionを公式払戻で精算する処理は将来の実行可能市場データ仕様とともに別モジュールで扱う。

Oracleはcore評価の成功条件から分離する。あるraceの全scored runnerにfiniteかつ1以上のfinal oddsが揃う場合だけ、そのrace全体をoracle対象とする。欠損runnerだけを除いてrace内確率を壊してはならない。primary prediction artifactにはfinal odds・人気を含めず、別market artifactへ保存する。

## 9. 公開API

主要なexport署名は次のとおりである。学習・予測のDataFrame入口は、splitで必要行を選択してfeature列だけを`float32`化し、pandas matrixのままLightGBMへ渡す。60万行×多数列をPythonのtuple/listへ全量複製しない。`ComparisonDataset`とsequence APIは小fixture・低水準検証用であり、全量統合runnerでは使わない。

```python
comparison_dataset_from_frame(frame, *, feature_columns, race_id_column="race_id",
                              finish_position_column="finish_position", split_column="split")

train_binary(frame, *, feature_columns, train_split="train",
             model_validation_split="model_validation", ..., params=None,
             early_stopping_rounds=50)
train_ranker(frame, *, feature_columns, train_split="train",
             model_validation_split="model_validation", ..., params=None,
             early_stopping_rounds=50)
predict(model, frame, *, feature_columns, model_kind)

uniform_baseline(frame_or_race_ids, *, race_id_column="race_id")
history_rate_probabilities(history_wins, history_starts, race_ids, *, prior_strength=2.0)

fit_temperature(scores, race_ids, finish_positions, **fit_options)
fit_temperature_from_frame(frame, *, score_column, ..., calibration_split="calibration")
fit_platt_from_frame(frame, *, score_column, ..., calibration_split="calibration",
                     input_kind="probability")
apply_temperature(calibrator, scores, race_ids)
coherent_binary_probabilities(raw_probabilities, race_ids, *, calibrator=None)

evaluate_predictions(probabilities, finish_positions, race_ids, *,
                     ranking_scores=None, conditions=None, reliability_bins=10)
evaluate_prediction_frame(frame, *, probability_column, ..., 
                          evaluation_split="development", final_odds_column=None)
final_odds_oracle_diagnostic(probabilities, final_odds, finish_positions, race_ids)
run_mvp(..., include_retrospective_test=False)
```

低水準のsequence APIは小fixtureと独立部品テストに、DataFrame APIは統合runnerに使う。

## 10. Artifact契約

`metrics.json`相当には最低限次を保存する。

- experiment ID、model family、feature columns/group、split絶対期間。
- data fingerprint、git commit、seed、LightGBM parameter、best iteration。
- probability mapping名、校正input、校正期間、epsilon、temperatureまたはPlatt係数。
- 対象race/runner数、除外数、group-size分布、同着race数。
- ranking/probability/reliability/conditional/final-odds-oracleの各payload。
- final oddsはoracle診断専用でselectionに未使用というflag。
- 2025 retrospective testを評価したか、明示opt-inだったか、結果を再選択へ使わないflag。

予測artifactは少なくとも `race_id`, `runner_key`, `split`, `finish_position`, `raw_binary_probability`, `raw_rank_score`, `coherent_probability`, `mapping_name` を持つ。市場列は別artifactとしてprovenanceを保持する。2025行は明示opt-inしない限り予測・保存しない。

## 11. 依存と未決事項

実行依存はLightGBM、pandas、NumPyである。現実装のbaseline、写像、校正、評価の数値計算自体は標準ライブラリで動き、LightGBMだけを遅延importする。テストはpytestを使う。依存定義は本タスクでは変更しない。

未決事項は次である。

- 同一特徴でのBinary/LambdaRankのcategorical feature指定方法。
- LightGBM hyperparameterの固定値とearly-stopping patienceの最終値。
- beta/isotonic等、MVP固定方式以外のcalibration比較。
- 条件別cellの最小race数とconfidence interval方式。
- 2025年は既にデータ監査に使われているため、「完全未使用final」ではなくretrospective testである。真のfinalは将来prospective期間で確保する。
