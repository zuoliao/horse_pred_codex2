# PIT-safe baseline feature specification

**Status:** PIPE-02 / FEAT-01～04 / QA-01 の先行実装（2026-08-30）  
**Implementation:** `src/horse_pred/features.py`  
**Scope:** JRA平地、承認済みraw相当のrunner-level `pandas.DataFrame`。モデル学習・払戻精算は対象外。

## 1. 公開API

```python
from horse_pred.features import FeatureConfig, FeatureDataset, build_features

dataset: FeatureDataset = build_features(
    normalized,
    split_config=split_config,
    config=FeatureConfig(),
)
X = dataset.frame.loc[:, dataset.feature_columns]
```

`FeatureDataset` は次を返す。

- `frame`: 入力raw/normalized列、監査用`meta__*`、生成特徴を同じ行に保持する。race ID、entity ID、着順label、最終単勝、人気はここに残してよい。
- `feature_columns`: **生成済み数値特徴だけ**の閉じたallowlist。モデル入力は必ずこれを使う。
- `feature_groups`: 実験configと一致する論理group別の数値列。`race_context`、`horse_history_basic`、`form_workload`、`connections_pit`、`field_relative`、`rating_strength`。

低水準の `build_pit_features(raw, config)` は生成特徴と`meta__*`だけを返す。`model_feature_allowlist()` と `validate_model_feature_columns()` は、rawの数値列をdtypeだけで誤採用することを防ぐ。

## 2. 入力契約

既定列名は採用rawに合わせる。

| 意味 | 既定列 | 用途 |
|---|---|---|
| race key / date | `raceid`, `date` | group、日次as-of、metadata |
| entity state key | `horse_id`, `jockey_id`, `trainer` | join/state更新だけ。model特徴にはしない |
| result | `着順` | target日をemitした後のstate更新だけ |
| race condition | `distance`, `race_class`, `course_type`, `around` | current contextと過去条件集計 |
| runner context | `sex`, `age`, `枠番`, `馬番` | current context |
| metadata only | `ground_state`, `weather`, `単勝`, `人気`等 | 初期prevday model特徴にはしない |

`trainer` は現rawに正規trainer IDがないため暫定state keyである。同姓・表記変更・改名の衝突を解消しないため、正規ID入手後に`trainer_id_col`を差し替える。race IDは一つの日付だけに属し、`(raceid, horse_id)`は一意でなければならない。違反は黙って集約せず例外にする。

## 3. PIT-C更新規則

rawには発走時刻・公表時刻がない。したがって、時刻やrace番号から同日内の先後関係を推測せず、次の保守的な規則を採る。

```text
for date in ascending dates:
    for every race on date:
        emit features from state whose source_date < date
    after all races on date were emitted:
        update state from all eligible results on date
```

すなわち、同一race内だけでなく**同一開催日内の別race結果も相互に見えない**。入力行順やrace番号順を変えても同じ特徴になり、未来行を末尾追加しても既存行の特徴は不変である。splitは特徴生成後に日付で付けるため、warmupの結果は後続trainの履歴に使えるが、split境界当日の結果は同日中に漏れない。

現在のsplit mappingは、`warmup=2013`、`train=2014–2021`、`model_validation=2022`、`calibration=2023`、`development=2024`、`retrospective_test=2025`、`prospective_final=2026+`を扱える。APIは固定名列挙ではなく、非重複の絶対日付intervalを開始日順に検証する。

## 4. race/runner context

raw文字カテゴリは直接modelへ渡さない。未来データのカテゴリ集合で符号が変わらない固定encodeだけを生成する。

- 競馬場: race IDの場コードを`1..10`へ変換した`venue_code`
- surface: turf / dirt / unknown one-hot。障害raceは`course_type`だけでは判定せず、`race_class`に`障害`を含むraceをsurface表記に関係なく非flatとする。行は監査用に保持するが`meta__is_flat_race=False`、`meta__is_scored_race=False`かつflat履歴stateを更新しない
- direction: right / left / straight / unknown one-hot
- sex: male / female / gelding / unknown one-hot
- class: 新馬、未勝利、1勝/500万、2勝/1000万、3勝/1600万、open/重賞を`0..5`へ固定変換
- class補助: 牝馬限定flag、最低年齢、`以上`flag
- 数値: distance、age、gate、horse number、raw行ベースfield size

`ground_state`と`weather`は結果ページ由来PIT-Cでprevday時点の利用可能性を証明できないため、初期model特徴から除外する。rawは`FeatureDataset.frame`にmetadataとして残る。取消・除外を含むraceのraw fieldはas-of再現不能なので、そのraceは`meta__is_scored_race=False`で評価対象外にする。

## 5. 履歴特徴

### Horse

- career: starts、wins、win rate、完走数、数値着順平均、累積公称距離
- 走数窓: 直近1/3/5/10走の上記集計
- 暦日窓: 過去14/30/90/180/365日の上記集計。境界日は含め、target日は含めない
- 指数減衰: half-life 30/90/180日のeffective starts、win rate、finish、distance
- 休養: 最終start日からtarget日までの日数
- 条件別career: same surface、same distance band、same venue
- opponent/value: 過去raceのpre-race field Elo平均と、後述の単純performance value平均

暦日窓より古い個票を無条件に捨ててcareerを作り直すことはない。career、条件別career、指数減衰stateは全過去updateを引き継ぎ、走数窓・暦日窓だけが指定範囲を参照する。

### Jockey / trainer

horseと同じ日次batch stateからcareer、走数窓、暦日窓、指数減衰のstarts/wins/win rate/完走数/着順平均を作る。同じtrainerが同一raceまたは同一日に複数頭を出しても、対象日の全runnerが同じ前日終了stateを見る。欠損IDは共通の「unknown entity」にまとめずcold startとして扱い、更新しない。

### Field-relative

対象race内で、過去のみから作ったhorse/jockey/trainer win rateと休養日数について、field平均差、z-score、percentileを計算する。現在raceの着順・人気・オッズは使わない。全欠損または分散0は、欠損を保持しつつ定義可能なz-scoreを0とする。

## 6. Forward-only Eloとrace strength

Horseだけに初期値1500、scale 400、K=24のpairwise Eloを使う。race前ratingを特徴としてemitし、その日の全race emit後に各raceのdeltaを同時適用する。

```text
expected(i > j) = 1 / (1 + 10 ** ((Rj - Ri) / 400))
delta_i = K / (field_size - 1) * sum_j(actual(i > j) - expected(i > j))
```

同着は0.5、数値着順は小さい方を勝ち、中止・失格は全数値着順より下、複数の中止・失格同士はtieとする。取消・除外は比較に入れない。Kを相手数で割るため、頭数だけで更新幅が線形増加しない。

生成するstrength特徴はhorse pre-Elo、field pre-Eloのmean/max/std、horse-minus-field mean、field内percentileである。過去走の簡易valueは「そのraceの着順percentile + (pre-race field mean Elo - 1500) / 400」とし、将来の相手成績で遡及更新しない。これは将来のRace-value encoderの代替ではなく、forward-only相手強度仮説を検証するための分離可能なbaselineである。

## 7. 結果例外

| raw着順 | 行保持 | start/history update | win | finish平均 | Elo |
|---|---:|---:|---:|---:|---|
| 数値 | ○ | ○ | `finish==1` | 更新 | 順位比較 |
| 数値付き降着表記 | ○ | ○ | parse後順位による | 更新 | parse後順位比較 |
| 中止・失格 | ○ | ○ | 0 | 更新しない | 数値完走馬より下 |
| 取消・除外 | ○ | × | 対象外 | 対象外 | 比較しない |
| 未知status | ○ | ×（未確定） | 対象外 | 対象外 | 比較しない |

同着1着は各馬をofficial win=1として履歴勝率を更新する。Binary labelでは各winnerを1とし、lossのrace weightをwinner数`m`で割るかwinner mass `1/m`を別途使う規則はmodeling側の責務であり、この特徴builderはtarget列をmodel allowlistへ入れない。未知statusはstartと推測せず、raceをscoring対象外にする。

中止・失格も出走負荷として公称distanceを累積する。これは実走距離ではなくschedule load proxyであり、実走距離が得られない不確実性を持つ。取消・除外を含むraceはscoringから外すが、実際にstartした他runnerの結果は将来履歴へ更新する。障害raceはflat-only MVPからrace全体を除き、将来のflat履歴も更新しない。

## 8. Leakage/QA invariants

`tests/test_features.py`で次をfixture検証する。

- 同一日全raceのemit後に一括updateされること
- 同日内の行順変更で特徴が変わらないこと
- future append invariance
- 30日前の境界を含み、target dateを含まないこと
- splitに`model_validation`が存在すること
- 中止・失格、取消・除外、同着をdropしないこと
- duplicate runner keyを拒否すること
- raw ID、target、最終単勝、人気が数値でもallowlistへ入らないこと
- `FeatureDataset.feature_columns`が全て数値で、論理groupの和と一致すること

## 9. 既知の制約と性能

- 実装依存は`pandas`と`numpy`。`pyproject.toml`の既存宣言を利用し、本タスクでは依存定義を変更しない。
- correctness優先のin-memory実装であり、runnerごとに複数windowをsnapshotする。約63万行の本rawではwide frameのメモリとjockey/trainerの365日deque走査が主なbottleneckになり得る。
- 2026-08-30の開発環境で採用raw先頭20,000行を測った参考値は約28.1秒、返却frameの`memory_usage(deep=True)`は約43.6 MB、数値allowlistは268列だった。単純線形外挿でも全629,967行は約15分・返却frame約1.37 GBで、concat中のpeak RSSはこれより大きい。これは正式な全量benchmarkではない。
- 本raw全量での時間・peak RSS benchmarkはPIPE統合後に必須。必要なら同じfixtureを保ったまま、日窓ごとのrolling aggregate、columnar buffer、date chunk出力へ置換する。
- trainer nameの擬似key、結果ページ由来contextのPIT-C、取消・除外raceのas-of field不明、DNFの公称距離proxyは解消していない。
- Eloとperformance valueは一方式だけを固定した比較baselineであり、BT/Plackett-Luce/動的ratingや将来Race-value encoderを先取りしていない。
