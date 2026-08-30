# IMP-005: expected-vs-actual race-value

## Pre-registration

**Decision period:** 2024 development only、3,051 races  
**2025:** cache読込直後に除外し、fit・calibration・evaluation・判断に使用しない  
**Seed:** 42  
**Uncertainty:** 4 race-date moving block、10,000 paired resamples、bootstrap seed=`20240830`  
**Odds:** feature・calibration・選択・採否に不使用

### Hypothesis

field-relative 15列を除いたlean 253-feature controlに、「過去raceでglobal Eloが期待した相対成績」と「実際の相対着順」の差を90日減衰集約した1列だけを追加すると、2024 OOSのrankingまたはprobability qualityが改善する。

現行`horse_history__career__mean_performance_value`は着順percentileとfield平均Elo水準の和であり、horse自身の事前能力に対して上振れ・下振れしたかを表さない。本実験はrating levelと独立した短期performance surpriseに追加情報があるかを検証する。

### Frozen definition

race `r`のstarter数を`n`、race前global Eloを`R`とする。

```text
expected_i = 1/(n-1) * sum_j 1 / (1 + 10 ** ((R_j - R_i) / 400))
actual_i   = 1/(n-1) * sum_j pairwise_result(i, j)
surprise_i = actual_i - expected_i = global_elo_delta_i / K
```

pairwise resultは上位=1、同着=.5、下位=0。数値着順は中止・失格より上、中止・失格同士はtie、取消・除外は非starterとして比較から除く。K=`24`だが、Kで割るためsurprise自体はKに依存しない。

target date `d`の特徴は、`date_r < d`の全過去観測についてhalf-life 90日の指数重みを付けた平均とする。

```text
weight_r = 2 ** (-(d - date_r) / 90)
feature  = sum(weight_r * surprise_r) / sum(weight_r)
```

追加列は`race_value__decay_90d__mean_global_elo_surprise`だけ。履歴なしはNaN。同日の結果は見えない。既存career performance valueはcontrol/candidate両方に残す。class、surface、距離、着差、時計、上がり、追加confidence列は混ぜない。

### Frozen control and candidate

| Arm | Feature selection | Count expected |
|---|---|---:|
| Control | semantic 6 groupsを明示include。`field_relative`と新race-value groupは非include | 253 |
| Candidate | control 6 groups + `race_value_expected_actual` | 254 |

`drop`ではなくexplicit `include`を使い、別の将来groupが暗黙混入しないよう固定する。

### Primary estimands and decision rule

BinaryとLambdaRankを別々に、candidate minus controlとして評価する。正のdeltaをcandidate改善に統一する。対象はNDCG@3、Top-1 winner mass、race Log Loss、race Brier。ECE/reliabilityとfeature importanceはsecondary diagnosticとする。

次のいずれかをmodel family別に満たした場合だけ支持とする。

1. **Probability path:** Log Loss改善point `>=.002`かつpaired 95% interval下限`>0`、Brier point改善`>=0`、NDCG point差`>=-.002`、Top-1 point差`>=-.005`。
2. **Ranking path:** NDCG改善のpaired 95% interval下限`>0`、Log Loss point改善`>=-.002`、Brier point改善`>=-.001`、Top-1 point差`>=-.005`。

いずれかのpathを満たせば、そのmodel family / pathだけを`accept`とする。区間が0を跨ぐがguardrail内なら`inconclusive`、主要区間が悪化側またはguardrail違反なら`reject`とする。

### Required controls

1. 新269-column cacheから、新race-value列を除く既存268列がdefault corrected cacheと同じ順序・値・NaN位置で全533,853 rows完全一致する。不一致ならcandidateを評価しない。
2. 同cacheから再学習する253-feature controlが既存`abl_006_drop_field_relative`を再現する。feature count=`253`、ordered feature SHA-256=`fd8735cf6f8472a5c7322e3622c83fde1ff7720b022b75637745c96a1bc1062f`、raw score・probability・主要metric差`<=1e-12`を必須とする。
3. Controlとcandidateのrunner population、split、model config、seedは同一。Candidate追加列は上記1列だけ。
4. 2025 rows used=`0`、odds used=`false`、実験artifactのgit state=`dirty=false`を確認する。

これは同じ2024で行う5件目の限定改善仮説であり、nominal intervalはselection optimismを補正しない。acceptしても2026+ prospectiveで確認する候補に限定する。

## Results

未実行。
