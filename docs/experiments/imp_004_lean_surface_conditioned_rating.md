# IMP-004: lean config × surface-conditioned rating

## Pre-registration

**Decision period:** 2024 development only、3,051 races  
**2025:** cache読込直後に除外し、fit・calibration・evaluation・判断に使用しない  
**Seed:** 42  
**Uncertainty:** 4 race-date moving block、10,000 paired resamples、bootstrap seed=`20240830`  
**Odds:** feature・calibration・選択・採否に不使用

### Hypothesis

既存field-relative 15列を除いたlean 253-feature controlに、芝・ダート別に独立更新するsurface-conditioned horse Elo familyだけを追加すると、2024 OOSのrankingまたはprobability qualityが改善する。

これは「field-relative削除」と「surface rating追加」が相補的かを検証するinteraction仮説であり、両変更が個別に良かったことから合成結果を仮定しない。

### Frozen control and candidate

| Arm | Feature selection | Count expected |
|---|---|---:|
| Control | semantic 6 groupsを明示include。`field_relative`と`surface_conditioned_rating`は非include | 253 |
| Candidate | control 6 groups + `surface_conditioned_rating`を明示include | 256 |

Candidateが追加する列は次の3列だけである。

- `surface_rating__horse_elo_pre`
- `surface_rating__horse_minus_field_mean_elo`
- `surface_rating__horse_elo_percentile`

`drop`ではなくexplicit `include`を使い、将来taxonomyへ別groupが追加されても本実験へ暗黙混入しないよう固定する。

Eloはinitial `1500`、K `24`、scale `400`。芝/ダートstateを分離し、同日全race emit後に更新し、障害raceは更新しない。既存268列はcache controlで全533,853 rowsにわたり完全一致済みである。

### Primary estimands

BinaryとLambdaRankを別々に、candidate minus controlとして評価する。正のdeltaをcandidate改善に統一する。

- Ranking: NDCG@3、Top-1 winner mass
- Probability: race Log Loss、race Brier
- Calibration: ECE/reliabilityはsecondary diagnostic

### Decision rule

次のいずれかをmodel family別に満たした場合だけ支持とする。

全deltaは`candidate improvement`を正とする。

1. **Probability path:** Log Loss改善point `>=.002`かつpaired 95% interval下限`>0`、Brier point改善`>=0`、NDCG point差`>=-.002`、Top-1 point差`>=-.005`。
2. **Ranking path:** NDCG改善のpaired 95% interval下限`>0`、Log Loss point改善`>=-.002`、Brier point改善`>=-.001`、Top-1 point差`>=-.005`。

いずれかのpathを満たせば、そのmodel family / pathだけを`accept`とする。区間が0を跨ぐがguardrail内なら`inconclusive`、主要区間が悪化側またはguardrail違反なら`reject`とする。surface変更runner sliceはmechanism診断だけに使い、採否を上書きしない。

### Required controls

1. 同じ271-column cacheから再学習する253-feature controlが、既存`abl_006_drop_field_relative`を再現する。feature count=`253`、ordered feature SHA-256=`fd8735cf6f8472a5c7322e3622c83fde1ff7720b022b75637745c96a1bc1062f`を必須とし、raw score・probabilityの最大絶対差`<=1e-12`、集約metric差`<=1e-12`を許容上限とする。不一致ならcandidateを評価しない。
2. Controlとcandidateのrunner population、split、model config、seedは同一。
3. Candidate追加列がsurface rating 3列だけである。
4. 2025 rows used=`0`、odds used=`false`、git state=`dirty=false`をartifactで確認する。

これは同じ2024で行う4件目の限定改善仮説であり、nominal intervalはselection optimismを補正しない。acceptしても2026+ prospectiveで確認する候補に限定する。

## Results

事前登録commit `846b3879f38be22a230769cbe2ef77d3abd76260`（両armとも`dirty=false`）から実行した。2025 rows used=`0`、odds used=`false`。

### Control validation

同じ271-column cacheから再学習した253-feature controlは、既存`abl_006_drop_field_relative`を完全再現した。

- ordered feature SHA: `fd8735cf6f8472a5c7322e3622c83fde1ff7720b022b75637745c96a1bc1062f`
- 2024 runner identity: 41,946 / mismatch 0
- raw score・全coherent probability最大絶対差: `0`
- 主要・条件別metric最大絶対差: `0`
- best iteration・temperature: 一致

### Candidate result

| Model | Candidate NDCG@3 | NDCG delta [95% CI] | Candidate Log Loss | LL improvement [95% CI] | Brier improvement | Top-1 delta | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Binary | .49424 | +.00299 `[-.00036,+.00648]` | 2.08740 | −.00188 `[-.00640,+.00273]` | −.00121 | −.00098 | reject |
| LambdaRank | .49056 | −.00188 `[-.00558,+.00172]` | 2.08731 | −.00265 `[-.00827,+.00305]` | −.00059 | −.00393 | reject |

Binaryはranking pointが改善方向だがNDCG interval下限が0以下で、Brier point改善`−.00121`がranking-path guardrail `>=−.001`を外れた。LambdaRankはNDCGが悪化方向で、Log Loss point改善`−.00265`もguardrailを外れた。両familyとも事前登録pathを満たさず、明示guardrail違反があるため`reject`とする。

### Interpretation

surface ratingは268-feature baseline上のLambdaRankで追加signalを示したが、field-relative 15列を除いたlean configとは相補的でなかった。3列は強く相関するabsolute/race-relative表現をまとめたfamilyなので、この結果からsurface能力概念そのものを否定せず、「現3列表現をlean configへそのまま足す」仮説を棄却する。

Current bestは253-feature `abl_006_drop_field_relative`のまま。次段ではこのnegative resultを見てsurface ratingの式を事後調整せず、事前監査で特定したrace-value定義の問題を独立仮説として扱う。

Full local artifacts:

- `artifacts/imp_004_control_lean_global_elo/`
- `artifacts/imp_004_lean_surface_conditioned_rating/`
- `artifacts/imp_004_primary_comparison/`

Tracked summary: [`experiments/rating_race_value_20260830/imp_004_summary.json`](../../experiments/rating_race_value_20260830/imp_004_summary.json)
