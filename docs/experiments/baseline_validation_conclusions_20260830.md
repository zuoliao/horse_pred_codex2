# Baseline validation and improvement conclusions

**Primary decision period:** 2024 development、3,051 races / 41,946 runners / 106 dates  
**2025 usage:** 0 rows。cacheに物理的に含まれる44,179 rowsは読込直後に除外  
**Raw fingerprint:** `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`

## Executive decision

現baselineは、**観測済みのstrict PIT-C scoring母集団内では信頼できる基準点**である。直接ID、odds、結果の明白なリークはなく、LightGBMはHistory-rateを全主要metricで大幅かつ安定して上回る。一方、2024年末京都の構造欠損と取消・除外raceの全体除外により、JRA全平地raceへの外的妥当性は「中程度」に留まる。

現時点のbest feature configは、既存field-relative 15列を除いた`abl_006_drop_field_relative`（253特徴）である。point estimateではLambdaRankがNDCG@3とLog Loss、BinaryがTop-1とBrierで僅かに良いが、両者のpaired intervalは全て0を跨ぐ。したがってモデルfamilyは未決着のまま両方を残す。

限定改善5件のうち、surface-conditioned Eloは268列baselineに対するLambdaRankのranking経路だけで事前基準を満たした。ただし253特徴lean configへ同じ3列を追加したIMP-004は両familyで棄却した。Elo期待差を90日減衰したrace-value 1列もBinary inconclusive、LambdaRank rejectで、現bestは変わらない。

## 1. Baselineをどこまで信用できるか

### Evidence

- 旧Task 16にはsurfaceが芝/ダート表記された障害raceが混入していた。修正後は障害履歴もhorse/jockey/trainer/Elo stateを更新しない。corrected artifactは533,853 runners / 37,889 races / 268特徴、clean commitから再生成した。
- 2024 developmentは3,051 races。Uniform、History-rate、Binary、LambdaRankを同一race/date blockで比較した。
- 4 race-date moving block、10,000 resamplesで、Binary対History-rateのNDCG改善は`+0.2327 [0.2175, 0.2464]`、Log Loss改善は`+0.4555 [0.4161, 0.4933]`。全bootstrap sampleで改善した。
- 禁止feature名は0。保存modelの予測再現誤差は`4.5e-16`以下、TreeSHAP additivity誤差は`2.1e-14`以下。

### Remaining selection bias

- 既知の146 missing racesは非ランダムである。2024は7回京都の9日間108 racesに集中し、平地103・障害5。2024公式平地3,327 racesに対するraw coverageは96.90%、strict scoring coverageは91.70%。京都平地の14.57%、12月平地の37.30%が欠ける。
- 2024 raw平地3,224 racesのうち173 races（5.37%）は取消・除外を含むためprimary scoring対象外。取消発表時刻がないのでstarter-only再評価は`post-scratch-field oracle sensitivity`にしかならず、`T_prevday` primaryへ混ぜない。

### Finding

内部妥当性は高いが、「2024京都年末を含む全JRA平地」「取消・除外発生race」「live出走母集団」への一般化には構造的な不確実性が残る。京都・12月や非starter発生率の高いopen/newcomer等の条件別差を断定しない。

## 2. BinaryとLambdaRank

corrected 268-feature baselineの差は未解決である。

| Model | NDCG@3 | Top-1 | Log Loss | Brier |
|---|---:|---:|---:|---:|
| Binary | .4873 | .2814 | 2.0898 | .8312 |
| LambdaRank | .4864 | .2804 | **2.0892** | **.8301** |

LambdaRank minus Binaryのpaired 95% intervalは、NDCG `[-.0059, .0040]`、Top-1 `[-.0112, .0084]`、Log Loss改善`[-.0075, .0088]`、Brier改善`[-.0018, .0042]`。Q1/Q2はLambdaRank、Q3/Q4はBinaryのpoint NDCGが高いが、年内変動も含めfamily採否の根拠にならない。

253-feature best configでもLambdaRank minus BinaryはNDCG `+.00118 [-.00447, .00666]`、Log Loss改善`+.00086 [-.00613, .00869]`で未解決である。

## 3. Feature group contribution

| Groupを除く | 統合判断 |
|---|---|
| connections + dependent relative | 両modelの全主要metricが大幅悪化。最も強い追加情報 |
| form/workload + descendants | proper scoreが悪化。30/90日decay mean finishが特に重要 |
| rating/value | RankerのLog Loss/Brierが明確に悪化。Elo/opponent strengthは追加情報あり |
| horse absolute history | Binary proper scoreが悪化。絶対履歴を維持 |
| current context | 小幅悪化方向。維持 |
| suitability | 除去でpoint rankingは上がるがproper scoreは悪化し、採用根拠不足 |
| field-relative 15列 | 除去で両modelの全point metricが改善。現在の冗長なwin-rate/rest相対表現は汎化に有害 |

race-centered SHAP shareはBinaryでform 26.5%、field-relative 21.0%、connections 20.9%、rating 12.7%。LambdaRankはform 35.8%、field-relative 16.8%、connections 15.3%、rating 11.2%。field-relativeはpermutationで強く使われながら再学習dropで改善したため、「依存度が高い=有用」ではなくbrittle relianceと解釈する。

## 4. 得意・不得意条件

Binary baselineの代表値:

| Condition | Stronger | Weaker |
|---|---|---|
| surface | turf NDCG .496 / LL 2.055 | dirt .479 / 2.124 |
| distance | middle .513 / 1.993 | sprint .478 / 2.139 |
| class | 2-win .514、新馬以外のmaiden .505 | open/graded .427、新馬 .456 |
| venue | Tokyo .533 / 2.001 | Sapporo NDCG .458、Fukushima LL 2.201 |
| field size | 9頭以下 .583 / 1.686 | 17–18頭 .438 / 2.306 |

runner診断では181日以上休養のtop選択hitが.197、surface変更はBinary .201 / Ranker .241、400m以上距離変更はBinary .254 / Ranker .276。field sizeごとのraw Log Lossはentropyが異なるため単純比較せず、uniformからの改善も併読した。cold-start馬の平均校正とtop hitは全体と大差なく、現データでの最大弱点はdebut馬だけではない。

## 5. Limited improvement experiments

各実験は2024の同一3,051 races、4-date block、10,000 resamplesで評価した。5仮説×2 familyのnominal intervalであり、多重比較と2024 selection optimismを除かない。

| Experiment | Change | Binary | LambdaRank | Decision |
|---|---|---|---|---|
| IMP-001 | 253-feature drop controlへ90日form percentile 1列追加 | NDCG −.00739、LL改善 −.01434 | NDCG −.00440、LL改善 −.00751 | reject。両familyで主要区間が悪化側 |
| IMP-002 | global Elo baselineへ芝/ダート別Elo 3列追加 | LL +.00197だがCI跨ぎ、NDCGほぼ0 | NDCG +.00364 `[+.00005,+.00714]`、proper score非悪化 | Ranker ranking pathのみaccept |
| IMP-003 | 2023 field-size-band別temperature | LL −.00040、Brier悪化、ECE改善 | LL −.00070、Brier悪化、ECE改善 | reject。ranking不変、ECE単独では採用しない |
| IMP-004 | 253-feature lean controlへsurface Elo 3列追加 | NDCG +.00299だがBrier改善 −.00121 | NDCG −.00188、LL改善 −.00265 | 両family reject。変更は相補的でなかった |
| IMP-005 | 253-feature lean controlへ90日Elo期待差1列追加 | NDCG −.00158、LL改善 −.00172 | NDCG −.00392、LL改善 −.00299 | Binary inconclusive、Ranker reject。採用せず |

IMP-002とIMP-005のcache controlはいずれも533,853行×旧268列でidentity、順序、値、NaN位置が完全一致し、不一致0・最大差0。IMP-005の新featureはrecent 90日mean finishとSpearman `−.938`なのにgain importanceがBinary 28.7%、Ranker 42.6%を占め、強く利用されたがOOS改善はなかった。

## 6. Current best model/config

| Config | Model | Features | NDCG@3 | Top-1 | Log Loss | Brier |
|---|---|---:|---:|---:|---:|---:|
| `abl_006_drop_field_relative` | Binary | 253 | .4913 | **.2912** | 2.0855 | **.82985** |
| `abl_006_drop_field_relative` | LambdaRank | 253 | **.4924** | .2881 | **2.0847** | .82995 |
| `imp_002_surface_conditioned_elo` | LambdaRank | 271 | .4900 | .2832 | 2.0870 | .82963 |

point estimate上は253-feature LambdaRankをbest modelと呼べるが、Binaryとの差は決着していない。実務上の結論は「253-feature configがbest、Binary/LambdaRankは並行維持」である。surface Elo Rankerはcorrected baselineに対する有効signalだが、253 configより良いというinterval証拠はない。

## 7. Final-odds marketとの差

final-odds oracleはNDCG@3 `.5539`、Top-1 `.3468`、Log Loss `1.8828`、Brier `.7815`。corrected Binaryとの差はLog Loss `+.2070`、LambdaRankとの差は`+.2065`で、市場が明確に強い。1番人気の実勝率`.345`に対しBinary平均`.246`、30倍以上は実勝率`.0085`に対し`.0238`で、favoriteをunderpredictしlongshotをoverpredictする。

これは締切後final oddsを使う**事後oracle診断**であり、実行可能ROI、締切前market edge、購入収益性を意味しない。oddsはfeature、calibration、改善仮説選択、accept/rejectに使っていない。

## 8. Next priority hypothesis

最優先は、race-valueを「最近の着順残差」と「最近対戦したfieldの事前強度」に分離し、後者だけを1列で検証可能にすることである。

理由:

1. IMP-005のElo期待差はrecent mean finishとほぼ同じ情報になり、window追加の優先度が低い。
2. state内部には過去raceのpre-race field Eloが既に保存されるが、現在はcareer平均しかemitされず、recent opponent strengthは未検証である。
3. outcomeとfield qualityを分ければ、どちらが効くかを解釈できる。
4. ただし現`mean_opponent_elo`はself-inclusive field meanなので、まずopponent-only定義・命名とPIT fixtureを固定する必要がある。

次の候補はhalf-life 90日のopponent-only field-strength平均1列であり、career値とrecent値の差を一度に追加しない。先に定義監査とdefault cache不変を確認し、同じ2024を既に5回使ったselection optimismを明記して事前登録する。surface/global shrinkageは係数選択が恣意的でIMP-004のnegative resultを見た事後調整になりやすいため次点とする。データ取得拡張、DNN、購入戦略、Web UIにはまだ進まない。

## Sources of truth

- [Data health](baseline_data_health_20260830.md)
- [Uncertainty](baseline_uncertainty_20260830.md)
- [Semantic ablation](semantic_feature_ablation_20260830.md)
- [Diagnostics](baseline_diagnostics_20260830.md)
- [Improvement preregistration](improvement_experiments_20260830.md)
- [IMP-004 lean surface interaction](imp_004_lean_surface_conditioned_rating.md)
- [IMP-005 expected-actual race value](imp_005_expected_actual_race_value.md)
- [Tracked machine summary](../../experiments/baseline_validation_20260830/improvement_summary.json)
- Full local artifacts: `artifacts/imp_001_*`、`artifacts/imp_002_*`、`artifacts/imp_003_*`、`artifacts/best_candidate_*`
