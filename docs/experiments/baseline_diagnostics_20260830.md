# Corrected baseline feature and error diagnostics

**Models:** corrected Binary / LambdaRank baseline  
**Data:** 2024 development 3,051 races / 41,946 runners  
**Methods:** split+gain importance、全runner TreeSHAP、race-centered TreeSHAP、semantic-group permutation 5 repeats、race/runner condition analysis  
**2025:** cache読込直後に44,179 retrospective rowsを除外し、以後の処理では0 rows

## 1. Validation and leak-proxy audit

- 保存modelからbaseline予測を再生成し、calibrated probabilityの最大絶対誤差はBinary `2.2e-16`、LambdaRank `1.0e-16`だった。
- TreeSHAP additivity最大誤差は`2.1e-14`以下。
- model allowlistにraw horse/jockey/trainer ID、date/year、odds、人気、着順、time、margin等の禁止feature名は0件。
- 2024内dateとの最大Spearman相関は`context__class_age_min=-0.610`、`context__venue_code=-0.306`、`context__age=-0.283`。JRAの年齢条件・開催scheduleにより正当に生じ得るが、calendar proxyでもあるためsubgroup driftを監視する。
- `horse_history__career__mean_opponent_elo`のdate相関は+0.043で、強いera memorizationの証拠は得られなかった。

trainer nameはmodelへ直接入らず、過去集計stateのkeyだけに使う。connections groupを除くと全主要metricが明確に悪化したため有用signalは確認できるが、多数の累積統計がentity fingerprintに近づく可能性までは排除できない。

## 2. Feature importance: three views

### Race-centered TreeSHAP share

race内softmax/rankingで相殺される共通効果を弱めるため、各feature SHAPからrace内meanを引いた絶対値をgroup集約した。

| Group | Binary | LambdaRank |
|---|---:|---:|
| form/workload | 26.45% | **35.82%** |
| field-relative | 21.00% | 16.81% |
| connections | 20.93% | 15.33% |
| rating/value | 12.74% | 11.24% |
| suitability | 7.29% | 9.81% |
| horse performance | 6.39% | 6.65% |
| current context | 5.21% | 4.34% |

LambdaRankはformへより集中する。Binaryはfield-relativeとconnectionsへの配分が高い。両modelで上位に共通するfeatureは`decay_90d/30d__mean_finish`、`same_surface__mean_finish`、jockey win-rate relative、horse-vs-field Elo、historical opponent Eloである。

### Permutation dependence

runner-varying列はrace内でgroup列をjoint shuffleし、race-constant列はrace block単位でjoint shuffleした。正値は悪化量である。

| Model | Largest Log Loss degradation |
|---|---|
| Binary | field-relative +0.219、form/workload +0.170、rating/value +0.069 |
| LambdaRank | form/workload +0.167、field-relative +0.086、connections +0.049 |

これはfitted modelの依存度であり、causal contributionではない。相関group間の代替、off-manifold perturbationもある。特にfield-relativeはpermutationで最大級に悪化する一方、再学習ablationで除くと性能が改善した。モデルがこのgroupへ強く依存しながらOOSではnoiseを拾う、brittle relianceの証拠である。

### Ablationとの統合

- connections: permutation単独では相関featureが代替するが、source+relative descendantsを再学習dropすると大幅悪化。追加情報は強い。
- form/workload: SHAP、gain、permutation、ablationが一貫して有用。
- rating/value: permutationで正寄与し、group dropでproper scoreが悪化。Elo/opponent strengthは追加情報を持つ。
- horse absolute history vs field-relative: absolute history dropはBinary Log Lossを悪化させる。現relative groupは除去で改善するため、「relativeなら常に良い」のではなく現在のwin-rate/rest変換が冗長・noiseという結論。

## 3. Strong and weak race conditions

race-constant sliceだけにcoherent race-macro metricを用いた。異なるfield sizeのLog Lossを単純なmodel能力差と同一視しない。

| Dimension | Stronger | Weaker |
|---|---|---|
| surface | turf: Binary NDCG .496 / LL 2.055 | dirt: .479 / 2.124 |
| distance | middle: .513 / 1.993 | sprint: .478 / 2.139 |
| class | 2-win: .514 / 1.992、maiden .505 / 2.054 | open/graded .427 / 2.242、新馬 .456 / 2.163 |
| venue | Tokyo .533 / 2.001 | Sapporo .458 NDCG、Fukushima LL 2.201 |
| field size | ≤9: .583 / 1.686 | 17–18: .438 / 2.306 |

LambdaRankもほぼ同じ弱点を持つ。Q1/Q2ではLambdaRank NDCGがBinaryを上回り、Q3/Q4では下回るが、baseline uncertaintyでmodel間差は未解決である。京都・12月は構造欠損のため強い断定を避ける。

## 4. Runner-specific diagnostics

runner sliceではrace内確率和が1にならないためrace Log Lossを報告せず、runner-micro loss、calibration gap、modelがtop選択したhorseのhit rateを使った。

- no prior history 4,404 runners（10.5%）のBinary平均確率.0680、実勝率.0665で平均校正は近い。top選択hit .279で全体.281と大差なく、cold-startだけが最大弱点ではない。
- 181日以上休養は1,576 runners。Binaryは.0519対実勝率.0470で約0.005 overprediction、top選択hit .197。
- surface変更4,657 runnersはtop選択hit Binary .201 / Ranker .241、same-surfaceは.286 / .285。condition-specific state改善の根拠になる。
- 距離400m以上変更5,038 runnersはtop選択hit Binary .254 / Ranker .276。same-distance .285。
- 複数condition変更18,299 runnersのtop選択hit Binary .269、変更なし.283。差はあるがbase rateも異なるため因果解釈しない。

前走conditionは、cacheに存在するstrictly earlier dateのeligible flat raceから再構築した。公式欠損またはPIT-C除外startを飛ばす場合があるため、完全な全出走履歴ではない。

## 5. Final-odds market oracle

final oddsは事後診断だけに用い、feature、calibration、選択、ROIへ使っていない。

| Model | NDCG@3 | Top-1 | Log Loss | Brier |
|---|---:|---:|---:|---:|
| Binary | .4873 | .2814 | 2.0898 | .8312 |
| LambdaRank | .4864 | .2804 | 2.0892 | .8301 |
| final-odds market oracle | .5539 | .3468 | 1.8828 | .7815 |

Binaryのmarketとの差はLog Loss +0.2070、LambdaRank +0.2065。1番人気の実勝率.345に対しBinary平均確率.246、最終odds 1–2倍は.518対.355でunderpredictionする。一方30倍以上は実勝率.0085に対し.0238でoverpredictionする。これは市場情報が有用というoracle事実で、締切前oddsなしに同じ評価を実行可能という意味ではない。

market gapは新馬+0.332、京都+0.259、17–18頭+0.251、sprint+0.242で大きい。新馬はhorse historyが存在しないため、現データだけで解決できる余地はconnections/contextに限られる。改善実験は、既存signalがあり説明可能なfield-relative縮約、surface-conditioned rating、venue表現、field-size calibrationを優先する。

## 6. Diagnostic decisions

1. 明白な直接ID/market/outcome leakは見つからなかった。
2. Elo/ratingを全削除しない。proper scoreへの追加情報がある。
3. field-relative 15列は「よく使われているが汎化に有害」な最優先修正対象。
4. form/workload、とくに30/90日decay mean finishを維持・活用する。
5. 新馬、open/graded、大頭数、sprint、surface変更を次のerror reduction対象とする。ただし2024 supportとcoverage limitationを併記する。

Full artifact: `artifacts/baseline_diagnostics_20260830_corrected/`。  
Tracked summary: [`experiments/baseline_validation_20260830/diagnostics_summary.json`](../../experiments/baseline_validation_20260830/diagnostics_summary.json)
