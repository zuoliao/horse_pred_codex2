# Semantic feature-group ablation

**Reference:** corrected 268-feature baseline  
**Design:** one semantic source-family knockout per experiment; Binary and LambdaRank use the same remaining columns  
**Evaluation:** 2024 development 3,051 races; paired four-date-block bootstrap 10,000回; 2025未使用

## Taxonomy

construction都合の旧6 groupをそのまま解釈せず、268列を重複なく次の7 groupへ再分割した。

| Group | Columns | Dependency-aware rule |
|---|---:|---|
| current context | 21 | race/runner current context |
| horse performance | 25 | horse由来field-relative 6列も同時drop、計31 |
| form/workload | 54 | rest-relative 3列も同時drop、計57 |
| suitability | 15 | same surface/distance/venue |
| connections | 130 | jockey/trainer relative 6列も同時drop、計136 |
| field-relative | 15 | そのままdrop |
| rating/value | 8 | runner/field Elo 6 + historical opponent/race value 2 |

親groupを除いても決定論的なrelative派生列が残れば通常ablationは寄与を過小評価するため、horse performance、form/workload、connectionsはsource-family knockoutを主結果とした。

## Results

| Dropped group | Binary NDCG / Top-1 | Binary LL / Brier | Ranker NDCG / Top-1 | Ranker LL / Brier | Interpretation |
|---|---:|---:|---:|---:|---|
| none (268) | .4873 / .2814 | 2.0898 / .8312 | .4864 / .2804 | 2.0892 / .8301 | corrected baseline |
| current context | .4849 / .2793 | 2.0992 / .8333 | .4859 / .2793 | 2.0937 / .8313 | modest probability contribution |
| horse performance + descendants | .4877 / .2889 | 2.0990 / .8329 | .4863 / .2819 | 2.0938 / .8310 | top-1 and proper scores conflict; retain |
| form/workload + descendants | .4839 / .2884 | 2.1004 / .8329 | .4828 / .2765 | 2.0996 / .8324 | proper-score contribution; retain |
| suitability | .4892 / .2892 | 2.0928 / .8317 | .4880 / .2848 | 2.0938 / .8321 | mixed; no clean removal case |
| connections + descendants | .4588 / .2604 | 2.1507 / .8428 | .4578 / .2594 | 2.1557 / .8439 | largest positive contributor |
| field-relative | **.4913 / .2912** | **2.0855 / .8298** | **.4924 / .2881** | **2.0847 / .8299** | removal improves all point metrics |
| rating/value | .4862 / .2824 | 2.0966 / .8331 | .4885 / .2838 | 2.1006 / .8332 | ranking/probability conflict; retain pending decomposition |

太字は各familyのbaselineより良いpoint estimateで、単一metricによる採否を意味しない。

## Paired uncertainty and findings

### Connections

最も明確な寄与である。drop時のBinary NDCG差は−0.0284 [−0.0382, −0.0187]、Top-1差−0.0210 [−0.0366, −0.0059]。Log Loss改善量は−0.0609 [−0.0773, −0.0455]で、負値はcandidate悪化を意味する。LambdaRankも全4 metricでintervalが悪化側にある。多数列であるため「jockey/trainerのどのwindowが効くか」は未解決だが、group全体の追加情報は強い。

### Form/workload

drop時に両familyのLog Lossが明確に悪化し、LambdaRankはBrierも悪化した。Binary Top-1だけは+0.0070へ動くため、勝馬一点選択と分布品質のtrade-offがある。単一Top-1だけで除去しない。

### Horse performance / suitability / rating-value

いずれもTop-1またはNDCGの一部が改善する一方、proper scoreが悪化するmixed resultだった。horse performance dropのBinary Log Loss改善量は−0.0092 [−0.0153, −0.0029]。rating/value dropはLambdaRank Log Loss −0.0114 [−0.0168, −0.0059]、Brier −0.0031 [−0.0048, −0.0015]で、Elo/value全体を捨てる根拠はない。相関featureが多いため、各groupの条件付き限界寄与として解釈する。

### Field-relative

唯一、drop後に両familyの全4 point metricが改善した。特にLambdaRank NDCG差+0.0060 [0.0019, 0.0103]、Top-1差+0.0077 [0.0006, 0.0148]は0を跨がない。Binary側はNDCG +0.0040 [−0.0020, 0.0098]等で方向は良いがintervalは跨ぐ。既存relative列は絶対履歴と強く重複し、noiseまたはtree split競合を増やす仮説が最も強い。

## Decision

1. connections、form/workloadを維持する。
2. horse performance、suitability、rating/valueは一括削除せず、必要なら内部を分解する。
3. `abl_006_drop_field_relative`を現時点のbest candidate configとし、診断後の限定改善実験でrelative表現の縮約を検証する。
4. Incremental experimentは順序依存が強く、leave-one-outで明確な主要結論が得られたため今回は主結果に加えない。

Full local artifacts: `artifacts/abl_001_*` ～ `artifacts/abl_007_*`、paired aggregate: `artifacts/semantic_ablation_analysis_20260830/`。  
Machine-readable tracked summary: [`experiments/baseline_validation_20260830/ablation_summary.json`](../../experiments/baseline_validation_20260830/ablation_summary.json)
