# Corrected baseline uncertainty analysis

**Population:** corrected 2024 development, strict PIT-C no-nonstarter flat races  
**Support:** 3,051 races / 41,946 runners / 106 race dates  
**2025 retrospective:** not loaded into metric construction and not used for any decision

## Result

Corrected LightGBM Binary and LambdaRank both beat Uniform and History-rate by margins whose paired 95% intervals are favorable for all four primary metrics. Binary and LambdaRank cannot be distinguished at this sample size: their paired interval crosses zero for NDCG@3, Top-1, race Log Loss, and race Brier under every tested block length.

| Method | NDCG@3 [95% CI] | Top-1 [95% CI] | Log Loss [95% CI] | Brier [95% CI] |
|---|---:|---:|---:|---:|
| Uniform | 0.1859 [0.1752, 0.1969] | 0.0793 [0.0699, 0.0891] | 2.5949 [2.5766, 2.6127] | 0.9222 [0.9204, 0.9238] |
| History-rate | 0.2546 [0.2419, 0.2680] | 0.1242 [0.1111, 0.1381] | 2.5453 [2.5119, 2.5796] | 0.9111 [0.9062, 0.9161] |
| LightGBM Binary | 0.4873 [0.4780, 0.4964] | 0.2814 [0.2665, 0.2957] | 2.0898 [2.0638, 2.1157] | 0.8312 [0.8247, 0.8380] |
| LightGBM LambdaRank | 0.4864 [0.4761, 0.4963] | 0.2804 [0.2650, 0.2954] | 2.0892 [2.0620, 2.1159] | 0.8301 [0.8230, 0.8370] |

## Method

Primary intervalは、時系列順に並べたJRA race dateを4日単位で循環moving-block resamplingしたpaired bootstrap 10,000回である。各sample dateに属する全raceを同時に含め、日ごとのrace数が異なるためdate meanの単純平均ではなくrace metric総和 / sampled race数を計算した。全methodに同じdrawを使った。

感度分析はdate block 1、2、8日、およびrace IID bootstrapを同じ10,000回で実施した。Binary対LambdaRankのintervalは全方式・全primary metricでzeroを跨いだ。intervalは2024評価sampleの不確実性であり、年を跨ぐdriftや構造欠損を表さない。

ECEはsample-adaptive quantile binに依存するためdecision-grade intervalを付けなかった。確率品質の判断はproper scoring ruleであるLog LossとBrierを主に用いた。

## Paired comparisons

表の値はpositiveほどcandidateが良いよう、Log Loss/Brierだけ符号を反転した改善量である。

| Comparison | NDCG@3 | Top-1 | Log Loss improvement | Brier improvement |
|---|---:|---:|---:|---:|
| Binary − History | +0.2327 [0.2175, 0.2464] | +0.1572 [0.1395, 0.1745] | +0.4555 [0.4161, 0.4933] | +0.0799 [0.0718, 0.0875] |
| LambdaRank − Binary | −0.0009 [−0.0059, 0.0040] | −0.0010 [−0.0112, 0.0084] | +0.0006 [−0.0075, 0.0088] | +0.0012 [−0.0018, 0.0042] |

LightGBM対Historyの全resampleで4 metricが改善方向だった。LambdaRank対Binaryは、resampleで改善した割合がNDCG 36.0%、Top-1 41.9%、Log Loss 55.2%、Brier 77.6%とmetric間で方向も揃わない。

## Temporal and condition stability

- Binaryは12か月すべてでHistory-rateより高いNDCG@3を維持した。
- Binary NDCG@3は11月0.4560と8月0.4685が低く、10月0.5099が高い。ただし12月は公式平地の37.30%、京都12月の全raceを欠くため、月比較を完全母集団へ一般化できない。
- LambdaRank − Binary NDCGはQ1 +0.0049、Q2 +0.0060、Q3 −0.0063、Q4 −0.0103で、年後半に符号が反転した。これはmodel familyの優劣を確定するより、drift/condition interaction仮説を示す。
- Binary NDCG@3は16頭超0.4380、13–16頭0.4555に対し、小頭数0.5835。field sizeでrace難度が大きく違う。
- class別はopen/special 0.4268、newcomer 0.4564、class 3 0.4602が弱く、未勝利0.5052、class 2 0.5135が高い。
- 競馬場別は東京0.5329に対し、札幌0.4578、福島0.4608、新潟0.4633。supportとmissingnessを伴うためpoint estimateとして扱う。

Dead heat 5 raceを除いた感度分析でも、Binary NDCG@3 0.4874、LambdaRank 0.4864で結論は変わらなかった。

## Decision

1. corrected LightGBM baselineは、観測済みstrict PIT-C population上で単純baselineを明確に上回る基準点として採用できる。
2. BinaryとLambdaRankは現時点で実質tie。単純で確率目的に直接対応し、NDCG@3が僅かに高いBinaryをworking referenceとするが、LambdaRankをrejectしない。
3. 次の比較は同一3,051 raceのpaired designを維持する。
4. 京都年末block欠損と取消・除外selectionはbootstrap外のsystematic uncertaintyとして別掲する。

Machine-readable summary: [`experiments/baseline_validation_20260830/uncertainty_summary.json`](../../experiments/baseline_validation_20260830/uncertainty_summary.json)
