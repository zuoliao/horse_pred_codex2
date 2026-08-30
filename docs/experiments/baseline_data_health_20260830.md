# Baseline data and evaluation health audit

**Audit date:** 2026-08-30  
**Raw fingerprint:** `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`  
**Decision surface:** 2024 development only。2025 retrospectiveは読み込まず、判断にも使用していない。

> この監査で旧Task 16 baselineを無効化する重大なpopulation bugを発見した。修正commit `810a003` 以後に再生成したartifactだけを新しい基準点として扱う。

## 1. Evidence: 障害race混入

rawの`course_type`は競走種別ではなく、障害raceでも実際の走路に応じて`芝`または`ダート`を持つことがある。全期間の`race_class`が`障害...`である1,633 raceのうち、924だけが`course_type=障害`で、709は`芝`190 / `ダート`519だった。

旧実装はsurfaceだけで平地判定したため、2024 developmentへ障害36 race / 388 runnersをscoring対象として混入させた。さらに学習期間中の障害結果がhorse、jockey、trainer、Elo stateを更新していたため、2024予測から36 raceだけを後処理で除く修正では足りない。

### Finding / action

- 平地を「race全行のsurfaceが芝・ダート、かつrace classに`障害`を含まない」と再定義した。
- `meta__is_flat_race`を明示し、feature builderとmodel-frame selectorで二重に検証する。
- 非平地raceはscoringだけでなく、全flat history stateの更新から除外する。
- 旧Task 16 artifact `mvp_baseline_20260830_task16_v2`は比較資料としても使用しない。

## 2. Evidence: 既知の欠損146 race

manifestの欠損rangeを展開し、rawに存在しないことを再確認した。欠損はすべて京都、14開催日に集中している。

| 年 | 欠損 | 日程・race | surface | starters |
|---:|---:|---|---|---:|
| 2015 | 2 | 1/18 R7, R11 | 芝1、ダート1 | 27 |
| 2017 | 36 | 5/27 R7–12、5/28全race、10/7全race、10/8 R1–6 | 芝17、ダート16、障害3 | 483 |
| 2024 | 108 | 7回京都1–9日、11/30～12/28の各12 race | 芝45、ダート58、障害5 | 1,481 |

2024分は[JRA公式の2024年番組](https://www.jra.go.jp/keiba/program/2024/index.html)と日別成績PDF（[7回京都1日](https://www.jra.go.jp/datafile/seiseki/report/2024/2024-7kyoto1.pdf)、[7回京都9日](https://www.jra.go.jp/datafile/seiseki/report/2024/2024-7kyoto9.pdf)、同じURL規則の2～8日）をrace単位で照合した。PDF最終summaryの出走延頭数1,481と、個別fieldの合計も一致した。PDFは一時領域でのみ確認し、repoへ保存していない。

### 2024 missing 103 flat races

| 条件 | 欠損内訳 |
|---|---|
| surface | 芝45、ダート58 |
| distance | ≤1399m 20、1400–1799m 33、1800–2199m 47、≥2200m 3 |
| class | 未勝利32、新馬17、1勝18、2勝18、3勝9、open/graded 9 |
| actual starter field band | ≤9頭 7、10–12頭 26、13–15頭 29、16頭以上 41 |
| starters | 1,419、平均13.777頭 |

### Finding: missingness is not random

2024の公式平地母集団を`raw 3,224 + missing 103 = 3,327`と再構築した。raw coverageは96.90%である。

- 欠損の100%が京都で、公式京都平地707 raceの14.57%を失う。
- 12月平地はraw 153 + missing 91 = 244 raceで、37.30%を失う。京都の12月は全欠損である。
- observed rawとmissingのshareは、ダート50.25%対56.31%、1800–2199m 37.62%対45.63%、新馬8.84%対16.50%、1勝class 28.38%対17.48%で異なる。
- G1 2、G2 1、G3 1、Listed 2を含み、年末2歳戦と上級raceも失う。
- 欠損raceに出走した馬の以後の履歴も欠けるため、影響は評価denominatorだけではない。

### Implication

bootstrap intervalは「観測済みかつ取消・除外のない平地race」に条件付きであり、この構造欠損の不確実性を含まない。LightGBMと単純baselineの大差は、bounded ranking metricでは欠損share 3.10 percentage pointsだけで覆りにくい。一方、旧artifactのBinaryとLambdaRankの約0.1 point差は容易に反転し得る。京都、12月、新馬等のsubgroup結論も弱い。

## 3. Evidence: 取消・除外raceのselection

正しいflat populationで、少なくとも1頭の取消・除外を含むraceは全期間2,048 / 43,128 = 4.75%、2024は173 / 3,224 = 5.37%だった。2024 primary scoringは3,051 raceで、公式再構築母集団3,327に対するcoverageは91.70%となる。

2024の取消・除外raceはscratch 81、excluded 97、両方5。nonstarterは181頭で、除外race当たり1.046頭だった。

| 条件 | 高いrate | 低いrate / 比較 |
|---|---|---|
| venue | 函館8.33%、札幌7.14%、京都6.79%、中山6.29% | 新潟3.59%、東京3.84% |
| month | 8月7.96%、6月7.06%、1月6.14% | 10月3.26%、3月3.38% |
| surface | ダート5.56% | 芝5.17% |
| class | open 8.95%、新馬7.37%、2勝6.00% | 1勝4.04%、未勝利4.85% |
| declared field | 16頭以上6.15% | 13–15頭4.24% |

scored raceのdeclared field平均13.748に対し、除外raceは14.029だった。除外raceの最終starter平均は12.983である。したがって現評価はopen/newcomer、北海道・京都、夏、大きいdeclared fieldから取消後に小さくなったraceを過少代表する。

## 4. Interpretation and decisions

### Primary policy

厳格な`T_prevday PIT-C` primaryでは、取消・除外を含むrace全体をscoring対象外とする現仕様を維持する。raw結果ページには取消・除外の発表timestampがなく、最終starter集合、field-relative値、LambdaRank group sizeを前日時点で知れないためである。

### Secondary sensitivity

取消・除外行だけを除くstarter-only評価は2024に173 raceを戻せるため、`post-scratch-field oracle sensitivity`として追加する価値がある。ただし実行可能な前日予測、primary metric、ROIとは解釈しない。将来timestamp付き出馬表・変更snapshotをprospective収集し、cutoffを定義できた時だけprimary候補になる。

### Go / no-go

1. 障害混入修正前のbaselineはno-go。
2. 修正版baseline再生成後は、同じ3,051 raceを使うpaired uncertaintyとfeature ablationを続行してよい。
3. Binary対LambdaRankの小差、京都/12月の条件性能はdecision-gradeとしない。
4. 全結果にcoverage warningを付け、2025 retrospectiveを仮説・feature・parameter選択に使わない。

## 5. Uncertainty and source restrictions

- 2015/2017/2024のmissing conditionsは公式日別成績の手作業・座標ベース照合による。モデルrawへの補完ではなく、selection-bias監査だけに使用した。
- JRAの[利用案内](https://www.jra.go.jp/use/)には著作権・二次利用上の制約がある。この監査では少数の公式PDFを一時取得してaggregateだけをrepoへ記録し、PDFやrace-level再配布datasetは保存していない。
- official populationとのcoverageは、manifestと日別成績から再構築した監査値であり、第三者sourceのlink数は取消馬を含む場合があるため採用しなかった。

Machine-readable summary: [`experiments/baseline_validation_20260830/data_health_summary.json`](../../experiments/baseline_validation_20260830/data_health_summary.json)
