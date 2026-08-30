# Evidence-led limited improvement experiments

## Selection boundary

- Hypothesis selection、feature selection、accept/reject判断は2024 developmentまでに限定する。
- 2025 retrospectiveは読込直後に除外し、いずれのfit、calibration、評価、判断にも使わない。
- 2026+ prospective finalを維持する。
- 各実験はcorrected Task 16と同じsplit、seed、LightGBM configを用い、記載した1仮説だけを変更する。

## Pre-registered experiments

### IMP-001: compact form-relative representation

**Evidence:** 15列の既存field-relative groupはpermutationで強く使われる一方、再学習dropで両modelの全point metricが改善した。30/90日decay mean finishとsame-surface mean finishは両modelの上位SHAP featureだった。

**Change:** 既存field-relative 15列を除いた`abl_006`構成へ、`decay_90d__mean_finish`のrace内percentile 1列だけを追加する。source valueはtarget raceの同一rowに既に存在し、label・odds・日付・IDを使わない。

**Primary comparison:** `abl_006_drop_field_relative`との2024同一race/date-block paired bootstrap。corrected 268-feature baselineはsecondary controlとする。BinaryとLambdaRankのNDCG@3、Log Lossをprimary、top-1とBrierをsecondaryとする。

**Decision rule:** probability経路はLog Loss改善のpaired 95% interval下限`>0`かつpoint改善`>=.002`、Brier非悪化、NDCG差`>=-.002`、top-1差`>=-.005`。ranking経路はNDCG改善のinterval下限`>0`、Log Loss悪化`<=.002`、Brier悪化`<=.001`、top-1差`>=-.005`。どちらにも入らなければinconclusive/rejectとし、268 baselineより良いだけでは追加1列を採用しない。

**Pre-run amendment:** 当初は相関の高い30日・90日・same-surfaceの3 percentileを同時追加する案だった。独立design reviewで、268 baselineだけでは15列dropと3列addが交絡し、候補source間の2024 Spearman相関も30–90日`.951`、90日–same-surface`.851`と高いと判明した。cleanな登録実行前に1列・drop-control比較へ縮約した。変更中worktreeで作られた3列版artifactは正式評価に使わない。

### IMP-002: surface-conditioned Elo

**Evidence:** same-surface historyとElo/race-value groupはablation・SHAP・permutationで追加情報を示し、surface変更horseのtop選択hitが低かった。

**Pre-registered intent:** horse Eloを芝/ダート別に独立更新し、target surfaceのpre-race rating、field平均との差、race内percentileだけを追加する。既存global Eloと他268列は変えない。厳密な同日batch更新と障害除外を維持する。

**Required control:** 新cacheで旧268列の名前・順序・値が一致すること、同cacheから旧268列だけで再fitしたcontrolが旧artifactを許容誤差内で再現すること、candidateとの差がsurface 3列だけであることを確認する。

**Decision rule:** Log Loss改善のpaired 95% interval下限`>0`をprimaryとし、Brier非悪化、NDCG差`>=-.002`、top-1差`>=-.005`をguardrailとする。surface変更sliceの改善方向はmechanism診断だけに使い、小標本runner slice単独では採用しない。Eloのinitial=`1500`、K=`24`、scale=`400`、同日batch update、芝/ダート分離、障害更新なしを固定する。

### IMP-003: field-size-band temperature calibration

**Evidence:** field size別にproper score、ECE、market gapが異なる。ただしuniform比のLog Loss改善はsmall `.398`からvery-large `.570`へ大きく、単純に「大頭数ほど相対性能が悪い」とはいえない。より限定した仮説は、small-fieldを含めfield sizeにより2023 optimal temperatureが異なるか、である。

**Pre-registered intent:** 2023 calibrationだけで固定band `<=9 / 10–13 / 14–16 / 17–18`ごとにtemperatureをfitし、2024へ固定適用する。2023 race数は`296 / 915 / 1683 / 270`。最小200 raceを満たさないbandだけglobal 2023 temperatureへfallbackし、2024を見てbandを統合しない。raw scoreとranking順序は変更しない。

**Decision rule:** ranking metricの完全一致を必須とする。per-race NLL/Brierの4-date block paired bootstrapで、Log Loss改善interval下限`>0`かつpoint改善`>=.002`、Brier非悪化を満たす場合に採用する。200 race以上のbandでLog Lossが`.01`超悪化した場合は保留/棄却。ECEは非加法なので補助reliability診断に留め、単独採用根拠にしない。

## Results

| Experiment | Result | Decision |
|---|---|---|
| IMP-001 | drop-control比でBinary NDCG `-.00739` / LL改善`-.01434`、Ranker `-.00440` / `-.00751`。主要95% intervalも悪化側 | reject |
| IMP-002 | cache旧268列は全533,853 rowsで完全一致。Binaryは未解決。Ranker NDCG `+.00364 [+.00005,+.00714]`、proper score非悪化 | Ranker ranking pathをaccept。ただし全体bestではない |
| IMP-003 | ranking完全一致。Binary LL `+.00040`悪化、Ranker `+.00070`悪化。ECEは改善したがBrierも悪化 | reject |

full resultとbest config判断は[統合結論](baseline_validation_conclusions_20260830.md)および[機械可読summary](../../experiments/baseline_validation_20260830/improvement_summary.json)をsource of truthとする。

## Interpretation guardrails

- 同じ2024 development上で3仮説を見るため、多重比較とselection optimismが残る。採用は次のprospective期間で再検証する候補選択であり、finalな汎化証明ではない。
- final oddsはmarket oracle診断だけに用い、改善実験のfeature、calibration、選択基準、ROIには用いない。
- IMP-001、IMP-002、IMP-003を互いに合成したモデルは、この限定実験群とは別仮説になるため今回は自動的には作らない。
- nominal intervalは3仮説×2 model familyを探索したselection optimismを除かない。すべて同じ2024の3,051 race、4-date moving block、10,000 resamplesで比較し、採用候補も2026+ prospectiveで再検証する。
