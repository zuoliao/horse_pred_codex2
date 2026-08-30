# Evidence-led limited improvement experiments

## Selection boundary

- Hypothesis selection、feature selection、accept/reject判断は2024 developmentまでに限定する。
- 2025 retrospectiveは読込直後に除外し、いずれのfit、calibration、評価、判断にも使わない。
- 2026+ prospective finalを維持する。
- 各実験はcorrected Task 16と同じsplit、seed、LightGBM configを用い、記載した1仮説だけを変更する。

## Pre-registered experiments

### IMP-001: compact form-relative representation

**Evidence:** 15列の既存field-relative groupはpermutationで強く使われる一方、再学習dropで両modelの全point metricが改善した。30/90日decay mean finishとsame-surface mean finishは両modelの上位SHAP featureだった。

**Change:** 既存field-relative 15列を除き、上記3 absolute PIT featureのrace内percentile 3列だけを追加する。source valueはtarget raceの同一rowに既に存在し、label・odds・日付・IDを使わない。

**Primary comparison:** corrected 268-feature baselineとの2024同一race/date-block paired bootstrap。BinaryとLambdaRankのNDCG@3、Log Lossをprimary、top-1とBrierをsecondaryとする。

**Decision rule:** 少なくとも一方のmodelでNDCG@3またはLog Loss改善のpaired 95% intervalが0より上で、もう一方のprimary metricに明確な悪化がない場合のみ積極採用する。point estimateだけの微差は未解決とする。

### IMP-002: surface-conditioned Elo

**Evidence:** same-surface historyとElo/race-value groupはablation・SHAP・permutationで追加情報を示し、surface変更horseのtop選択hitが低かった。

**Pre-registered intent:** horse Eloを芝/ダート別に独立更新し、target surfaceのpre-race rating、field平均との差、race内percentileだけを追加する。既存global Eloと他268列は変えない。厳密な同日batch更新と障害除外を維持する。

**Required control:** 新cacheで追加3列を除いた旧268列のschema/valueとbaseline予測を先に再現する。再現できなければIMP-002を評価しない。

**Decision rule:** IMP-001と同じ。加えてsurface変更sliceの改善方向を診断するが、小標本slice単独では採用しない。

### IMP-003: field-size-band temperature calibration

**Evidence:** field size別にproper scoreとmarket gapが大きく異なる。単一temperatureが大頭数raceの確率集中度を十分表せない可能性がある。

**Pre-registered intent:** 2023 calibrationだけで既定field-size bandごとにtemperatureをfitし、2024へ固定適用する。raw scoreとranking順序は変更しない。band不足時だけglobal 2023 temperatureへfallbackする。

**Decision rule:** ranking metricは同一であることを必須とし、Log LossまたはBrier改善のpaired 95% intervalが0より上、かつ他方に明確な悪化がない場合にcalibration候補として採用する。ECEは補助診断に留める。

## Interpretation guardrails

- 同じ2024 development上で3仮説を見るため、多重比較とselection optimismが残る。採用は次のprospective期間で再検証する候補選択であり、finalな汎化証明ではない。
- final oddsはmarket oracle診断だけに用い、改善実験のfeature、calibration、選択基準、ROIには用いない。
- IMP-001、IMP-002、IMP-003を互いに合成したモデルは、この限定実験群とは別仮説になるため今回は自動的には作らない。
