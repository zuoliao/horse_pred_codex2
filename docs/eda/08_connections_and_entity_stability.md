# Workstream F: Connections and entity stability

## 1. Questions

本workstreamは、騎手・調教師情報について次を問う。

1. career、短期form、surface、distance、venue、class、馬齢、horse experience、休養条件の履歴は、race内で勝敗・走行内容と安定して関連するか。
2. 騎手変更、初騎乗・継続騎乗、新馬、長期休養馬という文脈は何を示すか。
3. entityの成績順位は翌年へ持続するか。どのsupport水準でraw rateの分散と平均回帰が問題になるか。
4. 現行connections 130列は情報量に対して冗長ではないか。また、累積統計の組合せがentity fingerprintへ近づく危険はないか。
5. direct IDを入力せず、smoothed rate、uncertainty、effective sample size、condition-specific deviationへ縮約する価値はあるか。

目的はproduction featureの追加ではなく、既存groupの強いablation寄与を分解するための仮説作成である。

## 2. Data scope

共通契約に従い、予測対象はeligibleなJRA平地starterだけとした。障害raceは独自assertで0件であり、初回common `connection_state`の汚染可能性を避けて、flat-only private cacheからstateを再構成した。

| 区分 | 対象期間 | runners | races | 用途 |
|---|---:|---:|---:|---|
| warm-up | 2013 | 48,293 | 3,324 | state初期化と固定prior推定のみ |
| discovery | 2014–2019 | 284,883 | 19,925 | exploratory |
| replication | 2020–2021 | 92,732 | 6,660 | replicated |
| confirmation | 2022 | 45,649 | 3,331 | confirmed / prioritization |

対象日は2013-01-05から2022-12-28まで、targetを使う分析は2014–2022の423,264 runners / 29,916 racesである。raw SHA-256は`270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`。2023以降、odds、人気、market情報は読み込んでいない。

## 3. Definitions

- **career raw rate**: target日より前のflat startにおける公式勝数 / start数。非完走は非勝者としてcountする。
- **EB rate**: 2013 flat warm-upの平均勝率0.06887を平均とし、有効標本数20のBeta priorで縮約した率。これはEDA用の固定例であり、prior強度20を最適値とはしない。
- **effective sample size**: 本分析では観測start数。decay weightを用いる場合は、将来別途定義が必要である。
- **short-term form**: 過去30 / 90 / 365日の勝率を同じfixed priorで縮約したもの。
- **condition state**: surface、4距離band、venue、race class band、horse age band、horse career-start band、およびtrainerについてlong-layoff文脈が一致する過去startだけのEB rate。
- **condition-specific deviation**: condition EB rate − career EB rate。supportと必ず対にする。
- **first / repeat ride**: 同じhorse–jockey pairのprior-date startが0 / 1以上。
- **jockey change**: horseの直前flat startと今回のjockey keyが異なること。初出走は欠損扱い。
- **race association**: race内でfeature rankと`1-(finish-1)/(field_size-1)`のrankをSpearman相関し、race macro平均する。値が高いほど同一race内の高いconnection stateが上位着順と対応する。
- **year stability**: 年tのentity勝率rankと年t+1の勝率rankのSpearman相関。両年のminimum startsを同じ閾値で課す。

ID・氏名はjoin/state keyとしてのみ用い、数値化したmodel featureにはしていない。

## 4. Methods

1. flat surface（turf / dirt）かつ非障害raceであること、`(race_id, horse_id)`と`(horse_id, date)`が一意であること、最大日が2022-12-31以下であることをassertした。
2. 日ごとに全race・runnerのstateをemitし、その日の全結果を後から更新した。同日race間、同一race内の結果は入力へ入らない。
3. jockeyとtrainerについてcareer、30 / 90 / 365日、条件別state、posterior SDを独立再構成した。horse–jockey pairと前回jockeyもprior-dateだけで更新した。
4. race内associationとtop-choice hit rateを計算し、95% intervalは開催日block 400 bootstrapで得た。単変量診断であり、他featureを条件付けたincremental utilityではない。
5. 年次entity集計では1 / 10 / 30 / 100 startsのsupport gateを比較し、翌年rank correlationとprior-year decileから翌年rateへの平均回帰を確認した。
6. production実装からconnections 130列の構造を監査した。独立再構成した64 diagnostic列の50,000-row deterministic sampleではSpearman redundancyも確認した。

多くの関連を探索しているため、intervalは多重比較補正済みのconfirmatory検定ではない。方向の期間再現、support、意味、PIT安全性を優先する。

## 5. Descriptive findings

### 5.1 Connectionsは明確なrace内signalを持つ

下表の分母母集団は各期間の全runner / raceである。効果の実計算は「少なくとも2件の非欠損かつrace内で値が変化するrace」に限定される。EB列のrunner欠損は0だが、同値だけのraceは相関から除外されるため、scope race数は厳密な相関分母の上限である。

| Feature | Discovery 284,883 / 19,925 | Replication 92,732 / 6,660 | Confirmation 45,649 / 3,331 | 判定 |
|---|---:|---:|---:|---|
| jockey career EB | .276 [.272, .280] | .262 [.255, .269] | .258 [.250, .268] | 3期間で正方向 |
| jockey 30d EB | .183 [.178, .188] | .183 [.176, .190] | .185 [.176, .195] | 短期だけでは弱い |
| jockey 90d EB | .231 [.227, .235] | .224 [.217, .230] | .234 [.226, .243] | 安定した中間signal |
| jockey 365d EB | .272 [.268, .276] | .258 [.252, .263] | .263 [.254, .273] | careerとほぼ同水準 |
| trainer career EB | .189 [.184, .193] | .180 [.172, .186] | .190 [.179, .202] | 3期間で正方向 |
| trainer 30d EB | .087 [.082, .091] | .087 [.080, .096] | .103 [.094, .113] | 単独signalは弱い |
| trainer 90d EB | .134 [.129, .139] | .123 [.116, .131] | .142 [.132, .152] | 正方向だがcareer未満 |
| trainer 365d EB | .179 [.174, .183] | .169 [.161, .177] | .186 [.176, .197] | careerに近い |

値はrace-macro Spearman平均 [date-block 95% interval]。集計単位はrace、bootstrap単位はdate、欠損は全EB列0。top-choice hit rateもjockey careerで.183 / .203 / .187、trainer careerで.147 / .139 / .137と一貫して一様選択を上回った。ただしこれはconnection単独の順位であり、現modelの確率ではない。

**Observation:** jockey associationはtrainerより大きく、30日windowはcareer / 365日より大幅に弱い。90日は中間にある。  
**Interpretation:** connection能力には持続成分があり、極短期勝率は標本分散が大きい。  
**Hypothesis:** 多数の短期raw rateを同列に置くより、長期縮約level、短期deviation、support / uncertaintyへ分解した方がよい可能性がある。

### 5.2 条件別率は方向を維持するが、単変量ではcareerを明確に超えない

jockey surface EBのrace associationはdiscovery / replication / confirmationで.272 / .258 / .255、distanceは.261 / .247 / .246、venueは.250 / .240 / .247だった。trainer surfaceは.185 / .176 / .186、distanceは.171 / .168 / .182、venueは.155 / .156 / .167だった。すべて正方向に再現した一方、ほぼすべてcareerまたは365日率と同水準以下である。

trainer experience EBだけは.170 / .185 / .198と後半ほど強く、2022ではtrainer career .190をわずかに上回った。しかしdiscoveryでは逆であり、定義・時代構成・horse experienceとの交絡を分離していない。trainer layoff-context EBも.190 / .186 / .194でcareerに近く、独立増分とは言えない。

**Observation:** 条件別率は有意味なsignalを保つが、無条件careerとの強い重複がある。  
**Interpretation:** 「condition-specific rateそのもの」より、「careerからの縮約付き偏差」とsupportの方が問いに合う。  
**Hypothesis:** 条件別levelを一括追加せず、surfaceまたはhorse-experience deviationを一群ずつretrained ablationすべきである。

### 5.3 support不足はrunner全体では少ないが、raw rateの境界問題は残る

career startsが50未満の割合はjockeyで1.67% / 0.87% / 1.71%、trainerで0.81% / 0.89% / 0.78%だった（discovery / replication / confirmation）。0-startはjockey 418 / 86 / 75 runners、trainer 162 / 38 / 23 runners。raw career rateの欠損数はこれと一致し、EBは全期間0欠損だった。posterior SDは0-startで約.055、20–49 startsで約.030、50+ではjockey .014 / .012 / .011、trainer .012 / .015 / .015へ縮小した。

この少ないconnection cold-start率を、「新馬が容易」と解釈してはいけない。horse history 0でも騎手・調教師には通常十分な履歴があり、connectionが新馬で利用できる主要signalになるという意味である。

### 5.4 初騎乗・騎手変更のraw差は大きいが、選択効果が強い

初騎乗の勝率は.0620 / .0632 / .0645、継続騎乗は.0787 / .0814 / .0819だった。騎手変更は.0625 / .0637 / .0640、据置は.0846 / .0877 / .0894で、方向は3期間一致した。対象は初騎乗147,379 / 47,788 / 23,212 runners、騎手変更164,176 / 52,490 / 25,573 runnersでsupportは十分である。

これは馬の能力、前走内容、陣営の選択、乗替り理由を調整していないraw associationである。継続騎乗効果や騎手変更の因果効果とは呼ばず、後続の条件付き診断候補に留める。新馬のraw勝率は.0660 / .0673 / .0689、180日以上休養馬は.0478 / .0478 / .0524だったが、同様にbase-rate差を含む。

### 5.5 130列には構造的冗長性がある

現行130列は、2 entities × 13 blocks（career、4 count windows、5 calendar windows、3 decay windows）× 5 statistics（starts、wins、win rate、completed、mean finish）である。`win_rate=wins/starts`、last-1でのwinsとwin rate、非完走が少ない場合のstartsとcompletedなど、決定論的または近似的関係を含む。

独立再構成した64 diagnostic列の50,000-runner sampleでは、絶対Spearman 0.90以上が27 pairs、0.95以上が11 pairsあった。これは130列全体の完全なcluster解析ではないが、冗長性を否定できない。多数の累積・条件統計の組合せがentityごとに固有に近づくfingerprint riskも残る。一方で、これだけから直接ID memorizationが起きたとは言えない。

## 6. Temporal replication

年tからt+1のrank安定性は、十分なstart数を持つentityで高かった。

| Entity / minimum starts（両年） | Discovery: 6 year-pairs | Replication: 2 year-pairs | Confirmation: 1 year-pair |
|---|---:|---:|---:|
| jockey / 30+ | .789（年別.722–.856、平均108 entities） | .775（.771–.778、108） | .776（109） |
| jockey / 100+ | .818（.790–.849、92） | .816（.812–.820、96） | .776（95） |
| trainer / 30+ | .700（.654–.755、195） | .722（.718–.726、186） | .788（187） |
| trainer / 100+ | .713（.659–.767、186） | .728（.728–.728、180） | .781（180） |

値は年次raw win-rate rankのSpearman平均（括弧内はyear-pair range、1 pairではrangeなし）。entityが反復するため、これは独立標本のCIではなく時間方向のばらつきである。1-start gateではraw rank correlationがjockey .740 / .804 / .811、trainer .741 / .790 / .798だったが、rare entityのrankとsupport構成が混在する。

prior-year下位decileから翌年への平均回帰は全期間で見られた。例としてdiscoveryのjockey最下位decileは.0040→.0240、最上位は.1709→.1623、trainerは.0181→.0265、.1380→.1233だった。replicationでもjockey .0061→.0204 / .1699→.1643、trainer .0192→.0306 / .1447→.1270と同方向だった。したがってentity rankには持続性があるが、極端なraw rateをそのまま将来期待値とはみなせない。

30日signalが3期間すべてcareer / 365日signalより弱いことは、短windowの高分散を示す診断と整合する。ただし月次分散そのものをentity階層modelで推定しておらず、短期formが無価値という結論ではない。

## 7. Uncertainty

- race associationの95% intervalはdate block bootstrap 400回。runnerを独立bootstrapしていない。
- 同一jockey / trainer / horseの反復依存は残る。年次rank表はyear-pair rangeを示すが、entity-cluster CIではない。
- race association表のscope race数にはfeatureがrace内同値のraceも含む。相関の実分母はnonmissingかつrace内変動のあるraceであり、local CSVはその件数を別列で保存していない。cross-reviewでの修正候補である。
- 22の主要signalと複数support gateを探索したため、個々のintervalをconfirmatory p-valueとして扱わない。
- 条件別rateはconditionの出現頻度とentity選択に依存する。特にclass、venue、debut、layoffはassignment confoundingが強い。
- 固定prior effective n=20は説明用の一案であり、最適化していない。rawとEBの小差は縮約方式の採否を決めない。
- 2022は1年のみで、制度変化・COVID期・開催構成差を完全には切り分けられない。

## 8. Failure cases

1. 初回common `connection_state`には障害race混入の可能性があったため使用せず、flat-only cacheから独立再構成した。最終manifestで障害0、最大日2022-12-28を確認した。
2. 初回集計で非完走runnerのnullable winner labelが累積勝数をNaN化する問題を検出した。非完走は公式非勝者0として再実行し、全EB列の欠損が0になった。最初の結果は破棄した。
3. trainerはcanonical IDではなくraw nameをkeyにしている。443 raw keysは空白除去後も443だったが、改名・所属変更・同姓同名を検出できない。
4. jockeyは384 IDs / 384表示名で、複数表示名を持つIDが1、複数IDを持つ表示名が1あった。個別名は出力しておらず、同一人物かsource anomalyかは断定できない。
5. horse–jockey関係はprior flat startsだけを数えるため、地方・海外・障害での騎乗歴を完全には表さない。
6. 130 production列すべてを同じpre-2023 frame上で再学習ablationしたわけではない。構造監査と64 diagnostic列の相関だけで、削除候補を確定しない。

## 9. Leakage / PIT considerations

- 全stateは`history_date < target_date`。同日の全prediction rowをemit後に更新した。
- current-raceの着順、時計、着差、上がり、通過順位、odds、人気はfeatureへ入れていない。
- jockey IDとtrainer名はstate keyだけであり、feature値として出力していない。horse–jockey pair keyもprior start countへのjoinだけに使った。
- 2013 priorはwarm-up結果だけから固定し、2014以降の全期間正規化はしていない。
- condition stateはそのentity自身の過去結果のみで、future opponent resultやtarget encoding fold混入はない。
- historical jockey assignmentはPIT-C event reconstructionであり、実際の公表時刻・直前変更時刻を保持していない。prospective運用では`published_at`が必要である。
- final marketはjoinしていない。既存2024 ablationは凍結済みの過去証拠として参照しただけで、本EDAで2024 prediction/resultを再読していない。

## 10. Modeling implications

| Component | Decision after EDA | Evidence and limitation |
|---|---|---|
| connection group全体 | **retain** | 既存2024 source-family ablationではdrop時にBinary NDCG −.0284 [−.0382, −.0187]、Top-1 −.0210 [−.0366, −.0059]。本EDAでもpre-2023 signalが再現。ただし130列全部の必要性ではない |
| career / 365d level | **retain then simplify** | jockey .258–.276、trainer .180–.190のrace association。互いに近く冗長性あり |
| 30d raw/EB form | **redesign** | 3期間一貫して長期levelより弱い。短期deviation、support、uncertaintyとして検証する価値はある |
| condition-specific level | **simplify / test deviation** | 正方向だがcareerを明確に超えず、単変量の情報新規性は弱い |
| starts / posterior uncertainty | **retain explicitly** | rare supportでposterior SDが大きく、平均回帰がある |
| jockey change / first-repeat | **defer conditional diagnostic** | raw差は大きく再現するが、horse abilityとassignment confoundingが強い |
| direct entity ID | **do not add** | key auditの不完全性とmemorization risk。既存方針を維持 |
| 130-column representation | **redesign before adding columns** | deterministic relationsと高相関pairs。group ablationの強さは全列の増分価値を意味しない |

既存のconnections source-family knockoutは「connectionsに情報がある」という再学習estimandである。本EDAのunivariate association、SHAP/permutation依存、個々のwindowの増分utilityは別物であり、混同しない。

## 11. Candidate hypotheses

### F-H01 — Hierarchical connection compression（priority A）

- **Evidence:** career / 365日signalが時間再現し、130列に構造的冗長性と高相関pairsがある。平均回帰も明瞭。
- **Proposed experiment:** entityごとに長期EB level、90日または365日からの短期deviation、effective n、posterior uncertaintyだけへ事前固定して、現130列とのretrained rolling-origin ablationを行う。
- **Expected direction:** 同等以上のproper scoreを保ちながら列数とfingerprint riskを減らす。
- **PIT risk:** low。priorはfold train / warm-upだけから固定し、同日更新禁止。
- **What would falsify it:** rolling foldsでLog Loss / Brierが悪化し、ranking改善もない。

### F-H02 — Trainer horse-experience deviation（priority B）

- **Evidence:** trainer experience EBは.170 / .185 / .198で方向再現し、後半ほどcareerとの差が相対的に大きい。新馬にはhorse historyがない。
- **Proposed experiment:** `trainer experience-specific EB − trainer career EB`、support、uncertaintyの3要素を一groupとして、debut vs experiencedの事前固定区分で検証する。
- **Expected direction:** 新馬・低履歴raceでのranking / proper-score改善。
- **PIT risk:** low。trainerのprior-date成績のみ。ただしtrainer名key品質はconditional。
- **What would falsify it:** discovery以外のrolling foldsで方向が揃わない、またはcareer levelとの再学習増分がない。

### F-H03 — Conditional jockey-change diagnostic（priority C）

- **Evidence:** change / unchangedのraw勝率差が3期間で同方向、supportも大きい。
- **Proposed experiment:** feature追加前に、horse prior performance、horse ability rating、rest、class/surface transition、jockey EB差で調整したpre-2023 shallow diagnosticをpreregisterする。
- **Expected knowledge:** 乗替り自体、jockey quality差、馬の選択効果のどれが観察差を説明するか。
- **PIT risk:** conditional。historical assignmentの公表時刻が不明。
- **What would falsify it:** 調整後の方向が年ごとに反転する、またはassignment timestampを保証できない。

これらはhypothesis registry候補であり、production実装も2024評価も行っていない。

## 12. What not to conclude

- connections 130列のgroup ablationが強いことから、130列すべてが必要とは言えない。
- jockeyの単変量associationがtrainerより高いことから、騎手の因果効果が大きいとは言えない。強い馬への騎乗選択を含む。
- 30日率が弱いことから、短期formが無価値とは言えない。長期levelからのdeviationや他featureとのinteractionは未検証である。
- 条件別率がcareerを超えないことから、適性差がないとは言えない。縮約なしlevelの比較は条件偏差の適切な検定ではない。
- 継続騎乗・据置の勝率が高いことから、乗替りを避ければ勝率が上がるとは言えない。
- connection cold-startが2%未満であることから、新馬予測が簡単、または新馬を学習から除外すべきとは言えない。
- 年次rank correlationが高いことから、将来rateが固定とは言えない。上下decileには平均回帰がある。
- key auditで不整合が少ないことから、trainer nameがcanonical IDと同等とは言えない。
- 64列sampleの相関から、production 130列の最適な削除集合は決められない。
- 本EDAはfeature/model acceptanceではない。次のretrained rolling-origin実験は人間レビュー後にのみ開始する。

Local reproducibility artifacts: `artifacts/eda_20260901/workstreams/f_connections/` (`analyze.py`, aggregate CSV/JSON, manifest). Raw・runner-level state・ID mappingはGitへ追加していない。
