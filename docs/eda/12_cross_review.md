# Phase 5A cross-review

## Scope and review method

Workstreams A–I、共通view、再現用code、集約artifact、manifestを、担当外の3 reviewerが独立に確認した。reviewはproduction modelの採用判定ではなく、Phase 5A EDAのPIT、統計、競馬意味論が統合可能かを判定するものである。各reviewerの詳細記録はlocal artifactの`artifacts/eda_20260901/reviews/`に保存し、重大指摘を修正・再生成した後に再reviewした。

| Review | Main checks | Initial blocking findings | Remediation | Final verdict |
|---|---|---|---|---|
| PIT / leakage | future/current-race outcome、same-date update、market隔離、entity encoding、期間firewall | 共通predictive viewへの障害race混入、Gがpost-2022を含むfull frameを物理load | flat-only viewを再生成し、Gをcutoff-safe common viewとchunk-filtered OOT sourceへ変更。hashとmaterialized max dateを再検証 | **PASS** |
| Statistics / validation | denominator、race macro、同着、date-block interval、temporal replication、多重比較 | Gのrunner-micro集計、winner同着weight、interval estimand、E/F/Hのsupport・label・bin記録 | race-cell macro、co-winner `1/m`、同一estimand date bootstrap、effective supportとfixed definitionへ再生成 | **PASS** |
| Domain / semantics | 平地定義、starter/status、着順・時計・着差・上がり・通過順位、class transition、opponent semantics | 新馬と未勝利の混同、history 0の曖昧さ、passing/tie/frame/opponentの表現不整合 | `new_to_maiden`を独立化し、`0_observed_history`へ改名。recorded transition、tie共有、馬番/枠番、cross-race estimandを明記 | **PASS** |

## PIT / leakage review

### Findings and fixes

1. 初回共通viewには障害raceが含まれ、flat predictive state契約と不一致だった。canonical viewを再生成し、`runner_pre_race` / `outcomes`は450,340 rows・31,689 races、`historical_performance`は471,557 rows・33,240 races、障害row 0、post-2022 row 0とした。
2. Workstream Gは当初full-period pickleをload後にcutoffしていた。common cutoff-safe viewだけをsourceとし、OOT predictionはchunkごとに期間を絞ってからmaterializeする経路へ変更した。
3. 同日更新、future append、market隔離、direct ID非投入をcode、test、manifestで照合した。marketはWorkstream Hの明示的oracle joinに限られ、feature、calibration、候補採否には使用していない。
4. 修正前artifactは削除せず`artifacts/eda_20260901_pre_contract_fix/`へ保存した。canonical resultではない。

### Final assessment

PIT reviewerの最終判定は**PASS**である。これはretrospective PIT-C EDAの境界を承認するものであり、historical publication timestampを持つPIT-A、production feature、model採用を承認するものではない。

## Statistics / validation review

### Findings and fixes

1. Workstream Gのinteractionをrunner-micro平均からrace-cell平均後のrace-macroへ変更した。runner count、effective race count、date countを保存した。
2. 同着raceのproper-score targetはofficial co-winnerへ各`1/m`を割り当て、winner-conditioned集計でもrace weightを1に保った。
3. OOT intervalはdate blockごとにweighted numerator/denominatorを再集計し、point estimateと同じestimandをbootstrapした。
4. Workstream Eのfield-quality/performance関連を同一estimandの日別Spearmanへ変更し、race内associationではなくcross-race descriptive associationと明記した。
5. connection supportはstart数だけでなくeffective race/date countを記録し、Hのhistory availabilityは`0_observed_history`、`1`、`2_3`、`4_9`、`10_plus`に固定した。
6. canonical source、artifact-local script、manifest/output hashの一致を最終確認した。

### Final assessment

Statistics reviewerの最終判定は**PASS**である。残る注意点は、探索数が多いこと、repeated horse/entity依存を完全に消せないこと、小sliceのintervalが広いこと、2022が1年confirmationであること。これらはhypothesis registryのmultiple-comparison risk、uncertainty、sample supportへ引き継いだ。

## Domain / semantics review

### Findings and fixes

1. 平地はsurfaceだけでなくrace classの非障害も要求する。取消・除外はchoice setから外し、DNF/DQはstarterとしてstatusを保持する。
2. 新馬raceと「2013以降に観測されたprior JRA平地履歴0」は別概念である。新馬戦除外をEDAから支持しない。
3. class transitionでは`new_to_maiden`と`new_related_other`をsame/up/downから分けた。grade/class token不足と制度driftを制約として残した。
4. last 3F最速はtieを含む共有minimum、passing positionはfirst/last recorded segment、margin tokenは隣接到達group間であり、course phaseやwinnerからの累積秒と同一視しない。
5. opponent-only runner-relative値とrace-constant field quality、馬番と枠番、workstream間で異なるdistance taxonomyを明示的に分離した。

### Final assessment

Domain reviewerの最終判定は**PASS（重大0、軽微0）**である。今後も`one race = one choice set`、official orderとperformance contentの分離、opponent-onlyとfield qualityの分離、history 0と新馬classの分離を固定する。

## Integrated disposition

三者reviewのblocking findingはすべて修正し、影響するworkstreamとtop-level artifactを再生成した。Phase 5A EDAは統合可能である。ただし次は主張しない。

- EDA候補がproduction metricを改善すること
- final marketとの差が特定の欠損情報だけで説明できること
- 2022 confirmationがuntouched final holdoutであること
- Top3 auxiliary target、condition residual、race-wise choiceのいずれが最適であること
- retrospective PIT-Cの結果がprospective運用へそのまま一般化すること

最終判断は[EDA synthesis](14_eda_synthesis.md)と[next research roadmap](15_next_research_roadmap.md)へ統合した。優先候補は最大3件に限定し、まだ実行していない。
