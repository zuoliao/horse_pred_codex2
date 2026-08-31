"""Build non-recoverable, tracked Phase 5A EDA summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from horse_pred.eda import public_data_contract_summary

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "artifacts/eda_20260901"
OUTPUT = ROOT / "experiments/eda_20260901"


HYPOTHESES = [
    {
        "hypothesis_id": "EDA-S01-RACE-VALUE-2AXIS",
        "question": "過去走の走行内容と当時の相手水準を別軸で履歴化すると予測が改善するか",
        "evidence": "performance residualとpre-race field qualityの関連がD/R/Cで正、既存表現は両者を十分分離しない",
        "discovery_period": "2014-2019",
        "replication_result": "daily Spearman 0.221/0.156/0.181 (D/R/C), all positive",
        "sample_support": "270686/88241/43400 runners; 18966/6356/3176 races",
        "expected_direction": "proper scoreとrankingの両方を改善",
        "information_source": "horse+opponent",
        "target": "Binary+ranking",
        "proposed_change": (
            "condition-adjusted performance residualとrace-constant field qualityを別々の90日履歴として各1列"
        ),
        "PIT_status": "safe",
        "leakage_risk": "low",
        "redundancy_risk": "medium",
        "implementation_cost": "medium",
        "expected_value": "過去race valueの二つの意味を分離できる",
        "validation_plan": "rolling-originで2列を1groupとしてretrained ablation; 2024/2025不使用",
        "priority": "S",
        "status": "ready-for-experiment",
    },
    {
        "hypothesis_id": "EDA-S02-RACEWISE-CHOICE",
        "question": "race-wise probability objectiveはpost-softmax Binary/Rankよりcoherent probabilityを改善するか",
        "evidence": "one-winner choice-set構造、約24%のfamily disagreement、Binary/Rank差はmarket gapより小さい",
        "discovery_period": "2014-2019 target structure; 2020-2022 OOT error",
        "replication_result": "choice-set base structure and disagreement stable across all registered periods",
        "sample_support": "28498 discovery-to-confirmation choice sets; 9532 OOT races",
        "expected_direction": "Log Loss/Brier改善、ranking大幅悪化なし",
        "information_source": "race choice set",
        "target": "race-wise probability",
        "proposed_change": "同一features/splitsの線形conditional logitまたはtop-choice Plackett-Luce baseline",
        "PIT_status": "safe",
        "leakage_risk": "low",
        "redundancy_risk": "low",
        "implementation_cost": "medium",
        "expected_value": "objective mismatchの寄与を切り分ける",
        "validation_plan": "rolling folds、fold内calibration、Binary/Rankとpaired比較",
        "priority": "S",
        "status": "ready-for-experiment",
    },
    {
        "hypothesis_id": "EDA-S03-PERFORMANCE-TARGET",
        "question": "condition-adjusted continuous performance targetはwinner/top3 labelより内容を保持するか",
        "evidence": "3-4着境界は連続、rankとtime gapは約0.995相関、fixed clock thresholdはdrift",
        "discovery_period": "2014-2019",
        "replication_result": "rank/time relation replicated; condition residual coverage >99.6%",
        "sample_support": "403855 strict runners; 28498 races",
        "expected_direction": "走行内容学習を改善しrace内順位へ転用可能",
        "information_source": "historical performance",
        "target": "continuous performance",
        "proposed_change": "fold-train条件補正residualをHuber regressionしrace内順位化・確率化",
        "PIT_status": "conditional",
        "leakage_risk": "medium",
        "redundancy_risk": "low",
        "implementation_cost": "high",
        "expected_value": "winner以外の教師情報の価値を直接測る",
        "validation_plan": "補正器をfold内fit、rolling OOT、同一NDCG/LogLoss/Brier評価",
        "priority": "S",
        "status": "ready-for-experiment",
    },
    {
        "hypothesis_id": "EDA-A01-TRANSITION-RELIABILITY",
        "question": "condition switch時にpast-performance stateの信頼度を下げる表現が有効か",
        "evidence": "same surface/venue/directionとdistance差200m以内でperformance persistenceが高い",
        "discovery_period": "2014-2019",
        "replication_result": "surface and venue direction replicated in 2020-2021 and 2022",
        "sample_support": "403855 runners; 28498 races",
        "expected_direction": "surface/distance switch sliceのloss改善",
        "information_source": "horse+race context",
        "target": "Binary+ranking",
        "proposed_change": "surface switchかdistance bandの一方だけをstate reliability interactionとして検証",
        "PIT_status": "safe",
        "leakage_risk": "low",
        "redundancy_risk": "medium",
        "implementation_cost": "low",
        "expected_value": "適性平均と履歴信頼度を分離",
        "validation_plan": "一変換ずつrolling retrained ablation",
        "priority": "A",
        "status": "proposed",
    },
    {
        "hypothesis_id": "EDA-A02-CONNECTION-COMPRESSION",
        "question": "connections 130列を階層的に縮約して安定性を保てるか",
        "evidence": "career signalはD/R/Cで安定、64診断列に|rho|>=.90が27 pair、raw rateは平均回帰",
        "discovery_period": "2014-2019",
        "replication_result": "jockey career rho .276/.262/.258; trainer .189/.180/.190",
        "sample_support": "423264 runners; 29916 races",
        "expected_direction": "proper score維持または改善、冗長性低下",
        "information_source": "connection",
        "target": "all current families",
        "proposed_change": "long-term EB level、short deviation、effective n、uncertaintyへ縮約",
        "PIT_status": "safe",
        "leakage_risk": "low",
        "redundancy_risk": "low",
        "implementation_cost": "medium",
        "expected_value": "entity fingerprint riskと列数を減らす",
        "validation_plan": "現130列とのrolling retrained equivalence/noninferiority比較",
        "priority": "A",
        "status": "proposed",
    },
    {
        "hypothesis_id": "EDA-A03-LAST3F-RELATIVE",
        "question": "過去race内last-3F percentileは着順以外の持続signalを加えるか",
        "evidence": "winner fastest last3Fは37-41%のみ、prior last3Fと次走はrho約.31",
        "discovery_period": "2014-2019",
        "replication_result": "direction replicated in 2020-2021 and 2022",
        "sample_support": "last3F missing only 3 among completed flat performances",
        "expected_direction": "ranking改善、proper scores非悪化",
        "information_source": "horse performance",
        "target": "Binary+ranking",
        "proposed_change": "race-relative last3F percentileの単一decay history",
        "PIT_status": "safe",
        "leakage_risk": "low",
        "redundancy_risk": "medium",
        "implementation_cost": "low",
        "expected_value": "着順と異なる末脚内容を測る",
        "validation_plan": "既存SEC-3F証拠と定義を照合後、重複しなければ単独rolling ablation",
        "priority": "A",
        "status": "proposed",
    },
    {
        "hypothesis_id": "EDA-B01-WORKLOAD-INTERACTION",
        "question": "低自由度rest×recent startsは非線形な負荷構造を表すか",
        "evidence": "14-29日かつ30日2走以上は1走よりD/R/Cで低い",
        "discovery_period": "2014-2019",
        "replication_result": "direction replicated; confounding remains high",
        "sample_support": "5762/1769/813 runners in focal high-workload cell",
        "expected_direction": "selected workload slice改善",
        "information_source": "horse workload",
        "target": "Binary+ranking",
        "proposed_change": "predeclared low-degree interaction only",
        "PIT_status": "safe",
        "leakage_risk": "low",
        "redundancy_risk": "high",
        "implementation_cost": "low",
        "expected_value": "単調休養仮説を避ける",
        "validation_plan": "age/class/history guardrails付きrolling ablation",
        "priority": "B",
        "status": "proposed",
    },
    {
        "hypothesis_id": "EDA-B02-MARGIN-TOKEN",
        "question": "margin tokenは0.1秒同時計を安全に補完できるか",
        "evidence": "token orderとdisplay gap rho約.957、equal-clock隣接約22-24%",
        "discovery_period": "2014-2019",
        "replication_result": "structure replicated, seconds mapping not identified",
        "sample_support": "263242/85503/42108 clean adjacent edges",
        "expected_direction": "close finishの順序情報補完",
        "information_source": "historical performance",
        "target": "performance",
        "proposed_change": "mapping採用前にofficial semantics/source auditを追加",
        "PIT_status": "safe",
        "leakage_risk": "low",
        "redundancy_risk": "high",
        "implementation_cost": "medium",
        "expected_value": "PV-06の不確実性を解消",
        "validation_plan": "mappingをtrain-only固定できなければrejected-by-EDA",
        "priority": "B",
        "status": "proposed",
    },
    {
        "hypothesis_id": "EDA-C01-NEW-HORSE-EXCLUSION",
        "question": "新馬戦をtrainingから除外すべきか",
        "evidence": "history0勝率は約10%、connectionは利用可能、既存fit除外ablationは悪化",
        "discovery_period": "2014-2019",
        "replication_result": "history0 base structure stable; exclusion already rejected",
        "sample_support": "history0 about 10% across periods",
        "expected_direction": "除外は改善しない",
        "information_source": "population",
        "target": "Binary",
        "proposed_change": "除外しない。cold-startを固定diagnostic sliceとして維持",
        "PIT_status": "safe",
        "leakage_risk": "low",
        "redundancy_risk": "none",
        "implementation_cost": "low",
        "expected_value": "再試行を防ぐnegative result",
        "validation_plan": "no new experiment unless new data changes information set",
        "priority": "C",
        "status": "rejected-by-EDA",
    },
]


def dump_json(name: str, payload: object) -> None:
    (OUTPUT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "hypothesis_registry.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HYPOTHESES[0]))
        writer.writeheader()
        writer.writerows(HYPOTHESES)

    feature_rows = [
        [
            "CF-01",
            "EDA-S01-RACE-VALUE-2AXIS",
            "90d performance residual history",
            "horse",
            "after past race",
            "<0.4% outcome-side",
            "safe",
            "medium",
            "ready",
        ],
        [
            "CF-02",
            "EDA-S01-RACE-VALUE-2AXIS",
            "90d race-constant field-quality history",
            "opponent",
            "pre-race rating",
            "rating cold-start structural",
            "safe",
            "medium",
            "ready",
        ],
        [
            "CF-03",
            "EDA-A01-TRANSITION-RELIABILITY",
            "past-state reliability under one condition switch",
            "horse+context",
            "pre-race",
            "history0 structural",
            "safe",
            "medium",
            "proposed",
        ],
        [
            "CF-04",
            "EDA-A02-CONNECTION-COMPRESSION",
            "EB level/deviation/support/uncertainty",
            "connection",
            "pre-race",
            "EB level none",
            "safe",
            "low",
            "proposed",
        ],
        [
            "CF-05",
            "EDA-A03-LAST3F-RELATIVE",
            "race-relative last3F decay history",
            "horse",
            "after past race",
            "3 completed rows",
            "safe",
            "medium",
            "proposed",
        ],
    ]
    with (OUTPUT / "candidate_feature_registry.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "candidate_id",
                "hypothesis_id",
                "representation",
                "source",
                "available_at",
                "missingness",
                "PIT_status",
                "redundancy_risk",
                "status",
            ]
        )
        writer.writerows(feature_rows)

    local_manifest = json.loads((LOCAL / "manifest.json").read_text(encoding="utf-8"))
    dump_json("data_contract_summary.json", public_data_contract_summary(local_manifest))
    dump_json(
        "data_quality_summary.json",
        {
            "max_target_date": "2022-12-31",
            "retained_raw_rows": 488715,
            "retained_raw_races": 34504,
            "flat_started_performances": 471557,
            "flat_races": 33240,
            "predictive_rows": 450340,
            "predictive_races": 31689,
            "duplicate_keys": 0,
            "post_2022_retained": 0,
            "raw_sha256": local_manifest["raw_sha256"],
            "main_risks": [
                "2014 history is left-truncated by 2013 warm-up",
                "race-class token regime drift",
                "historical publication timestamps unavailable",
            ],
        },
    )
    dump_json(
        "temporal_stability_summary.json",
        {
            "signals": [
                {
                    "signal": "90d horse-history decay",
                    "discovery": 0.457,
                    "replication": 0.430,
                    "confirmation": 0.459,
                    "direction_consistency": "3/3 positive",
                },
                {
                    "signal": "jockey career EB",
                    "discovery": 0.276,
                    "replication": 0.262,
                    "confirmation": 0.258,
                    "direction_consistency": "3/3 positive",
                },
                {
                    "signal": "trainer career EB",
                    "discovery": 0.189,
                    "replication": 0.180,
                    "confirmation": 0.190,
                    "direction_consistency": "3/3 positive",
                },
                {
                    "signal": "field quality vs performance residual daily Spearman",
                    "discovery": 0.221,
                    "replication": 0.156,
                    "confirmation": 0.181,
                    "direction_consistency": "3/3 positive",
                },
                {
                    "signal": "same vs switched surface persistence",
                    "discovery": "0.381-0.415 vs 0.253-0.281",
                    "replication": "0.361-0.398 vs 0.197-0.247",
                    "confirmation": "0.379-0.410 vs 0.210-0.247",
                    "direction_consistency": "3/3",
                },
            ],
            "interpretation": "directional replication, not untouched confirmatory inference",
        },
    )
    dump_json(
        "modeling_decisions.json",
        {
            "binary_winner": "retain as frozen control",
            "lambdarank_top_heavy": "retain as frozen control; relevance redesign remains unproven",
            "full_order": "do not use unweighted all-pairs; lower ranks dominate",
            "continuous_performance_target": "priority S experiment after review",
            "multitask": "defer until an auxiliary performance target has standalone OOT value",
            "race_wise_probability": "priority S transparent baseline",
            "lightgbm": "retain as primary nonlinear control",
            "alternative_gbdt": "defer until representation tests",
            "ensemble": "do not reopen without new component evidence",
            "condition_specific_models": "do not split; test reliability interactions first",
            "dnn": "defer",
            "production_changed": False,
        },
    )
    dump_json(
        "summary.json",
        {
            "analysis_id": "eda_20260901",
            "phase": "Phase 5A: systematic exploratory data analysis and problem reformulation",
            "periods": {"warmup": "2013", "discovery": "2014-2019", "replication": "2020-2021", "confirmation": "2022"},
            "top_hypotheses": [row["hypothesis_id"] for row in HYPOTHESES if row["priority"] == "S"],
            "production_model_changed": False,
            "status": "complete_awaiting_human_selection",
            "local_artifact": "artifacts/eda_20260901 (gitignored)",
            "review_status": "PASS: PIT/leakage, statistics/validation, and domain/semantics",
        },
    )


if __name__ == "__main__":
    main()
