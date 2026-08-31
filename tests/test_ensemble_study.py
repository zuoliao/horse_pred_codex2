from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import horse_pred.ensemble_study as ensemble
from horse_pred.artifacts import write_artifact_manifest, write_json
from horse_pred.cli import parser
from horse_pred.config import load_json
from horse_pred.data import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/performance/ens_01_fixed_5050.json"


def test_registered_config_freezes_source_blend_and_no_weight_search() -> None:
    config = load_json(CONFIG_PATH)
    ensemble.validate_ensemble_config(config)
    assert config["blend"]["binary_weight"] == 0.5
    assert config["blend"]["lambdarank_weight"] == 0.5
    assert config["selection_accounting"]["weight_search_candidates"] == 0

    changed = copy.deepcopy(config)
    changed["blend"]["binary_weight"] = 0.6
    with pytest.raises(ValueError, match="50:50"):
        ensemble.validate_ensemble_config(changed)


def test_cli_registers_ensemble_command() -> None:
    args = parser().parse_args(
        ["run-ensemble-study", "--config", "ens.json", "--output", "artifact"]
    )
    assert args.command == "run-ensemble-study"


def _source_frame(*, races_per_role: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in range(2020, 2024):
        for role, race_year in (("calibration", year - 1), ("evaluation", year)):
            for race_index in range(races_per_role):
                race_id = f"{year}-{role}-{race_index}"
                winner_horse = 1 if race_index % 2 == 0 else 2
                for method in ("binary_control", "lambdarank_candidate"):
                    binary_favors = 1 if race_index % 2 == 0 else 2
                    favored = binary_favors if method == "binary_control" else 3 - binary_favors
                    for horse in (1, 2):
                        probability = 0.9 if horse == favored else 0.1
                        rows.append(
                            {
                                "fold_id": f"roll_{year}",
                                "role": role,
                                "evaluation_year": year,
                                "race_id": race_id,
                                "race_date": pd.Timestamp(race_year, 1, race_index + 1),
                                "method": method,
                                "model_kind": (
                                    "binary" if method == "binary_control" else "lambdarank"
                                ),
                                "model_finish_position": 1 if horse == winner_horse else 2,
                                "horse_id": f"h{horse}",
                                "horse_number": horse,
                                "utility": float(np.log(probability)),
                                "probability_t1": probability,
                                "probability_calibrated": probability,
                            }
                        )
    return pd.DataFrame(rows)


def test_alignment_rejects_missing_keys_and_blend_uses_only_calibration_outcomes() -> None:
    config = load_json(CONFIG_PATH)
    source = _source_frame()
    aligned = ensemble.align_reference_predictions(source, config)
    candidate, temperatures = ensemble.build_ensemble_predictions(
        aligned, evaluation_years=[2020], config=config
    )
    assert temperatures.keys() == {"roll_2020"}
    assert np.allclose(candidate["raw_output"], 0.5)
    assert np.allclose(candidate["probability_t1"], candidate["raw_output"], atol=1e-12)

    changed = aligned.copy()
    evaluation = changed["role"].eq("evaluation")
    changed.loc[evaluation, "model_finish_position"] = 3 - changed.loc[
        evaluation, "model_finish_position"
    ]
    changed_candidate, changed_temperatures = ensemble.build_ensemble_predictions(
        changed, evaluation_years=[2020], config=config
    )
    assert changed_temperatures == temperatures
    assert np.allclose(
        changed_candidate["probability_calibrated"], candidate["probability_calibrated"]
    )

    missing = source.drop(source.index[0])
    with pytest.raises(ValueError, match="keys do not match"):
        ensemble.align_reference_predictions(missing, config)


def _comparison_metrics(
    *, ll: float, brier: float = 0.001, ndcg: float = 0.0, top1: float = 0.0
) -> dict[str, dict[str, float | int]]:
    values = {
        "race_log_loss": ll,
        "race_brier": brier,
        "ndcg_at_3": ndcg,
        "top_1_winner_mass": top1,
    }
    return {
        metric: {
            "year_macro_improvement": value,
            "improved_years": 3 if value > 0 else 0,
            "worsened_years": 0 if value > 0 else 3,
            "tied_years": 0,
            "direction_consistency": 1.0 if value > 0 else 0.0,
            "worst_year_improvement": value,
        }
        for metric, value in values.items()
    }


def test_screen_and_confirmation_must_pass_against_both_references() -> None:
    config = load_json(CONFIG_PATH)
    summary = {
        "comparisons": {
            "ensemble_vs_binary": {"metrics": _comparison_metrics(ll=0.003)},
            "ensemble_vs_lambdarank": {"metrics": _comparison_metrics(ll=0.004)},
        }
    }
    assert ensemble.screen_ensemble(summary, config)["passed"]
    summary["comparisons"]["ensemble_vs_lambdarank"]["metrics"]["race_log_loss"][
        "year_macro_improvement"
    ] = 0.001
    assert ensemble.screen_ensemble(summary, config)["decision"] == "reject"

    summary["comparisons"]["ensemble_vs_lambdarank"]["metrics"] = _comparison_metrics(ll=0.004)
    intervals = {
        "paired": {
            comparison: {
                metric: {"lower": 0.001, "upper": 0.006}
                for metric in (
                    "race_log_loss",
                    "race_brier",
                    "ndcg_at_3",
                    "top_1_winner_mass",
                )
            }
            for comparison in ("ensemble_vs_binary", "ensemble_vs_lambdarank")
        }
    }
    assert ensemble.confirmation_decision(
        summary=summary, bootstrap=intervals, config=config
    )["decision"] == "accept"
    intervals["paired"]["ensemble_vs_binary"]["race_log_loss"]["lower"] = -0.001
    assert ensemble.confirmation_decision(
        summary=summary, bootstrap=intervals, config=config
    )["decision"] == "inconclusive"


def _write_source_artifact(tmp_path: Path) -> tuple[Path, str, str]:
    artifact = tmp_path / "source"
    artifact.mkdir()
    predictions = artifact / "predictions_scoring.csv.gz"
    _source_frame().to_csv(predictions, index=False, compression="gzip")
    write_json(
        artifact / "run_meta.json",
        {
            "experiment_id": "sec_3f_001_rolling",
            "git": {"commit": "a" * 40, "dirty": False},
            "rows_used_2024": 0,
            "rows_used_2025": 0,
            "odds_used": False,
        },
    )
    write_json(
        artifact / "metrics.json",
        {
            "scope": {
                "rows_used_2024": 0,
                "rows_used_2025": 0,
                "odds_used": False,
            }
        },
    )
    write_artifact_manifest(artifact)
    return artifact, sha256_file(predictions), sha256_file(artifact / "artifact_manifest.json")


def test_source_hash_gate_and_atomic_runner(tmp_path) -> None:
    artifact, prediction_hash, manifest_hash = _write_source_artifact(tmp_path)
    config = load_json(CONFIG_PATH)
    config["source"].update(
        {
            "artifact_directory": str(artifact),
            "predictions_sha256": prediction_hash,
            "manifest_sha256": manifest_hash,
            "run_commit": "a" * 40,
        }
    )
    config["uncertainty"].update({"bootstrap_resamples": 30, "block_length_dates": 1})
    config_path = tmp_path / "ens.json"
    write_json(config_path, config)

    source, validation = ensemble.load_and_validate_source(root=ROOT, config=config)
    assert len(source) == len(_source_frame())
    assert validation["predictions_sha256"] == prediction_hash
    bad = copy.deepcopy(config)
    bad["source"]["predictions_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prediction SHA"):
        ensemble.load_and_validate_source(root=ROOT, config=bad)

    output = tmp_path / "output"
    result = ensemble.run_ensemble_study(
        repo_root=ROOT, config_path=config_path, output_dir=output
    )
    assert result["scope"]["rows_used_2024"] == 0
    assert result["scope"]["rows_used_2025"] == 0
    assert result["scope"]["weight_search_candidates"] == 0
    assert result["scope"]["confirmation_opened"] is False
    assert result["confirmation"]["confirmation_result"]["decision"] == "not_opened"
    saved = pd.read_csv(output / "predictions_scoring.csv.gz")
    assert not saved["evaluation_year"].eq(2023).any()
    assert (output / "artifact_manifest.json").is_file()
    with pytest.raises(FileExistsError):
        ensemble.run_ensemble_study(
            repo_root=ROOT, config_path=config_path, output_dir=output
        )
