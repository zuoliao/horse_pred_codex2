from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

import horse_pred.hpo_study as hpo
from horse_pred.artifacts import write_json
from horse_pred.cli import parser
from horse_pred.config import load_json

ROOT = Path(__file__).resolve().parents[1]
BINARY_CONFIG = ROOT / "configs/performance/hpo_01b_binary_rolling.json"
RANK_CONFIG = ROOT / "configs/performance/hpo_01r_lambdarank_rolling.json"


def test_registered_configs_freeze_separate_family_searches() -> None:
    binary = load_json(BINARY_CONFIG)
    rank = load_json(RANK_CONFIG)
    hpo.validate_hpo_config(binary)
    hpo.validate_hpo_config(rank)

    assert binary["model_kind"] == "binary"
    assert rank["model_kind"] == "lambdarank"
    assert len(binary["profiles"]) == 12
    assert rank["selection"]["primary_metric"] == "ndcg_at_3"
    assert rank["base_model_config"] == "configs/exp_002_lambdarank.json"

    changed = copy.deepcopy(rank)
    changed["profiles"][1]["parameter_overrides"]["num_leaves"] = 17
    with pytest.raises(ValueError, match="profiles"):
        hpo.validate_hpo_config(changed)

    changed = copy.deepcopy(binary)
    changed["confirmation_fold"]["evaluation_year"] = 2024
    with pytest.raises(ValueError, match="confirmation"):
        hpo.validate_hpo_config(changed)


def test_cli_registers_hpo_command() -> None:
    args = parser().parse_args(
        [
            "run-hpo-study",
            "--cache",
            "cache.pkl",
            "--config",
            "hpo.json",
            "--output",
            "artifact",
        ]
    )
    assert args.command == "run-hpo-study"


def _metric_payload(
    *, primary: str, primary_value: float, secondary: dict[str, float] | None = None
) -> dict[str, dict[str, float | int]]:
    values = {
        "race_log_loss": 0.0,
        "race_brier": 0.0,
        "ndcg_at_3": 0.0,
        "top_1_winner_mass": 0.0,
    }
    values[primary] = primary_value
    values.update(secondary or {})
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


def test_selection_is_deterministic_and_uses_registered_guardrails() -> None:
    config = load_json(BINARY_CONFIG)
    comparisons = {}
    for profile in config["profiles"][1:]:
        comparisons[f"{profile['id']}_vs_control"] = {
            "metrics": _metric_payload(primary="race_log_loss", primary_value=-0.01)
        }
    comparisons["leaves_15_vs_control"] = {
        "metrics": _metric_payload(
            primary="race_log_loss",
            primary_value=0.0030,
            secondary={"race_brier": 0.0002, "ndcg_at_3": 0.0, "top_1_winner_mass": 0.0},
        )
    }
    comparisons["depth_6_vs_control"] = {
        "metrics": _metric_payload(
            primary="race_log_loss",
            primary_value=0.00305,
            secondary={"race_brier": 0.0001, "ndcg_at_3": 0.0, "top_1_winner_mass": 0.0},
        )
    }

    selected = hpo.select_hpo_profile({"comparisons": comparisons}, config)

    # Primary values within 1e-4 tie; Brier deterministically selects leaves_15.
    assert selected["selected_profile"] == "leaves_15"
    comparisons["leaves_15_vs_control"]["metrics"]["race_brier"][
        "year_macro_improvement"
    ] = -0.001
    selected = hpo.select_hpo_profile({"comparisons": comparisons}, config)
    assert not selected["profiles"]["leaves_15"]["eligible"]


def test_confirmation_requires_primary_interval_and_guardrails() -> None:
    config = load_json(BINARY_CONFIG)
    improvement = {
        "race_log_loss": 0.003,
        "race_brier": 0.0001,
        "ndcg_at_3": 0.0,
        "top_1_winner_mass": 0.0,
    }
    interval = {metric: {"lower": 0.001, "upper": 0.005} for metric in improvement}
    assert hpo.confirmation_decision(
        improvement=improvement, interval=interval, config=config
    )["decision"] == "accept"

    interval["race_log_loss"] = {"lower": -0.001, "upper": 0.005}
    assert hpo.confirmation_decision(
        improvement=improvement, interval=interval, config=config
    )["decision"] == "inconclusive"
    improvement["race_brier"] = -0.001
    assert hpo.confirmation_decision(
        improvement=improvement, interval=interval, config=config
    )["decision"] == "reject"


def _prediction_rows(years: list[int], methods: list[str]) -> pd.DataFrame:
    rows = []
    for year in years:
        for date_index in range(4):
            race_id = f"{year}r{date_index}"
            race_date = pd.Timestamp(year=year, month=1, day=date_index + 1)
            for method in methods:
                winner_probability = 0.8 if method == "leaves_15" else 0.6
                for position, probability in ((1, winner_probability), (2, 1 - winner_probability)):
                    rows.append(
                        {
                            "stage": "fixture",
                            "fold_id": f"roll_{year}",
                            "role": "evaluation",
                            "evaluation_year": year,
                            "race_id": race_id,
                            "race_date": race_date,
                            "method": method,
                            "model_kind": "binary",
                            "model_finish_position": position,
                            "raw_output": probability,
                            "utility": probability,
                            "probability_t1": probability,
                            "probability_calibrated": probability,
                        }
                    )
    return pd.DataFrame(rows)


def test_runner_exposes_only_selected_candidate_to_2023(tmp_path, monkeypatch) -> None:
    config = load_json(BINARY_CONFIG)
    config["uncertainty"]["bootstrap_resamples"] = 20
    config_path = tmp_path / "hpo.json"
    write_json(config_path, config)
    cache_path = tmp_path / "cache.pkl"
    cache_path.write_bytes(b"fixture")
    fixture_frame = pd.DataFrame(
        {
            "race_id": ["fixture"],
            "race_date": ["2020-01-01"],
            "model_finish_position": [1],
        }
    )
    monkeypatch.setattr(
        hpo,
        "read_model_frame_cache",
        lambda path: (fixture_frame, {"feature_columns": ["x"], "data_fingerprint": "fixture"}),
    )
    monkeypatch.setattr(
        hpo,
        "isolate_rolling_source",
        lambda frame, maximum_outcome_year: (
            frame,
            {
                "cache_rows": 1,
                "rolling_rows": 1,
                "rows_excluded_by_year": {},
                "rows_used_2024": 0,
                "rows_used_2025": 0,
                "maximum_outcome_year": 2023,
            },
        ),
    )
    monkeypatch.setattr(
        hpo,
        "resolve_hpo_scope",
        lambda **kwargs: {
            "feature_columns": ("x",),
            "feature_groups": {"fixture": ("x",)},
            "feature_resolution": {},
            "feature_columns_sha256": "a" * 64,
            "feature_config": {},
            "feature_config_path": tmp_path / "feature.json",
            "model_config": {"parameters": {}},
            "model_config_path": tmp_path / "model.json",
        },
    )
    calls: list[tuple[list[int], list[str]]] = []

    def fake_fit_stage(*, folds, profiles, **kwargs):
        years = [int(fold["evaluation_year"]) for fold in folds]
        methods = [profile["id"] for profile in profiles]
        calls.append((years, methods))
        return _prediction_rows(years, methods), {}, []

    monkeypatch.setattr(hpo, "_fit_stage", fake_fit_stage)
    monkeypatch.setattr(hpo, "git_state", lambda root: {"commit": "fixture", "dirty": False})
    output = tmp_path / "artifact"

    result = hpo.run_hpo_study(
        repo_root=ROOT, cache_path=cache_path, config_path=config_path, output_dir=output
    )

    assert calls[0] == ([2020, 2021, 2022], [profile["id"] for profile in config["profiles"]])
    assert calls[1] == ([2023], ["control", "leaves_15"])
    assert result["scope"]["nonselected_profiles_scored_on_2023"] == 0
    saved = pd.read_csv(output / "predictions_scoring.csv.gz")
    assert set(saved.loc[saved["evaluation_year"].eq(2023), "method"]) == {
        "control",
        "leaves_15",
    }
    assert result["confirmation"]["confirmation_result"]["decision"] == "accept"
