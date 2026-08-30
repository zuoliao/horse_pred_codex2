from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import horse_pred.cached_experiment as cached
from horse_pred.artifacts import write_json
from horse_pred.modeling import TemperatureCalibrator


def _config(selection: dict[str, list[str]] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": "test_cached",
        "hypothesis": "one registered test hypothesis",
        "seed": 42,
        "model_configs": {
            "binary": "binary.json",
            "lambdarank": "ranker.json",
        },
        "feature_selection": selection or {"drop": []},
    }


def test_semantic_selection_removes_descendants_of_dropped_source() -> None:
    columns = (
        "context__distance",
        "horse_history__career__starts",
        "horse_history__days_30__starts",
        "horse_history__same_surface__starts",
        "jockey_history__career__starts",
        "field_relative__jockey_win_rate__zscore",
        "field_relative__horse_rest_days__rank_pct",
        "rating__horse_elo",
    )
    selected, groups, resolution = cached.resolve_semantic_feature_selection(
        columns, _config({"drop": ["connections"]})
    )

    assert "jockey_history__career__starts" not in selected
    assert "field_relative__jockey_win_rate__zscore" not in selected
    assert "field_relative__horse_rest_days__rank_pct" in selected
    assert "connections" not in groups
    assert resolution["dependency_removed_columns"] == [
        "field_relative__jockey_win_rate__zscore"
    ]


def test_feature_selection_requires_exactly_one_operation() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        cached.validate_cached_experiment_config(
            _config({"include": ["current_context"], "drop": []})
        )


def test_registered_relative_features_require_selected_pit_sources() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["r1", "r1", "r2", "r2"],
            "horse_history__decay_30d__mean_finish": [1.0, 3.0, np.nan, 2.0],
        }
    )
    config = _config({"drop": []})
    config["derived_features"] = [
        {
            "operation": "within_race_percentile",
            "source": "horse_history__decay_30d__mean_finish",
            "output": "experimental__decay_30d_mean_finish__percentile",
        }
    ]
    columns, groups, resolution = cached.add_registered_derived_features(
        frame,
        ("horse_history__decay_30d__mean_finish",),
        {"form_workload": ("horse_history__decay_30d__mean_finish",)},
        {"taxonomy": "semantic_feature_groups_v2"},
        config,
    )

    output = "experimental__decay_30d_mean_finish__percentile"
    assert columns[-1] == output
    assert groups["experimental_derived"] == (output,)
    assert frame[output].tolist()[:2] == [0.5, 1.0]
    assert np.isnan(frame.loc[2, output])
    assert frame.loc[3, output] == 1.0
    assert resolution["derived_features"][0]["source"].startswith("horse_history__")

    with pytest.raises(ValueError, match="not in the selected allowlist"):
        cached.add_registered_derived_features(
            frame.drop(columns=output),
            (),
            {},
            {},
            config,
        )


def test_derived_feature_config_rejects_nonexperimental_output() -> None:
    config = _config()
    config["derived_features"] = [
        {
            "operation": "within_race_percentile",
            "source": "horse_history__career__starts",
            "output": "winner_label",
        }
    ]
    with pytest.raises(ValueError, match="experimental__"):
        cached.validate_cached_experiment_config(config)


def test_isolate_pre_2025_frame_removes_retrospective_and_checks_dates() -> None:
    rows = []
    for split, date in (
        ("train", "2014-01-01"),
        ("model_validation", "2022-01-01"),
        ("calibration", "2023-01-01"),
        ("development", "2024-01-01"),
        ("retrospective_test", "2025-01-01"),
    ):
        rows.extend(
            {
                "race_id": f"{date}-{number}",
                "race_date": date,
                "split": split,
                "model_finish_position": number,
            }
            for number in (1, 2)
        )
    # Give the two runners a common race ID in each split.
    frame = pd.DataFrame(rows)
    frame["race_id"] = frame["race_id"].str.rsplit("-", n=1).str[0]
    isolated, report = cached.isolate_pre_2025_frame(frame)

    assert set(isolated["split"]) == {
        "train",
        "model_validation",
        "calibration",
        "development",
    }
    assert isolated["race_date"].dt.year.max() == 2024
    assert report["retrospective_2025_rows_in_cache"] == 2
    assert report["retrospective_2025_rows_used"] == 0

    bad = frame.copy()
    bad.loc[bad["split"].eq("development"), "race_date"] = "2025-02-01"
    with pytest.raises(ValueError, match="labeled retrospective_test"):
        cached.isolate_pre_2025_frame(bad)


class _FakeBooster:
    def __init__(self, width: int) -> None:
        self.width = width

    def save_model(self, path: str) -> None:
        Path(path).write_text("fake model\n", encoding="utf-8")

    def feature_importance(self, *, importance_type: str) -> np.ndarray:
        if importance_type == "split":
            return np.arange(1, self.width + 1)
        return np.arange(1, self.width + 1, dtype=float) * 2.0


class _FakeModel:
    def __init__(self, width: int) -> None:
        self.best_iteration_ = 3
        self.booster_ = _FakeBooster(width)


def test_cached_run_writes_atomic_2024_only_odds_free_artifact(
    tmp_path, monkeypatch
) -> None:
    rows = []
    for split, year in (
        ("train", 2014),
        ("model_validation", 2022),
        ("calibration", 2023),
        ("development", 2024),
        ("retrospective_test", 2025),
    ):
        race_id = f"{year}010101"
        for position in (1, 2):
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": f"{year}-01-01",
                    "split": split,
                    "model_finish_position": position,
                    "finish_position": position,
                    "winner_label": int(position == 1),
                    "horse_id": f"h{position}",
                    "horse_number": position,
                    "course_type": "芝",
                    "distance": 1600,
                    "race_class": "3歳未勝利",
                    "field_size": 2,
                    "distance_band": "mile",
                    "field_size_band": "small",
                    "final_win_odds": float(position + 1),
                    "final_popularity": position,
                    "context__distance": 1600.0,
                    "horse_history__career__wins": float(position == 1),
                    "horse_history__career__starts": 2.0,
                }
            )
    frame = pd.DataFrame(rows)
    cache_path = tmp_path / "cache.pkl"
    frame.to_pickle(cache_path)
    write_json(
        cache_path.with_suffix(".pkl.meta.json"),
        {
            "schema_version": 1,
            "data_fingerprint": "raw-sha",
            "row_count": len(frame),
            "race_count": frame["race_id"].nunique(),
            "feature_columns": [
                "context__distance",
                "horse_history__career__wins",
                "horse_history__career__starts",
            ],
        },
    )
    binary_path = tmp_path / "binary.json"
    ranker_path = tmp_path / "ranker.json"
    write_json(
        binary_path,
        {
            "model_family": "lightgbm_binary",
            "seed": 42,
            "parameters": {"objective": "binary", "random_state": 42},
        },
    )
    write_json(
        ranker_path,
        {
            "model_family": "lightgbm_lambdarank",
            "seed": 42,
            "parameters": {"objective": "lambdarank", "random_state": 42},
        },
    )
    config = _config()
    config["model_configs"] = {
        "binary": str(binary_path),
        "lambdarank": str(ranker_path),
    }
    config_path = tmp_path / "experiment.json"
    write_json(config_path, config)

    monkeypatch.setattr(cached, "train_binary", lambda *args, **kwargs: _FakeModel(3))
    monkeypatch.setattr(cached, "train_ranker", lambda *args, **kwargs: _FakeModel(3))

    def fake_predict(model, selected, *, feature_columns, model_kind):
        if model_kind == "binary":
            return [0.7 if value == 1 else 0.3 for value in selected["model_finish_position"]]
        return [-float(value) for value in selected["model_finish_position"]]

    monkeypatch.setattr(cached, "predict", fake_predict)
    monkeypatch.setattr(
        cached,
        "fit_temperature",
        lambda *args, **kwargs: TemperatureCalibrator(temperature=1.0, fitted=True),
    )

    output = tmp_path / "artifact"
    metrics = cached.run_cached_experiment(
        repo_root=Path.cwd(),
        cache_path=cache_path,
        config_path=config_path,
        output_dir=output,
    )

    predictions = pd.read_csv(output / "predictions_2024.csv.gz")
    assert predictions["race_date"].str.startswith("2024").all()
    assert "final_win_odds" not in predictions
    assert "final_popularity" not in predictions
    assert metrics["scope"]["retrospective_used"] is False
    assert metrics["data"]["retrospective_2025_rows_used"] == 0
    importance = pd.read_csv(output / "feature_importance.csv")
    assert {"importance_split", "importance_gain"}.issubset(importance)
    assert (output / "models" / "binary.txt").is_file()
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert any(item["path"] == "predictions_2024.csv.gz" for item in manifest["files"])
    assert not list(tmp_path.glob(".artifact.tmp-*"))


def test_cached_run_cleans_temporary_artifact_on_failure(tmp_path) -> None:
    bad_config = tmp_path / "bad.json"
    write_json(bad_config, {"schema_version": 1})
    with pytest.raises(ValueError, match="missing"):
        cached.run_cached_experiment(
            repo_root=Path.cwd(),
            cache_path=tmp_path / "missing.pkl",
            config_path=bad_config,
            output_dir=tmp_path / "failed",
        )
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed.tmp-*"))
