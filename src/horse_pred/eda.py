"""Reproducible Phase 5A exploratory-data-analysis views and aggregates.

The module deliberately stops before model/feature promotion.  Runner-level
views are private local artifacts; only non-recoverable aggregate summaries are
intended for the tracked ``experiments/`` directory.
"""

from __future__ import annotations

import hashlib
import html
import json
import subprocess
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horse_pred.config import load_json
from horse_pred.data import load_manifest, load_raw, normalize_raw, sha256_file
from horse_pred.features import FeatureConfig, build_features

EDA_MAX_DATE = pd.Timestamp("2022-12-31")
PERIODS: tuple[tuple[str, str, str], ...] = (
    ("warmup", "2013-01-01", "2013-12-31"),
    ("discovery", "2014-01-01", "2019-12-31"),
    ("replication", "2020-01-01", "2021-12-31"),
    ("confirmation", "2022-01-01", "2022-12-31"),
)
MARKET_COLUMNS = frozenset({"final_win_odds", "final_popularity", "単勝", "人気"})
CURRENT_OUTCOME_COLUMNS = frozenset(
    {
        "winner_label",
        "coherent_win_target",
        "finish_position",
        "finish_raw",
        "time_raw",
        "margin_raw",
        "passing_order_raw",
        "last_3f_seconds",
        "着順",
        "タイム",
        "着差",
        "通過順位",
        "上がり3F",
    }
)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_value(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _schema_hash(frame: pd.DataFrame) -> str:
    contract = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    encoded = json.dumps(contract, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _period_label(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="raise")
    labels = pd.Series(pd.NA, index=values.index, dtype="string")
    for label, start, end in PERIODS:
        labels.loc[dates.between(start, end)] = label
    return labels


def load_eda_population(
    raw_path: str | Path,
    *,
    max_date: str | date | pd.Timestamp,
    expected_sha256: str | None = None,
) -> pd.DataFrame:
    """Load and physically retain only rows permitted by the Phase 5A firewall."""

    requested = pd.Timestamp(max_date).normalize()
    if requested != EDA_MAX_DATE:
        raise ValueError(
            f"Phase 5A requires max_date={EDA_MAX_DATE.date()}, got {requested.date()}"
        )
    raw = load_raw(raw_path, expected_sha256=expected_sha256)
    raw_dates = pd.to_datetime(raw["date"], errors="raise")
    retained = raw.loc[raw_dates.le(requested)].copy()
    if retained.empty:
        raise ValueError("EDA population is empty")
    retained_dates = pd.to_datetime(retained["date"], errors="raise")
    if retained_dates.max() > EDA_MAX_DATE:
        raise AssertionError("EDA loader retained a target after 2022-12-31")
    if retained_dates.min() < pd.Timestamp("2013-01-01"):
        raise AssertionError("EDA population precedes the registered warm-up")
    return normalize_raw(retained)


def _time_seconds(values: pd.Series) -> pd.Series:
    text = values.astype("string")
    parts = text.str.extract(r"^(?P<m>[0-9]+):(?P<s>[0-9]+(?:\.[0-9]+)?)$")
    return pd.to_numeric(parts["m"], errors="coerce") * 60 + pd.to_numeric(
        parts["s"], errors="coerce"
    )


def build_race_table(normalized: pd.DataFrame) -> pd.DataFrame:
    work = normalized.copy()
    work["race_time_seconds"] = _time_seconds(work["time_raw"])
    started = work["started"].fillna(False).astype(bool)
    completed = work["finish_position"].notna()
    group = work.groupby("race_id", sort=True, observed=True)
    race = group.agg(
        date=("race_date", "first"),
        venue=("venue", "first"),
        surface=("surface", "first"),
        distance=("distance_m", "first"),
        direction=("around", "first"),
        race_class=("race_class", "first"),
        declared_count=("race_id", "size"),
        starter_count=("started", "sum"),
        dead_heat_size=("dead_heat_size", "max"),
        winner_time_seconds=("race_time_seconds", "min"),
        field_median_time_seconds=("race_time_seconds", "median"),
        last_3f_median=("last_3f_seconds", "median"),
        source_rows=("race_id", "size"),
    ).reset_index()
    status_counts = pd.crosstab(work["race_id"], work["status"]).add_prefix("status__")
    race = race.merge(status_counts, left_on="race_id", right_index=True, how="left")
    winners = (
        work.loc[work["winner_label"].eq(1)]
        .groupby("race_id", sort=True)["horse_id"]
        .agg(lambda values: "|".join(sorted(map(str, values))))
        .rename("winner_horse_ids")
    )
    race = race.merge(winners, left_on="race_id", right_index=True, how="left")
    race["scratch_exclusion_count"] = (~started).groupby(work["race_id"]).sum().to_numpy()
    race["completed_count"] = completed.groupby(work["race_id"]).sum().to_numpy()
    race["source_coverage_complete"] = race["declared_count"].eq(race["source_rows"])
    class_text = race["race_class"].astype("string")
    race["age_restriction"] = class_text.str.extract(r"(2歳|3歳以上|3歳|4歳以上)", expand=False)
    race["sex_restriction"] = np.where(class_text.str.contains("牝", na=False), "female_only", "mixed")
    race["analysis_period"] = _period_label(race["date"])
    return race


def build_raw_status_population(normalized: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "race_id",
        "race_date",
        "horse_id",
        "horse_number",
        "status",
        "started",
        "history_update_eligible",
        "finish_position",
        "is_dead_heat",
        "dead_heat_size",
        "pit_c_scoring_eligible",
    ]
    result = normalized.loc[:, columns].copy()
    result["analysis_period"] = _period_label(result["race_date"])
    return result


def build_market_oracle(normalized: pd.DataFrame) -> pd.DataFrame:
    return normalized.loc[
        :,
        ["race_id", "race_date", "horse_id", "horse_number", "final_win_odds", "final_popularity"],
    ].copy()


def build_historical_performance(normalized: pd.DataFrame) -> pd.DataFrame:
    work = normalized.loc[normalized["started"].fillna(False)].copy()
    work = work.sort_values(["horse_id", "race_date", "race_id"], kind="stable")
    work["race_time_seconds"] = _time_seconds(work["time_raw"])
    winner_time = work.groupby("race_id", sort=False)["race_time_seconds"].transform("min")
    field_size = work.groupby("race_id", sort=False)["race_id"].transform("size")
    work["winner_relative_time_gap"] = work["race_time_seconds"] - winner_time
    work["finish_percentile"] = (work["finish_position"] - 1) / (field_size - 1).replace(0, np.nan)
    work["last_3f_percentile"] = work.groupby("race_id", sort=False)["last_3f_seconds"].rank(
        method="average", pct=True, ascending=True
    )
    work["available_from"] = pd.to_datetime(work["race_date"]) + pd.offsets.Day(1)
    work["next_target_date"] = work.groupby("horse_id", sort=False)["race_date"].shift(-1)
    work["elapsed_days_to_next_target"] = (
        pd.to_datetime(work["next_target_date"]) - pd.to_datetime(work["race_date"])
    ).dt.days
    work["analysis_period"] = _period_label(work["race_date"])
    columns = [
        "horse_id",
        "race_id",
        "race_date",
        "available_from",
        "next_target_date",
        "elapsed_days_to_next_target",
        "venue",
        "surface",
        "distance_m",
        "ground_state",
        "race_class",
        "status",
        "finish_position",
        "finish_percentile",
        "race_time_seconds",
        "winner_relative_time_gap",
        "margin_raw",
        "last_3f_seconds",
        "last_3f_percentile",
        "passing_order_raw",
        "analysis_period",
    ]
    return work.loc[:, columns]


def build_private_views(
    normalized: pd.DataFrame, *, split_config: Mapping[str, Any]
) -> dict[str, pd.DataFrame]:
    dataset = build_features(normalized, config=FeatureConfig(), split_config=split_config)
    safe_meta = [
        "race_id",
        "race_date",
        "horse_id",
        "jockey_id",
        "trainer",
        "course_type",
        "distance",
        "race_class",
        "horse_number",
    ]
    runner = dataset.frame.loc[
        dataset.frame["started"].fillna(False),
        [column for column in safe_meta if column in dataset.frame] + list(dataset.feature_columns),
    ].copy()
    forbidden = MARKET_COLUMNS.union(CURRENT_OUTCOME_COLUMNS).intersection(runner.columns)
    if forbidden:
        raise AssertionError(f"runner_pre_race contains forbidden columns: {sorted(forbidden)}")
    runner["analysis_period"] = _period_label(runner["race_date"])
    outcomes = dataset.frame.loc[
        dataset.frame["started"].fillna(False),
        [
            "race_id",
            "race_date",
            "horse_id",
            "winner_label",
            "coherent_win_target",
            "finish_position",
            "time_raw",
            "margin_raw",
            "passing_order_raw",
            "last_3f_seconds",
        ],
    ].copy()
    connection_columns = [
        column
        for column in runner.columns
        if column.startswith("jockey_history__") or column.startswith("trainer_history__")
    ]
    connection = runner.loc[
        :, ["race_id", "race_date", "horse_id", "jockey_id", "trainer"] + connection_columns
    ].copy()
    for entity in ("jockey", "trainer"):
        starts_column = f"{entity}_history__career__starts"
        rate_column = f"{entity}_history__career__win_rate"
        starts = pd.to_numeric(connection[starts_column], errors="coerce").fillna(0)
        wins = pd.to_numeric(connection[rate_column], errors="coerce").fillna(0) * starts
        alpha = wins + 1
        beta = starts - wins + 1
        connection[f"{entity}__effective_sample_size"] = starts
        connection[f"{entity}__posterior_win_rate_sd"] = np.sqrt(
            alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
        )
    historical = build_historical_performance(normalized)
    rating_columns = [
        "rating__horse_elo_pre",
        "rating__field_mean_elo_pre",
        "rating__field_max_elo_pre",
        "rating__field_std_elo_pre",
    ]
    historical = historical.merge(
        runner[["race_id", "horse_id"] + rating_columns],
        on=["race_id", "horse_id"],
        how="left",
        validate="one_to_one",
    )
    starter_count = historical.groupby("race_id", sort=False)["horse_id"].transform("size")
    historical["opponent_only_mean_elo_pre"] = (
        historical["rating__field_mean_elo_pre"] * starter_count
        - historical["rating__horse_elo_pre"]
    ) / (starter_count - 1).replace(0, np.nan)
    return {
        "race_table": build_race_table(normalized),
        "runner_pre_race": runner,
        "outcomes": outcomes,
        "historical_performance": historical,
        "connection_state": connection,
        "market_oracle": build_market_oracle(normalized),
        "raw_status_population": build_raw_status_population(normalized),
    }


def _view_summary(
    name: str, frame: pd.DataFrame, *, raw_sha256: str, availability: str
) -> dict[str, Any]:
    date_column = "race_date" if "race_date" in frame else "date"
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    key_map = {
        "race_table": ["race_id"],
        "runner_pre_race": ["race_id", "horse_id"],
        "outcomes": ["race_id", "horse_id"],
        "historical_performance": ["race_id", "horse_id"],
        "connection_state": ["race_id", "horse_id"],
        "market_oracle": ["race_id", "horse_id"],
        "raw_status_population": ["race_id", "horse_id"],
    }
    keys = key_map[name]
    return {
        "view": name,
        "row_count": int(len(frame)),
        "race_count": int(frame["race_id"].nunique()),
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
        "key_columns": keys,
        "key_unique": not bool(frame.duplicated(keys).any()),
        "source_fingerprint": raw_sha256,
        "schema_hash": _schema_hash(frame),
        "feature_availability_time": availability,
        "missing_rate": {
            str(column): round(float(frame[column].isna().mean()), 8) for column in frame.columns
        },
    }


def _race_macro_rate(frame: pd.DataFrame, value: str) -> float:
    return float(frame.groupby("race_id", sort=False)[value].mean().mean())


def build_aggregate_tables(views: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    race = views["race_table"].copy()
    status = views["raw_status_population"].copy()
    hist = views["historical_performance"].copy()
    outcomes = views["outcomes"].copy()
    runner = views["runner_pre_race"].copy()

    yearly = race.assign(year=pd.to_datetime(race["date"]).dt.year).groupby("year", observed=True).agg(
        races=("race_id", "nunique"),
        declared=("declared_count", "sum"),
        starters=("starter_count", "sum"),
        mean_field_size=("starter_count", "mean"),
        dead_heat_races=("dead_heat_size", lambda x: int((x > 1).sum())),
        median_winner_time=("winner_time_seconds", "median"),
        median_last_3f=("last_3f_median", "median"),
    ).reset_index()
    yearly["phase"] = _period_label(pd.to_datetime(yearly["year"].astype(str) + "-07-01"))

    outcome_join = outcomes.merge(
        race[["race_id", "starter_count", "surface", "distance", "race_class", "analysis_period"]],
        on="race_id",
        how="left",
        validate="many_to_one",
    )
    outcome_join["top3"] = pd.to_numeric(outcome_join["finish_position"], errors="coerce").le(3)
    field = outcome_join.groupby(["analysis_period", "starter_count"], observed=True).agg(
        runners=("horse_id", "size"),
        races=("race_id", "nunique"),
        empirical_win_rate=("winner_label", "mean"),
        top3_rate=("top3", "mean"),
    ).reset_index()
    field["uniform_win_rate"] = 1 / field["starter_count"]

    missing_columns = ["last_3f_seconds", "winner_relative_time_gap", "margin_raw", "passing_order_raw"]
    missing = (
        hist.groupby("analysis_period", observed=True)[missing_columns]
        .agg(lambda x: float(x.isna().mean()))
        .reset_index()
    )
    counts = hist.groupby("analysis_period", observed=True).agg(
        runners=("horse_id", "size"), races=("race_id", "nunique")
    ).reset_index()
    missing = counts.merge(missing, on="analysis_period", validate="one_to_one")

    rank_gap = hist.dropna(subset=["finish_position"]).copy()
    rank_gap["finish_band"] = pd.cut(
        pd.to_numeric(rank_gap["finish_position"]),
        [0, 1, 2, 3, 5, 9, np.inf],
        labels=["1", "2", "3", "4-5", "6-9", "10+"],
    )
    rank_gap = rank_gap.groupby(["analysis_period", "finish_band"], observed=True).agg(
        runners=("horse_id", "size"),
        races=("race_id", "nunique"),
        median_time_gap=("winner_relative_time_gap", "median"),
        iqr_time_gap=("winner_relative_time_gap", lambda x: float(x.quantile(.75) - x.quantile(.25))),
        median_last_3f_pct=("last_3f_percentile", "median"),
    ).reset_index()

    history_col = "horse_history__career__starts"
    history = runner[["race_id", "horse_id", "analysis_period", history_col]].merge(
        outcomes[["race_id", "horse_id", "winner_label", "finish_position"]],
        on=["race_id", "horse_id"],
        validate="one_to_one",
    )
    history["history_band"] = pd.cut(
        history[history_col], [-1, 0, 1, 3, 9, np.inf], labels=["0", "1", "2-3", "4-9", "10+"]
    )
    history["top3"] = pd.to_numeric(history["finish_position"], errors="coerce").le(3)
    history_curve = history.groupby(["analysis_period", "history_band"], observed=True).agg(
        runners=("horse_id", "size"),
        races=("race_id", "nunique"),
        win_rate=("winner_label", "mean"),
        top3_rate=("top3", "mean"),
    ).reset_index()

    margin_vocab = hist.assign(token=hist["margin_raw"].astype("string").fillna("<NA>")).groupby(
        ["analysis_period", "token"], observed=True
    ).agg(runners=("horse_id", "size"), races=("race_id", "nunique")).reset_index()

    status_table = status.groupby(["analysis_period", "status"], observed=True).agg(
        runners=("horse_id", "size"), races=("race_id", "nunique")
    ).reset_index()
    return {
        "yearly_coverage_drift": yearly,
        "field_size_base_rates": field,
        "content_missingness": missing,
        "finish_time_gap_structure": rank_gap,
        "history_availability_curve": history_curve,
        "margin_token_vocabulary": margin_vocab,
        "status_population": status_table,
    }


def _svg_bar(table: pd.DataFrame, x: str, y: str, title: str) -> str:
    values = pd.to_numeric(table[y], errors="coerce").fillna(0)
    maximum = max(float(values.max()), 1e-12)
    width, height, margin = 900, 420, 60
    slot = (width - 2 * margin) / max(len(table), 1)
    bars: list[str] = []
    for index, (_, row) in enumerate(table.iterrows()):
        value = float(row[y]) if pd.notna(row[y]) else 0.0
        bar_height = (height - 2 * margin) * value / maximum
        left = margin + index * slot + slot * 0.15
        top = height - margin - bar_height
        label = html.escape(str(row[x]))
        bars.append(
            f'<rect x="{left:.1f}" y="{top:.1f}" width="{slot * .7:.1f}" height="{bar_height:.1f}" fill="#356a8a"/>'
            f'<text x="{left + slot * .35:.1f}" y="{height - margin + 18}" '
            f'text-anchor="middle" font-size="11">{label}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="white"/><text x="{width / 2}" y="28" '
        f'text-anchor="middle" font-size="18">{html.escape(title)}</text>{"".join(bars)}</svg>'
    )


def _write_report(output: Path, summaries: Mapping[str, Any], tables: Mapping[str, pd.DataFrame]) -> None:
    items = "".join(
        f"<li><b>{html.escape(name)}</b>: {value['row_count']:,} rows / {value['race_count']:,} races, "
        f"{value['date_min']}–{value['date_max']}</li>"
        for name, value in summaries.items()
    )
    links = "".join(
        f'<li><a href="tables/{html.escape(name)}.csv">{html.escape(name)}</a></li>' for name in tables
    )
    body = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Phase 5A EDA</title>
<style>
body{{font-family:system-ui;max-width:1000px;margin:2rem auto;line-height:1.5}}
code{{background:#eee;padding:.1rem .3rem}}
</style></head><body><h1>Phase 5A systematic EDA</h1>
<p>Target-aware scope is physically capped at <code>2022-12-31</code>.
Runner-level views are private local artifacts.</p><h2>Views</h2><ul>{items}</ul>
<h2>Aggregate source tables</h2><ul>{links}</ul><h2>Plots</h2>
<img src="plots/yearly_race_count.svg" alt="yearly race count">
<p>This report is descriptive. It does not promote a production feature or model.</p>
</body></html>"""
    (output / "report.html").write_text(body, encoding="utf-8")


def run_eda(
    *,
    repo_root: str | Path,
    raw_path: str | Path,
    output_dir: str | Path,
    max_date: str | date,
    config_path: str | Path = "configs/eda/phase_5a.json",
    manifest_path: str | Path = "configs/data_manifest.json",
    split_path: str | Path = "configs/splits.json",
    resume: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        manifest_file = root / manifest_file
    split_file = Path(split_path)
    if not split_file.is_absolute():
        split_file = root / split_file
    config = load_json(config_file)
    if str(max_date) != config["max_target_date"]:
        raise ValueError("CLI max-date differs from registered EDA cutoff")
    if output.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite EDA output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("views", "tables", "plots", "workstreams", "logs"):
        (output / name).mkdir(exist_ok=True)
    existing_manifest = output / "manifest.json"
    if resume and existing_manifest.is_file():
        with existing_manifest.open(encoding="utf-8") as handle:
            completed = json.load(handle)
        if completed.get("max_target_date") != str(EDA_MAX_DATE.date()):
            raise ValueError("existing EDA artifact has an incompatible cutoff")
        return completed

    _json_dump(output / "analysis_config.json", config)
    (output / "logs" / "run.log").write_text(
        "Phase 5A run started; target-aware data capped at 2022-12-31.\n",
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_file)
    raw_sha256 = sha256_file(raw_path)
    if raw_sha256 != manifest["raw_file"]["sha256"]:
        raise ValueError("raw fingerprint differs from approved manifest")
    normalized = load_eda_population(
        raw_path, max_date=max_date, expected_sha256=raw_sha256
    )
    if pd.to_datetime(normalized["race_date"]).max() > EDA_MAX_DATE:
        raise AssertionError("post-normalization date firewall failed")
    views = build_private_views(normalized, split_config=load_json(split_file))
    del normalized

    availability = {
        "race_table": "mixed pre-race context and explicitly named analysis outcomes",
        "runner_pre_race": "strict PIT-C before target date",
        "outcomes": "analysis-only current-race outcome",
        "historical_performance": "available only from the following date",
        "connection_state": "strict PIT-C before target date",
        "market_oracle": "final market; explicit oracle join only",
        "raw_status_population": "analysis-only result/status audit",
    }
    summaries: dict[str, Any] = {}
    for name, frame in views.items():
        if pd.to_datetime(frame["race_date" if "race_date" in frame else "date"]).max() > EDA_MAX_DATE:
            raise AssertionError(f"{name} crosses the cutoff")
        frame.to_csv(output / "views" / f"{name}.csv.gz", index=False, compression="gzip")
        summaries[name] = _view_summary(
            name, frame, raw_sha256=raw_sha256, availability=availability[name]
        )
    tables = build_aggregate_tables(views)
    for name, table in tables.items():
        table.to_csv(output / "tables" / f"{name}.csv", index=False)
    yearly = tables["yearly_coverage_drift"]
    (output / "plots" / "yearly_race_count.svg").write_text(
        _svg_bar(yearly, "year", "races", "Race coverage by year (2013–2022)"),
        encoding="utf-8",
    )
    _write_report(output, summaries, tables)
    (output / "logs" / "run.log").write_text(
        "Phase 5A common views and aggregates completed; cutoff checks passed.\n",
        encoding="utf-8",
    )

    git_commit = _git_value(root, "rev-parse", "HEAD")
    git_status = _git_value(root, "status", "--short")
    artifact_manifest = {
        "analysis_id": config["analysis_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "max_target_date": str(EDA_MAX_DATE.date()),
        "raw_sha256": raw_sha256,
        "config_sha256": sha256_file(config_file),
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "seed": config["seed"],
        "workstreams": config["workstreams"],
        "views": summaries,
        "files": {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    _json_dump(output / "manifest.json", artifact_manifest)
    _json_dump(output / "data_contract_summary.json", {"views": summaries})
    return artifact_manifest


def public_data_contract_summary(local_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Remove workstation paths and column-level recoverability from a local manifest."""

    return {
        "analysis_id": local_manifest["analysis_id"],
        "max_target_date": local_manifest["max_target_date"],
        "raw_sha256": local_manifest["raw_sha256"],
        "git_commit": local_manifest["git_commit"],
        "views": {
            name: {
                key: payload[key]
                for key in (
                    "row_count",
                    "race_count",
                    "date_min",
                    "date_max",
                    "key_columns",
                    "key_unique",
                    "schema_hash",
                    "feature_availability_time",
                )
            }
            for name, payload in local_manifest["views"].items()
        },
    }
