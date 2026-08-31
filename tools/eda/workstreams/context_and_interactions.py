"""Workstream G: bounded context/interaction and OOT-error diagnostics.

This is a local/private EDA artifact.  It writes aggregate tables only.  The
canonical transition counts based on raw history are reported in the chapter;
this script complements them with pre-2023 cached PIT states and frozen rolling
OOT predictions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from horse_pred.data import sha256_file
from horse_pred.eda import EDA_MAX_DATE, load_eda_population

OUT = Path(os.environ.get("HORSE_EDA_CONTEXT_OUTPUT", "artifacts/eda_20260901/workstreams/g_context"))
RUNNER_VIEW = Path(os.environ["HORSE_EDA_RUNNER_VIEW"])
OUTCOMES_VIEW = Path(os.environ["HORSE_EDA_OUTCOMES_VIEW"])
PRED_PATH = Path(
    os.environ.get(
        "HORSE_EDA_ROLLING_PREDICTIONS",
        "artifacts/eval_roll_001_current_best_20260831/predictions_scoring.csv.gz",
    )
)
RAW_PATH = Path(os.environ["HORSE_EDA_RAW_PATH"])
MAX_DATE = pd.Timestamp("2022-12-31")


def period(year: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=year.index, dtype="string")
    out.loc[year.between(2014, 2019)] = "discovery"
    out.loc[year.between(2020, 2021)] = "replication"
    out.loc[year.eq(2022)] = "confirmation"
    return out


def centered_summary(frame: pd.DataFrame, interaction: str, cell: str) -> pd.DataFrame:
    work = frame[["period", "race_id", "race_date", "winner", "base", cell]].dropna()
    work["centered_win"] = work["winner"] - work["base"]
    rows = []
    for (phase, value), group in work.groupby(["period", cell], observed=True):
        race_values = group.groupby(["race_id", "race_date"], observed=True)["centered_win"].mean().reset_index()
        daily = race_values.groupby("race_date", observed=True)["centered_win"].mean()
        rows.append(
            {
                "interaction": interaction,
                "period": phase,
                "cell": str(value),
                "runners": len(group),
                "races": group["race_id"].nunique(),
                "dates": group["race_date"].nunique(),
                "missing": 0,
                "centered_win": race_values["centered_win"].mean(),
                "date_sd": daily.std(ddof=1),
                "aggregation": "race macro of within-race cell mean; date-block SD",
            }
        )
    return pd.DataFrame(rows)


def class_tier(value: pd.Series) -> pd.Series:
    text = value.astype("string")
    out = pd.Series(np.nan, index=text.index, dtype=float)
    out.loc[text.str.contains("新馬|未勝利", na=False)] = 0
    out.loc[text.str.contains("500万|1勝", na=False)] = 1
    out.loc[text.str.contains("1000万|2勝", na=False)] = 2
    out.loc[text.str.contains("1600万|3勝", na=False)] = 3
    out.loc[text.str.contains("オープン|G[123]|重賞", na=False)] = 4
    return out


def build_raw_transitions() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | str]]:
    raw = load_eda_population(RAW_PATH, max_date=EDA_MAX_DATE)
    raw["race_date"] = pd.to_datetime(raw["race_date"], errors="raise")
    assert raw["race_date"].max() <= MAX_DATE
    flat_row = raw["surface"].isin(["turf", "dirt"]) & ~raw["race_class"].astype("string").str.contains(
        "障害", na=False
    )
    flat_race = flat_row.groupby(raw["race_id"], observed=True).transform("all")
    started = raw.loc[flat_race & raw["started"].fillna(False)].copy()
    started["field_size"] = started.groupby("race_id", observed=True)["race_id"].transform("size")
    finish = pd.to_numeric(started["finish_position"], errors="coerce")
    started["performance"] = 1 - (finish - 1) / (started["field_size"] - 1).replace(0, np.nan)
    started.loc[finish.isna(), "performance"] = 0.0
    started["winner"] = finish.eq(1).astype(float)
    started["class_tier"] = class_tier(started["race_class"])
    started["direction"] = started["around"].astype("string").replace("", pd.NA)
    started = started.sort_values(["horse_id", "race_date", "race_id"], kind="stable")
    for current, previous in [
        ("race_date", "prev_date"),
        ("surface", "prev_surface"),
        ("distance_m", "prev_distance"),
        ("venue", "prev_venue"),
        ("direction", "prev_direction"),
        ("class_tier", "prev_class_tier"),
        ("horse_age", "prev_age"),
        ("field_size", "prev_field_size"),
        ("performance", "prev_performance"),
    ]:
        started[previous] = started.groupby("horse_id", observed=True)[current].shift(1)
    invalid = started["prev_date"].notna() & started["prev_date"].ge(started["race_date"])
    if invalid.any():
        raise AssertionError("strict-prior transition firewall violated")
    target = started.loc[
        started["pit_c_scoring_eligible"].fillna(False) & started["race_date"].dt.year.between(2014, 2022)
    ].copy()
    target["period"] = period(target["race_date"].dt.year)
    target["base"] = 1 / target["field_size"]
    target["surface_transition"] = np.select(
        [
            target["prev_surface"].eq(target["surface"]),
            target["prev_surface"].eq("dirt") & target["surface"].eq("turf"),
            target["prev_surface"].eq("turf") & target["surface"].eq("dirt"),
        ],
        [target["surface"].astype(str) + "_same", "dirt_to_turf", "turf_to_dirt"],
        default="missing",
    )
    delta = target["distance_m"] - target["prev_distance"]
    target["distance_transition"] = np.select(
        [delta.abs().le(200).fillna(False), delta.ge(400).fillna(False), delta.le(-400).fillna(False)],
        ["within_200", "longer_400+", "shorter_400+"],
        default="change_201_399",
    )
    target.loc[delta.isna(), "distance_transition"] = "missing"
    target["venue_transition"] = np.where(
        target["prev_venue"].isna(),
        "missing",
        np.where(target["venue"].eq(target["prev_venue"]).fillna(False), "same", "switch"),
    )
    target["direction_transition"] = np.where(
        target["prev_direction"].isna() | target["direction"].isna(),
        "missing",
        np.where(target["direction"].eq(target["prev_direction"]).fillna(False), "same", "switch"),
    )
    class_delta = target["class_tier"] - target["prev_class_tier"]
    target["class_transition"] = np.select(
        [class_delta.lt(0).fillna(False), class_delta.eq(0).fillna(False), class_delta.gt(0).fillna(False)],
        ["down", "same", "up"],
        default="missing",
    )
    rest = (target["race_date"] - target["prev_date"]).dt.days
    target["rest_transition"] = (
        pd.cut(rest, [-np.inf, 13, 29, 89, 179, np.inf], labels=["0-13", "14-29", "30-89", "90-179", "180+"])
        .astype("string")
        .fillna("history0")
    )
    fs_delta = target["field_size"] - target["prev_field_size"]
    target["field_size_transition"] = np.select(
        [
            fs_delta.abs().le(1).fillna(False),
            fs_delta.between(2, 4).fillna(False),
            fs_delta.between(-4, -2).fillna(False),
            fs_delta.ge(5).fillna(False),
            fs_delta.le(-5).fillna(False),
        ],
        ["similar", "larger_2-4", "smaller_2-4", "larger_5+", "smaller_5+"],
        default="missing",
    )

    rows = []
    for transition in [
        "surface_transition",
        "distance_transition",
        "venue_transition",
        "direction_transition",
        "class_transition",
        "rest_transition",
        "field_size_transition",
    ]:
        for (phase, cell), group in target.loc[~target[transition].eq("missing")].groupby(
            ["period", transition], observed=True
        ):
            corrs = []
            for _, race in group.dropna(subset=["prev_performance", "performance"]).groupby("race_id", observed=True):
                if len(race) >= 3 and race["prev_performance"].nunique() > 1 and race["performance"].nunique() > 1:
                    corrs.append(race["prev_performance"].corr(race["performance"], method="spearman"))
            rows.append(
                {
                    "transition": transition,
                    "period": phase,
                    "cell": str(cell),
                    "runners": len(group),
                    "races": group["race_id"].nunique(),
                    "dates": group["race_date"].nunique(),
                    "missing_previous": int(group["prev_performance"].isna().sum()),
                    "win_rate": group["winner"].mean(),
                    "uniform_base": group["base"].mean(),
                    "win_minus_base": (group["winner"] - group["base"]).mean(),
                    "persistence_race_spearman": float(np.nanmean(corrs)) if corrs else np.nan,
                    "persistence_races": len(corrs),
                    "aggregation": "runner win minus uniform base; race-macro Spearman",
                }
            )
    keep = [
        "race_id",
        "race_date",
        "horse_id",
        "surface_transition",
        "distance_transition",
        "venue_transition",
        "direction_transition",
        "class_transition",
        "rest_transition",
        "field_size_transition",
    ]
    manifest = {
        "raw_sha256": sha256_file(RAW_PATH),
        "max_target_date": str(target["race_date"].max().date()),
        "rows_2023_or_later_analyzed": 0,
        "strict_prior_violations": 0,
        "target_runners": int(len(target)),
        "target_races": int(target["race_id"].nunique()),
    }
    return target[keep], pd.DataFrame(rows), manifest


def load_frame() -> tuple[pd.DataFrame, dict[str, int | str]]:
    cols = [
        "race_date",
        "race_id",
        "horse_id",
        "context__field_size_rows",
        "context__age",
        "context__distance",
        "context__class_tier",
        "horse_history__career__starts",
        "horse_history__days_since_last_start",
        "horse_history__days_30__starts",
        "rating__field_std_elo_pre",
        "jockey_history__decay_90d__win_rate",
        "jockey_history__career__starts",
        "horse_history__career__mean_opponent_elo",
    ]
    source = pd.read_csv(RUNNER_VIEW, usecols=cols)
    outcomes = pd.read_csv(
        OUTCOMES_VIEW,
        usecols=["race_id", "horse_id", "winner_label", "coherent_win_target"],
    )
    frame = source.merge(
        outcomes,
        on=["race_id", "horse_id"],
        how="inner",
        validate="one_to_one",
    )
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="raise")
    assert frame["race_date"].max() <= MAX_DATE
    assert not frame["race_date"].dt.year.ge(2023).any()
    frame["period"] = period(frame["race_date"].dt.year)
    frame = frame.loc[frame["period"].notna()].copy()
    frame["winner"] = frame["winner_label"].astype(float)
    frame["winner_weight"] = frame["coherent_win_target"].astype(float)
    frame["field_size"] = frame["context__field_size_rows"].astype(float)
    frame["base"] = 1.0 / frame["field_size"]
    manifest = {
        "source_rows": int(len(source)),
        "rows_after_pre2023_firewall": int(len(frame)),
        "races_after_pre2023_firewall": int(frame["race_id"].nunique()),
        "max_target_date": str(frame["race_date"].max().date()),
        "rows_2023_or_later_analyzed": 0,
        "source": "cutoff-enforced common runner_pre_race and outcomes views",
    }
    return frame, manifest


def interaction_tables(frame: pd.DataFrame) -> pd.DataFrame:
    frame["rest_band"] = (
        pd.cut(
            frame["horse_history__days_since_last_start"],
            [-np.inf, 13, 29, 89, 179, np.inf],
            labels=["0-13", "14-29", "30-89", "90-179", "180+"],
        )
        .astype("string")
        .fillna("history0")
    )
    frame["workload30"] = pd.cut(
        frame["horse_history__days_30__starts"], [-np.inf, 0, 1, np.inf], labels=["0", "1", "2+"]
    ).astype("string")
    frame["rest_x_workload"] = frame["rest_band"] + "|starts30=" + frame["workload30"]

    frame["age_band"] = pd.cut(
        frame["context__age"], [-np.inf, 2, 3, 5, np.inf], labels=["2", "3", "4-5", "6+"]
    ).astype("string")
    frame["starts_band"] = pd.cut(
        frame["horse_history__career__starts"], [-np.inf, 0, 3, 9, np.inf], labels=["0", "1-3", "4-9", "10+"]
    ).astype("string")
    frame["age_x_starts"] = frame["age_band"] + "|starts=" + frame["starts_band"]

    frame["experience"] = pd.cut(
        frame["horse_history__career__starts"], [-np.inf, 0, 3, np.inf], labels=["0", "1-3", "4+"]
    ).astype("string")
    # Quantile edges are frozen from discovery only, then reused.
    discovery = frame.loc[frame["period"].eq("discovery")]
    q1, q2 = discovery["jockey_history__decay_90d__win_rate"].quantile([1 / 3, 2 / 3])
    frame["jockey_form"] = pd.cut(
        frame["jockey_history__decay_90d__win_rate"],
        [-np.inf, q1, q2, np.inf],
        labels=["low", "mid", "high"],
        include_lowest=True,
    ).astype("string")
    frame["jockey_x_experience"] = frame["jockey_form"] + "|horse_starts=" + frame["experience"]

    race = frame.drop_duplicates("race_id").copy()
    fs_edges = [0, 11, 14, np.inf]
    race["field_band"] = pd.cut(race["field_size"], fs_edges, labels=["<=11", "12-14", "15+"])
    spread_q = discovery.drop_duplicates("race_id")["rating__field_std_elo_pre"].quantile([1 / 3, 2 / 3])
    race["spread_band"] = pd.cut(
        race["rating__field_std_elo_pre"],
        [-np.inf, spread_q.iloc[0], spread_q.iloc[1], np.inf],
        labels=["low", "mid", "high"],
        include_lowest=True,
    )
    race_cell = race[["race_id", "field_band", "spread_band"]]
    frame = frame.merge(race_cell, on="race_id", how="left", validate="many_to_one")
    frame["field_x_spread"] = frame["field_band"].astype("string") + "|spread=" + frame["spread_band"].astype("string")

    tables = [
        centered_summary(frame, "rest_x_workload30", "rest_x_workload"),
        centered_summary(frame, "age_x_career_starts", "age_x_starts"),
        centered_summary(frame, "jockey_form_x_horse_experience", "jockey_x_experience"),
        centered_summary(frame, "field_size_x_rating_spread", "field_x_spread"),
    ]
    return pd.concat(tables, ignore_index=True)


def pace_style_table() -> tuple[pd.DataFrame, dict[str, int | str]]:
    summary = pd.DataFrame(
        columns=[
            "interaction",
            "period",
            "cell",
            "runners",
            "races",
            "dates",
            "centered_win",
            "date_sd",
        ]
    )
    manifest = {
        "source_rows": 0,
        "pre2023_scored_rows": 0,
        "rows_2023_or_later_analyzed": 0,
        "status": "not rerun; only frozen PACE-01/PACE-02 evidence is cited",
        "reason": "the cutoff-safe common view excludes the rejected PACE-02 candidate",
    }
    return summary, manifest


def shallow_tree_diagnostic(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    from sklearn.tree import DecisionTreeRegressor, export_text

    features = [
        "context__field_size_rows",
        "context__age",
        "context__distance",
        "context__class_tier",
        "horse_history__career__starts",
        "horse_history__days_since_last_start",
        "horse_history__days_30__starts",
        "rating__field_std_elo_pre",
        "jockey_history__decay_90d__win_rate",
        "horse_history__career__mean_opponent_elo",
    ]
    discovery = frame.loc[frame["period"].eq("discovery"), features + ["winner", "base"]].copy()
    medians = discovery[features].median()
    x = discovery[features].fillna(medians)
    y = discovery["winner"] - discovery["base"]
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5000, random_state=20260901)
    tree.fit(x, y)
    out = frame[["period", "race_id", "race_date", "winner", "base"]].copy()
    out["leaf"] = tree.apply(frame[features].fillna(medians)).astype(str)
    summary = centered_summary(out, "discovery_shallow_tree_leaf", "leaf")
    return summary, export_text(tree, feature_names=features, decimals=5)


def oot_errors(frame: pd.DataFrame, transitions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | str]]:
    usecols = [
        "role",
        "evaluation_year",
        "race_id",
        "race_date",
        "method",
        "horse_id",
        "model_finish_position",
        "field_size",
        "probability_calibrated",
    ]
    retained = []
    source_rows = 0
    for chunk in pd.read_csv(PRED_PATH, usecols=usecols, chunksize=50_000):
        source_rows += len(chunk)
        keep = chunk["role"].eq("evaluation") & chunk["evaluation_year"].isin([2020, 2021, 2022])
        retained.append(chunk.loc[keep].copy())
    pred = pd.concat(retained, ignore_index=True)
    pred["race_date"] = pd.to_datetime(pred["race_date"], errors="raise")
    assert pred["race_date"].max() <= MAX_DATE
    assert set(pred["evaluation_year"].unique()) == {2020, 2021, 2022}
    assert not pred["evaluation_year"].ge(2023).any()
    pred["race_id"] = pred["race_id"].astype("string")
    pred["horse_id"] = pred["horse_id"].astype("string")

    context = frame[
        [
            "race_id",
            "horse_id",
            "winner",
            "winner_weight",
            "field_size",
            "context__age",
            "horse_history__career__starts",
            "horse_history__days_since_last_start",
            "rating__field_std_elo_pre",
        ]
    ].copy()
    context["race_id"] = context["race_id"].astype("string")
    context["horse_id"] = context["horse_id"].astype("string")
    joined = pred.merge(
        context, on=["race_id", "horse_id"], how="inner", validate="many_to_one", suffixes=("_pred", "")
    )
    transitions = transitions.copy()
    transitions["race_id"] = transitions["race_id"].astype("string")
    transitions["horse_id"] = transitions["horse_id"].astype("string")
    joined = joined.merge(
        transitions.drop(columns=["race_date"]),
        on=["race_id", "horse_id"],
        how="left",
        validate="many_to_one",
    )
    assert not joined["evaluation_year"].ge(2023).any()
    joined["period"] = np.where(joined["evaluation_year"].le(2021), "replication", "confirmation")
    joined["history_band"] = pd.cut(
        joined["horse_history__career__starts"],
        [-np.inf, 0, 3, 9, np.inf],
        labels=["0", "1-3", "4-9", "10+"],
    ).astype("string")
    joined["rest_band"] = (
        pd.cut(
            joined["horse_history__days_since_last_start"],
            [-np.inf, 13, 29, 89, 179, np.inf],
            labels=["0-13", "14-29", "30-89", "90-179", "180+"],
        )
        .astype("string")
        .fillna("history0")
    )
    joined["field_band"] = pd.cut(joined["field_size"], [0, 11, 14, np.inf], labels=["<=11", "12-14", "15+"]).astype(
        "string"
    )
    spread_edges = (
        frame.loc[frame["period"].eq("discovery")]
        .drop_duplicates("race_id")["rating__field_std_elo_pre"]
        .quantile([1 / 3, 2 / 3])
    )
    joined["spread_band"] = pd.cut(
        joined["rating__field_std_elo_pre"],
        [-np.inf, spread_edges.iloc[0], spread_edges.iloc[1], np.inf],
        labels=["low", "mid", "high"],
        include_lowest=True,
    ).astype("string")
    joined["field_x_spread"] = joined["field_band"] + "|spread=" + joined["spread_band"]

    max_probability = joined.groupby(["method", "race_id"], observed=True)["probability_calibrated"].transform("max")
    joined["predicted_top1"] = joined["probability_calibrated"].eq(max_probability)
    winners = joined.loc[joined["winner_weight"].gt(0)].copy()
    winners["top1_correct"] = winners["predicted_top1"].astype(float)
    winners["winner_log_loss"] = -np.log(winners["probability_calibrated"].clip(1e-15, 1))
    rows = []
    for slice_name in [
        "history_band",
        "rest_band",
        "field_band",
        "field_x_spread",
        "surface_transition",
        "distance_transition",
        "venue_transition",
        "direction_transition",
        "class_transition",
        "field_size_transition",
    ]:
        for keys, group in winners.groupby(["method", "period", slice_name], observed=True):
            method, phase, cell = keys
            weight = group["winner_weight"]
            weight_sum = float(weight.sum())
            daily = (
                group.assign(
                    weighted_loss=group["winner_log_loss"] * weight,
                    weighted_top1=group["top1_correct"] * weight,
                )
                .groupby("race_date", observed=True)
                .agg(
                    weight=("winner_weight", "sum"),
                    weighted_loss=("weighted_loss", "sum"),
                    weighted_top1=("weighted_top1", "sum"),
                )
            )
            daily["loss"] = daily["weighted_loss"] / daily["weight"]
            daily["top1"] = daily["weighted_top1"] / daily["weight"]
            loss = float(np.average(group["winner_log_loss"], weights=weight))
            top1 = float(np.average(group["top1_correct"], weights=weight))
            loss_half_width = float(1.96 * daily["loss"].std(ddof=1) / np.sqrt(len(daily)))
            top1_half_width = float(1.96 * daily["top1"].std(ddof=1) / np.sqrt(len(daily)))
            rows.append(
                {
                    "slice": slice_name,
                    "method": method,
                    "period": phase,
                    "cell": str(cell),
                    "races": group["race_id"].nunique(),
                    "race_equivalents": weight_sum,
                    "dates": group["race_date"].nunique(),
                    "missing": 0,
                    "top1": top1,
                    "top1_ci_low": top1 - top1_half_width,
                    "top1_ci_high": top1 + top1_half_width,
                    "winner_probability": float(np.average(group["probability_calibrated"], weights=weight)),
                    "winner_log_loss": loss,
                    "winner_log_loss_ci_low": loss - loss_half_width,
                    "winner_log_loss_ci_high": loss + loss_half_width,
                    "aggregation": "race-macro via coherent dead-heat winner weights",
                }
            )
    manifest = {
        "prediction_source_rows_seen_by_firewall": int(source_rows),
        "evaluation_rows_2020_2022_analyzed": int(len(pred)),
        "joined_rows": int(len(joined)),
        "joined_races": int(joined["race_id"].nunique()),
        "methods": sorted(joined["method"].unique().tolist()),
        "roles_analyzed": ["evaluation"],
        "evaluation_years_analyzed": [2020, 2021, 2022],
        "rows_2023_analyzed": 0,
        "max_prediction_date": str(pred["race_date"].max().date()),
    }
    return pd.DataFrame(rows), manifest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    transitions, transition_summary, raw_manifest = build_raw_transitions()
    transition_summary.to_csv(OUT / "transition_summary.csv", index=False)
    frame, frame_manifest = load_frame()
    interactions = interaction_tables(frame)
    pace_interactions, pace_manifest = pace_style_table()
    leaf_interactions, tree_text = shallow_tree_diagnostic(frame)
    interaction_parts = [interactions, leaf_interactions]
    if not pace_interactions.empty:
        interaction_parts.append(pace_interactions)
    interactions = pd.concat(interaction_parts, ignore_index=True)
    interactions.to_csv(OUT / "bounded_interactions.csv", index=False)
    (OUT / "discovery_shallow_tree.txt").write_text(tree_text)
    errors, pred_manifest = oot_errors(frame, transitions)
    errors.to_csv(OUT / "oot_error_slices.csv", index=False)
    manifest = {
        "workstream": "G_context_suitability_interactions",
        "status": "diagnostic_not_production",
        "frame": frame_manifest,
        "raw_transitions": raw_manifest,
        "predictions": pred_manifest,
        "pace_style": pace_manifest,
        "discovery_rule": "interaction cut points frozen on 2014-2019 only",
        "replication_rule": "2020-2021",
        "confirmation_rule": "2022",
        "market_columns_used": [],
        "all_pair_search": False,
        "output_files": [
            "transition_summary.csv",
            "bounded_interactions.csv",
            "oot_error_slices.csv",
            "discovery_shallow_tree.txt",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
