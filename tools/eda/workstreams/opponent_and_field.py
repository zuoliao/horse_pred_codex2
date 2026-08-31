from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from horse_pred.data import RAW_COLUMNS, normalize_raw, sha256_file
from horse_pred.features import FeatureConfig, is_flat_race
from horse_pred.pace_recent import PACE_RECENT_COLUMN, build_pace_recent_history
from horse_pred.rating import RatingEvent, RatingSpec, build_rating_history_from_events
from horse_pred.speed_figure import SpeedFigureSpec, _solve_ridge, condition_design_vector

RAW = Path(os.environ["HORSE_EDA_RAW_PATH"])
OUTPUT = Path(
    os.environ.get(
        "HORSE_EDA_OPPONENT_OUTPUT",
        "artifacts/eda_20260901/workstreams/e_opponent/summary.json",
    )
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
SEED = 20260901
RESAMPLES = 1000


def period_for_year(year: int) -> str | None:
    if 2014 <= year <= 2019:
        return "discovery_2014_2019"
    if 2020 <= year <= 2021:
        return "replication_2020_2021"
    if year == 2022:
        return "confirmation_2022"
    return None


def scalar(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    return value


def describe(values: pd.Series) -> dict[str, object]:
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if x.empty:
        return {"n": 0, "mean": None, "sd": None, "q10": None, "median": None, "q90": None}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "q10": float(x.quantile(0.1)),
        "median": float(x.median()),
        "q90": float(x.quantile(0.9)),
    }


def date_bootstrap_mean(frame: pd.DataFrame, value: str) -> dict[str, object]:
    source = frame.loc[np.isfinite(pd.to_numeric(frame[value], errors="coerce")), ["date", value]].copy()
    if source.empty:
        return {"estimate": None, "ci95": [None, None], "n": 0, "dates": 0}
    source[value] = pd.to_numeric(source[value], errors="raise").astype(float)
    daily = source.groupby("date", observed=True)[value].agg(["sum", "count"])
    sums, counts = daily["sum"].to_numpy(float), daily["count"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    draws = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        sample = rng.integers(0, len(daily), len(daily))
        draws[i] = sums[sample].sum() / counts[sample].sum()
    return {
        "estimate": float(source[value].mean()),
        "ci95": [float(v) for v in np.quantile(draws, [0.025, 0.975])],
        "n": int(len(source)),
        "dates": int(len(daily)),
        "unit": "race_date bootstrap",
        "resamples": RESAMPLES,
    }


def date_bootstrap_corr(frame: pd.DataFrame, left: str, right: str) -> dict[str, object]:
    source = frame.loc[:, ["date", left, right]].copy()
    source[left] = pd.to_numeric(source[left], errors="coerce")
    source[right] = pd.to_numeric(source[right], errors="coerce")
    source = source.dropna()
    daily = (
        source.groupby("date", observed=True)
        .apply(
            lambda x: x[left].corr(x[right], method="spearman") if len(x) >= 3 else np.nan,
            include_groups=False,
        )
        .dropna()
        .to_numpy(float)
    )
    estimate = float(daily.mean()) if len(daily) else np.nan
    rng = np.random.default_rng(SEED)
    draws = (
        [float(rng.choice(daily, size=len(daily), replace=True).mean()) for _ in range(RESAMPLES)] if len(daily) else []
    )
    return {
        "estimate": None if not np.isfinite(estimate) else estimate,
        "ci95": [float(v) for v in np.quantile(draws, [0.025, 0.975])] if draws else [None, None],
        "n": int(len(source)),
        "dates": int(source.date.nunique()),
        "unit": "race_date bootstrap of daily Spearman means; point uses the same estimand",
        "resamples": RESAMPLES,
    }


def class_group(value: object) -> str:
    text = re.sub(r"\s+", "", "" if pd.isna(value) else str(value).replace("\u00a0", ""))
    if "新馬" in text:
        return "new"
    if "未勝利" in text:
        return "maiden"
    if "1勝" in text or "500万" in text:
        return "1win"
    if "2勝" in text or "1000万" in text:
        return "2win"
    if "3勝" in text or "1600万" in text:
        return "3win"
    if any(token in text.upper() for token in ("OPEN", "OP", "G1", "G2", "G3")) or "オープン" in text or "重賞" in text:
        return "open"
    return "unknown"


def field_band(n: int) -> str:
    if n <= 9:
        return "02_09_or_less"
    if n <= 13:
        return "10_13"
    if n <= 16:
        return "14_16"
    return "17_plus"


raw_chunks = []
raw_rows_read = 0
for chunk in pd.read_csv(
    RAW,
    encoding="utf-8-sig",
    dtype=str,
    keep_default_na=False,
    na_filter=False,
    chunksize=50_000,
):
    if tuple(chunk.columns) != RAW_COLUMNS:
        raise AssertionError("approved raw schema changed")
    raw_rows_read += len(chunk)
    raw_chunks.append(chunk.loc[chunk["date"].le("2022-12-31")].copy())
raw = normalize_raw(pd.concat(raw_chunks, ignore_index=True))
raw["date"] = pd.to_datetime(raw["race_date"], errors="raise").dt.normalize()
if raw["date"].dt.year.ge(2023).any() or raw["date"].max().year != 2022:
    raise AssertionError("time firewall failed")
raw["year"] = raw["date"].dt.year
raw["period"] = raw["year"].map(period_for_year)
raw["source_position"] = np.arange(len(raw), dtype=np.int64)

flat_ids = []
for race_id, race in raw.groupby("race_id", sort=False):
    if is_flat_race(race, FeatureConfig()):
        flat_ids.append(str(race_id))
flat = raw.loc[raw["race_id"].astype(str).isin(flat_ids)].copy()

# The canonical ordinal rating state does not consume clocks. Construct the
# same event stream directly to avoid parsing unused runner times.
events = []
for race_id, race in flat.sort_values(["date", "race_id", "source_position"], kind="stable").groupby(
    "race_id", sort=False
):
    starters = race.loc[race["started"].eq(True).fillna(False)].copy()
    if starters.empty:
        continue
    finishes = pd.to_numeric(starters["finish_position"], errors="coerce")
    events.append(
        RatingEvent(
            race_id=str(race_id),
            race_date=pd.Timestamp(starters["date"].iloc[0]),
            surface_key=str(starters["course_type"].iloc[0]),
            horse_ids=tuple(starters["horse_id"]),
            finishes=tuple(float(v) for v in finishes.to_numpy(dtype=float, na_value=np.nan)),
            source_positions=tuple(int(v) for v in starters["source_position"]),
        )
    )
events.sort(key=lambda event: (event.race_date, event.race_id))
rating = build_rating_history_from_events(
    events,
    RatingSpec(family="pairwise_elo", initial_rating=1500.0, k=24.0, scale=400.0),
)
rating["race_id"] = rating["race_id"].astype(str)
rating["horse_id"] = rating["horse_id"].astype(str)
metadata = raw.loc[
    :,
    [
        "source_position",
        "race_id",
        "horse_id",
        "date",
        "period",
        "pit_c_scoring_eligible",
        "started",
        "winner_label",
        "race_class",
        "course_type",
        "time_raw",
        "distance",
        "status",
    ],
].copy()
metadata["race_id"] = metadata["race_id"].astype(str)
metadata["horse_id"] = metadata["horse_id"].astype(str)
frame = rating.merge(metadata, on=["source_position", "race_id", "horse_id"], how="left", validate="one_to_one")
if frame["date"].isna().any():
    raise AssertionError("rating/source-position join failed")

# Exact strict predictive choice-set denominator used by Workstream B.
eligible_ids = flat.loc[flat["pit_c_scoring_eligible"].eq(True), "race_id"].astype(str).unique()
strict = frame.loc[frame["race_id"].isin(eligible_ids) & frame["started"].eq(True).fillna(False)].copy()
winner_counts = strict.groupby("race_id", observed=True)["winner_label"].sum()
strict_ids = winner_counts.loc[winner_counts.gt(0)].index.astype(str)
strict = strict.loc[strict["race_id"].isin(strict_ids)].copy()
strict["field_size"] = strict.groupby("race_id", observed=True)["race_id"].transform("size").astype(int)
strict["class_group"] = strict["race_class"].map(class_group)
strict["field_band"] = strict["field_size"].map(field_band)

# Current field composition uses only ratings emitted before this race date.
field_rows = []
for race_id, race in strict.groupby("race_id", sort=False):
    ratings = race["global_state_pre"].to_numpy(float)
    starts = race["modular_rating__global_starts_pre"].to_numpy(float)
    n = len(race)
    scores = (ratings - 1500.0) * math.log(10.0) / 400.0
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    max_mask = ratings == ratings.max()
    winner = pd.to_numeric(race["finish_position"], errors="coerce").eq(1).to_numpy()
    pace_vals = pd.to_numeric(race.get(PACE_RECENT_COLUMN, pd.Series(np.nan, index=race.index)), errors="coerce")
    front_excess = np.maximum(pace_vals.to_numpy(float) - 0.5, 0.0)
    experienced = starts >= 3
    field_rows.append(
        {
            "race_id": str(race_id),
            "date": race["date"].iloc[0],
            "period": race["period"].iloc[0],
            "class_group": race["class_group"].iloc[0],
            "field_band": race["field_band"].iloc[0],
            "field_size": n,
            "field_rating_mean": float(ratings.mean()),
            "field_rating_max": float(ratings.max()),
            "field_rating_sd": float(ratings.std(ddof=0)),
            "field_rating_spread": float(ratings.max() - ratings.min()),
            "field_top3_mean": float(np.sort(ratings)[-min(3, n) :].mean()),
            "normalized_entropy": float(-(probs * np.log(probs)).sum() / math.log(n)),
            "effective_contenders": float(1.0 / (probs**2).sum()),
            "max_choice_probability": float(probs.max()),
            "top_rating_hit_tie_fraction": float(winner[max_mask].sum() / max_mask.sum()),
            "cold_share": float((starts == 0).mean()),
            "experienced_share": float(experienced.mean()),
            "median_starts": float(np.median(starts)),
            "mean_uncertainty": float((1.0 / np.sqrt(starts + 1.0)).mean()),
            "experienced_field_mean": float(ratings[experienced].mean()) if experienced.any() else np.nan,
            "front_known_share": float(pace_vals.notna().mean()),
            "front_candidate_count": int(np.nansum(front_excess > 0)),
            "front_pressure_sum": float(np.nansum(front_excess)) if pace_vals.notna().any() else np.nan,
        }
    )
fields = pd.DataFrame(field_rows)

# Target-specific leave-one-out summaries. These diagnose semantics, not candidates.
for _race_id, indexes in strict.groupby("race_id", sort=False).groups.items():
    idx = np.asarray(list(indexes))
    values = strict.loc[idx, "global_state_pre"].to_numpy(float)
    n = len(values)
    strict.loc[idx, "inclusive_mean"] = values.mean()
    strict.loc[idx, "opponent_mean"] = (values.sum() - values) / (n - 1)
    opp_max, opp_top3, opp_spread = [], [], []
    for i in range(n):
        other = np.delete(values, i)
        opp_max.append(other.max())
        opp_top3.append(np.sort(other)[-min(3, len(other)) :].mean())
        opp_spread.append(other.max() - other.min())
    strict.loc[idx, "opponent_max"] = opp_max
    strict.loc[idx, "opponent_top3_mean"] = opp_top3
    strict.loc[idx, "opponent_spread"] = opp_spread

# Frozen PACE-01 state, joined only after all PIT-safe values were emitted.
pace = build_pace_recent_history(raw, through_year=2022)
pace["race_id"] = pace["race_id"].astype(str)
pace["horse_id"] = pace["horse_id"].astype(str)
strict = strict.merge(
    pace[["race_id", "horse_id", PACE_RECENT_COLUMN]], on=["race_id", "horse_id"], how="left", validate="one_to_one"
)
# Refresh current field front summaries now that pace is available.
front_by_race = strict.groupby("race_id", observed=True)[PACE_RECENT_COLUMN].agg(
    front_known=lambda x: int(x.notna().sum()),
    front_candidate_count=lambda x: int((pd.to_numeric(x, errors="coerce") > 0.5).sum()),
    front_pressure_sum=lambda x: float(np.maximum(pd.to_numeric(x, errors="coerce") - 0.5, 0).sum()),
)
fields = fields.drop(columns=["front_known_share", "front_candidate_count", "front_pressure_sum"]).merge(
    front_by_race, left_on="race_id", right_index=True, how="left", validate="one_to_one"
)
fields["front_known_share"] = fields["front_known"] / fields["field_size"]

# Prior-surface switchers (same-day values are emitted before any date update).
last_surface: dict[object, str] = {}
switch_values = pd.Series(np.nan, index=strict.index, dtype=float)
ordered_all = frame.sort_values(["date", "race_id", "source_position"], kind="stable")
switch_map = {}
for _event_date, day in ordered_all.groupby("date", sort=True):
    for row in day.itertuples():
        previous = last_surface.get(row.horse_id)
        switch_map[(row.race_id, row.horse_id)] = np.nan if previous is None else float(previous != row.course_type)
    for row in day.itertuples():
        last_surface[row.horse_id] = row.course_type
strict["surface_switcher"] = [switch_map.get((r, h), np.nan) for r, h in zip(strict.race_id, strict.horse_id)]
switch = strict.groupby("race_id", observed=True)["surface_switcher"].agg(switch_known="count", switcher_share="mean")
fields = fields.merge(switch, left_on="race_id", right_index=True, how="left", validate="one_to_one")

# Historical past-race opponent observations: target result is never used in the observation.
all_obs = frame.copy()
for _race_id, indexes in all_obs.groupby("race_id", sort=False).groups.items():
    idx = np.asarray(list(indexes))
    vals = all_obs.loc[idx, "global_state_pre"].to_numpy(float)
    n = len(vals)
    all_obs.loc[idx, "obs_inclusive"] = vals.mean()
    all_obs.loc[idx, "obs_opponent"] = (vals.sum() - vals) / (n - 1) if n > 1 else np.nan

history_state: dict[object, dict[str, object]] = {}
history_records = []
decay_rate = math.log(2.0) / 90.0
for event_date, day in all_obs.sort_values(["date", "race_id", "source_position"], kind="stable").groupby(
    "date", sort=True
):
    date = pd.Timestamp(event_date)
    for row in day.itertuples():
        s = history_state.get(row.horse_id)
        out = {"race_id": row.race_id, "horse_id": row.horse_id}
        if s is None or not s["values"]:
            out.update(
                {
                    k: np.nan
                    for k in (
                        "hist_career_opp",
                        "hist_career_inclusive",
                        "hist_last1_opp",
                        "hist_last3_opp",
                        "hist_last5_opp",
                        "hist_decay90_opp",
                        "hist_trend_opp",
                    )
                }
            )
            out["hist_count"] = 0
        else:
            vals = list(s["values"])
            inc = list(s["inclusive"])
            factor = math.exp(-decay_rate * (date - s["decay_date"]).days)
            dec_total = s["decay_total"] * factor
            dec_weight = s["decay_weight"] * factor
            career = float(np.mean(vals))
            out.update(
                {
                    "hist_count": len(vals),
                    "hist_career_opp": career,
                    "hist_career_inclusive": float(np.mean(inc)),
                    "hist_last1_opp": vals[-1],
                    "hist_last3_opp": float(np.mean(vals[-3:])),
                    "hist_last5_opp": float(np.mean(vals[-5:])),
                    "hist_decay90_opp": dec_total / dec_weight,
                    "hist_trend_opp": dec_total / dec_weight - career,
                }
            )
        history_records.append(out)
    for row in day.itertuples():
        if not np.isfinite(row.obs_opponent):
            continue
        s = history_state.get(row.horse_id)
        if s is None:
            s = {"values": [], "inclusive": [], "decay_total": 0.0, "decay_weight": 0.0, "decay_date": date}
            history_state[row.horse_id] = s
        factor = math.exp(-decay_rate * (date - s["decay_date"]).days)
        s["decay_total"] = s["decay_total"] * factor + float(row.obs_opponent)
        s["decay_weight"] = s["decay_weight"] * factor + 1.0
        s["decay_date"] = date
        s["values"].append(float(row.obs_opponent))
        s["inclusive"].append(float(row.obs_inclusive))
history = pd.DataFrame(history_records)
strict = strict.merge(history, on=["race_id", "horse_id"], how="left", validate="one_to_one")

# Current performance residual × pre-race field quality. The outcome residual is analysis-only.
clock_parts = raw["time_raw"].fillna("").astype(str).str.extract(r"^(\d+):(\d{2})\.(\d)$", expand=True)
clock_num = clock_parts.apply(pd.to_numeric, errors="coerce")
raw["clock_sec_fast"] = (clock_num[0] * 60 + clock_num[1] + clock_num[2] / 10).where(clock_num[1].lt(60))
flat["clock_sec_fast"] = raw.loc[flat.index, "clock_sec_fast"]
speed_spec = SpeedFigureSpec()
xtx = np.zeros((51, 51))
xty = np.zeros(51)
clean_count = 0
residual_records = []
strict_id_set = set(strict["race_id"].astype(str).unique())
flat_ordered = flat.sort_values(["date", "race_id", "source_position"], kind="stable")
for event_date, day in flat_ordered.groupby("date", sort=True):
    beta = _solve_ridge(xtx, xty, speed_spec.ridge_alpha) if clean_count >= speed_spec.min_prior_clean_races else None
    updates = []
    for race_id, race in day.groupby("race_id", sort=False):
        design = condition_design_vector(race)
        if design is None or race["status"].astype(str).isin(("demoted", "disqualified")).any():
            continue
        distances = pd.to_numeric(race["distance"], errors="coerce").dropna().unique()
        finish = pd.to_numeric(race["finish_position"], errors="coerce")
        times = pd.to_numeric(race["clock_sec_fast"], errors="coerce")
        timed = race["started"].eq(True).fillna(False) & finish.notna() & times.notna()
        winners = timed & finish.eq(1)
        if len(distances) != 1 or distances[0] <= 0 or not winners.any():
            continue
        winner_times = times.loc[winners].astype(float)
        if (
            winner_times.max() - winner_times.min() > 1e-12
            or (timed & finish.gt(1) & times.lt(winner_times.min())).any()
        ):
            continue
        winner_clock = float(winner_times.min()) * 1000 / float(distances[0])
        if beta is not None and str(race_id) in strict_id_set and pd.Timestamp(event_date).year >= 2014:
            expected = float(design @ beta)
            for idx in race.index[timed]:
                residual_records.append(
                    {
                        "race_id": str(race_id),
                        "horse_id": str(race.at[idx, "horse_id"]),
                        "performance_residual": float(
                            np.clip(expected - float(times.at[idx]) * 1000 / float(distances[0]), -5, 5)
                        ),
                    }
                )
        updates.append((design, winner_clock))
    for design, winner_clock in updates:
        xtx += np.outer(design, design)
        xty += design * winner_clock
        clean_count += 1
residuals = pd.DataFrame(residual_records)
strict = strict.merge(residuals, on=["race_id", "horse_id"], how="left", validate="one_to_one")
strict = strict.merge(
    fields[["race_id", "field_rating_mean", "field_rating_spread"]], on="race_id", how="left", validate="many_to_one"
)

# Aggregate-only output.
period_counts = {}
current_semantics = {}
history_summary = {}
two_axis = {}
field_structure = {}
for period, runners in strict.loc[strict["period"].notna()].groupby("period", sort=True):
    race_part = fields.loc[fields["period"].eq(period)].copy()
    period_counts[period] = {
        "races": int(runners.race_id.nunique()),
        "runners": int(len(runners)),
        "dates": int(runners.date.nunique()),
    }
    within = []
    for _, race in runners.groupby("race_id", sort=False):
        within.append(race["global_state_pre"].corr(race["opponent_mean"], method="spearman"))
    current_semantics[period] = {
        "self_vs_opponent_mean_within_race_spearman": describe(pd.Series(within)),
        "opponent_minus_inclusive": describe(runners["opponent_mean"] - runners["inclusive_mean"]),
        "self_vs_opponent_mean_global_spearman": float(
            runners["global_state_pre"].corr(runners["opponent_mean"], method="spearman")
        ),
        "opponent_top3_minus_inclusive_top3": describe(
            runners["opponent_top3_mean"]
            - runners.groupby("race_id")["global_state_pre"].transform(lambda x: np.sort(x)[-min(3, len(x)) :].mean())
        ),
    }
    history_summary[period] = {
        "coverage_any": float(runners["hist_count"].gt(0).mean()),
        "coverage_3plus": float(runners["hist_count"].ge(3).mean()),
        "coverage_5plus": float(runners["hist_count"].ge(5).mean()),
        "career_opp_minus_inclusive": describe(runners["hist_career_opp"] - runners["hist_career_inclusive"]),
        "career_vs_decay90_spearman": float(
            runners["hist_career_opp"].corr(runners["hist_decay90_opp"], method="spearman")
        ),
        "last3_vs_last5_spearman": float(runners["hist_last3_opp"].corr(runners["hist_last5_opp"], method="spearman")),
        "trend": describe(runners["hist_trend_opp"]),
    }
    valid = runners.dropna(subset=["performance_residual", "field_rating_mean"]).copy()
    valid["field_quality_z"] = (valid["field_rating_mean"] - valid["field_rating_mean"].mean()) / valid[
        "field_rating_mean"
    ].std(ddof=1)
    valid["good_performance"] = valid["performance_residual"].gt(0).astype(float)
    valid["strong_field"] = valid["field_rating_mean"].gt(valid["field_rating_mean"].median()).astype(float)
    valid["strong_and_good"] = valid["good_performance"] * valid["strong_field"]
    two_axis[period] = {
        "coverage": float(len(valid) / len(runners)),
        "runners": int(len(valid)),
        "races": int(valid.race_id.nunique()),
        "runner_spearman": date_bootstrap_corr(valid, "field_rating_mean", "performance_residual"),
        "good_performance_rate": date_bootstrap_mean(valid, "good_performance"),
        "strong_field_and_good_rate": date_bootstrap_mean(valid, "strong_and_good"),
        "performance_residual": describe(valid["performance_residual"]),
    }
    field_structure[period] = {
        "field_rating_mean": describe(race_part["field_rating_mean"]),
        "field_rating_spread": describe(race_part["field_rating_spread"]),
        "normalized_entropy": date_bootstrap_mean(race_part, "normalized_entropy"),
        "effective_contenders": describe(race_part["effective_contenders"]),
        "top_rating_hit": date_bootstrap_mean(race_part, "top_rating_hit_tie_fraction"),
        "cold_share": date_bootstrap_mean(race_part, "cold_share"),
        "experienced_share": date_bootstrap_mean(race_part, "experienced_share"),
        "switcher_share": date_bootstrap_mean(race_part, "switcher_share"),
        "front_known_share": date_bootstrap_mean(race_part, "front_known_share"),
        "front_candidate_count": describe(race_part["front_candidate_count"]),
        "front_pressure_sum": describe(race_part["front_pressure_sum"]),
    }

by_class = {}
for (period, group), part in fields.loc[fields.period.notna()].groupby(["period", "class_group"], sort=True):
    by_class[f"{period}|{group}"] = {
        "races": int(len(part)),
        "field_mean": float(part.field_rating_mean.mean()),
        "spread": float(part.field_rating_spread.mean()),
        "cold_share": float(part.cold_share.mean()),
        "experienced_share": float(part.experienced_share.mean()),
    }
by_size = {}
for (period, band), part in fields.loc[fields.period.notna()].groupby(["period", "field_band"], sort=True):
    by_size[f"{period}|{band}"] = {
        "races": int(len(part)),
        "field_mean": float(part.field_rating_mean.mean()),
        "spread": float(part.field_rating_spread.mean()),
        "entropy": float(part.normalized_entropy.mean()),
    }

payload = {
    "contract": {
        "raw_sha256": sha256_file(RAW),
        "raw_rows_read_in_chunks": raw_rows_read,
        "retained_rows": int(len(raw)),
        "min_date": str(raw.date.min().date()),
        "max_date": str(raw.date.max().date()),
        "post_2022_retained": int(raw.date.dt.year.ge(2023).sum()),
        "rating_spec": {"family": "pairwise_elo", "initial": 1500, "k": 24, "scale": 400},
        "bootstrap_seed": SEED,
        "bootstrap_resamples": RESAMPLES,
    },
    "period_counts": period_counts,
    "current_opponent_semantics": current_semantics,
    "historical_opponent": history_summary,
    "field_structure": field_structure,
    "by_class": by_class,
    "by_field_size": by_size,
    "performance_residual_x_field_quality": two_axis,
    "missingness": {
        p: {
            "history_cold_start": float(g.hist_count.eq(0).mean()),
            "pace_history": float(g[PACE_RECENT_COLUMN].isna().mean()),
            "switcher_no_prior": float(g.surface_switcher.isna().mean()),
            "performance_residual": float(g.performance_residual.isna().mean()),
        }
        for p, g in strict.loc[strict.period.notna()].groupby("period", sort=True)
    },
    "invariants": {
        "target_max_year": int(strict.date.dt.year.max()),
        "same_date_rating_batch": "provided by build_rating_history",
        "future_opponent_results_used": False,
        "target_outcome_used_in_field_quality": False,
        "runner_level_artifact_written": False,
        "leave_one_out_identity": "opponent_mean=(field_sum-self_rating)/(n-1)",
        "speed_clean_races_through_2022": clean_count,
    },
}
OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=scalar) + "\n", encoding="utf-8")
print(
    json.dumps(
        {"output": str(OUTPUT), "period_counts": period_counts, "invariants": payload["invariants"]},
        ensure_ascii=False,
        indent=2,
    )
)
