"""Workstream F: strict prior-date connection-state diagnostics.

This is a local/private analysis artifact.  It emits aggregates only and does
not alter the production feature pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(
    os.environ.get(
        "HORSE_EDA_CONNECTIONS_OUTPUT",
        str(ROOT / "artifacts/eda_20260901/workstreams/f_connections"),
    )
)
CACHE = Path(
    os.environ.get(
        "HORSE_EDA_HISTORY_CACHE",
        str(ROOT / "artifacts/eda_20260901/workstreams/c_history/runner_history_flat_private.pkl"),
    )
)
SEED = 20260901
BOOTSTRAPS = 400
PRIOR_N = 20.0


def period(year: int) -> str:
    if 2014 <= year <= 2019:
        return "discovery"
    if 2020 <= year <= 2021:
        return "replication"
    if year == 2022:
        return "confirmation"
    return "warmup"


def stable_seed(label: str) -> int:
    return SEED + int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")


def smoothed(wins: float, starts: float, prior_mean: float) -> float:
    return (wins + PRIOR_N * prior_mean) / (starts + PRIOR_N)


def posterior_sd(wins: float, starts: float, prior_mean: float) -> float:
    alpha = wins + PRIOR_N * prior_mean
    beta = starts - wins + PRIOR_N * (1.0 - prior_mean)
    total = alpha + beta
    return float(np.sqrt(alpha * beta / (total * total * (total + 1.0))))


def distance_band(value: float) -> str:
    if value < 1400:
        return "sprint_short"
    if value < 1800:
        return "sprint_mile"
    if value < 2200:
        return "middle"
    return "long"


def class_band(value: object) -> str:
    text = str(value)
    for key, name in (
        ("新馬", "debut"),
        ("未勝利", "maiden"),
        ("1勝", "1win"),
        ("2勝", "2win"),
        ("3勝", "3win"),
        ("G1", "graded"),
        ("G2", "graded"),
        ("G3", "graded"),
        ("オープン", "open"),
    ):
        if key in text:
            return name
    return "other"


def age_band(value: float) -> str:
    return "2" if value <= 2 else "3" if value == 3 else "4" if value == 4 else "5+"


def experience_band(value: float) -> str:
    return "debut" if value == 0 else "1" if value == 1 else "2-3" if value <= 3 else "4-9" if value <= 9 else "10+"


class State:
    def __init__(self) -> None:
        self.starts = 0.0
        self.wins = 0.0
        self.completed = 0.0
        self.finish_sum = 0.0
        self.events: deque[tuple[pd.Timestamp, float, float, bool]] = deque()
        self.conditions: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])

    def base(self, date: pd.Timestamp, prior_mean: float) -> dict[str, float]:
        while self.events and (date - self.events[0][0]).days > 365:
            self.events.popleft()
        out = {
            "career_starts": self.starts,
            "career_raw": self.wins / self.starts if self.starts else np.nan,
            "career_eb": smoothed(self.wins, self.starts, prior_mean),
            "career_sd": posterior_sd(self.wins, self.starts, prior_mean),
            "career_mean_finish": self.finish_sum / self.completed if self.completed else np.nan,
        }
        event_ages = [((date - e[0]).days, e) for e in self.events]
        for days in (30, 90, 365):
            eligible = [e for age, e in event_ages if age <= days]
            n = float(len(eligible))
            w = float(sum(e[1] for e in eligible))
            out[f"days{days}_starts"] = n
            out[f"days{days}_raw"] = w / n if n else np.nan
            out[f"days{days}_eb"] = smoothed(w, n, prior_mean)
        return out

    def conditional(self, name: str, value: str, prior_mean: float) -> tuple[float, float, float]:
        n, w = self.conditions.get((name, value), [0.0, 0.0])
        value_eb = smoothed(w, n, prior_mean)
        career_eb = smoothed(self.wins, self.starts, prior_mean)
        return n, value_eb, value_eb - career_eb

    def update(self, date: pd.Timestamp, win: float, finish: float, conditions: dict[str, str]) -> None:
        self.starts += 1.0
        self.wins += win
        if np.isfinite(finish):
            self.completed += 1.0
            self.finish_sum += finish
        self.events.append((date, win, finish, bool(np.isfinite(finish))))
        for name, value in conditions.items():
            aggregate = self.conditions[(name, value)]
            aggregate[0] += 1.0
            aggregate[1] += win


def load_flat() -> tuple[pd.DataFrame, dict[str, object]]:
    with CACHE.open("rb") as stream:
        runners, source_manifest = pickle.load(stream)
    runners = runners.copy()
    runners["race_date"] = pd.to_datetime(runners["race_date"], errors="raise")
    assert runners["race_date"].max() <= pd.Timestamp("2022-12-31")
    assert runners["surface"].isin(["turf", "dirt"]).all()
    assert not runners["race_class"].astype("string").str.contains("障害", na=False).any()
    assert runners["started"].fillna(False).all()
    assert runners.duplicated(["race_id", "horse_id"]).sum() == 0
    assert runners.duplicated(["horse_id", "race_date"]).sum() == 0
    # Eligible non-finishers have no numeric finish and are non-winners.  The
    # nullable exploratory label must not poison cumulative connection state.
    runners["winner"] = pd.to_numeric(runners["winner"], errors="coerce").fillna(0.0)
    runners["year"] = runners["race_date"].dt.year
    runners["period"] = runners["year"].map(period)
    return runners.sort_values(["race_date", "race_id", "horse_number"], kind="stable"), source_manifest


def construct_states(runners: pd.DataFrame, prior_mean: float) -> pd.DataFrame:
    states = {"jockey": defaultdict(State), "trainer": defaultdict(State)}
    horse_last_jockey: dict[object, object] = {}
    pair_starts: dict[tuple[object, object], int] = defaultdict(int)
    rows: list[dict[str, object]] = []
    for date, day in runners.groupby("race_date", sort=True):
        base_cache: dict[tuple[str, object], dict[str, float]] = {}
        for row in day.itertuples(index=False):
            conditions = {
                "surface": str(row.surface),
                "distance": distance_band(float(row.distance_m)),
                "venue": str(row.venue),
                "class": class_band(row.race_class),
                "age": age_band(float(row.horse_age)),
                "experience": experience_band(float(row.history_starts)),
                "layoff": "long" if np.isfinite(row.rest_days) and row.rest_days >= 180 else "not_long",
            }
            result: dict[str, object] = {
                "race_id": row.race_id,
                "race_date": date,
                "year": int(row.year),
                "period": row.period,
                "winner": float(row.winner),
                "current_performance": float(row.current_performance),
                "field_size": int(row.field_size),
                "history_starts": float(row.history_starts),
                "current_long_layoff": float(np.isfinite(row.rest_days) and row.rest_days >= 180),
            }
            for entity, key in (("jockey", row.jockey_id), ("trainer", row.trainer)):
                state = states[entity][key]
                cache_key = (entity, key)
                base = base_cache.get(cache_key)
                if base is None:
                    base = state.base(date, prior_mean)
                    base_cache[cache_key] = base
                for name, value in base.items():
                    result[f"{entity}_{name}"] = value
                for condition_name, condition_value in conditions.items():
                    n, eb, deviation = state.conditional(condition_name, condition_value, prior_mean)
                    result[f"{entity}_{condition_name}_starts"] = n
                    result[f"{entity}_{condition_name}_eb"] = eb
                    result[f"{entity}_{condition_name}_deviation"] = deviation
            previous = horse_last_jockey.get(row.horse_id)
            result["jockey_first_ride"] = float(pair_starts[(row.horse_id, row.jockey_id)] == 0)
            result["jockey_repeat_ride"] = 1.0 - result["jockey_first_ride"]
            result["jockey_change"] = np.nan if previous is None else float(previous != row.jockey_id)
            result["horse_jockey_pair_starts"] = float(pair_starts[(row.horse_id, row.jockey_id)])
            rows.append(result)

        # All states are updated only after every race on this date was emitted.
        for row in day.itertuples(index=False):
            conditions = {
                "surface": str(row.surface),
                "distance": distance_band(float(row.distance_m)),
                "venue": str(row.venue),
                "class": class_band(row.race_class),
                "age": age_band(float(row.horse_age)),
                "experience": experience_band(float(row.history_starts)),
                "layoff": "long" if np.isfinite(row.rest_days) and row.rest_days >= 180 else "not_long",
            }
            finish = float(row.finish_position) if pd.notna(row.finish_position) else np.nan
            for entity, key in (("jockey", row.jockey_id), ("trainer", row.trainer)):
                states[entity][key].update(date, float(row.winner), finish, conditions)
            pair_starts[(row.horse_id, row.jockey_id)] += 1
            horse_last_jockey[row.horse_id] = row.jockey_id
    result = pd.DataFrame(rows)
    assert len(result) == len(runners)
    assert result["race_date"].max() <= pd.Timestamp("2022-12-31")
    return result


def date_ci(race: pd.DataFrame, column: str, label: str) -> tuple[float, float, float]:
    clean = race[["race_date", column]].dropna()
    if clean.empty:
        return np.nan, np.nan, np.nan
    daily = clean.groupby("race_date")[column].agg(["sum", "count"])
    point = float(daily["sum"].sum() / daily["count"].sum())
    rng = np.random.default_rng(stable_seed(label))
    values = daily.to_numpy(float)
    draws = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    estimates = values[draws, 0].sum(1) / values[draws, 1].sum(1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return point, float(low), float(high)


def signal_summary(frame: pd.DataFrame) -> pd.DataFrame:
    features = [
        "jockey_career_raw",
        "jockey_career_eb",
        "jockey_days30_eb",
        "jockey_days90_eb",
        "jockey_days365_eb",
        "jockey_surface_eb",
        "jockey_distance_eb",
        "jockey_venue_eb",
        "jockey_class_eb",
        "jockey_age_eb",
        "jockey_experience_eb",
        "trainer_career_raw",
        "trainer_career_eb",
        "trainer_days30_eb",
        "trainer_days90_eb",
        "trainer_days365_eb",
        "trainer_surface_eb",
        "trainer_distance_eb",
        "trainer_venue_eb",
        "trainer_class_eb",
        "trainer_age_eb",
        "trainer_experience_eb",
        "trainer_layoff_eb",
    ]
    output: list[dict[str, object]] = []
    analysis = frame.loc[frame["year"].between(2014, 2022)]
    for scope, part in analysis.groupby("period", sort=False):
        for feature in features:
            valid = part[["race_id", "race_date", feature, "current_performance", "winner"]].dropna()
            group = valid.groupby("race_id", observed=True)
            valid["x_rank"] = group[feature].rank(method="average")
            valid["y_rank"] = group["current_performance"].rank(method="average")
            rank_group = valid.groupby("race_id", observed=True)
            valid["x_center"] = valid.x_rank - rank_group.x_rank.transform("mean")
            valid["y_center"] = valid.y_rank - rank_group.y_rank.transform("mean")
            valid["cross"] = valid.x_center * valid.y_center
            valid["x_sq"] = valid.x_center**2
            valid["y_sq"] = valid.y_center**2
            agg = (
                valid.groupby(["race_id", "race_date"], observed=True)
                .agg(
                    cross=("cross", "sum"),
                    x_sq=("x_sq", "sum"),
                    y_sq=("y_sq", "sum"),
                    maximum=(feature, "max"),
                )
                .reset_index()
            )
            agg["rho"] = agg.cross / np.sqrt(agg.x_sq * agg.y_sq)
            selected = valid.loc[valid[feature].eq(rank_group[feature].transform("max"))]
            selected = (
                selected.groupby(["race_id", "race_date"], observed=True).winner.mean().rename("top1").reset_index()
            )
            race = agg.merge(selected, on=["race_id", "race_date"], how="left")
            rho, rho_lo, rho_hi = date_ci(race, "rho", f"{scope}-{feature}-rho")
            top1, top1_lo, top1_hi = date_ci(race, "top1", f"{scope}-{feature}-top1")
            output.append(
                {
                    "period": scope,
                    "feature": feature,
                    "runner_count": len(part),
                    "race_count": part["race_id"].nunique(),
                    "effective_race_count": int(race["rho"].notna().sum()),
                    "effective_date_count": int(race.loc[race["rho"].notna(), "race_date"].nunique()),
                    "missing_count": int(part[feature].isna().sum()),
                    "race_spearman": rho,
                    "race_spearman_ci_low": rho_lo,
                    "race_spearman_ci_high": rho_hi,
                    "top1": top1,
                    "top1_ci_low": top1_lo,
                    "top1_ci_high": top1_hi,
                }
            )
    return pd.DataFrame(output)


def support_summary(frame: pd.DataFrame) -> pd.DataFrame:
    bins = [-1, 0, 4, 19, 49, np.inf]
    labels = ["0", "1-4", "5-19", "20-49", "50+"]
    rows = []
    for entity in ("jockey", "trainer"):
        data = frame.loc[frame["year"].between(2014, 2022)].copy()
        data["support"] = pd.cut(data[f"{entity}_career_starts"], bins=bins, labels=labels)
        for (scope, support), part in data.groupby(["period", "support"], observed=True):
            rows.append(
                {
                    "entity": entity,
                    "period": scope,
                    "support": str(support),
                    "runner_count": len(part),
                    "race_count": part.race_id.nunique(),
                    "runner_share": len(part) / len(data.loc[data.period.eq(scope)]),
                    "win_rate": part.winner.mean(),
                    "mean_posterior_sd": part[f"{entity}_career_sd"].mean(),
                }
            )
    return pd.DataFrame(rows)


def relationship_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    definitions = {
        "jockey_first_ride": frame.jockey_first_ride.eq(1),
        "jockey_repeat_ride": frame.jockey_repeat_ride.eq(1),
        "jockey_change": frame.jockey_change.eq(1),
        "jockey_unchanged": frame.jockey_change.eq(0),
        "trainer_debut_horse": frame.history_starts.eq(0),
        "trainer_long_layoff_horse": frame.current_long_layoff.eq(1),
    }
    for name, mask in definitions.items():
        for scope in ("discovery", "replication", "confirmation"):
            part = frame.loc[mask & frame.period.eq(scope)]
            rows.append(
                {
                    "slice": name,
                    "period": scope,
                    "runner_count": len(part),
                    "race_count": part.race_id.nunique(),
                    "win_rate": part.winner.mean(),
                    "mean_jockey_eb": part.jockey_career_eb.mean(),
                    "mean_trainer_eb": part.trainer_career_eb.mean(),
                }
            )
    return pd.DataFrame(rows)


def annual_tables(runners: pd.DataFrame, prior_mean: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual = (
        runners.loc[runners.year.between(2013, 2022)]
        .groupby(["year", "jockey_id"], observed=True)
        .agg(starts=("winner", "size"), wins=("winner", "sum"))
        .reset_index()
    )
    results = []
    regression = []
    for entity, key in (("jockey", "jockey_id"), ("trainer", "trainer")):
        annual = (
            runners.loc[runners.year.between(2013, 2022)]
            .groupby(["year", key], observed=True)
            .agg(starts=("winner", "size"), wins=("winner", "sum"))
            .reset_index()
        )
        annual["raw"] = annual.wins / annual.starts
        annual["eb"] = (annual.wins + PRIOR_N * prior_mean) / (annual.starts + PRIOR_N)
        nxt = annual.rename(
            columns={
                "year": "next_year",
                "starts": "next_starts",
                "wins": "next_wins",
                "raw": "next_raw",
                "eb": "next_eb",
            }
        )
        pairs = annual.merge(nxt, on=key)
        pairs = pairs.loc[pairs.next_year.eq(pairs.year + 1)]
        for destination, block in pairs.groupby("next_year"):
            scope = period(int(destination))
            if scope == "warmup":
                continue
            for minimum in (1, 10, 30, 100):
                valid = block.loc[(block.starts >= minimum) & (block.next_starts >= minimum)]
                for estimate in ("raw", "eb"):
                    rho = valid[estimate].rank().corr(valid.next_raw.rank()) if len(valid) >= 3 else np.nan
                    results.append(
                        {
                            "entity": entity,
                            "destination_year": destination,
                            "period": scope,
                            "minimum_starts_both_years": minimum,
                            "estimate": estimate,
                            "entity_count": len(valid),
                            "spearman": rho,
                        }
                    )
        # Regression-to-mean table pooled by destination-period, based on prior-year deciles.
        for scope, block in pairs.loc[pairs.next_year.between(2014, 2022)].groupby(
            pairs.loc[pairs.next_year.between(2014, 2022), "next_year"].map(period)
        ):
            valid = block.loc[(block.starts >= 10) & (block.next_starts >= 10)].copy()
            valid["prior_decile"] = pd.qcut(valid.raw.rank(method="first"), 10, labels=False) + 1
            for decile, group in valid.groupby("prior_decile"):
                regression.append(
                    {
                        "entity": entity,
                        "period": scope,
                        "prior_decile": int(decile),
                        "entity_count": len(group),
                        "prior_raw_rate": group.raw.mean(),
                        "prior_eb_rate": group.eb.mean(),
                        "next_raw_rate": group.next_raw.mean(),
                    }
                )
    return pd.DataFrame(results), pd.DataFrame(regression)


def name_audit(runners: pd.DataFrame) -> dict[str, object]:
    jockey = runners[["jockey_id", "騎手"]].dropna().drop_duplicates()
    trainer = runners[["trainer"]].dropna().drop_duplicates().copy()
    trainer["normalized"] = trainer.trainer.astype(str).str.replace(r"\s+", "", regex=True)
    return {
        "jockey_distinct_ids": int(jockey.jockey_id.nunique()),
        "jockey_distinct_display_names": int(jockey["騎手"].nunique()),
        "jockey_ids_with_multiple_display_names": int((jockey.groupby("jockey_id")["騎手"].nunique() > 1).sum()),
        "jockey_display_names_with_multiple_ids": int((jockey.groupby("騎手").jockey_id.nunique() > 1).sum()),
        "trainer_raw_keys": int(trainer.trainer.nunique()),
        "trainer_normalized_keys": int(trainer.normalized.nunique()),
        "trainer_whitespace_variant_groups": int((trainer.groupby("normalized").trainer.nunique() > 1).sum()),
        "interpretation": "aggregate key audit only; it cannot establish biological/person identity continuity",
    }


def redundancy_summary(frame: pd.DataFrame) -> dict[str, object]:
    columns = [
        c
        for c in frame
        if c.startswith(("jockey_", "trainer_"))
        and any(
            t in c
            for t in (
                "career_",
                "days30_",
                "days90_",
                "days365_",
                "surface_",
                "distance_",
                "venue_",
                "class_",
                "age_",
                "experience_",
            )
        )
    ]
    sample = frame.loc[frame.year.between(2014, 2022), columns].sample(
        n=min(50000, int(frame.year.between(2014, 2022).sum())), random_state=SEED
    )
    corr = sample.corr(method="spearman", min_periods=100)
    upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).abs()
    return {
        "production_connection_columns": 130,
        "production_columns_per_entity": 65,
        "production_blocks_per_entity": 13,
        "statistics_per_block": 5,
        "structural_formula": "2 entities * (career + 4 count + 5 day + 3 decay blocks) * 5 statistics",
        "diagnostic_columns_examined": len(columns),
        "sample_rows": len(sample),
        "absolute_spearman_pairs_ge_0_90": int((upper >= 0.90).sum().sum()),
        "absolute_spearman_pairs_ge_0_95": int((upper >= 0.95).sum().sum()),
        "deterministic_relations": [
            "win_rate=wins/starts",
            "last_1 wins equals last_1 win_rate when starts=1",
            "completed approximates starts except non-finish",
        ],
        "note": (
            "correlations use independently reconstructed pre-2023 flat-only "
            "diagnostic states, not the production frame"
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runners, source_manifest = load_flat()
    prior = runners.loc[runners.year.eq(2013), "winner"].mean()
    states = construct_states(runners, float(prior))
    signals = signal_summary(states)
    support = support_summary(states)
    relationships = relationship_summary(states)
    stability, regression = annual_tables(runners, float(prior))
    signals.to_csv(OUT / "connection_signal_summary.csv", index=False)
    support.to_csv(OUT / "support_cold_start.csv", index=False)
    relationships.to_csv(OUT / "relationship_context.csv", index=False)
    stability.to_csv(OUT / "annual_stability.csv", index=False)
    regression.to_csv(OUT / "regression_to_mean.csv", index=False)
    names = name_audit(runners)
    redundancy = redundancy_summary(states)
    (OUT / "name_key_audit.json").write_text(json.dumps(names, ensure_ascii=False, indent=2) + "\n")
    (OUT / "redundancy_summary.json").write_text(json.dumps(redundancy, ensure_ascii=False, indent=2) + "\n")
    analysis = runners.loc[runners.year.between(2014, 2022)]
    manifest = {
        "raw_sha256": source_manifest["raw_sha256"],
        "max_target_date": str(runners.race_date.max().date()),
        "flat_only_asserted": True,
        "obstacle_rows": 0,
        "analysis_runner_count": len(analysis),
        "analysis_race_count": analysis.race_id.nunique(),
        "period_counts": {
            p: {"runners": int(len(g)), "races": int(g.race_id.nunique())} for p, g in analysis.groupby("period")
        },
        "same_date_emit_before_update": True,
        "prior_source": "2013 flat warm-up official wins",
        "prior_mean": float(prior),
        "prior_effective_n": PRIOR_N,
        "bootstrap_unit": "race_date",
        "bootstrap_replicates": BOOTSTRAPS,
        "market_columns_used": [],
        "direct_ids_as_model_features": False,
        "source_code_feature_shape": {"connection_columns": 130, "per_entity": 65},
        "outputs": [
            "connection_signal_summary.csv",
            "support_cold_start.csv",
            "relationship_context.csv",
            "annual_stability.csv",
            "regression_to_mean.csv",
            "name_key_audit.json",
            "redundancy_summary.json",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
