from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from horse_pred.data import RAW_COLUMNS, normalize_raw, sha256_file
from horse_pred.features import FeatureConfig, is_flat_race
from horse_pred.speed_figure import (
    SpeedFigureSpec,
    _solve_ridge,
    condition_design_vector,
)

RAW = Path(os.environ["HORSE_EDA_RAW_PATH"])
OUTPUT = Path(
    os.environ.get(
        "HORSE_EDA_TARGET_OUTPUT",
        "artifacts/eda_20260901/workstreams/b_target/summary.json",
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def date_bootstrap_mean(frame: pd.DataFrame, value: str) -> dict[str, object]:
    source = frame.loc[np.isfinite(pd.to_numeric(frame[value], errors="coerce")), ["date", value]].copy()
    if source.empty:
        return {"estimate": None, "ci95": [None, None], "n": 0, "dates": 0}
    source[value] = pd.to_numeric(source[value], errors="raise").astype(float)
    daily = source.groupby("date", observed=True)[value].agg(["sum", "count"])
    estimate = float(source[value].mean())
    rng = np.random.default_rng(SEED)
    sums = daily["sum"].to_numpy(float)
    counts = daily["count"].to_numpy(float)
    draws = np.empty(RESAMPLES, dtype=float)
    for index in range(RESAMPLES):
        sample = rng.integers(0, len(daily), len(daily))
        draws[index] = sums[sample].sum() / counts[sample].sum()
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci95": [float(lo), float(hi)],
        "n": int(len(source)),
        "dates": int(len(daily)),
        "unit": "race_date bootstrap",
        "resamples": RESAMPLES,
    }


raw_chunks = []
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
    raw_chunks.append(chunk.loc[chunk["date"].le("2022-12-31")].copy())
raw = normalize_raw(pd.concat(raw_chunks, ignore_index=True))
raw["date"] = pd.to_datetime(raw["race_date"], errors="raise").dt.normalize()
if raw["date"].max().year != 2022:
    raise AssertionError("approved raw unexpectedly lacks the 2022 confirmation period")
frame = raw.loc[raw["date"].le(pd.Timestamp("2022-12-31"))].copy()
if frame["date"].dt.year.ge(2023).any():
    raise AssertionError("post-2022 target row survived")
frame["year"] = frame["date"].dt.year
frame["period"] = frame["year"].map(period_for_year)
clock_parts = frame["time_raw"].fillna("").astype(str).str.extract(r"^(\d+):(\d{2})\.(\d)$", expand=True)
clock_numeric = clock_parts.apply(pd.to_numeric, errors="coerce")
frame["clock_sec_fast"] = (clock_numeric[0] * 60.0 + clock_numeric[1] + clock_numeric[2] / 10.0).where(
    clock_numeric[1].lt(60)
)

flat_ids: list[str] = []
config = FeatureConfig()
for race_id, race in frame.groupby("race_id", sort=False):
    if is_flat_race(race, config):
        flat_ids.append(str(race_id))
flat = frame.loc[frame["race_id"].astype(str).isin(flat_ids)].copy()

# Conservative predictive choice sets: flat, no raw nonstarter, started rows,
# and at least one official winner. DNF/DQ remain starters in the set.
strict_ids = flat.loc[flat["pit_c_scoring_eligible"].eq(True), "race_id"].astype(str).unique()
strict = flat.loc[flat["race_id"].astype(str).isin(strict_ids) & flat["started"].eq(True).fillna(False)].copy()
winner_counts = strict.groupby("race_id", observed=True)["winner_label"].sum()
strict_ids = winner_counts.loc[winner_counts.gt(0)].index.astype(str)
strict = strict.loc[strict["race_id"].astype(str).isin(strict_ids)].copy()
strict["field_size"] = strict.groupby("race_id", observed=True)["race_id"].transform("size").astype(int)
strict["model_rank"] = (
    pd.to_numeric(strict["finish_position"], errors="coerce").fillna(strict["field_size"] + 1).astype(int)
)
strict["rank_percentile"] = np.where(
    strict["field_size"].gt(1),
    np.maximum(0.0, 1.0 - (strict["model_rank"] - 1.0) / (strict["field_size"] - 1.0)),
    np.nan,
)
strict["top3"] = pd.to_numeric(strict["finish_position"], errors="coerce").le(3).fillna(False).astype(int)
finish = pd.to_numeric(strict["finish_position"], errors="coerce")
strict["graded"] = np.select(
    [
        finish.eq(1).fillna(False).to_numpy(dtype=bool),
        finish.eq(2).fillna(False).to_numpy(dtype=bool),
        finish.eq(3).fillna(False).to_numpy(dtype=bool),
    ],
    [3, 2, 1],
    default=0,
).astype(int)
strict["clock_sec"] = strict["clock_sec_fast"]

race_rows = []
pair_rows = []
adjacent_rows = []
timed_runner_rows = []
margin_tokens: dict[str, Counter] = {
    "discovery_2014_2019": Counter(),
    "replication_2020_2021": Counter(),
    "confirmation_2022": Counter(),
}
for race_id, race in strict.groupby("race_id", sort=False):
    one = race.iloc[0]
    period = one["period"]
    if period is None:
        continue
    n = int(len(race))
    numeric = pd.to_numeric(race["finish_position"], errors="coerce")
    co_winners = int(numeric.eq(1).sum())
    top3 = int(numeric.le(3).sum())
    rank_counts = numeric.dropna().astype(int).value_counts()
    tied_pairs = int(sum(int(c * (c - 1) // 2) for c in rank_counts if c > 1))
    missing_rank = int(numeric.isna().sum())
    all_pairs = n * (n - 1) // 2
    comparable_pairs = all_pairs - tied_pairs - (missing_rank * (missing_rank - 1) // 2)
    winner_pairs = co_winners * (n - co_winners)
    bottom = max(n - top3, 0)
    top3_involving_pairs = all_pairs - bottom * (bottom - 1) // 2
    hard_winner_rate = co_winners / n
    top3_rate = top3 / n

    def bernoulli_entropy(p: float) -> float:
        if p <= 0 or p >= 1:
            return 0.0
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    graded_counts = race["graded"].value_counts().to_numpy(float)
    gp = graded_counts / graded_counts.sum()
    race_rows.append(
        {
            "race_id": str(race_id),
            "date": one["date"],
            "period": period,
            "field_size": n,
            "co_winners": co_winners,
            "hard_winner_rate": hard_winner_rate,
            "coherent_winner_rate": 1.0 / n,
            "top3_count": top3,
            "top3_rate": top3_rate,
            "winner_binary_entropy_bits": bernoulli_entropy(1.0 / n),
            "top3_binary_entropy_bits": bernoulli_entropy(top3_rate),
            "graded_label_entropy_bits": float(-(gp * np.log2(gp)).sum()),
            "choice_uniform_logloss_nats": math.log(n),
            "choice_uniform_brier": 1.0 / max(co_winners, 1) - 1.0 / n,
            "winner_identity_capacity_bits": math.log2(n),
            "full_order_capacity_bits": math.lgamma(n + 1) / math.log(2),
            "top3_unordered_capacity_bits": math.log2(math.comb(n, min(3, n))),
            "top3_ordered_capacity_bits": (
                sum(math.log2(v) for v in range(max(1, n - 2), n + 1)) if n >= 3 else math.log2(math.factorial(n))
            ),
            "all_pairs": all_pairs,
            "comparable_pairs": comparable_pairs,
            "tied_pairs": tied_pairs,
            "winner_pairs": winner_pairs,
            "top3_involving_pairs": top3_involving_pairs,
            "missing_rank_starters": missing_rank,
            "has_any_dead_heat": int((rank_counts > 1).any()),
            "has_first_dead_heat": int(co_winners > 1),
            "has_dnf": int(race["status"].astype(str).eq("did_not_finish").any()),
            "has_dq": int(race["status"].astype(str).eq("disqualified").any()),
        }
    )
    pair_rows.append(
        {
            "date": one["date"],
            "period": period,
            "all_pairs": all_pairs,
            "comparable_pairs": comparable_pairs,
            "winner_pairs": winner_pairs,
            "top3_involving_pairs": top3_involving_pairs,
            "tied_pairs": tied_pairs,
        }
    )

    distance = float(pd.to_numeric(race["distance"], errors="coerce").iloc[0])
    winner_clock = race.loc[numeric.eq(1), "clock_sec"].dropna()
    if not winner_clock.empty and distance > 0:
        wt = float(winner_clock.min())
        timed = race.loc[numeric.notna() & race["clock_sec"].notna()].copy()
        timed["time_gap_sec_per_1000m"] = (timed["clock_sec"].astype(float) - wt) * 1000.0 / distance
        for _, row in timed.iterrows():
            timed_runner_rows.append(
                {
                    "race_id": str(race_id),
                    "date": one["date"],
                    "period": period,
                    "finish": int(row["finish_position"]),
                    "rank_percentile": float(row["rank_percentile"]),
                    "time_gap_sec_per_1000m": float(row["time_gap_sec_per_1000m"]),
                }
            )
        by_rank = timed.groupby("finish_position", observed=True)["clock_sec"].mean().sort_index()
        ranks = [int(v) for v in by_rank.index]
        for left, right in zip(ranks, ranks[1:]):
            if right != left + 1:
                continue
            gap = (float(by_rank.loc[right]) - float(by_rank.loc[left])) * 1000.0 / distance
            boundary = f"{left}-{right}" if left <= 4 else "5+"
            adjacent_rows.append(
                {"race_id": str(race_id), "date": one["date"], "period": period, "boundary": boundary, "gap": gap}
            )

    nonwinner = race.loc[numeric.gt(1), "margin_raw"].fillna("").astype(str).str.strip()
    margin_tokens[period].update(nonwinner.tolist())

races = pd.DataFrame(race_rows)
pairs = pd.DataFrame(pair_rows)
adjacent = pd.DataFrame(adjacent_rows)
timed_runners = pd.DataFrame(timed_runner_rows)

# Strict prequential condition residual observations. 2013 is state warm-up
# only; summaries start in 2014 and end in 2022.
speed_spec = SpeedFigureSpec()
dimension = 51
xtx = np.zeros((dimension, dimension), dtype=float)
xty = np.zeros(dimension, dtype=float)
clean_count = 0
speed_rows = []
flat_ordered = flat.sort_values(["date", "race_id"], kind="stable")


def clean_race_times_fast(race: pd.DataFrame):
    if race["status"].astype("string").isin(("demoted", "disqualified")).any():
        return None
    distances = pd.to_numeric(race["distance"], errors="coerce").dropna().unique()
    if len(distances) != 1 or not np.isfinite(distances[0]) or distances[0] <= 0:
        return None
    distance = float(distances[0])
    times = pd.to_numeric(race["clock_sec_fast"], errors="coerce")
    started = race["started"].eq(True).fillna(False)
    finishes = pd.to_numeric(race["finish_position"], errors="coerce")
    timed = started & times.notna() & finishes.notna()
    winners = timed & finishes.eq(1)
    if not winners.any():
        return None
    winner_times = times.loc[winners].astype(float)
    if winner_times.max() - winner_times.min() > 1e-12:
        return None
    winner_time = float(winner_times.min())
    nonwinners = timed & finishes.gt(1)
    if nonwinners.any() and times.loc[nonwinners].astype(float).lt(winner_time).any():
        return None
    return winner_time * 1000.0 / distance, times.loc[timed].astype(float) * 1000.0 / distance


for date, day in flat_ordered.groupby("date", sort=True):
    beta = _solve_ridge(xtx, xty, speed_spec.ridge_alpha) if clean_count >= speed_spec.min_prior_clean_races else None
    updates = []
    for race_id, race in day.groupby("race_id", sort=False):
        design = condition_design_vector(race)
        clean = clean_race_times_fast(race)
        if design is None or clean is None:
            continue
        winner_pace, runner_paces = clean
        if beta is not None and date.year >= 2014:
            expected = float(design @ beta)
            finish_values = pd.to_numeric(race["finish_position"], errors="coerce")
            for idx, runner_pace in runner_paces.items():
                if not np.isfinite(finish_values.loc[idx]):
                    continue
                raw_residual = expected - float(runner_pace)
                speed_rows.append(
                    {
                        "race_id": str(race_id),
                        "date": date,
                        "period": period_for_year(date.year),
                        "finish": int(finish_values.loc[idx]),
                        "raw_residual": raw_residual,
                        "clipped_residual": float(np.clip(raw_residual, -5.0, 5.0)),
                        "winner_condition_residual": expected - winner_pace,
                        "within_race_time_gap": float(runner_pace - winner_pace),
                    }
                )
        updates.append((design, winner_pace))
    for design, winner_pace in updates:
        xtx += np.outer(design, design)
        xty += design * winner_pace
        clean_count += 1
speed = pd.DataFrame(speed_rows)
strict_race_ids = set(strict.loc[strict["period"].notna(), "race_id"].astype(str))
speed = speed.loc[speed["period"].notna() & speed["race_id"].astype(str).isin(strict_race_ids)].copy()

summary: dict[str, object] = {
    "schema_version": 1,
    "analysis_id": "eda_20260901_workstream_b_target",
    "raw_sha256": sha256_file(RAW),
    "max_target_date": str(frame["date"].max().date()),
    "firewall": {
        "2013_target_claims": False,
        "post_2022_rows_loaded_into_target_summary": 0,
        "periods": {"discovery": [2014, 2019], "replication": [2020, 2021], "confirmation": [2022, 2022]},
    },
    "population": {
        "through_2022_raw_rows": int(len(frame)),
        "through_2022_raw_races": int(frame["race_id"].nunique()),
        "flat_raw_rows": int(len(flat)),
        "flat_raw_races": int(flat["race_id"].nunique()),
        "strict_choice_rows_2014_2022": int(strict["period"].notna().sum()),
        "strict_choice_races_2014_2022": int(strict.loc[strict["period"].notna(), "race_id"].nunique()),
        "definition": "flat races with no raw scratch/exclusion, started runners, and at least one official winner",
    },
    "raw_status_by_period": {},
    "choice_set_by_period": {},
    "field_size_bands": {},
    "target_entropy_and_capacity": {},
    "pairwise_structure": {},
    "time_gap": {},
    "margin_tokens": {},
    "performance_residual": {},
}

for period in ("discovery_2014_2019", "replication_2020_2021", "confirmation_2022"):
    raw_period = flat.loc[flat["period"].eq(period)]
    rp = races.loc[races["period"].eq(period)].copy()
    pp = pairs.loc[pairs["period"].eq(period)].copy()
    summary["raw_status_by_period"][period] = {
        "declared_rows": int(len(raw_period)),
        "races": int(raw_period["race_id"].nunique()),
        "status_rows": {str(k): int(v) for k, v in raw_period["status"].value_counts().items()},
        "races_with_scratch_or_exclusion": int(
            raw_period.groupby("race_id", observed=True)["started"].apply(lambda s: s.eq(False).any()).sum()
        ),
    }
    summary["choice_set_by_period"][period] = {
        "races": int(len(rp)),
        "runners": int(rp["field_size"].sum()),
        "dates": int(rp["date"].nunique()),
        "field_size_mean_ci": date_bootstrap_mean(rp, "field_size"),
        "field_size_median": float(rp["field_size"].median()),
        "field_size_q25_q75": [float(v) for v in rp["field_size"].quantile([0.25, 0.75])],
        "hard_winner_runner_rate_ci": date_bootstrap_mean(rp, "hard_winner_rate"),
        "coherent_winner_runner_rate_ci": date_bootstrap_mean(rp, "coherent_winner_rate"),
        "top3_runner_rate_ci": date_bootstrap_mean(rp, "top3_rate"),
        "dead_heat_any_races": int(rp["has_any_dead_heat"].sum()),
        "dead_heat_first_races": int(rp["has_first_dead_heat"].sum()),
        "dnf_races": int(rp["has_dnf"].sum()),
        "dq_races": int(rp["has_dq"].sum()),
        "missing_rank_starters": int(rp["missing_rank_starters"].sum()),
    }
    bands = pd.cut(
        rp["field_size"], [0, 9, 13, 16, 100], labels=["small_<=9", "medium_10_13", "large_14_16", "very_large_17+"]
    )
    by_band = (
        rp.assign(band=bands)
        .groupby("band", observed=True)
        .agg(
            races=("race_id", "size"),
            mean_field_size=("field_size", "mean"),
            coherent_winner_rate=("coherent_winner_rate", "mean"),
            top3_rate=("top3_rate", "mean"),
        )
    )
    summary["field_size_bands"][period] = {
        str(index): {k: scalar(v) for k, v in row.items()} for index, row in by_band.to_dict("index").items()
    }
    summary["target_entropy_and_capacity"][period] = {
        column: date_bootstrap_mean(rp, column)
        for column in (
            "winner_binary_entropy_bits",
            "top3_binary_entropy_bits",
            "graded_label_entropy_bits",
            "choice_uniform_logloss_nats",
            "choice_uniform_brier",
            "winner_identity_capacity_bits",
            "top3_unordered_capacity_bits",
            "top3_ordered_capacity_bits",
            "full_order_capacity_bits",
        )
    }
    totals = pp[["all_pairs", "comparable_pairs", "winner_pairs", "top3_involving_pairs", "tied_pairs"]].sum()
    summary["pairwise_structure"][period] = {
        "total_pairs": int(totals["all_pairs"]),
        "comparable_pairs": int(totals["comparable_pairs"]),
        "tied_pairs": int(totals["tied_pairs"]),
        "winner_pair_fraction": float(totals["winner_pairs"] / totals["all_pairs"]),
        "top3_involving_pair_fraction": float(totals["top3_involving_pairs"] / totals["all_pairs"]),
        "mean_pairs_per_race_ci": date_bootstrap_mean(pp, "all_pairs"),
    }
    time_period = timed_runners.loc[timed_runners["period"].eq(period)].copy()
    race_corr = (
        time_period.groupby(["race_id", "date"], observed=True)
        .apply(
            lambda g: g["rank_percentile"].corr(-g["time_gap_sec_per_1000m"], method="spearman"),
            include_groups=False,
        )
        .rename("spearman")
        .reset_index()
    )
    ap = adjacent.loc[adjacent["period"].eq(period)]
    summary["time_gap"][period] = {
        "timed_numeric_runners": int(len(time_period)),
        "numeric_runner_coverage": float(len(time_period) / rp["field_size"].sum()),
        "time_gap_sec_per_1000m_quantiles": {
            str(q): float(v)
            for q, v in time_period["time_gap_sec_per_1000m"].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]).items()
        },
        "race_macro_rank_vs_negative_gap_spearman_ci": date_bootstrap_mean(
            race_corr.rename(columns={"date": "date"}), "spearman"
        ),
        "adjacent_boundary_mean_ci": {
            boundary: date_bootstrap_mean(ap.loc[ap["boundary"].eq(boundary)], "gap")
            for boundary in ("1-2", "2-3", "3-4", "4-5", "5+")
        },
        "adjacent_equal_clock_fraction": float(ap["gap"].abs().lt(1e-12).mean()),
        "adjacent_negative_clock_inversions": int(ap["gap"].lt(-1e-12).sum()),
    }
    tokens = margin_tokens[period]
    total_tokens = sum(tokens.values())
    summary["margin_tokens"][period] = {
        "nonwinner_rows": int(total_tokens),
        "blank_or_missing": int(tokens.get("", 0)),
        "unique_tokens": int(len(tokens)),
        "top_tokens": [
            {"token": key, "count": int(value), "share": value / total_tokens} for key, value in tokens.most_common(15)
        ],
    }
    sp = speed.loc[speed["period"].eq(period)].copy()
    summary["performance_residual"][period] = {
        "rows": int(len(sp)),
        "race_coverage": float(sp["race_id"].nunique() / len(rp)),
        "runner_coverage": float(len(sp) / rp["field_size"].sum()),
        "raw_quantiles": {
            str(q): float(v) for q, v in sp["raw_residual"].quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]).items()
        },
        "clipped_quantiles": {
            str(q): float(v) for q, v in sp["clipped_residual"].quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]).items()
        },
        "clipped_fraction": float(sp["raw_residual"].abs().gt(5).mean()),
        "winner_condition_residual_sd": float(sp.drop_duplicates("race_id")["winner_condition_residual"].std()),
        "within_race_time_gap_sd": float(sp["within_race_time_gap"].std()),
        "identity_check_max_abs": float(
            (sp["raw_residual"] - (sp["winner_condition_residual"] - sp["within_race_time_gap"])).abs().max()
        ),
    }

OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=scalar) + "\n", encoding="utf-8")
print(
    json.dumps(
        {"output": str(OUTPUT), "population": summary["population"], "firewall": summary["firewall"]},
        ensure_ascii=False,
        indent=2,
    )
)
