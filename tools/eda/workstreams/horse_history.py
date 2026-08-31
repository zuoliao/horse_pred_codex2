"""Workstream C: strict-PIT horse history and temporal dynamics diagnostics.

Local/private artifact only.  No runner-level output is written.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from horse_pred.data import sha256_file
from horse_pred.eda import EDA_MAX_DATE, load_eda_population

RAW_PATH = Path(os.environ["HORSE_EDA_RAW_PATH"])
OUT = Path(os.environ.get("HORSE_EDA_HISTORY_OUTPUT", "artifacts/eda_20260901/workstreams/c_history"))
SEED = 20260901
BOOTSTRAPS = 500
DAY_WINDOWS = (14, 30, 60, 90, 180, 365)
HALF_LIVES = (30, 90, 180)
COUNT_WINDOWS = (1, 2, 3, 5, 10)


def period_for_year(year: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=year.index, dtype="string")
    result.loc[year.between(2013, 2013)] = "warmup"
    result.loc[year.between(2014, 2019)] = "discovery"
    result.loc[year.between(2020, 2021)] = "replication"
    result.loc[year.eq(2022)] = "confirmation"
    return result


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode()).digest()
    return SEED + int.from_bytes(digest[:4], "little")


def date_block_ci(frame: pd.DataFrame, value: str, label: str) -> tuple[float, float, float]:
    clean = frame[["race_date", value]].copy()
    clean[value] = pd.to_numeric(clean[value], errors="coerce").astype("float64")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan, np.nan, np.nan
    daily = clean.groupby("race_date", observed=True)[value].agg(["sum", "count"])
    denominator = float(daily["count"].sum())
    if denominator <= 0:
        return np.nan, np.nan, np.nan
    point = float(daily["sum"].sum()) / denominator
    if len(daily) < 2:
        return point, np.nan, np.nan
    rng = np.random.default_rng(stable_seed(label))
    sums = daily["sum"].to_numpy(float)
    counts = daily["count"].to_numpy(float)
    draws = rng.integers(0, len(daily), size=(BOOTSTRAPS, len(daily)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return point, float(low), float(high)


def build_pre_race_history() -> tuple[pd.DataFrame, dict[str, object]]:
    normalized = load_eda_population(RAW_PATH, max_date=EDA_MAX_DATE)
    row_is_flat = normalized["surface"].isin(["turf", "dirt"]) & ~normalized["race_class"].astype(
        "string"
    ).str.contains("障害", na=False)
    race_is_flat = row_is_flat.groupby(normalized["race_id"], observed=True).transform("all")
    flat = normalized.loc[race_is_flat].copy()
    runners = flat.loc[flat["started"].fillna(False)].copy()
    runners["race_date"] = pd.to_datetime(runners["race_date"], errors="raise")
    if runners["race_date"].max() > EDA_MAX_DATE:
        raise AssertionError("target date firewall violated")
    duplicate_horse_date = int(runners.duplicated(["horse_id", "race_date"]).sum())
    if duplicate_horse_date:
        raise AssertionError("analysis requires one start per horse/date")

    runners["year"] = runners["race_date"].dt.year
    runners["period"] = period_for_year(runners["year"])
    runners["field_size"] = runners.groupby("race_id", observed=True)["race_id"].transform("size")
    finish = pd.to_numeric(runners["finish_position"], errors="coerce")
    denom = (runners["field_size"] - 1).replace(0, np.nan)
    runners["current_performance"] = 1.0 - (finish - 1.0) / denom
    runners.loc[finish.isna(), "current_performance"] = 0.0
    runners["winner"] = finish.eq(1).astype(float)
    runners["top3"] = finish.le(3).astype(float)
    runners["centered_win"] = runners["winner"] - 1.0 / runners["field_size"]
    runners["centered_top3"] = runners["top3"] - np.minimum(3, runners["field_size"]) / runners["field_size"]
    runners = runners.sort_values(["horse_id", "race_date", "race_id"], kind="stable").reset_index()

    size = len(runners)
    arrays: dict[str, np.ndarray] = {
        "history_starts": np.zeros(size, dtype=float),
        "rest_days": np.full(size, np.nan),
        "last_surface_same": np.full(size, np.nan),
        "last_distance_delta": np.full(size, np.nan),
        "consecutive_weeks": np.zeros(size, dtype=float),
    }
    for n in COUNT_WINDOWS:
        arrays[f"history_mean_last{n}"] = np.full(size, np.nan)
        arrays[f"lag{n}_performance"] = np.full(size, np.nan)
    arrays["history_mean_career"] = np.full(size, np.nan)
    for window in DAY_WINDOWS:
        arrays[f"starts_{window}d"] = np.zeros(size, dtype=float)
        arrays[f"distance_{window}d"] = np.zeros(size, dtype=float)
        arrays[f"performance_mean_{window}d"] = np.full(size, np.nan)
    for half_life in HALF_LIVES:
        arrays[f"performance_decay_{half_life}d"] = np.full(size, np.nan)

    for _, positions in runners.groupby("horse_id", sort=False).indices.items():
        history: list[float] = []
        event_history: deque[tuple[pd.Timestamp, float, float]] = deque()
        career_sum = 0.0
        last_date: pd.Timestamp | None = None
        last_surface: str | None = None
        last_distance = np.nan
        decay_weight = {half_life: 0.0 for half_life in HALF_LIVES}
        decay_sum = {half_life: 0.0 for half_life in HALF_LIVES}
        decay_date: pd.Timestamp | None = None

        for position in positions:
            row = runners.iloc[position]
            current_date = pd.Timestamp(row["race_date"])
            arrays["history_starts"][position] = len(history)
            if last_date is not None:
                arrays["rest_days"][position] = (current_date - last_date).days
                arrays["last_surface_same"][position] = float(row["surface"] == last_surface)
                arrays["last_distance_delta"][position] = float(row["distance_m"] - last_distance)

            if history:
                arrays["history_mean_career"][position] = career_sum / len(history)
            for n in COUNT_WINDOWS:
                if history:
                    arrays[f"history_mean_last{n}"][position] = float(np.mean(history[-n:]))
                if len(history) >= n:
                    arrays[f"lag{n}_performance"][position] = history[-n]

            while event_history and (current_date - event_history[0][0]).days > 365:
                event_history.popleft()
            for window in DAY_WINDOWS:
                eligible = [event for event in event_history if (current_date - event[0]).days <= window]
                arrays[f"starts_{window}d"][position] = len(eligible)
                arrays[f"distance_{window}d"][position] = float(sum(event[2] for event in eligible))
                if eligible:
                    arrays[f"performance_mean_{window}d"][position] = float(np.mean([event[1] for event in eligible]))

            consecutive = 0
            for week in range(1, 9):
                older = current_date - pd.Timedelta(7 * week, unit="D")
                newer = current_date - pd.Timedelta(7 * (week - 1), unit="D")
                if any(older <= event[0] < newer for event in event_history):
                    consecutive += 1
                else:
                    break
            arrays["consecutive_weeks"][position] = consecutive

            if decay_date is not None:
                elapsed = (current_date - decay_date).days
                for half_life in HALF_LIVES:
                    factor = np.exp(-np.log(2.0) * elapsed / half_life)
                    decay_weight[half_life] *= factor
                    decay_sum[half_life] *= factor
                    if decay_weight[half_life] > 0:
                        arrays[f"performance_decay_{half_life}d"][position] = (
                            decay_sum[half_life] / decay_weight[half_life]
                        )

            performance = float(row["current_performance"])
            distance = float(row["distance_m"])
            history.append(performance)
            career_sum += performance
            event_history.append((current_date, performance, distance))
            for half_life in HALF_LIVES:
                decay_weight[half_life] += 1.0
                decay_sum[half_life] += performance
            decay_date = current_date
            last_date = current_date
            last_surface = str(row["surface"])
            last_distance = distance

    for name, values in arrays.items():
        runners[name] = values
    runners = runners.sort_values(["race_date", "race_id", "horse_number"], kind="stable")
    if not runners.loc[runners["year"].ge(2014), "history_starts"].ge(0).all():
        raise AssertionError("invalid history count")

    manifest = {
        "raw_path": str(RAW_PATH),
        "raw_sha256": sha256_file(RAW_PATH),
        "max_target_date": str(runners["race_date"].max().date()),
        "rows_loaded_through_2022": int(len(normalized)),
        "flat_declared_rows": int(len(flat)),
        "flat_races": int(flat["race_id"].nunique()),
        "eligible_starters": int(len(runners)),
        "races": int(runners["race_id"].nunique()),
        "analysis_runners_2014_2022": int(runners["year"].between(2014, 2022).sum()),
        "analysis_races_2014_2022": int(runners.loc[runners["year"].between(2014, 2022), "race_id"].nunique()),
        "horse_date_duplicates": duplicate_horse_date,
        "warmup_target_claims": 0,
        "market_columns_used": [],
        "same_date_update": False,
        "performance_definition": "1-(finish-1)/(starter_count-1); non-finish=0",
        "bootstrap_unit": "race_date",
        "bootstrap_replicates": BOOTSTRAPS,
        "random_seed": SEED,
    }
    return runners, manifest


def race_signal_rows(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    work = frame[
        [
            "race_id",
            "race_date",
            "year",
            "field_size",
            "winner",
            "centered_win",
            "current_performance",
            feature,
        ]
    ].copy()
    valid_race = work.groupby("race_id", observed=True)[feature].transform("count").gt(0)
    work = work.loc[valid_race]
    work["feature_rank"] = work.groupby("race_id", observed=True)[feature].rank(
        method="average", ascending=True, pct=True, na_option="bottom"
    )
    max_value = work.groupby("race_id", observed=True)[feature].transform("max")
    selected = work[feature].eq(max_value)
    work["selected_win"] = np.where(selected, work["winner"], 0.0)
    work["selected_count"] = selected.astype(float)
    winner_rank = work.groupby("race_id", observed=True)[feature].rank(
        method="average", ascending=False, na_option="bottom"
    )
    work["winner_ndcg3"] = np.where(work["winner"].eq(1) & winner_rank.le(3), 1.0 / np.log2(winner_rank + 1.0), 0.0)

    race = (
        work.groupby(["race_id", "race_date", "year"], observed=True)
        .agg(
            selected_wins=("selected_win", "sum"),
            selected_count=("selected_count", "sum"),
            uniform=("field_size", lambda x: float(np.mean(1.0 / x))),
            ndcg3=("winner_ndcg3", "sum"),
            feature_coverage=(feature, lambda x: float(x.notna().mean())),
        )
        .reset_index()
    )
    race["top1"] = race["selected_wins"] / race["selected_count"]
    race["top1_lift"] = race["top1"] - race["uniform"]

    ranked = work.loc[work[feature].notna()].copy()
    ranked["within_race_pct"] = ranked.groupby("race_id", observed=True)[feature].rank(method="average", pct=True)
    ranked["high_value"] = ranked["centered_win"].where(ranked["within_race_pct"].ge(0.8))
    ranked["low_value"] = ranked["centered_win"].where(ranked["within_race_pct"].le(0.2))
    high_low = ranked.groupby("race_id", observed=True).agg(high=("high_value", "mean"), low=("low_value", "mean"))
    high_low["high_low_win"] = high_low["high"] - high_low["low"]
    race = race.merge(high_low[["high_low_win"]], left_on="race_id", right_index=True, how="left")

    valid = work[feature].notna() & work["current_performance"].notna()
    corr = work.loc[valid, ["race_id", feature, "current_performance"]].copy()
    corr["xrank"] = corr.groupby("race_id", observed=True)[feature].rank(method="average")
    corr["yrank"] = corr.groupby("race_id", observed=True)["current_performance"].rank(method="average")
    corr["xd"] = corr["xrank"] - corr.groupby("race_id", observed=True)["xrank"].transform("mean")
    corr["yd"] = corr["yrank"] - corr.groupby("race_id", observed=True)["yrank"].transform("mean")
    corr["xy"] = corr["xd"] * corr["yd"]
    corr["xx"] = corr["xd"] ** 2
    corr["yy"] = corr["yd"] ** 2
    corr_race = corr.groupby("race_id", observed=True).agg(
        xy=("xy", "sum"), xx=("xx", "sum"), yy=("yy", "sum"), n=("xrank", "size")
    )
    corr_race["race_spearman"] = corr_race["xy"] / np.sqrt(corr_race["xx"] * corr_race["yy"])
    corr_race.loc[corr_race["n"].lt(3), "race_spearman"] = np.nan
    corr_race["race_spearman"] = pd.to_numeric(corr_race["race_spearman"], errors="coerce").astype("float64")
    return race.merge(corr_race[["race_spearman"]], left_on="race_id", right_index=True, how="left")


def signal_summary(frame: pd.DataFrame, feature: str) -> list[dict[str, object]]:
    races = race_signal_rows(frame, feature)
    rows: list[dict[str, object]] = []
    for period, years in (
        ("discovery", range(2014, 2020)),
        ("replication", range(2020, 2022)),
        ("confirmation", range(2022, 2023)),
    ):
        target = races.loc[races["year"].isin(years)].copy()
        top, top_low, top_high = date_block_ci(target, "top1_lift", f"{feature}-{period}-top")
        corr, corr_low, corr_high = date_block_ci(target, "race_spearman", f"{feature}-{period}-corr")
        high_low, hl_low, hl_high = date_block_ci(target, "high_low_win", f"{feature}-{period}-highlow")
        annual = target.groupby("year", observed=True)["top1_lift"].mean()
        rows.append(
            {
                "period": period,
                "feature": feature,
                "runners": int(frame.loc[frame["year"].isin(years), feature].notna().sum()),
                "races": int(len(target)),
                "race_dates": int(target["race_date"].nunique()),
                "missing_rate": float(frame.loc[frame["year"].isin(years), feature].isna().mean()),
                "mean_feature_coverage": float(target["feature_coverage"].mean()),
                "top1_lift": top,
                "top1_lift_ci_low": top_low,
                "top1_lift_ci_high": top_high,
                "ndcg3": float(target["ndcg3"].mean()),
                "race_spearman": corr,
                "race_spearman_ci_low": corr_low,
                "race_spearman_ci_high": corr_high,
                "high_low_centered_win": high_low,
                "high_low_ci_low": hl_low,
                "high_low_ci_high": hl_high,
                "positive_years": int(annual.gt(0).sum()),
                "years": int(annual.notna().sum()),
                "aggregation_unit": "race macro; CI date-block bootstrap",
                "evidence_stage": period,
                "multiple_comparison_risk": "high/exploratory",
                "pit_risk": "low: strict prior-date history",
            }
        )
    return rows


def category_curve(frame: pd.DataFrame, dimension: str, category: str, *, label: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    work = frame.loc[frame["year"].between(2014, 2022)].copy()
    for (period, band), part in work.groupby(["period", category], observed=True, dropna=False):
        race_values = (
            part.groupby(["race_id", "race_date", "year"], observed=True)
            .agg(
                centered_win=("centered_win", "mean"),
                centered_top3=("centered_top3", "mean"),
                performance=("current_performance", "mean"),
            )
            .reset_index()
        )
        point, low, high = date_block_ci(race_values, "centered_win", f"curve-{label}-{period}-{band}")
        annual = race_values.groupby("year", observed=True)["centered_win"].mean()
        records.append(
            {
                "dimension": dimension,
                "period": str(period),
                "band": str(band),
                "runners": int(len(part)),
                "races": int(part["race_id"].nunique()),
                "race_dates": int(part["race_date"].nunique()),
                "centered_win": point,
                "centered_win_ci_low": low,
                "centered_win_ci_high": high,
                "centered_top3": float(race_values["centered_top3"].mean()),
                "mean_current_performance": float(race_values["performance"].mean()),
                "positive_years": int(annual.gt(0).sum()),
                "years": int(annual.notna().sum()),
                "aggregation_unit": "race-band macro; CI date-block bootstrap",
                "multiple_comparison_risk": "high/exploratory",
            }
        )
    return pd.DataFrame(records)


def persistence_summary(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.loc[frame["year"].between(2014, 2022) & frame["lag1_performance"].notna()].copy()
    work["age_band"] = pd.cut(
        pd.to_numeric(work["horse_age"], errors="coerce"),
        [1, 2, 3, 4, 5, np.inf],
        labels=["2", "3", "4", "5", "6+"],
    )
    work["surface_transition"] = work["last_surface_same"].map({1.0: "same_surface", 0.0: "surface_switch"})
    absolute_delta = work["last_distance_delta"].abs()
    work["distance_transition"] = pd.cut(
        absolute_delta,
        [-1, 100, 399, np.inf],
        labels=["0-100m", "200-399m", "400m+"],
    )
    work["rest_band"] = pd.cut(
        work["rest_days"],
        [0, 13, 27, 55, 89, 179, 364, np.inf],
        labels=["1-13", "14-27", "28-55", "56-89", "90-179", "180-364", "365+"],
    )
    output: list[dict[str, object]] = []
    for slice_type in ("age_band", "surface_transition", "distance_transition", "rest_band"):
        for (period, slice_value), part in work.groupby(["period", slice_type], observed=True, dropna=False):
            race = race_signal_rows(part, "lag1_performance")
            corr, low, high = date_block_ci(race, "race_spearman", f"persistence-{slice_type}-{slice_value}-{period}")
            annual = race.groupby("year", observed=True)["race_spearman"].mean()
            output.append(
                {
                    "period": str(period),
                    "slice_type": slice_type,
                    "slice_value": str(slice_value),
                    "runners": int(len(part)),
                    "races": int(part["race_id"].nunique()),
                    "signal_races": int(race["race_spearman"].notna().sum()),
                    "lag1_race_spearman": corr,
                    "ci_low": low,
                    "ci_high": high,
                    "positive_years": int(annual.gt(0).sum()),
                    "years": int(annual.notna().sum()),
                    "missing_rate": 0.0,
                    "aggregation_unit": "race macro; CI date-block bootstrap",
                    "multiple_comparison_risk": "high/exploratory",
                    "pit_risk": "low",
                }
            )
    return pd.DataFrame(output)


def write_line_svg(
    path: Path,
    table: pd.DataFrame,
    *,
    x: str,
    y: str,
    series: str,
    title: str,
    x_order: list[str],
) -> None:
    width, height = 920, 480
    left, right, top, bottom = 90, 35, 55, 80
    work = table[[x, y, series]].dropna().copy()
    work[x] = work[x].astype("string")
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work = work.dropna()
    y_min = min(0.0, float(work[y].min()))
    y_max = max(0.0, float(work[y].max()))
    pad = max((y_max - y_min) * 0.08, 0.002)
    y_min -= pad
    y_max += pad
    x_pos = {
        label: left + index * (width - left - right) / max(len(x_order) - 1, 1) for index, label in enumerate(x_order)
    }

    def py(value: float) -> float:
        return top + (y_max - value) * (height - top - bottom) / (y_max - y_min)

    colors = {"discovery": "#31688e", "replication": "#35b779", "confirmation": "#fde725"}
    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="19">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{py(0):.1f}" x2="{width - right}" y2="{py(0):.1f}" stroke="#aaa"/>',
    ]
    for label, xpos in x_pos.items():
        elements.append(
            f'<text x="{xpos:.1f}" y="{height - bottom + 25}" '
            f'text-anchor="middle" font-size="12">{html.escape(label)}</text>'
        )
    for index, (name, part) in enumerate(work.groupby(series, observed=True)):
        points = []
        for label in x_order:
            match = part.loc[part[x].eq(label), y]
            if not match.empty:
                points.append((x_pos[label], py(float(match.iloc[0]))))
        if not points:
            continue
        color = colors.get(str(name), "#666")
        coords = " ".join(f"{xp:.1f},{yp:.1f}" for xp, yp in points)
        elements.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
        for xp, yp in points:
            elements.append(f'<circle cx="{xp:.1f}" cy="{yp:.1f}" r="4" fill="{color}"/>')
        elements.append(
            f'<text x="{left + index * 180}" y="{height - 18}" '
            f'font-size="12" fill="{color}">{html.escape(str(name))}</text>'
        )
    elements.append(
        f'<text transform="translate(20 {height / 2}) rotate(-90)" '
        f'text-anchor="middle" font-size="13">{html.escape(y)}</text>'
    )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">' + "".join(elements) + "</svg>\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    private_cache = OUT / "runner_history_flat_private.pkl"
    if private_cache.exists():
        runners, manifest = pd.read_pickle(private_cache)
    else:
        runners, manifest = build_pre_race_history()
        pd.to_pickle((runners, manifest), private_cache)
    manifest["raw_sha256"] = sha256_file(RAW_PATH)
    analysis = runners.loc[runners["year"].between(2014, 2022)].copy()

    signal_features = [
        *(f"history_mean_last{n}" for n in COUNT_WINDOWS),
        "history_mean_career",
        *(f"performance_mean_{n}d" for n in (30, 90, 180, 365)),
        *(f"performance_decay_{n}d" for n in HALF_LIVES),
    ]
    signal_rows: list[dict[str, object]] = []
    for feature in signal_features:
        signal_rows.extend(signal_summary(analysis, feature))
    signal_table = pd.DataFrame(signal_rows)
    signal_table.to_csv(OUT / "history_window_signal.csv", index=False)

    lag_rows: list[dict[str, object]] = []
    for lag in COUNT_WINDOWS:
        lag_rows.extend(signal_summary(analysis, f"lag{lag}_performance"))
    lag_table = pd.DataFrame(lag_rows)
    lag_table.to_csv(OUT / "lag_decay_signal.csv", index=False)
    lag_plot = lag_table.copy()
    lag_plot["lag"] = lag_plot["feature"].str.extract(r"lag([0-9]+)")[0]
    write_line_svg(
        OUT / "lag_decay_race_spearman.svg",
        lag_plot,
        x="lag",
        y="race_spearman",
        series="period",
        title="Past-performance persistence by starts ago (race macro)",
        x_order=["1", "2", "3", "5", "10"],
    )

    analysis["history_band"] = pd.cut(
        analysis["history_starts"],
        [-1, 0, 1, 3, 9, np.inf],
        labels=["0", "1", "2-3", "4-9", "10+"],
    )
    learning_parts = [category_curve(analysis, "history_starts", "history_band", label="history-band")]
    learning = pd.concat(learning_parts, ignore_index=True)
    for period in ("discovery", "replication", "confirmation"):
        for band in ("0", "1", "2-3", "4-9", "10+"):
            part = analysis.loc[analysis["period"].eq(period) & analysis["history_band"].astype("string").eq(band)]
            if part.empty or band == "0":
                corr = low = high = np.nan
                signal_races = 0
            else:
                race = race_signal_rows(part, "history_mean_career")
                corr, low, high = date_block_ci(race, "race_spearman", f"learning-{period}-{band}")
                signal_races = int(race["race_spearman"].notna().sum())
            mask = learning["period"].eq(period) & learning["band"].eq(band)
            learning.loc[mask, "career_signal_races"] = signal_races
            learning.loc[mask, "career_race_spearman"] = corr
            learning.loc[mask, "career_race_spearman_ci_low"] = low
            learning.loc[mask, "career_race_spearman_ci_high"] = high
    learning.to_csv(OUT / "history_learning_curve.csv", index=False)

    analysis["rest_band"] = pd.cut(
        analysis["rest_days"],
        [-np.inf, 13, 27, 55, 89, 179, 364, np.inf],
        labels=["1-13", "14-27", "28-55", "56-89", "90-179", "180-364", "365+"],
    )
    analysis["rest_band"] = analysis["rest_band"].cat.add_categories(["debut"])
    analysis.loc[analysis["history_starts"].eq(0), "rest_band"] = "debut"
    analysis["starts30_band"] = pd.cut(analysis["starts_30d"], [-1, 0, 1, 2, np.inf], labels=["0", "1", "2", "3+"])
    analysis["starts90_band"] = pd.cut(
        analysis["starts_90d"], [-1, 0, 1, 2, 3, np.inf], labels=["0", "1", "2", "3", "4+"]
    )
    analysis["consecutive_week_band"] = pd.cut(
        analysis["consecutive_weeks"], [-1, 0, 1, 2, np.inf], labels=["0", "1", "2", "3+"]
    )
    workload = pd.concat(
        [
            category_curve(analysis, "rest_days", "rest_band", label="rest"),
            category_curve(analysis, "starts_30d", "starts30_band", label="starts30"),
            category_curve(analysis, "starts_90d", "starts90_band", label="starts90"),
            category_curve(analysis, "consecutive_weeks", "consecutive_week_band", label="consecutive"),
        ],
        ignore_index=True,
    )
    workload.to_csv(OUT / "rest_workload_curves.csv", index=False)
    write_line_svg(
        OUT / "rest_non_linear_centered_win.svg",
        workload.loc[workload["dimension"].eq("rest_days")],
        x="band",
        y="centered_win",
        series="period",
        title="Rest interval and field-size-centered win outcome (descriptive)",
        x_order=["debut", "1-13", "14-27", "28-55", "56-89", "90-179", "180-364", "365+"],
    )

    analysis["age_band"] = pd.cut(
        pd.to_numeric(analysis["horse_age"], errors="coerce"),
        [1, 2, 3, 4, 5, np.inf],
        labels=["2", "3", "4", "5", "6+"],
    )
    analysis["distance_band"] = pd.cut(
        pd.to_numeric(analysis["distance_m"], errors="coerce"),
        [0, 1400, 1800, 2200, np.inf],
        labels=["<1400", "1400-1799", "1800-2199", "2200+"],
        right=False,
    )
    class_text = analysis["race_class"].astype("string")
    analysis["class_band"] = "other"
    analysis.loc[class_text.str.contains("新馬", na=False), "class_band"] = "debut"
    analysis.loc[class_text.str.contains("未勝利", na=False), "class_band"] = "maiden"
    analysis.loc[class_text.str.contains("1勝|500万", na=False), "class_band"] = "allowance_1"
    analysis.loc[class_text.str.contains("2勝|1000万", na=False), "class_band"] = "allowance_2"
    analysis.loc[class_text.str.contains("3勝|1600万", na=False), "class_band"] = "allowance_3"
    analysis.loc[class_text.str.contains("オープン|OP|Ｇ|G[123]", na=False), "class_band"] = "open_graded"
    interaction_rows: list[pd.DataFrame] = []
    for context in ("age_band", "distance_band", "class_band"):
        analysis["interaction_band"] = (
            analysis[context].astype("string") + " | rest=" + analysis["rest_band"].astype("string")
        )
        table = category_curve(analysis, f"rest_days_x_{context}", "interaction_band", label=f"rest-{context}")
        table["context"] = context
        interaction_rows.append(table)
    pd.concat(interaction_rows, ignore_index=True).to_csv(OUT / "workload_context_interactions.csv", index=False)

    persistence_summary(analysis).to_csv(OUT / "persistence_slices.csv", index=False)

    manifest["outputs"] = sorted(path.name for path in OUT.iterdir() if path.suffix in {".csv", ".svg"})
    manifest["analysis_feature_count"] = len(signal_features)
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
