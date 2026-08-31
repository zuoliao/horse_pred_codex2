from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from horse_pred.data import RAW_COLUMNS, normalize_raw, sha256_file
from horse_pred.features import FeatureConfig, is_flat_race

ROOT = Path(os.environ.get("HORSE_EDA_ERRORS_OUTPUT", "artifacts/eda_20260901/workstreams/h_errors"))
PREDICTIONS = Path(
    os.environ.get(
        "HORSE_EDA_ROLLING_PREDICTIONS",
        "artifacts/eval_roll_001_current_best_20260831/predictions_scoring.csv.gz",
    )
)
RAW = Path(os.environ["HORSE_EDA_RAW_PATH"])
YEARS = {2020, 2021, 2022}
METHODS = ("binary_current", "lambdarank_current")
SEED = 20260901
RESAMPLES = 1000

ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "plots").mkdir(exist_ok=True)
(ROOT / "tables").mkdir(exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    return value


def class_group(value: object) -> str:
    text = "" if pd.isna(value) else str(value).replace(" ", "")
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


def distance_band(value: object) -> str:
    v = float(value)
    if v < 1400:
        return "sprint_lt1400"
    if v < 1800:
        return "mile_1400_1799"
    if v < 2200:
        return "middle_1800_2199"
    return "stayer_2200_plus"


def field_band(value: int) -> str:
    if value <= 9:
        return "09_or_less"
    if value <= 13:
        return "10_13"
    if value <= 16:
        return "14_16"
    return "17_plus"


def history_band(starts: float) -> str:
    if starts <= 0:
        return "0_new"
    if starts <= 1:
        return "1"
    if starts <= 3:
        return "2_3"
    if starts <= 9:
        return "4_9"
    return "10_plus"


def connection_band(rate: float) -> str:
    if rate < 0.06:
        return "low_lt06"
    if rate < 0.08:
        return "mid_06_08"
    if rate < 0.10:
        return "upper_08_10"
    return "high_10_plus"


def date_bootstrap_mean(frame: pd.DataFrame, value: str) -> dict[str, object]:
    source = frame.loc[np.isfinite(pd.to_numeric(frame[value], errors="coerce")), ["date", value]].copy()
    if source.empty:
        return {"estimate": None, "ci95": [None, None], "races": 0, "dates": 0}
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
        "races": int(len(source)),
        "dates": int(len(daily)),
        "unit": "race_date bootstrap",
        "resamples": RESAMPLES,
    }


def ndcg_at_3(finish: np.ndarray, score: np.ndarray) -> float:
    relevance = np.select([finish == 1, finish == 2, finish == 3], [3.0, 2.0, 1.0], default=0.0)
    gains = np.power(2.0, relevance) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, min(3, len(score)) + 2))
    order = np.argsort(-score, kind="stable")[:3]
    ideal = np.argsort(-relevance, kind="stable")[:3]
    dcg = float((gains[order] * discounts).sum())
    idcg = float((gains[ideal] * discounts).sum())
    return dcg / idcg if idcg > 0 else np.nan


# Predictions firewall: only evaluation rows for 2020--2022 are ever retained.
prediction_parts = []
source_rows = 0
for chunk in pd.read_csv(PREDICTIONS, compression="gzip", chunksize=50_000):
    source_rows += len(chunk)
    keep = chunk.loc[chunk["role"].eq("evaluation") & chunk["evaluation_year"].isin(YEARS)].copy()
    prediction_parts.append(keep)
pred = pd.concat(prediction_parts, ignore_index=True)
pred["evaluation_year"] = pd.to_numeric(pred["evaluation_year"], errors="raise").astype(int)
pred["race_date"] = pd.to_datetime(pred["race_date"], errors="raise").dt.normalize()
pred["race_id"] = pred["race_id"].astype(str)
pred["horse_id"] = pred["horse_id"].astype(str)
assert set(pred["role"].unique()) == {"evaluation"}
assert set(pred["evaluation_year"].unique()) == YEARS
assert int(pred["race_date"].dt.year.ge(2023).sum()) == 0
assert int(pred["evaluation_year"].eq(2023).sum()) == 0
assert set(pred["method"].unique()) == set(METHODS)
assert not pred.duplicated(["evaluation_year", "method", "race_id", "horse_id"]).any()

# Approved raw is independently firewalled and becomes three physically separate logical tables.
raw_parts = []
for chunk in pd.read_csv(
    RAW, encoding="utf-8-sig", dtype=str, keep_default_na=False, na_filter=False, chunksize=50_000
):
    if tuple(chunk.columns) != RAW_COLUMNS:
        raise AssertionError("approved raw schema changed")
    raw_parts.append(chunk.loc[chunk["date"].le("2022-12-31")].copy())
raw = normalize_raw(pd.concat(raw_parts, ignore_index=True))
raw["date"] = pd.to_datetime(raw["race_date"], errors="raise").dt.normalize()
assert int(raw["date"].dt.year.ge(2023).sum()) == 0
raw["race_id"] = raw["race_id"].astype(str)
raw["horse_id"] = raw["horse_id"].astype(str)

flat_ids = []
for race_id, race in raw.groupby("race_id", sort=False):
    if is_flat_race(race, FeatureConfig()):
        flat_ids.append(str(race_id))
starters = raw.loc[raw["race_id"].isin(flat_ids) & raw["started"].eq(True).fillna(False)].copy()
starters = starters.sort_values(["date", "race_id"], kind="stable")

# Minimal strictly-prior diagnostic states. IDs remain state/join keys only.
horse_starts: dict[str, int] = defaultdict(int)
jockey_starts: dict[str, int] = defaultdict(int)
trainer_starts: dict[str, int] = defaultdict(int)
jockey_wins: dict[str, float] = defaultdict(float)
trainer_wins: dict[str, float] = defaultdict(float)
state_records = []
for _event_date, day in starters.groupby("date", sort=True):
    for row in day.itertuples():
        hs = horse_starts[row.horse_id]
        js = jockey_starts[str(row.jockey_id)]
        ts = trainer_starts[str(row.trainer)]
        state_records.append(
            {
                "race_id": row.race_id,
                "horse_id": row.horse_id,
                "horse_starts_pre": hs,
                "jockey_starts_pre": js,
                "trainer_starts_pre": ts,
                "jockey_smoothed_win_rate": (jockey_wins[str(row.jockey_id)] + 1.0) / (js + 20.0),
                "trainer_smoothed_win_rate": (trainer_wins[str(row.trainer)] + 1.0) / (ts + 20.0),
            }
        )
    for row in day.itertuples():
        horse_starts[row.horse_id] += 1
        jockey_starts[str(row.jockey_id)] += 1
        trainer_starts[str(row.trainer)] += 1
        win = float(row.finish_position == 1) if not pd.isna(row.finish_position) else 0.0
        jockey_wins[str(row.jockey_id)] += win
        trainer_wins[str(row.trainer)] += win
states = pd.DataFrame(state_records)
assert not states.duplicated(["race_id", "horse_id"]).any()

# Outcome/context table excludes market columns.
outcomes = starters.loc[
    starters["date"].dt.year.between(2014, 2022),
    [
        "race_id",
        "horse_id",
        "date",
        "finish_position",
        "course_type",
        "distance",
        "race_class",
        "ground_state",
    ],
].copy()
outcomes = outcomes.merge(states, on=["race_id", "horse_id"], how="left", validate="one_to_one")
outcomes["class_group"] = outcomes["race_class"].map(class_group)
outcomes["distance_band"] = outcomes["distance"].map(distance_band)
outcomes["history_uncertainty"] = 1.0 / np.sqrt(outcomes["horse_starts_pre"] + 1.0)
field_diag = (
    outcomes.groupby("race_id", observed=True)
    .agg(
        cold_share=("horse_starts_pre", lambda x: float((x == 0).mean())),
        field_uncertainty=("history_uncertainty", "mean"),
        field_size_actual=("race_id", "size"),
    )
    .reset_index()
)
outcomes = outcomes.merge(field_diag, on="race_id", how="left", validate="many_to_one")
discovery_uncertainty = outcomes.loc[outcomes["date"].dt.year.between(2014, 2019)].drop_duplicates("race_id")[
    "field_uncertainty"
]
uncertainty_cuts = discovery_uncertainty.quantile([0.25, 0.5, 0.75]).to_numpy(float)
outcomes = outcomes.loc[outcomes["date"].dt.year.isin(YEARS)].copy()

# Final odds are a separate market table and enter only the explicit oracle join below.
market = raw.loc[raw["date"].dt.year.isin(YEARS), ["race_id", "horse_id", "final_win_odds", "final_popularity"]].copy()
market["final_win_odds"] = pd.to_numeric(market["final_win_odds"], errors="coerce")
assert not market.duplicated(["race_id", "horse_id"]).any()

analysis = pred.merge(outcomes, on=["race_id", "horse_id"], how="left", validate="many_to_one")
assert analysis["finish_position"].notna().any() and analysis["date"].notna().all()
assert "final_win_odds" not in analysis.columns

# Per-race, per-model metrics and error signatures.
race_records = []
for (year, method, race_id), race in analysis.groupby(["evaluation_year", "method", "race_id"], sort=False):
    p = race["probability_calibrated"].to_numpy(float)
    finish = pd.to_numeric(race["model_finish_position"], errors="coerce").to_numpy(float)
    winners = finish == 1
    target = winners.astype(float) / winners.sum()
    order = np.argsort(-p, kind="stable")
    top = race.iloc[int(order[0])]
    winner_rows = race.loc[winners]
    winner_prob = float((target * p).sum())
    winner_rank = float(np.mean([1 + int(np.sum(p > p[i])) for i in np.flatnonzero(winners)]))
    record = {
        "year": int(year),
        "method": method,
        "race_id": race_id,
        "date": race["date"].iloc[0],
        "race_log_loss": float(-(target * np.log(np.clip(p, 1e-15, 1))).sum()),
        "race_brier": float(np.square(p - target).sum()),
        "ndcg_at_3": ndcg_at_3(finish, p),
        "top1": float(target[order[0]]),
        "winner_probability": winner_prob,
        "winner_rank": winner_rank,
        "top_probability": float(p[order[0]]),
        "high_confidence_selected": float(p[order[0]] >= 0.30),
        "high_confidence_error": float(p[order[0]] >= 0.30 and not winners[order[0]]),
        "winner_history_band": history_band(float(winner_rows["horse_starts_pre"].mean())),
        "winner_horse_starts": float(winner_rows["horse_starts_pre"].mean()),
        "winner_connection_starts": float(
            (winner_rows["jockey_starts_pre"] + winner_rows["trainer_starts_pre"]).mean()
        ),
        "winner_connection_rate": float(
            ((winner_rows["jockey_smoothed_win_rate"] + winner_rows["trainer_smoothed_win_rate"]) / 2).mean()
        ),
        "cold_share": float(race["cold_share"].iloc[0]),
        "field_uncertainty": float(race["field_uncertainty"].iloc[0]),
        "surface": str(race["course_type"].iloc[0]),
        "class_group": str(race["class_group"].iloc[0]),
        "distance_band": str(race["distance_band"].iloc[0]),
        "field_band": field_band(int(len(race))),
        "top_horse_starts": float(top["horse_starts_pre"]),
    }
    race_records.append(record)
races = pd.DataFrame(race_records)
races["winner_connection_band"] = races["winner_connection_rate"].map(connection_band)

# Binary/ranker disagreement uses a paired race table.
pair = races.pivot(
    index=["year", "race_id", "date"],
    columns="method",
    values=["top1", "race_log_loss", "winner_probability", "top_probability"],
).reset_index()
top_ids = analysis.loc[
    analysis.groupby(["evaluation_year", "method", "race_id"])["probability_calibrated"].idxmax(),
    ["evaluation_year", "method", "race_id", "horse_id"],
]
top_ids = top_ids.pivot(index=["evaluation_year", "race_id"], columns="method", values="horse_id").reset_index()
top_ids["top_choice_disagree"] = (top_ids[METHODS[0]] != top_ids[METHODS[1]]).astype(float)
paired_metrics = races.merge(
    top_ids[["evaluation_year", "race_id", "top_choice_disagree"]],
    left_on=["year", "race_id"],
    right_on=["evaluation_year", "race_id"],
    how="left",
    validate="many_to_one",
)

# Runner-level race-weighted calibration, with predeclared fixed bins.
calibration_rows = []
bins = np.array([0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 1.0000001])
for (year, method), part in analysis.groupby(["evaluation_year", "method"], sort=True):
    part = part.copy()
    finish = pd.to_numeric(part["model_finish_position"], errors="coerce")
    winner_count = finish.eq(1).groupby(part["race_id"]).transform("sum")
    part["target"] = finish.eq(1).astype(float) / winner_count
    part["race_weight"] = 1.0 / part.groupby("race_id")["race_id"].transform("size")
    part["bin"] = pd.cut(part["probability_calibrated"], bins=bins, right=False, include_lowest=True)
    for band, g in part.groupby("bin", observed=True):
        w = g["race_weight"].to_numpy(float)
        pred_mean = float(np.average(g["probability_calibrated"], weights=w))
        obs = float(np.average(g["target"], weights=w))
        calibration_rows.append(
            {
                "year": int(year),
                "method": method,
                "bin": str(band),
                "runners": int(len(g)),
                "race_weight": float(w.sum()),
                "mean_probability": pred_mean,
                "observed_rate": obs,
                "gap": obs - pred_mean,
            }
        )
calibration = pd.DataFrame(calibration_rows)

# Explicit final-odds oracle join. Complete races only; never used for selection.
oracle = analysis.merge(market, on=["race_id", "horse_id"], how="left", validate="many_to_one")
valid_odds = oracle["final_win_odds"].notna() & np.isfinite(oracle["final_win_odds"]) & oracle["final_win_odds"].ge(1.0)
complete = valid_odds.groupby([oracle["evaluation_year"], oracle["method"], oracle["race_id"]]).transform("all")
oracle = oracle.loc[complete].copy()
oracle["inverse_odds"] = 1.0 / oracle["final_win_odds"]
oracle["market_probability"] = oracle["inverse_odds"] / oracle.groupby(["evaluation_year", "method", "race_id"])[
    "inverse_odds"
].transform("sum")
market_race = []
for (year, method, race_id), race in oracle.groupby(["evaluation_year", "method", "race_id"], sort=False):
    finish = pd.to_numeric(race["model_finish_position"], errors="coerce").to_numpy(float)
    winners = finish == 1
    target = winners.astype(float) / winners.sum()
    mp = race["market_probability"].to_numpy(float)
    modelp = race["probability_calibrated"].to_numpy(float)
    market_ll = float(-(target * np.log(np.clip(mp, 1e-15, 1))).sum())
    model_ll = float(-(target * np.log(np.clip(modelp, 1e-15, 1))).sum())
    winner = race.loc[winners]
    market_race.append(
        {
            "year": int(year),
            "method": method,
            "race_id": race_id,
            "date": race["date"].iloc[0],
            "market_log_loss": market_ll,
            "model_log_loss": model_ll,
            "model_market_gap": model_ll - market_ll,
            "market_brier": float(np.square(mp - target).sum()),
            "market_ndcg_at_3": ndcg_at_3(finish, mp),
            "market_top1": float(target[np.argmax(mp)]),
            "winner_history_band": history_band(float(winner["horse_starts_pre"].mean())),
            "class_group": str(race["class_group"].iloc[0]),
            "field_uncertainty": float(race["field_uncertainty"].iloc[0]),
            "cold_share": float(race["cold_share"].iloc[0]),
        }
    )
market_races = pd.DataFrame(market_race)

# Race-macro aggregate tables.
metric_table = (
    races.groupby(["year", "method"], observed=True)
    .agg(
        races=("race_id", "size"),
        ndcg_at_3=("ndcg_at_3", "mean"),
        top1=("top1", "mean"),
        race_log_loss=("race_log_loss", "mean"),
        race_brier=("race_brier", "mean"),
        winner_rank=("winner_rank", "mean"),
        high_confidence_selected=("high_confidence_selected", "mean"),
        high_confidence_error=("high_confidence_error", "mean"),
    )
    .reset_index()
)
metric_table.to_csv(ROOT / "tables" / "year_metrics.csv", index=False)
calibration.to_csv(ROOT / "tables" / "calibration.csv", index=False)

disagreement = []
for year, part in paired_metrics.groupby("year", sort=True):
    one = part.drop_duplicates("race_id")
    disagree = one["top_choice_disagree"].eq(1)
    disagreement.append(
        {
            "year": int(year),
            "races": int(len(one)),
            "top_choice_disagree_rate": float(disagree.mean()),
            "binary_top1_when_disagree": float(
                one.loc[disagree, "top1"].loc[one.loc[disagree, "method"].eq("binary_current")].mean()
            )
            if False
            else np.nan,
        }
    )
# Compute paired correctness directly from top IDs and outcome keys.
winner_keys = analysis.loc[
    pd.to_numeric(analysis["model_finish_position"], errors="coerce").eq(1), ["evaluation_year", "race_id", "horse_id"]
].drop_duplicates()
top_pair = (
    top_ids.merge(
        winner_keys.assign(is_winner=1),
        left_on=["evaluation_year", "race_id", "binary_current"],
        right_on=["evaluation_year", "race_id", "horse_id"],
        how="left",
    )
    .rename(columns={"is_winner": "binary_correct"})
    .drop(columns="horse_id")
)
top_pair = (
    top_pair.merge(
        winner_keys.assign(is_winner=1),
        left_on=["evaluation_year", "race_id", "lambdarank_current"],
        right_on=["evaluation_year", "race_id", "horse_id"],
        how="left",
    )
    .rename(columns={"is_winner": "rank_correct"})
    .drop(columns="horse_id")
)
top_pair[["binary_correct", "rank_correct"]] = top_pair[["binary_correct", "rank_correct"]].fillna(0)
disagreement_table = (
    top_pair.groupby("evaluation_year", observed=True)
    .agg(
        races=("race_id", "size"),
        top_choice_disagree_rate=("top_choice_disagree", "mean"),
        binary_top1=("binary_correct", "mean"),
        rank_top1=("rank_correct", "mean"),
    )
    .reset_index()
)
for year, g in top_pair.groupby("evaluation_year", sort=True):
    mask = g.top_choice_disagree.eq(1)
    disagreement_table.loc[disagreement_table.evaluation_year.eq(year), "binary_top1_when_disagree"] = g.loc[
        mask, "binary_correct"
    ].mean()
    disagreement_table.loc[disagreement_table.evaluation_year.eq(year), "rank_top1_when_disagree"] = g.loc[
        mask, "rank_correct"
    ].mean()
disagreement_table.to_csv(ROOT / "tables" / "disagreement.csv", index=False)

slice_rows = []
slice_specs = [
    ("winner_history", "winner_history_band"),
    ("winner_connection", "winner_connection_band"),
    ("surface", "surface"),
    ("class", "class_group"),
    ("distance", "distance_band"),
    ("field_size", "field_band"),
]
# Cut points are frozen on discovery (2014--2019), then transferred unchanged.
cuts = np.unique(np.concatenate(([-np.inf], uncertainty_cuts, [np.inf])))
races["uncertainty_band"] = pd.cut(
    races.field_uncertainty,
    bins=cuts,
    labels=["q1_low", "q2", "q3", "q4_high"],
    include_lowest=True,
).astype(str)
slice_specs.append(("field_uncertainty", "uncertainty_band"))
for (year, method), part in races.groupby(["year", "method"], sort=True):
    for dimension, column in slice_specs:
        for level, g in part.groupby(column, observed=True):
            if len(g) < 100:
                continue
            slice_rows.append(
                {
                    "year": int(year),
                    "method": method,
                    "dimension": dimension,
                    "level": str(level),
                    "races": int(len(g)),
                    "ndcg_at_3": float(g.ndcg_at_3.mean()),
                    "top1": float(g.top1.mean()),
                    "race_log_loss": float(g.race_log_loss.mean()),
                    "winner_rank": float(g.winner_rank.mean()),
                    "high_confidence_error": float(g.high_confidence_error.mean()),
                }
            )
slices = pd.DataFrame(slice_rows)
slices.to_csv(ROOT / "tables" / "error_slices.csv", index=False)

market_slice = (
    market_races.groupby(["year", "method", "winner_history_band"], observed=True)
    .agg(
        races=("race_id", "size"),
        market_log_loss=("market_log_loss", "mean"),
        model_log_loss=("model_log_loss", "mean"),
        model_market_gap=("model_market_gap", "mean"),
    )
    .reset_index()
)
market_slice.to_csv(ROOT / "tables" / "market_gap_by_history.csv", index=False)
market_metrics = (
    market_races.groupby(["year", "method"], observed=True)
    .agg(
        races=("race_id", "size"),
        market_log_loss=("market_log_loss", "mean"),
        market_brier=("market_brier", "mean"),
        market_ndcg_at_3=("market_ndcg_at_3", "mean"),
        market_top1=("market_top1", "mean"),
        model_market_gap=("model_market_gap", "mean"),
    )
    .reset_index()
)
market_metrics.to_csv(ROOT / "tables" / "market_oracle.csv", index=False)

# Dependency-free aggregate SVG plots.
COLORS = {"binary_current": "#2563eb", "lambdarank_current": "#dc2626"}


def svg_line(
    path: Path,
    title: str,
    series: dict[str, list[tuple[float, float]]],
    x_labels: list[str],
    y_min: float,
    y_max: float,
) -> None:
    width, height = 760, 420
    left, right, top, bottom = 75, 25, 55, 60
    pw = width - left - right
    ph = height - top - bottom

    def xy(x, y):
        return left + x * pw, top + (y_max - y) / (y_max - y_min) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
    ]
    for i in range(6):
        y = y_min + (y_max - y_min) * i / 5
        _, py = xy(0, y)
        parts.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" '
            f'y2="{py:.1f}" stroke="#ddd"/><text x="{left - 8}" '
            f'y="{py + 4:.1f}" text-anchor="end" font-family="sans-serif" '
            f'font-size="11">{y:.3f}</text>'
        )
    denom = max(1, len(x_labels) - 1)
    for i, label in enumerate(x_labels):
        px, _ = xy(i / denom, y_min)
        parts.append(
            f'<text x="{px:.1f}" y="{height - 28}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{label}</text>'
        )
    for name, points in series.items():
        coords = []
        for x, y in points:
            px, py = xy(x / denom, y)
            coords.append(f"{px:.1f},{py:.1f}")
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{COLORS[name]}"/>')
        parts.append(
            f'<polyline points="{" ".join(coords)}" fill="none" '
            f'stroke="{COLORS[name]}" stroke-width="2"/>'
            f'<text x="{width - right - 145}" '
            f'y="{45 + 18 * list(series).index(name)}" font-family="sans-serif" '
            f'font-size="12" fill="{COLORS[name]}">{name}</text>'
        )
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


years = sorted(metric_table.year.unique())
for metric, title in (("race_log_loss", "OOT race log loss"), ("ndcg_at_3", "OOT NDCG@3")):
    series = {
        m: [
            (i, float(metric_table.loc[(metric_table.year.eq(y)) & (metric_table.method.eq(m)), metric].iloc[0]))
            for i, y in enumerate(years)
        ]
        for m in METHODS
    }
    values = [v for points in series.values() for _, v in points]
    pad = max((max(values) - min(values)) * 0.2, 0.005)
    svg_line(
        ROOT / "plots" / f"{metric}.svg", title, series, [str(y) for y in years], min(values) - pad, max(values) + pad
    )

plot_market = (
    market_slice.groupby(["method", "winner_history_band"], observed=True)["model_market_gap"].mean().unstack(0)
)
bands = list(plot_market.index)
series = {m: [(i, float(plot_market.loc[b, m])) for i, b in enumerate(bands)] for m in METHODS}
values = [v for points in series.values() for _, v in points]
svg_line(
    ROOT / "plots" / "market_gap_by_history.svg",
    "Model minus final-market log loss",
    series,
    bands,
    min(values) - 0.02,
    max(values) + 0.02,
)

cal_series = {}
for method, g in calibration.groupby("method"):
    gg = g.groupby("bin", sort=False).agg(
        mean_probability=("mean_probability", "mean"), observed_rate=("observed_rate", "mean")
    )
    cal_series[method] = [(i, float(v)) for i, v in enumerate(gg.observed_rate)]
svg_line(
    ROOT / "plots" / "calibration.svg",
    "Race-weighted observed winner mass by probability bin",
    cal_series,
    [str(v) for v in range(len(next(iter(cal_series.values()))))],
    0,
    0.4,
)

summary = {
    "contract": {
        "prediction_path": str(PREDICTIONS),
        "prediction_sha256": sha256(PREDICTIONS),
        "source_rows_streamed": source_rows,
        "retained_role": sorted(pred.role.unique()),
        "retained_evaluation_years": sorted(int(v) for v in pred.evaluation_year.unique()),
        "retained_rows": int(len(pred)),
        "retained_2023_rows": int(pred.evaluation_year.eq(2023).sum()),
        "retained_2024_2025_rows": int(pred.evaluation_year.ge(2024).sum()),
        "raw_sha256": sha256_file(RAW),
        "market_join": "explicit separate market table; complete races only; oracle diagnostic",
        "final_odds_used_for_feature_or_acceptance": False,
        "bootstrap_seed": SEED,
        "bootstrap_resamples": RESAMPLES,
        "uncertainty_cut_source": "2014-2019 discovery races only",
        "uncertainty_cut_points": [float(value) for value in uncertainty_cuts],
        "history_bands": ["0", "1", "2-3", "4-9", "10+"],
        "minimum_error_slice_races": 100,
        "market_top1_definition": "coherent winner mass at market top choice",
    },
    "counts": metric_table.to_dict("records"),
    "metrics_with_ci": {},
    "calibration": {},
    "disagreement": disagreement_table.to_dict("records"),
    "winner_rank_distribution": {},
    "high_confidence_errors": {},
    "market_oracle": market_metrics.to_dict("records"),
    "market_gap_by_history": market_slice.to_dict("records"),
    "missingness": {},
    "invariants": {
        "prediction_2023_rows_retained": 0,
        "prediction_2024_2025_rows_retained": 0,
        "market_columns_in_primary_analysis": False,
        "production_model_fit": False,
        "ensemble_fit": False,
        "selection_performed": False,
    },
}
for (year, method), part in races.groupby(["year", "method"], sort=True):
    key = f"{year}|{method}"
    summary["metrics_with_ci"][key] = {
        metric: date_bootstrap_mean(part, metric) for metric in ("ndcg_at_3", "top1", "race_log_loss", "race_brier")
    }
    cal = calibration.loc[(calibration.year.eq(year)) & (calibration.method.eq(method))]
    summary["calibration"][key] = {
        "race_weighted_ece": float(np.average(np.abs(cal.gap), weights=cal.race_weight)),
        "bins": cal.to_dict("records"),
    }
    summary["winner_rank_distribution"][key] = {
        "median": float(part.winner_rank.median()),
        "p75": float(part.winner_rank.quantile(0.75)),
        "p90": float(part.winner_rank.quantile(0.9)),
        "winner_outside_top3": float(part.winner_rank.gt(3).mean()),
    }
    selected = date_bootstrap_mean(part, "high_confidence_selected")
    wrong = date_bootstrap_mean(part, "high_confidence_error")
    summary["high_confidence_errors"][key] = {
        "selected": selected,
        "wrong_all_races": wrong,
        "conditional_wrong_rate": float(part.high_confidence_error.sum() / part.high_confidence_selected.sum())
        if part.high_confidence_selected.sum()
        else None,
    }
    summary["missingness"][key] = {
        "outcome_join": float(part.race_log_loss.isna().mean()),
        "market_incomplete_race_rate": float(
            1
            - market_races.loc[(market_races.year.eq(year)) & (market_races.method.eq(method)), "race_id"].nunique()
            / part.race_id.nunique()
        ),
    }
(ROOT / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, default=scalar) + "\n", encoding="utf-8"
)

log_lines = [
    "Workstream H completed",
    f"prediction rows streamed: {source_rows}",
    f"retained rows: {len(pred)}",
    f"retained years: {sorted(YEARS)}",
    "retained 2023 rows: 0",
    "retained 2024/2025 rows: 0",
    "market: explicit complete-race final-odds oracle join only",
    "production model / ensemble / selection: not run",
]
(ROOT / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
files = []
for path in sorted(ROOT.rglob("*")):
    if path.is_file() and path.name != "manifest.json" and "__pycache__" not in path.parts:
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "size_bytes": path.stat().st_size})
(ROOT / "manifest.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "files": files,
            "firewall": {"evaluation_years": [2020, 2021, 2022], "retained_2023": 0, "retained_2024_2025": 0},
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "retained_rows": len(pred),
            "metric_table": metric_table.to_dict("records"),
            "market": market_metrics.to_dict("records"),
            "artifacts": [f["path"] for f in files],
        },
        ensure_ascii=False,
        indent=2,
        default=scalar,
    )
)
