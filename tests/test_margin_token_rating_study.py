from __future__ import annotations

import pandas as pd

from horse_pred.margin_token_rating_study import derive_margin_token_audit
from horse_pred.rating import RatingEvent

ORDERED = ["ハナ", "アタマ", "クビ", "1/2"]
MAPPING = {"ハナ": 0.02, "アタマ": 0.04, "クビ": 0.06, "1/2": 0.08}


def _event(year: int, *, last_gap: float = 0.1) -> RatingEvent:
    return RatingEvent(
        race_id=f"{year}05010101",
        race_date=pd.Timestamp(f"{year}-01-01"),
        surface_key="turf",
        horse_ids=("h1", "h2", "h3"),
        finishes=(1.0, 2.0, 3.0),
        source_positions=(0, 1, 2),
        result_times_seconds=(100.0, 100.0, 100.0 + last_gap),
        finish_statuses=("finished", "finished", "finished"),
        margin_tokens=(None, "ハナ", "1/2"),
        distance_m=1600.0,
    )


def test_train_only_audit_ignores_future_sentinel() -> None:
    baseline = derive_margin_token_audit(
        (_event(2021),),
        years={2021},
        ordered_tokens=ORDERED,
        tie_token_seconds=MAPPING,
    )
    with_future = derive_margin_token_audit(
        (_event(2021), _event(2022, last_gap=-10.0)),
        years={2021},
        ordered_tokens=ORDERED,
        tie_token_seconds=MAPPING,
    )

    assert baseline == with_future
    assert baseline["eligible_adjacent_distinct_rank_edges"] == 2
    assert baseline["equal_clock"]["token_counts"] == {"ハナ": 1}
    assert baseline["clock_inversion_count"] == 0


def test_audit_reports_status_exclusion_and_token_order() -> None:
    clean = _event(2021)
    excluded = RatingEvent(
        **{
            **clean.__dict__,
            "race_id": "202105010102",
            "finish_statuses": ("finished", "demoted", "finished"),
        }
    )
    audit = derive_margin_token_audit(
        (clean, excluded),
        years={2021},
        ordered_tokens=ORDERED,
        tie_token_seconds=MAPPING,
    )

    assert audit["race_count"] == 2
    assert audit["clean_race_count"] == 1
    assert audit["excluded_status_race_count"] == 1
    assert audit["recognized_edge_fraction"] == 1.0
