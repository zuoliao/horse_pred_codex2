import base64
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from horse_pred.cli import main
from horse_pred.live_data_archive import (
    EnvelopeValidationError,
    archive_jvlink_batch,
    load_live_data_archive_config,
    validate_jvlink_batch_envelope,
)

CONFIG_PATH = Path("configs/live_data/jvlink_archive.json")


def _config() -> dict:
    return load_live_data_archive_config(CONFIG_PATH)


def _odds_envelope(payload: bytes = b"O1 normalized fixture bytes") -> dict:
    return {
        "schema_version": 1,
        "source": {
            "provider": "jra_van_data_lab",
            "transport": "jv_link",
            "service_data_id": "0B41",
            "jv_link_version": "5.0.0",
            "jvdata_spec_version": "4.9.0.1",
        },
        "batch": {
            "batch_id": "batch-20260901-001",
            "collector_id": "private-windows-collector-01",
            "requested_at": "2026-09-01T08:51:00+09:00",
            "observed_at": "2026-09-01T08:52:00+09:00",
            "source_cursor": "fixture-cursor",
        },
        "records": [
            {
                "record_type": "O1",
                "source_key": "202609010101:O1:0850",
                "data_division": "1",
                "source_created_date": "2026-09-01",
                "published_at": "2026-09-01T08:50:00+09:00",
                "published_at_precision": "minute",
                "payload_encoding": "base64",
                "payload_base64": base64.b64encode(payload).decode("ascii"),
                "source_flags": [
                    "source_intermediate_odds",
                    "source_timestamp_minute_precision",
                ],
            }
        ],
    }


def _private_root(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    root = repo / "data" / "live_data_private" / "archive"
    repo.mkdir()
    return repo, root


def test_archive_writes_content_hash_manifest_and_receipt(tmp_path) -> None:
    repo, root = _private_root(tmp_path)
    payload = b"exact private JV-Link fixture"
    result = archive_jvlink_batch(
        _odds_envelope(payload),
        archive_root=root,
        repo_root=repo,
        config=_config(),
        ingested_at=datetime(2026, 8, 31, 23, 53, tzinfo=timezone.utc),
        receipt_id="receipt-001",
    )

    assert result["collection_status"] == "groundwork_archive_only_not_live_collection"
    assert result["summary"] == {
        "record_count": 1,
        "new_payload_count": 1,
        "deduplicated_payload_count": 0,
        "new_record_count": 1,
        "deduplicated_record_count": 0,
    }
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    record = receipt["records"][0]
    assert receipt["clock_flags"] == ["clock_order_ok"]
    assert record["clock_flags"] == ["clock_order_ok"]
    assert record["published_at"] == "2026-09-01T08:50:00+09:00"
    assert record["observed_at"] == "2026-08-31T23:52:00+00:00"
    assert record["ingested_at"] == "2026-08-31T23:53:00+00:00"
    assert record["payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert (root / record["payload_object"]).read_bytes() == payload
    manifest = json.loads((root / record["record_manifest"]).read_text(encoding="utf-8"))
    assert manifest["payload_sha256"] == record["payload_sha256"]
    assert "payload_base64" not in json.dumps(receipt)
    assert "payload_base64" not in json.dumps(manifest)


def test_duplicate_record_reuses_objects_but_appends_receipt(tmp_path) -> None:
    repo, root = _private_root(tmp_path)
    envelope = _odds_envelope()
    first = archive_jvlink_batch(
        envelope,
        archive_root=root,
        repo_root=repo,
        config=_config(),
        ingested_at=datetime(2026, 8, 31, 23, 53, tzinfo=timezone.utc),
        receipt_id="receipt-001",
    )
    second = archive_jvlink_batch(
        envelope,
        archive_root=root,
        repo_root=repo,
        config=_config(),
        ingested_at=datetime(2026, 8, 31, 23, 54, tzinfo=timezone.utc),
        receipt_id="receipt-002",
    )

    assert first["batch_content_hash"] == second["batch_content_hash"]
    assert second["summary"]["deduplicated_payload_count"] == 1
    assert second["summary"]["deduplicated_record_count"] == 1
    assert len(list((root / "receipts").rglob("*.json"))) == 2
    assert len(list((root / "objects").rglob("*.bin"))) == 1
    assert len(list((root / "records").rglob("*.json"))) == 1


def test_fail_closed_on_non_jvlink_source_before_writing(tmp_path) -> None:
    repo, root = _private_root(tmp_path)
    envelope = _odds_envelope()
    envelope["source"]["provider"] = "jra_web"
    envelope["source"]["transport"] = "http_scraper"

    with pytest.raises(EnvelopeValidationError, match="Only the configured official"):
        archive_jvlink_batch(
            envelope,
            archive_root=root,
            repo_root=repo,
            config=_config(),
        )
    assert not root.exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value["source"].update({"service_data_id": "0B30"}), "service_data_id"),
        (lambda value: value["records"][0].update({"record_type": "WH"}), "record_type"),
        (lambda value: value["records"][0].update({"payload_base64": "***"}), "payload_base64"),
        (
            lambda value: value["records"][0].update({"published_at": "2026-09-01T08:50:30+09:00"}),
            "minute precision",
        ),
        (
            lambda value: value["records"][0].update({"source_flags": ["unreviewed_flag"]}),
            "unsupported values",
        ),
    ],
)
def test_envelope_validation_rejects_unreviewed_inputs(change, message) -> None:
    envelope = _odds_envelope()
    change(envelope)
    with pytest.raises(EnvelopeValidationError, match=message):
        validate_jvlink_batch_envelope(envelope, config=_config())


def test_missing_source_publish_time_is_retained_and_flagged(tmp_path) -> None:
    repo, root = _private_root(tmp_path)
    envelope = _odds_envelope(b"RA fixture")
    envelope["source"]["service_data_id"] = "0B15"
    record = envelope["records"][0]
    record.update(
        {
            "record_type": "RA",
            "source_key": "202609010101:RA:race-card",
            "published_at": None,
            "published_at_precision": "date",
            "source_flags": ["source_created_date_only", "source_timestamp_unavailable"],
        }
    )
    result = archive_jvlink_batch(
        envelope,
        archive_root=root,
        repo_root=repo,
        config=_config(),
        ingested_at=datetime(2026, 8, 31, 23, 53, tzinfo=timezone.utc),
        receipt_id="receipt-ra",
    )
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["records"][0]["published_at"] is None
    assert receipt["records"][0]["clock_flags"] == ["source_published_at_unavailable"]


def test_future_times_create_flags_instead_of_rewriting_evidence(tmp_path) -> None:
    repo, root = _private_root(tmp_path)
    envelope = _odds_envelope()
    envelope["records"][0]["published_at"] = "2026-09-01T09:10:00+09:00"
    result = archive_jvlink_batch(
        envelope,
        archive_root=root,
        repo_root=repo,
        config=_config(),
        ingested_at=datetime(2026, 8, 31, 23, 40, tzinfo=timezone.utc),
        receipt_id="receipt-clock-flags",
    )
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["clock_flags"] == ["observed_after_ingested"]
    assert receipt["records"][0]["clock_flags"] == [
        "source_published_after_observed",
        "observed_after_ingested",
    ]


def test_in_repo_archive_must_stay_below_ignored_private_prefix(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError, match="data/live_data_private"):
        archive_jvlink_batch(
            _odds_envelope(),
            archive_root=repo / "tracked_raw",
            repo_root=repo,
            config=_config(),
        )
    assert not (repo / "tracked_raw").exists()


def test_ingested_at_cannot_be_supplied_by_adapter() -> None:
    envelope = deepcopy(_odds_envelope())
    envelope["batch"]["ingested_at"] = "2026-09-01T08:53:00+09:00"
    with pytest.raises(EnvelopeValidationError, match="unsupported keys"):
        validate_jvlink_batch_envelope(envelope, config=_config())


def test_checked_in_schema_and_policy_are_explicitly_groundwork_only() -> None:
    schema = json.loads(Path("schemas/live_data/jvlink_batch_envelope.schema.json").read_text())
    config = _config()
    assert schema["properties"]["source"]["properties"]["transport"]["const"] == "jv_link"
    assert config["operational_collection_started"] is False
    assert config["archive_policy"]["raw_payload_git_policy"] == "ignored_private_never_commit"


def test_cli_archives_local_envelope_without_echoing_payload(tmp_path, capsys) -> None:
    input_path = tmp_path / "normalized_batch.json"
    input_path.write_text(json.dumps(_odds_envelope(b"private CLI fixture")), encoding="utf-8")
    archive_root = tmp_path / "private-archive"

    assert (
        main(
            [
                "archive-jvlink-batch",
                "--input",
                str(input_path),
                "--archive-root",
                str(archive_root),
                "--repo-root",
                str(Path.cwd()),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    summary = json.loads(output)
    assert summary["summary"]["record_count"] == 1
    assert summary["collection_status"] == "groundwork_archive_only_not_live_collection"
    assert "private CLI fixture" not in output
