"""Fail-closed local archive for normalized JV-Link batch envelopes.

This module deliberately has no JV-Link, HTTP, or JRA Web transport code.  A
private Windows-side adapter may produce the normalized JSON envelope defined
in ``schemas/live_data/jvlink_batch_envelope.schema.json``; this module only
validates and durably archives that envelope on the local filesystem.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from horse_pred.config import canonical_json_hash, load_json


class EnvelopeValidationError(ValueError):
    """Raised before any archive write when an envelope violates the contract."""


_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]+$")
_TOP_LEVEL_KEYS = {"schema_version", "source", "batch", "records"}
_SOURCE_KEYS = {
    "provider",
    "transport",
    "service_data_id",
    "jv_link_version",
    "jvdata_spec_version",
}
_BATCH_KEYS = {"batch_id", "collector_id", "requested_at", "observed_at", "source_cursor"}
_RECORD_REQUIRED_KEYS = {
    "record_type",
    "source_key",
    "data_division",
    "source_created_date",
    "published_at",
    "published_at_precision",
    "payload_encoding",
    "payload_base64",
    "source_flags",
}
_RECORD_OPTIONAL_KEYS = {"observed_at"}


def load_live_data_archive_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the checked-in archive policy."""

    config = load_json(path)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported live-data archive config schema_version")
    if config.get("operational_collection_started") is not False:
        raise ValueError("Groundwork config must not claim operational collection has started")
    required = {"source_policy", "archive_policy", "clock_policy"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Live-data archive config is missing: {sorted(missing)}")
    return config


def archive_jvlink_batch(
    envelope: Mapping[str, Any],
    *,
    archive_root: str | Path,
    repo_root: str | Path,
    config: Mapping[str, Any],
    ingested_at: datetime | None = None,
    receipt_id: str | None = None,
) -> dict[str, Any]:
    """Validate and append one local receipt plus content-addressed records.

    The envelope never supplies ``ingested_at``.  The archive boundary assigns
    it.  Repeated source records reuse the immutable content object and record
    manifest, but every accepted batch gets a new append-only receipt.
    """

    validated = validate_jvlink_batch_envelope(envelope, config=config)
    root = _validate_archive_root(archive_root, repo_root=repo_root, config=config)
    accepted_at = _as_utc(ingested_at or datetime.now(timezone.utc))
    accepted_at_text = accepted_at.isoformat()
    identifier = receipt_id or f"{accepted_at.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}"
    _validate_safe_id(identifier, field="receipt_id")

    root.mkdir(parents=True, exist_ok=True)
    marker = {
        "schema_version": 1,
        "archive_kind": "private_local_jvlink_append_only",
        "source_provider": config["source_policy"]["provider"],
        "transport": config["source_policy"]["transport"],
        "config_hash": canonical_json_hash(config),
        "warning": "Private raw data. Do not commit, share, or send to external AI services.",
    }
    _ensure_exact_file(root / ".private_archive.json", _json_bytes(marker))

    source = validated["source"]
    batch = validated["batch"]
    receipt_records: list[dict[str, Any]] = []
    new_payload_count = 0
    new_record_count = 0

    for record in validated["records"]:
        payload = record.pop("_payload_bytes")
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        payload_path = root / "objects" / "sha256" / payload_sha256[:2] / f"{payload_sha256}.bin"
        payload_is_new = _ensure_content_addressed_file(payload_path, payload, expected_hash=payload_sha256)
        new_payload_count += int(payload_is_new)

        semantic_record = {
            "schema_version": 1,
            "source": source,
            "service_data_id": source["service_data_id"],
            "record_type": record["record_type"],
            "source_key": record["source_key"],
            "data_division": record["data_division"],
            "source_created_date": record["source_created_date"],
            "published_at": record["published_at"],
            "published_at_precision": record["published_at_precision"],
            "source_flags": record["source_flags"],
            "payload_sha256": payload_sha256,
            "payload_size_bytes": len(payload),
        }
        record_sha256 = hashlib.sha256(_canonical_json_bytes(semantic_record)).hexdigest()
        record_path = root / "records" / "sha256" / record_sha256[:2] / f"{record_sha256}.json"
        record_is_new = _ensure_exact_file(record_path, _json_bytes(semantic_record))
        new_record_count += int(record_is_new)

        observed_at = _parse_timestamp(record["observed_at"], field="record.observed_at")
        published_at = (
            _parse_timestamp(record["published_at"], field="record.published_at")
            if record["published_at"] is not None
            else None
        )
        clock_flags = _clock_flags(
            published_at=published_at,
            observed_at=observed_at,
            ingested_at=accepted_at,
            tolerance_seconds=int(config["clock_policy"]["future_tolerance_seconds"]),
        )
        receipt_records.append(
            {
                "record_type": record["record_type"],
                "source_key": record["source_key"],
                "published_at": record["published_at"],
                "published_at_precision": record["published_at_precision"],
                "observed_at": record["observed_at"],
                "ingested_at": accepted_at_text,
                "source_flags": record["source_flags"],
                "clock_flags": clock_flags,
                "payload_sha256": payload_sha256,
                "record_sha256": record_sha256,
                "payload_object": str(payload_path.relative_to(root)),
                "record_manifest": str(record_path.relative_to(root)),
                "payload_deduplicated": not payload_is_new,
                "record_deduplicated": not record_is_new,
            }
        )

    batch_semantics = {
        "source": source,
        "batch_id": batch["batch_id"],
        "collector_id": batch["collector_id"],
        "requested_at": batch["requested_at"],
        "observed_at": batch["observed_at"],
        "source_cursor": batch["source_cursor"],
        "record_sha256": [item["record_sha256"] for item in receipt_records],
    }
    batch_content_hash = hashlib.sha256(_canonical_json_bytes(batch_semantics)).hexdigest()
    batch_clock_flags = _batch_clock_flags(
        requested_at=_parse_timestamp(batch["requested_at"], field="batch.requested_at"),
        observed_at=_parse_timestamp(batch["observed_at"], field="batch.observed_at"),
        ingested_at=accepted_at,
        tolerance_seconds=int(config["clock_policy"]["future_tolerance_seconds"]),
    )
    receipt = {
        "schema_version": 1,
        "receipt_id": identifier,
        "batch_id": batch["batch_id"],
        "batch_content_hash": batch_content_hash,
        "source": source,
        "collector_id": batch["collector_id"],
        "requested_at": batch["requested_at"],
        "observed_at": batch["observed_at"],
        "ingested_at": accepted_at_text,
        "source_cursor": batch["source_cursor"],
        "clock_flags": batch_clock_flags,
        "records": receipt_records,
        "summary": {
            "record_count": len(receipt_records),
            "new_payload_count": new_payload_count,
            "deduplicated_payload_count": len(receipt_records) - new_payload_count,
            "new_record_count": new_record_count,
            "deduplicated_record_count": len(receipt_records) - new_record_count,
        },
        "collection_status": "groundwork_archive_only_not_live_collection",
    }
    receipt_date = accepted_at.date().isoformat()
    receipt_path = root / "receipts" / receipt_date / f"{identifier}.json"
    if not _write_new_file(receipt_path, _json_bytes(receipt)):
        raise FileExistsError(f"Receipt already exists: {identifier}")

    return {
        "receipt_id": identifier,
        "receipt_path": str(receipt_path),
        "batch_content_hash": batch_content_hash,
        "ingested_at": accepted_at_text,
        "summary": receipt["summary"],
        "clock_flags": batch_clock_flags,
        "collection_status": receipt["collection_status"],
    }


def validate_jvlink_batch_envelope(envelope: Mapping[str, Any], *, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a transport-neutral batch without writing files."""

    if not isinstance(envelope, Mapping):
        raise EnvelopeValidationError("Envelope must be a JSON object")
    _require_exact_keys(envelope, required=_TOP_LEVEL_KEYS, allowed=_TOP_LEVEL_KEYS, field="envelope")
    if envelope["schema_version"] != 1:
        raise EnvelopeValidationError("Unsupported envelope schema_version")

    source = envelope["source"]
    if not isinstance(source, Mapping):
        raise EnvelopeValidationError("source must be an object")
    _require_exact_keys(source, required=_SOURCE_KEYS, allowed=_SOURCE_KEYS, field="source")
    policy = config["source_policy"]
    if source["provider"] != policy["provider"] or source["transport"] != policy["transport"]:
        raise EnvelopeValidationError("Only the configured official JRA-VAN JV-Link source is accepted")
    service_data_id = source["service_data_id"]
    allowed_records = policy["service_data_records"].get(service_data_id)
    if allowed_records is None:
        raise EnvelopeValidationError(f"Unsupported JV-Link service_data_id: {service_data_id}")
    for field in ("jv_link_version", "jvdata_spec_version"):
        _validate_nonempty_string(source[field], field=f"source.{field}", maximum=64)

    batch = envelope["batch"]
    if not isinstance(batch, Mapping):
        raise EnvelopeValidationError("batch must be an object")
    _require_exact_keys(batch, required=_BATCH_KEYS, allowed=_BATCH_KEYS, field="batch")
    for field in ("batch_id", "collector_id"):
        _validate_safe_id(batch[field], field=f"batch.{field}")
    requested = _parse_timestamp(batch["requested_at"], field="batch.requested_at")
    observed = _parse_timestamp(batch["observed_at"], field="batch.observed_at")
    if batch["source_cursor"] is not None:
        _validate_nonempty_string(batch["source_cursor"], field="batch.source_cursor", maximum=512)

    records = envelope["records"]
    maximum_records = int(config["archive_policy"]["maximum_records_per_batch"])
    if not isinstance(records, list) or not records:
        raise EnvelopeValidationError("records must be a non-empty array")
    if len(records) > maximum_records:
        raise EnvelopeValidationError(f"records exceeds configured maximum of {maximum_records}")

    allowed_source_flags = set(policy["allowed_source_flags"])
    normalized_records: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        field_root = f"records[{index}]"
        if not isinstance(item, Mapping):
            raise EnvelopeValidationError(f"{field_root} must be an object")
        _require_exact_keys(
            item,
            required=_RECORD_REQUIRED_KEYS,
            allowed=_RECORD_REQUIRED_KEYS | _RECORD_OPTIONAL_KEYS,
            field=field_root,
        )
        if item["record_type"] not in allowed_records:
            raise EnvelopeValidationError(
                f"{field_root}.record_type is not valid for {service_data_id}: {item['record_type']}"
            )
        _validate_nonempty_string(item["source_key"], field=f"{field_root}.source_key", maximum=256)
        _validate_nonempty_string(item["data_division"], field=f"{field_root}.data_division", maximum=16)
        try:
            date.fromisoformat(item["source_created_date"])
        except (TypeError, ValueError) as exc:
            raise EnvelopeValidationError(f"{field_root}.source_created_date must be YYYY-MM-DD") from exc

        precision = item["published_at_precision"]
        if precision not in {"minute", "date", "unavailable"}:
            raise EnvelopeValidationError(f"{field_root}.published_at_precision is invalid")
        published_text = item["published_at"]
        if published_text is None:
            if precision == "minute":
                raise EnvelopeValidationError(f"{field_root}.published_at cannot be null at minute precision")
        else:
            if precision != "minute":
                raise EnvelopeValidationError(f"{field_root}.published_at requires minute precision")
            published = _parse_timestamp(published_text, field=f"{field_root}.published_at")
            required_offset = int(config["clock_policy"]["source_timezone_offset_minutes"])
            offset = published.utcoffset()
            if offset is None or int(offset.total_seconds() // 60) != required_offset:
                raise EnvelopeValidationError(f"{field_root}.published_at must use the configured JST offset")
            if published.second or published.microsecond:
                raise EnvelopeValidationError(f"{field_root}.published_at must reflect minute precision")

        record_observed = _parse_timestamp(
            item.get("observed_at", batch["observed_at"]), field=f"{field_root}.observed_at"
        )
        if item["payload_encoding"] != "base64":
            raise EnvelopeValidationError(f"{field_root}.payload_encoding must be base64")
        try:
            payload = base64.b64decode(item["payload_base64"], validate=True)
        except (TypeError, ValueError, binascii.Error) as exc:
            raise EnvelopeValidationError(f"{field_root}.payload_base64 is invalid") from exc
        maximum_bytes = int(config["archive_policy"]["maximum_payload_bytes_per_record"])
        if not payload or len(payload) > maximum_bytes:
            raise EnvelopeValidationError(f"{field_root} payload size must be between 1 and {maximum_bytes} bytes")
        source_flags = item["source_flags"]
        if not isinstance(source_flags, list) or any(not isinstance(flag, str) for flag in source_flags):
            raise EnvelopeValidationError(f"{field_root}.source_flags must be an array of strings")
        if len(source_flags) != len(set(source_flags)):
            raise EnvelopeValidationError(f"{field_root}.source_flags must not contain duplicates")
        unsupported_flags = set(source_flags).difference(allowed_source_flags)
        if unsupported_flags:
            raise EnvelopeValidationError(f"{field_root}.source_flags contains unsupported values")

        normalized_records.append(
            {
                "record_type": item["record_type"],
                "source_key": item["source_key"],
                "data_division": item["data_division"],
                "source_created_date": item["source_created_date"],
                "published_at": _timestamp_text(published_text),
                "published_at_precision": precision,
                "observed_at": record_observed.astimezone(timezone.utc).isoformat(),
                "source_flags": sorted(source_flags),
                "_payload_bytes": payload,
            }
        )

    return {
        "schema_version": 1,
        "source": dict(source),
        "batch": {
            **dict(batch),
            "requested_at": requested.astimezone(timezone.utc).isoformat(),
            "observed_at": observed.astimezone(timezone.utc).isoformat(),
        },
        "records": normalized_records,
    }


def _validate_archive_root(archive_root: str | Path, *, repo_root: str | Path, config: Mapping[str, Any]) -> Path:
    root = Path(archive_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    if root == repo:
        raise ValueError("Archive root must not be the repository root")
    try:
        relative = root.relative_to(repo)
    except ValueError:
        return root
    allowed_prefix = Path(config["archive_policy"]["allowed_in_repo_prefix"])
    if relative != allowed_prefix and allowed_prefix not in relative.parents:
        raise ValueError(f"In-repository private archive must be below {allowed_prefix}")
    return root


def _clock_flags(
    *,
    published_at: datetime | None,
    observed_at: datetime,
    ingested_at: datetime,
    tolerance_seconds: int,
) -> list[str]:
    flags: list[str] = []
    if published_at is None:
        flags.append("source_published_at_unavailable")
    elif (published_at - observed_at).total_seconds() > tolerance_seconds:
        flags.append("source_published_after_observed")
    if (observed_at - ingested_at).total_seconds() > tolerance_seconds:
        flags.append("observed_after_ingested")
    return flags or ["clock_order_ok"]


def _batch_clock_flags(
    *, requested_at: datetime, observed_at: datetime, ingested_at: datetime, tolerance_seconds: int
) -> list[str]:
    flags: list[str] = []
    if (requested_at - observed_at).total_seconds() > tolerance_seconds:
        flags.append("requested_after_observed")
    if (observed_at - ingested_at).total_seconds() > tolerance_seconds:
        flags.append("observed_after_ingested")
    return flags or ["clock_order_ok"]


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise EnvelopeValidationError(f"{field} must be a timezone-aware ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EnvelopeValidationError(f"{field} must be a timezone-aware ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EnvelopeValidationError(f"{field} must include a timezone offset")
    return parsed


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("ingested_at must be timezone-aware")
    return value.astimezone(timezone.utc)


def _timestamp_text(value: str | None) -> str | None:
    if value is None:
        return None
    return _parse_timestamp(value, field="published_at").isoformat()


def _validate_safe_id(value: Any, *, field: str) -> None:
    _validate_nonempty_string(value, field=field, maximum=128)
    if not _SAFE_ID.fullmatch(value):
        raise EnvelopeValidationError(f"{field} contains unsupported characters")


def _validate_nonempty_string(value: Any, *, field: str, maximum: int) -> None:
    if not isinstance(value, str) or not value or len(value) > maximum or "\n" in value or "\r" in value:
        raise EnvelopeValidationError(f"{field} must be a non-empty bounded string")


def _require_exact_keys(value: Mapping[str, Any], *, required: set[str], allowed: set[str], field: str) -> None:
    missing = required.difference(value)
    unknown = set(value).difference(allowed)
    if missing:
        raise EnvelopeValidationError(f"{field} is missing required keys: {sorted(missing)}")
    if unknown:
        raise EnvelopeValidationError(f"{field} contains unsupported keys: {sorted(unknown)}")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _ensure_content_addressed_file(path: Path, payload: bytes, *, expected_hash: str) -> bool:
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"Content-addressed archive corruption detected: {path}")
        return False
    created = _write_new_file(path, payload)
    if not created and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError(f"Content-addressed archive race/corruption detected: {path}")
    return created


def _ensure_exact_file(path: Path, payload: bytes) -> bool:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Immutable archive metadata mismatch: {path}")
        return False
    created = _write_new_file(path, payload)
    if not created and path.read_bytes() != payload:
        raise RuntimeError(f"Immutable archive metadata race/mismatch: {path}")
    return created


def _write_new_file(path: Path, payload: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return False
    return True
