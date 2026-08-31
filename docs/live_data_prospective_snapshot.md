# LIVE-DATA prospective snapshot design

**Status:** groundwork implemented; operational collection deferred by the user
**Decision date:** 2026-08-31 JST  
**Scope:** private, individual, no-odds-model research support only; no wagering automation

## 1. Decision

Use JRA-VAN Data Lab. through the official JV-Link interface as the only allowed
prospective machine-ingestion source. Do not implement a JRA Web crawler, a
direct Data Lab. server client, or an adapter for a private Web endpoint.

This repository now defines only the transport-neutral boundary after JV-Link:
a Windows-side private adapter may export a normalized batch envelope, and the
local archive validates and stores it. The repository does not contain a paid
key, call JV-Link, contact JRA/JRA-VAN, or claim that live collection is active.

On 2026-08-31 the user deferred operational collection because the current
machine is a Mac and the official JV-Link transport is Windows-only. Keep this
design dormant until the user explicitly revisits a supported private Windows
environment. Do not seek an unofficial Mac transport or substitute JRA Web
scraping.

## 2. Official-source evidence

### Terms and permitted boundary

- The current [JRA-VAN terms](https://jra-van.jp/info/rule.html) limit service
  territory to Japan, restrict ordinary information use to the subscriber's
  personal/private use, and prohibit Data Lab. server access other than through
  JV-Link. The design therefore has exactly one official transport and keeps raw
  records local and private.
- In an [official staff answer about AI-assisted development](https://developer.jra-van.jp/t/topic/964),
  JRA-VAN staff said that the described individual Data Lab. use was acceptable
  where the official SDK/JV-Link was used for a private research environment and
  raw or bulk racing data was not sent to an external AI service. The answer also
  warns about excessive or unexpected access. This design matches those facts;
  it is not treated as a general legal opinion.
- The [official commercial-use notice](https://developer.jra-van.jp/t/topic/899)
  says an ordinary Data Lab. contract does not permit delivering its data to end
  users through an app or service. Commercial redistribution requires the
  corporate JRADB route. This groundwork does not authorize publication,
  multi-user sharing, or commercial use.
- JRA's public [site-use page](https://jra.jp/use/) establishes copyright and
  reuse controls but does not provide the systematic machine-ingestion and
  private archive contract required here. Public visibility is not treated as
  crawler permission.

### Platform and record availability

- The [current SDK page](https://jra-van.jp/dlb/sdv/sdk.html) lists SDK 5.0.0,
  including a 64-bit Python structure/sample package, while continuing to list
  JVData specification 4.9.0.1.
- The [Data Lab. product page](https://jra-van.jp/dlb/) lists JV-Link 5.0.0 for
  Windows 10/11. The current macOS repository environment cannot itself operate
  the official transport.
- The [official update schedule](https://jra-van.jp/dlb/ddata.html) describes
  Thursday runner-name tables, Friday/Saturday race cards, race-day changes,
  odds, and body weight. It describes time-series odds at roughly 5-10-minute
  intervals and body weight around 60 minutes before post.
- The [JVData specification 4.9.0.1](https://jra-van.jp/dlb/sdv/sdk/JV-Data4901.pdf)
  defines the source records and source timestamp fields used below. It also
  states that real-time meeting information may remove a formerly announced
  change, so current-state replacement must not erase the locally observed
  transaction history.

| Information | JV-Link service / record | Source-time contract |
|---|---|---|
| runner-name table / race card | `0B15`, `RA` / `SE` | exact announcement minute is not a record field; keep `published_at=null` and use first actual observation conservatively |
| body weight | `0B11`, `WH` | `発表月日時分`, minute precision |
| weather / turf and dirt going | `0B14` or `0B16`, `WE` | initial state and changes have `発表月日時分` |
| scratch / exclusion | `0B14` or `0B16`, `AV` | `発表月日時分` |
| jockey change | `0B14` or `0B16`, `JC` | before/after state and `発表月日時分` |
| scheduled-post change | `0B14` or `0B16`, `TC` | before/after time and `発表月日時分` |
| course change | `0B14` or `0B16`, `CC` | before/after course and `発表月日時分` |
| win/place/bracket time series | `0B41`, `O1` | intermediate record `発表月日時分`; one-year official availability |
| quinella time series | `0B42`, `O2` | intermediate record `発表月日時分`; one-year official availability |

Initial normalization may retain `O2`, but the first market snapshot target is
win odds in `O1`. Final/confirmed odds remain oracle/settlement information and
must not become an executable pre-deadline snapshot.

## 3. Architecture and lawful operating boundary

```text
JRA-VAN Data Lab. server
    -> official JV-Link cache on one private Windows host in Japan
    -> private adapter emitting the normalized envelope
    -> archive-jvlink-batch CLI (no network capability)
    -> ignored append-only raw objects, record manifests, and receipts
    -> later as-of snapshot materialization
```

Rules:

1. Only `provider=jra_van_data_lab` and `transport=jv_link` pass validation.
2. Service-data ID and record-type pairs are allow-listed in
   `configs/live_data/jvlink_archive.json`.
3. The adapter must not send an `ingested_at`; the local archive boundary owns
   that timestamp.
4. Raw payloads stay under an explicit private root. Within this repository the
   only allowed location is `data/live_data_private/`, which is ignored by Git.
   An external path is allowed only on the same private local host.
5. Raw payloads, reconstructed source data, and reversible extracts must not be
   committed, shared, or sent to an external AI service.
6. The archive code has no HTTP dependency and cannot fall back to JRA Web.

## 4. Time semantics

The envelope and receipt separate four facts:

| Field | Meaning |
|---|---|
| `source_created_date` | JVData record creation date; not assumed to be its exact availability time |
| `published_at` | official source announcement minute when explicitly present |
| `observed_at` | adapter clock when the normalized record was actually observed |
| `ingested_at` | UTC archive-boundary clock assigned by this repository |

`published_at` is nullable. A scheduled publication time from a Web page is not
substituted for a missing record timestamp. JVData's source value contains month,
day, hour, and minute; the Windows adapter must resolve its year from race/feed
context, emit an offset-aware JST value, and retain the resolution rule in its
own audit log. The archive requires exact `+09:00` and zero seconds for a
minute-precision source timestamp.

Clock flags are evidence, not automatic repairs:

- `source_published_at_unavailable`
- `source_published_after_observed`
- `observed_after_ingested`
- `requested_after_observed`
- `clock_order_ok`

For an actual live prediction, eligibility requires `ingested_at <= cutoff`.
A record published before cutoff but received afterward stays excluded from that
live snapshot. A later source-time replay using only `published_at` is PIT-B and
must be labeled separately.

## 5. Append-only archive contract

The schema is
`schemas/live_data/jvlink_batch_envelope.schema.json`; the executable policy is
`configs/live_data/jvlink_archive.json`.

The archive creates:

```text
private archive root/
  .private_archive.json
  objects/sha256/<prefix>/<payload_sha256>.bin
  records/sha256/<prefix>/<record_sha256>.json
  receipts/YYYY-MM-DD/<receipt_id>.json
```

- Payload objects are keyed by the SHA-256 of exact decoded bytes.
- Record manifests hash source identity, version, published time, flags, and
  payload hash; `observed_at` is intentionally excluded so a repeat observation
  deduplicates to the same source record.
- Every accepted batch still creates a new receipt. The receipt records the
  repeated observation, `ingested_at`, content hashes, source flags, clock flags,
  and whether payload/record storage was deduplicated.
- Existing content-addressed files are verified before reuse. A hash/path content
  mismatch fails rather than overwriting evidence.
- Envelope validation finishes before archive directories are created. Invalid
  source, record mapping, timestamp, flag, base64, or size fails closed.

Example groundwork-only invocation:

```bash
uv run horse-pred archive-jvlink-batch \
  --input /private/local/normalized_batch.json \
  --archive-root data/live_data_private \
  --config configs/live_data/jvlink_archive.json
```

This command archives a file already produced locally. It does not contact
JRA/JRA-VAN and does not mean that prospective collection has started.

## 6. Low-frequency, cache-first operational plan

The future Windows adapter should let JV-Link own authentication, network access,
and official caching.

1. Retrieve `0B15` after the official Thursday/Friday/Saturday availability
   windows; treat schedules as wake-up hints, not exact `published_at` values.
2. Use official JV-Link event notification / `0B16` for changes and reconcile
   against `0B14` at a low fixed interval. Store every observed current-state
   version. If a prior change disappears, emit a retraction observation instead
   of deleting history.
3. Receive `WH` when announced and archive only changed content objects while
   preserving every batch receipt.
4. Request `0B41` no faster than its roughly 5-10-minute source update interval.
   Keep all source intervals; do not tune a cutoff during collection.
5. After restart, backfill within the official one-week/one-year feed windows and
   log the outage. Do not claim that a short-lived event was reconstructed if it
   appeared and disappeared during downtime.
6. Synchronize the host clock and record clock health with each batch. Collector
   liveness, zero-record batches, and errors are evidence and must not be silently
   omitted.

## 7. Future as-of rules

Once real receipts exist, a T-close manifest should select:

- the latest ingested runner version and all ingested AV/JC/TC/CC state as of the
  cutoff;
- the latest applicable ingested WE venue-day state;
- the latest ingested WH record for each runner;
- the last intermediate O1 record satisfying both `published_at <= cutoff` and
  `ingested_at <= cutoff`;
- the latest scheduled-post version known at the time, without retroactively
  moving an already frozen decision snapshot.

T-10 versus T-15 is deliberately not chosen here. It should be preregistered
only after at least two meetings establish feed completeness and receipt latency.
No ROI or purchase-strategy optimization belongs in that activation audit.

## 8. Activation checklist and honest status

Design/code groundwork is complete when the schema, policy, fail-closed archive,
CLI, and tests are present. That is not the same as collection start.

Operational collection may be called **started** only after all applicable boxes
below have real evidence:

- [ ] The user has an appropriate individual/private Data Lab. contract and key.
- [ ] An in-Japan private Windows 10/11 host runs the official current JV-Link.
- [ ] The raw archive is outside Git/sharing/external-AI boundaries.
- [ ] Host clock synchronization and a collector identifier are recorded.
- [ ] One real race day has durable `0B15`, WE, WH, and pre-post O1 receipts.
- [ ] A `0B14`/`0B16` reconciliation receipt exists even if no change occurred.
- [ ] Duplicate, latency, missing-record, restart, and zero-record behavior has an
  audit report.
- [ ] A scheduler/monitor remains enabled for the next meeting.

At the date of this document none of those live-feed boxes has been evidenced in
the repository. The honest state is therefore:

```text
design and local archive groundwork: implemented
paid key: not present
JRA/JRA-VAN network access by this code: none
real prospective snapshots: zero demonstrated
operational collection: not started
```

## 9. Uncertainty and stop conditions

- The official staff answer is conditional on the described personal/private
  facts; it is not blanket permission for teams, cloud raw storage, or model/API
  redistribution.
- Exact arrival latency, event loss during downtime, and the completeness of each
  live feed remain unknown until genuine receipts exist.
- A minute-level odds record does not prove a price was executable at an exact
  second or that the eventual pari-mutuel return was fixed.
- JV-Link and JVData versions may change. Unknown service IDs, record types,
  source flags, or envelope fields fail until policy/schema changes are reviewed.
- If the intended use becomes shared, public, corporate, commercial, or sends
  raw data to an external service, stop collection and obtain written terms for
  that concrete use before proceeding.
