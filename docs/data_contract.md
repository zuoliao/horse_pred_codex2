# Data contract: `race_results_merged.csv`

**Contract version:** 1  
**Frozen:** 2026-08-30 (JST)  
**Tasks covered:** GOV-01, DATA-01, DATA-02, DATA-03

## 1. Scope and governance

The canonical raw source for the first JRA flat-racing baseline is the existing local file named `race_results_merged.csv`. Its bytes are not copied into or tracked by this repository. The path must be injected either with `--raw-path` or `HORSE_PRED_RAW_CSV`; code and configuration must not contain a workstation-specific absolute path.

The source is an existing netkeiba-derived local dataset whose use has been approved for this private local project. That approval does not authorize new scraping, redistribution, third-party sharing, commercial use, or upload to an external AI/SaaS. Raw, intermediate, processed, or reconstructable row-level data must not be committed.

The machine-readable source of truth is [`configs/data_manifest.json`](../configs/data_manifest.json). This document explains decisions; it does not replace the manifest.

## 2. Frozen byte identity

| Property | Frozen value |
|---|---:|
| File name | `race_results_merged.csv` |
| SHA-256 | `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87` |
| Size | 118,918,481 bytes |
| Encoding | UTF-8 with BOM; open as `utf-8-sig` |
| Header columns | **26** |
| Runner rows | 629,967 |
| Races | 44,761 |
| Event range | 2013-01-05 through 2025-12-28 |

Earlier research text described 27 columns. Direct byte and pandas audits both show **26 columns**; the earlier number was a documentation error. There is no raw `venue` column. Venue is derived from positions 5–6 of the 12-digit `raceid` and is not invented as a 27th raw field.

The loader checks bytes before parsing when an expected hash is supplied. Size, BOM, ordered header, row count, race count, and event range are separately checked by the manifest audit. A changed file must receive a new manifest/dataset ID; updating the expected hash merely to make a failure disappear is prohibited.

## 3. Raw schema

Nullability below describes observed source tokens. Empty string, `--`, and `---` normalize to null for numeric derivatives, while the original token remains in the raw column.

| # | Raw column | Logical type | Nullable | Unit / meaning | Layer |
|---:|---|---|---:|---|---|
| 1 | `raceid` | 12-digit string | no | `YYYY + venue + meeting + day + race` | key |
| 2 | `race_class` | string | no | source condition/class label; NBSP retained | context |
| 3 | `course_type` | enum string | no | `芝`, `ダート`, `障害` | context |
| 4 | `distance` | integer | no | metres | context |
| 5 | `ground_state` | enum string | no | `良`, `稍重`, `重`, `不良` | result-page context |
| 6 | `around` | enum string | yes | `右`, `左`, `直線`; blank is observed mainly for jump | context |
| 7 | `weather` | enum string | no | `晴`, `曇`, `雨`, `小雨`, `小雪`, `雪` | result-page context |
| 8 | `着順` | outcome string | no | numeric or exceptional code | outcome |
| 9 | `枠番` | integer | no | 1–8 observed | runner context |
| 10 | `馬番` | integer | no | 1–18 observed | race-runner key |
| 11 | `馬名` | string | no | display identity | identity only |
| 12 | `horse_id` | string | no | 10 digits observed | source-native key |
| 13 | `sex` | enum string | no | `牡`, `牝`, `セ` | runner context |
| 14 | `age` | integer | no | years; 2–13 observed | runner context |
| 15 | `騎手` | string | no | display identity | identity only |
| 16 | `jockey_id` | string | no | 3 or 4 digits observed; keep as string | source-native key |
| 17 | `trainer` | string | no | display identity | identity only |
| 18 | `タイム` | duration string | yes | source `M:SS.t`; preserved before later parsing | outcome |
| 19 | `着差` | margin string | yes | numeric/text margin including `同着` | outcome |
| 20 | `通過順位` | string | yes | hyphen-separated source order | outcome |
| 21 | `上がり3F` | number | yes | seconds | outcome |
| 22 | `単勝` | number | yes | **final** win odds only | final market |
| 23 | `人気` | integer | yes | final popularity rank | final market |
| 24 | `馬体重` | integer | yes | kg | result-page context |
| 25 | `馬体重増減` | integer | yes | kg vs previous published value | result-page context |
| 26 | `date` | ISO date | no | event date | event time |

Observed null-token counts include `タイム` 5,147, `着差` 49,921, `通過順位` 3,102, `上がり3F` 5,151, `単勝`/`人気` 2,212 each, `馬体重`/`馬体重増減` 1,096 each, and `around` 19,697. Missing or unknown values are preserved and reported; they are never silently filtered.

### Keys

- Race key: `raceid`.
- Race-runner uniqueness checks: (`raceid`, `horse_id`) and (`raceid`, `馬番`). Both have zero observed duplicates in the frozen raw.
- `horse_id` and `jockey_id` are source-native join keys but are not direct initial model features.
- Names are display/audit fields, not safe unique identifiers.

## 4. Venue and surface normalization

Venue comes only from `raceid[4:6]`:

| Code | Canonical venue | Code | Canonical venue |
|---|---|---|---|
| 01 | `sapporo` | 06 | `nakayama` |
| 02 | `hakodate` | 07 | `chukyo` |
| 03 | `fukushima` | 08 | `kyoto` |
| 04 | `niigata` | 09 | `hanshin` |
| 05 | `tokyo` | 10 | `kokura` |

Surface maps `芝→turf`, `ダート→dirt`, `障害→jump`. Unknown venue/surface values become canonical `unknown` while `venue_code`/`surface_raw` retain the source value and the audit increments an unknown counter. Unknown values are not dropped.

The initial modeling population is JRA **flat** racing. Jump rows remain in the normalized/audit and feature layers but are excluded by an explicit race-level population filter, never by the raw loader. `course_type` alone is not a valid discriminator because the source retains `芝` or `ダート` for some obstacle races; any race whose `race_class` contains `障害` is non-flat regardless of its surface label.

## 5. Outcome and exception contract

### 5.1 Row-level normalization

| Raw `着順` | `status` | `finish_position` | `started` | history update | `winner_label` |
|---|---|---:|---:|---:|---:|
| numeric `N` | `finished` | N | true | yes | 1 iff N=1 |
| `N(降)` / full-width parentheses | `demoted` | N | true | yes | 1 iff official N=1 |
| `中` | `did_not_finish` | null | true | yes | 0 |
| `失` | `disqualified` | null | true | yes | 0 |
| `取` | `scratched` | null | false | no | null |
| `除` | `excluded` | null | false | no | null |
| anything else | `unknown` | null | unknown | no | null |

`中` and `失` remain members of the starting field and opponent set. Their position-based target is missing; they must not be removed before starter count or workload/history update. `取` and `除` remain declared rows but are excluded from starter count and history update.

### 5.2 Dead heats and probability mass

Dead heat is detected by duplicate numeric official `finish_position` within a race, not solely by `着差 == 同着`: the source writes `同着` on only one member in observed examples. Every member of a duplicate-rank group receives `is_dead_heat=true` and its group size.

For a dead heat for first:

- every official co-winner has `winner_label=1` for the direct binary outcome;
- a separate `coherent_win_target` assigns `1/m` to each of `m` co-winners and zero to other starters, so race target mass sums to one;
- non-starters receive null for both targets.

The frozen raw contains 1,240 races with at least one duplicate official rank and 1,264 runners beyond one-per-rank. These counts cover dead heats at any finishing position, not only first.

### 5.3 Field sizes, scoring, and settlement boundary

- `declared_runner_count`: every raw row, including `取`/`除`.
- `starter_count`: rows with `started=true`, including `中`/`失`.
- A race containing `取` or `除` receives `pit_c_scoring_eligible=false` for the initial scoring dataset. The event time of that status is unavailable in this result-page reconstruction, so the project will not pretend the exact pre-race field was known at a historical cutoff.
- All such races remain in audit output and raw/normalized storage.
- Refund and payout settlement cannot be inferred from this file: it has final win odds but no payout/vote table. DATA-03 freezes labels and field membership only; BET-01 must later join official payout data and test dead heat/refund rules separately.

Observed exceptional runner counts are `取` 1,006; `除` 1,206; `中` 2,935; `失` 2; and demotion forms 24. The normalized starter count is 627,755; non-starter count is 2,212. Because some races contain more than one `取`/`除`, the distinct count of initially PIT-C-ineligible races is 2,096.

## 6. Coverage audit and known missing race IDs

JRA annual race totals are taken from the official [競走回数（JRAの概要）](https://www.jra.go.jp/company/about/outline/growth/kyoso/index.html), accessed 2026-08-30. All raw years except these three match the official annual total:

| Year | Raw | JRA official | Shortfall |
|---:|---:|---:|---:|
| 2015 | 3,452 | 3,454 | 2 |
| 2017 | 3,419 | 3,455 | 36 |
| 2024 | 3,346 | 3,454 | 108 |
| **Total** |  |  | **146** |

The 146 absent IDs were reconstructed and checked against the raw at race-ID level:

- 2015: `201508010707`, `201508010711`.
- 2017: `201708031107`–`201708031112`, `201708031201`–`201708031212`, `201708040101`–`201708040112`, `201708040201`–`201708040206`.
- 2024: every race 1–12 on `2024080701` through `2024080709` (the nine-day 7th Kyoto meeting), 108 IDs.

The manifest stores inclusive day/race ranges that expand deterministically to exactly 146 IDs. The audit fails if any manifest-declared absent ID unexpectedly appears. It also emits race coverage by year, venue code, month, and surface, plus runner counts by surface. A future corrected source must be a new data version; missing races must be added as a separate provenance-preserving supplement rather than silently overwriting this raw.

Because 2024 is both development data and has a 108-race coverage hole, all 2024 metrics must carry a coverage warning and, when supplementation is available, a before/after sensitivity comparison.

## 7. Frozen time split

This split is preregistered before model metrics:

| Role | Period |
|---|---|
| warm-up only | 2013 |
| train | 2014–2021 |
| model validation | 2022 |
| probability calibration | 2023 |
| development/backtest iteration | 2024 |
| retrospective test | 2025 |
| prospective final | 2026 onward |

The same-family repositories have already inspected 2025, so it must not be called a perfectly untouched final holdout. Only future 2026+ observations collected under a frozen protocol can serve as prospective final evidence.

## 8. APIs and audit command

Public module APIs:

```python
load_raw(path, expected_sha256=None)       # -> pandas.DataFrame, raw strings
normalize_raw(raw)                         # -> pandas.DataFrame, no exception drops
audit_raw(path, expected_sha256=None)       # -> dict, streaming aggregate audit
```

The normalized DataFrame retains all 26 raw columns and adds at least `race_id`, `race_date`, `venue`, `surface`, `finish_position`, `status`, `started`, `winner_label`, `coherent_win_target`, and `pit_c_scoring_eligible`. `finish_status` is retained as a compatibility alias of `status`.

Full manifest verification:

```bash
uv run python scripts/audit_data.py \
  --raw-path /absolute/path/to/race_results_merged.csv
```

or:

```bash
HORSE_PRED_RAW_CSV=/absolute/path/to/race_results_merged.csv \
  uv run python scripts/audit_data.py
```

`--skip-sha256` exists only for repeated local diagnostics; a release/experiment audit must not use it. The script is read-only and prints aggregate JSON to stdout.

## 9. Remaining decisions

- Obtain the 146 missing official result rows under an approved source gate and version them as a supplement.
- Define the later flat-population filter and confirm whether rare flat races with unusual course metadata require exclusions.
- Parse time, margin, and passing-order strings in PIPE-01 with separate fixtures; this contract intentionally preserves them verbatim.
- Join payout/refund data before any realized-return calculation.
- PIT-01 must classify result-page weather, going, body weight, final odds, and popularity; this contract does not promote them to historically timestamped pre-race inputs.
