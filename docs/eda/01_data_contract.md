# Phase 5A EDA Data Contract

## Time and population firewall

- `max_target_date` is fixed to `2022-12-31`; the loader rejects a later value or retained row.
- 2013 is state warm-up and quality coverage only. Discovery is 2014–2019, replication 2020–2021, confirmation 2022.
- 2023 is not loaded into target-aware EDA. 2024 and 2025 are not loaded or queried.
- Predictive views use eligible flat-race starters. Raw population views retain scratches, exclusions, DNF, DQ, demotions, and dead heats.
- All races on one date are emitted before that date updates historical state.

## Logical views

| View | Grain and key | Allowed content | Prohibited/isolated content | Availability |
|---|---|---|---|---|
| `race_table` | one `race_id` | context, restrictions, field/status counts, outcome summaries, source flags | runner IDs as features; future results | context pre-race; outcomes analysis-only |
| `runner_pre_race` | eligible race × horse | context and strict-PIT history/form/connection/rating/availability | current time, margin, last 3F, passing, popularity, odds; direct IDs in features | immediately before target date |
| `historical_performance` | horse × completed past race | past content, elapsed time to target, anomaly flags, opponent pre-race state | performance date not strictly before target date | later target dates only |
| `connection_state` | entity type/key × target race | starts, wins, recent/conditional form, effective sample size | direct entity key as model input; future target encoding | immediately before target date |
| `market_oracle` | race × runner | final odds/popularity and coverage flags | implicit primary join or candidate selection | final market only |
| `raw_status_population` | declared raw runner | normalized status and exception flags | predictive feature use | outcome audit only |

IDs are join/state keys only. Outcome labels use an explicit analysis namespace; market columns exist only in `market_oracle`.

## Machine-readable summary

Every view records row/race counts, date range, key uniqueness, raw SHA-256, schema hash, availability class, missing rates, and excluded population. Tracked files are aggregate JSON only; runner-level views remain under ignored `artifacts/`.

## PIT invariants

1. `historical_performance.performance_date < target_date`.
2. Same-date races cannot affect one another.
3. Appending future rows cannot change earlier view values.
4. `runner_pre_race` contains no final-market or current-race outcome columns.
5. `market_oracle` requires an explicit diagnostic join.
6. Full-period normalization, entity target encoding, and future opponent results are forbidden.

## Privacy and versioning

The approved local CSV is private. Raw, caches, models, predictions, and recoverable runner-level tables are ignored by Git. Only code/config, hashes, aggregate counts, non-recoverable summaries, and conclusions are tracked.
