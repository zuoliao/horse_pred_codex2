# Prospective activation plan

## Honest current state

```text
official-source/archive design: implemented
operational JV-Link collection: not started
real prospective receipts: 0 demonstrated
as-of snapshots: not implemented
shadow prediction: not available
```

The repository has a fail-closed, transport-neutral envelope, official service/record allow-list, append-only content-addressed archive, local archive CLI, and tests. It deliberately has no JV-Link transport, HTTP fallback, or JRA scraper.

## Missing activation prerequisites

| Area | Current evidence | Required before activation |
|---|---|---|
| Contract | no key or contract evidence | individual/private JRA-VAN Data Lab contract and local key setup |
| Host | current development host is macOS | supported private Windows 10/11 host operated in Japan |
| Official software | not evidenced | current official JV-Link/SDK installed and version recorded |
| Adapter | absent | Windows JV-Link reader plus RA/SE/WH/WE/AV/JC/TC/CC/O1 parser that emits the checked-in envelope |
| Scheduling | absent | low-frequency service/task scheduler, cursors, retries, restart/backfill rules |
| Monitoring | absent | clock, liveness, zero-record/error, gaps, disk, version and cursor continuity evidence |
| Snapshot materializer | absent | dual `published_at`/`ingested_at` as-of state, revisions/retractions and completeness reason codes |
| Frozen deployment | absent | model/schema/preprocessing/calibration bundle with registry hashes |
| Shadow runner | absent | common `decision_snapshot_id` with fundamental, market, combined, policy and later settlement |

There is one concrete contract mismatch inside the groundwork: the operational design says zero-record batches must be logged, while the data envelope rejects an empty `records` array. Do not weaken the raw envelope. Add a separate health/error/zero-record event ledger before activation.

## User decisions and costs

1. **Subscription:** confirm an individual/private JRA-VAN Data Lab contract. The official product page accessed 2026-09-01 displays **¥2,090/month including tax** and JV-Link 5.0.0 for Windows 10/11. Reconfirm the price, trial conditions, and versions at signup: <https://jra-van.jp/dlb/>.
2. **Windows host:** choose an existing always-on private Windows 10/11 PC or purchase/provision one in Japan. Hardware, electricity, storage, and backup costs are variable and are not estimated by this repository.
3. **Topology:** safest default is collector, private archive, scheduler, and monitoring on the same Windows host. If scoring stays on Mac, the user must approve a private encrypted handoff that does not send raw/reconstructable JVData to cloud, shared storage, Git, chat, or an external AI service.
4. **Storage/security:** choose archive volume, encryption, backup/retention, disk budget, and local secret storage. Never place the service key in Git or chat.
5. **Operations:** identify who keeps the host running on meeting days and which local/private channel receives alerts.
6. **Use boundary:** confirm the use remains individual, private, and shadow-only. Shared/public/commercial delivery requires a fresh written terms review and potentially the corporate JRADB route.

No unofficial Mac/Wine transport, Web crawler, or unlicensed alternative will be implemented.

## Activation sequence

### Immediately executable repository work

- define Windows adapter interface/conformance fixtures without claiming a live adapter;
- add a separate health/error/zero-record event schema and audit CLI;
- implement fixture-only as-of materialization and completeness/latency audit;
- define immutable deployment-bundle and shadow-ledger schemas;
- add dry-run monitoring and decision-snapshot fixtures.

These are infrastructure tasks, not feature/model research. They still require a separate implementation authorization if the user wants them now.

### After user provisioning

1. Install official JV-Link/SDK and subscriber setup on the private Windows host.
2. Implement and acceptance-test the real adapter on that host.
3. Start scheduler/archive/health ledger and collect one meeting without scoring claims.
4. Demonstrate reconciliation, duplicate, restart, zero-record/error, cursor, clock, and disk behavior.
5. Collect at least a second complete meeting and create the completeness/latency report.
6. Freeze the cutoff using only feed availability—not outcomes, metrics, ROI, or profit.
7. Package the canonical fundamental and begin the frozen shadow protocol.

### Must wait for prospective data

- proper-score and incremental-market conclusions;
- edge-band calibration;
- fixed-stake shadow ROI, profit, bet count, drawdown and monthly stability;
- A1/A2, nonlinear race-wise, Inter-horse, DNN, and additional feature work.

Minute-level O1 odds are a displayed pari-mutuel snapshot, not a locked execution price. Shadow profit remains diagnostic even after collection begins.
