# Phase 5C model registry

## Decision

The canonical frozen fundamental candidate is the LightGBM Binary `PV-01 + PACE-01 + SPEED-01` 256-feature configuration, identified by ordered-feature SHA-256 `3b6104ec33f6bf2b02b64685bf3ebf6bb828f14683c262e922e696da43bb4940`.

This decision uses no new metric. SPEED-01 was already accepted on 2020–2023 rolling evidence and supported by its preregistered 2024 Binary milestone. Its experiment document explicitly promoted the unchanged candidate to development incumbent. Phase 5B documents later reused PV-01 254 as a matched experimental control and incorrectly allowed “formal control” to read like the accumulated incumbent.

The terms now mean:

| Term | Meaning |
|---|---|
| formal historical comparison control | Stable matched-experiment reference; not necessarily the best accumulated candidate |
| rolling candidate | Supported on exposed rolling years but not frozen as the prospective fundamental |
| development incumbent | Passed an already preregistered historical development milestone |
| canonical frozen fundamental candidate | The single no-odds identity to package for prospective comparison |

## Lineage

| Model | Count | Dependency | Evidence period | Registry status |
|---|---:|---|---|---|
| lean field-relative-drop | 253 | base | fit 2014–21; ES 2022; cal 2023; dev 2024 | LambdaRank conservative formal control; Binary historical base |
| Binary PV-01 | 254 | lean + signed time-gap | selection 2018–21; gate 2022; cal 2023; dev 2024 | formal historical comparison control |
| LambdaRank PV-01 | 254 | lean + signed time-gap | same historical contract | interval-inconclusive branch |
| LambdaRank SEC-3F | 254 | lean + last-3F percentile history | rolling eval 2020–23 | rolling component |
| Binary PACE-01 | 255 | PV-01 + early-position history | rolling eval 2020–23 | accepted parent candidate |
| LambdaRank PACE-01 | 255 | lean + SEC-3F + PACE-01 | rolling eval 2020–23 | accepted parent candidate |
| **Binary SPEED-01** | **256** | **PV-01 + PACE-01 + SPEED-01** | rolling 2020–23; prereg dev 2024 | **development incumbent; canonical frozen fundamental** |
| LambdaRank SPEED-01 | 256 | lean + SEC-3F + PACE-01 + SPEED-01 | rolling 2020–23; dev 2024 directional | rolling candidate only |
| Binary S1 performance | 255 | PV-01 + fold-train-frozen performance | eval 2020–22; reproduced in S2 | supported overlapping branch, not composed |
| LambdaRank S1 performance | 254 | lean + fold-train-frozen performance | eval 2020–22 | supported overlapping branch, not composed |
| S3 Huber target | matched 254/253 | target replacement | eval 2020–22 | rejected |
| S2 linear Conditional Logit | matched 254/255 | objective replacement | eval 2020–22 | rejected as Binary replacement |

S1 performance and SPEED-01 are semantically close but have different temporal fitting contracts: SPEED-01 is fully prequential, whereas S1 refits the normalizer inside each fold train and freezes it afterward. Their union with the 256-feature incumbent was never evaluated. It is not constructed during this audit.

Feature count alone is not an identity. S1 joint C3 is also a 256-feature Binary branch but has a different ordered-feature hash. Every frozen reference must include family, ordered-feature hash, configs, raw fingerprint, and source commit.

## Frozen identity

- feature config: `configs/performance/speed_01_binary_candidate.json`, file SHA-256 `3ddcc75e0f2e96a3f3ce0b61a438212cff9cf975f62e1ae3280aa893933f51bb`
- model config: `configs/exp_001_binary.json`, SHA-256 `46afba67d03f28c48f6177dbb43960a2eb38326df6c2a8e68cd28763e748d2b5`
- milestone config: `configs/evaluation/speed_01_2024_milestone.json`, SHA-256 `c76618cae7fe2f4a5a5a8c70ecf529cdb5ec747b7bdb71440fc2cde03fd04e09`
- rolling commit: `f4b38b8744dc7a49d701b23c1c14badc6a663571`
- milestone commit: `dbc656e16a167b3bdcb41691682d10260eed5c38`
- historical raw SHA-256: `270923ce73c4441e64173f242a8719de7d1e9b205508140463ca547ef7b1ca87`

The local model artifact is reproducibility evidence, not the registry source of truth and remains outside Git. Before shadow operation, the frozen identity still needs an immutable deployment bundle containing the model, ordered schema, preprocessing semantics, calibration, and hashes.

The complete machine-readable registry is `experiments/program_audit_20260901/model_registry.json`.
