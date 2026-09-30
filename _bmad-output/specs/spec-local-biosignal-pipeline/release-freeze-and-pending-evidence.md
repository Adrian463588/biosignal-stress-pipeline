# Release Freeze and Pending Evidence Ledger

## Decision

Release, ONNX promotion, and Android model packaging are **blocked** as of
2026-07-29. The freeze applies to every current model version and clinical
feature. Historical artifacts remain preserved but non-deployable.

## Triggering evidence

- The fusion threshold was selected using ID-29..ID-32 labels, so that cohort
  is no longer untouched.
- A fusion manifest recorded an unauthorized `0.70` gate.
- ADR-007 and ADR-010 attempted to authorize sub-`0.80` release; ADR-011 now
  supersedes those clauses.

## Blocking evidence ledger

| Gate | Required evidence | Current state | Promotion rule |
|---|---|---|---|
| Exact performance gate | Config/local gate/export prove the only accepted value is `0.80` | Passed locally; `20260729T115214690Z/01_exact_gate.*` | Any other value blocks before publication |
| Frozen model design | Checksums for code, config, preprocessing, features, architecture, threshold, and original-32 training input | Pending | Must exist before any ID-33..ID-36 access |
| New untouched cohort | One-time per-participant ID-33..ID-36 report with frozen threshold and BA `>=0.80` | Blocked: cohort evidence not present | Failure spends cohort and keeps release blocked |
| Canonical ONNX | Per-sensor static model, manifest, checker/forward inference, SHA-256, parity `<1e-5`, vectors, accepted index | Pending | Incomplete/legacy/quarantined artifact cannot enter index |
| Quantization | BA remains `>=0.80`; degradation is at most two percentage points | Pending | FP32 evidence must pass first |
| Android parity/runtime | 100 deterministic Python→ONNX→Kotlin vectors `<1e-5`; native runtime loads supported ABI | Pending | Host evidence plus physical-device instrumentation required |
| Clinical AI assets | User-supplied Gemma and approved KB manifest: source, license, version, size, SHA-256, signature | Blocked: supplied/validated assets not evidenced | Missing/corrupt/unlicensed asset exposes unavailable state |
| Verbatim/Live Monitor safety | Consent, encryption, real engines or typed unavailable states, 60-minute stability | Pending | No fake output or silent fallback |
| Physical phone and wearable | ONNX/LiteRT/Sherpa smoke, P50/P95, thermal, battery, lifecycle, 60-minute memory run | Blocked: connected-device evidence not present | Emulator/host-only results cannot promote release |
| Release bundle | Traceability, ADRs, commands, tests, metrics, confusion matrices, parity/checksums, device logs, limitations, rollback | Pending | All upstream gates must pass |

## Quarantine and rollback

- Do not delete or overwrite raw data, canonical legacy artifacts, current
  models, manifests, or unrelated worktree changes.
- Existing lower-gate or contaminated-holdout artifacts may be retained only
  as historical/quarantined evidence and must be absent from the accepted
  `model_index.json` and Android packaged assets.
- Publication is append-only and atomic. Rollback repoints the accepted index
  to the last fully validated version; if none exists, the safe rollback state
  is no deployable model and an explicit unavailable UI state.

## Unfreeze authority

QA/Release may unfreeze only after every ledger row is backed by reviewable
evidence and ADR-011 invariants pass in CI. Documentation edits alone cannot
unfreeze a release.
