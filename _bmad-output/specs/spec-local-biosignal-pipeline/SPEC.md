# Local Biosignal Pipeline v3

Status: in progress
Owner: ML and biosignal pipeline
Implementation repository: `ProsesData`

## Product outcome

Produce a reproducible local PPG-only stress model, a validated ONNX bundle, and
fail-closed Android inference from immutable renamed raw recordings. ECG is
clock-corrected before segmentation and used only to validate PPG timing.

## Epics and acceptance criteria

| Epic | Behaviour | Acceptance criterion |
|---|---|---|
| 0 | Audit and recoverable cleanup | Raw/canonical hashes are recorded; only allowlisted generated outputs move to an external backup. |
| 1 | Local configuration | Every required command works without Colab or remote services. |
| 2 | Raw ingestion | 32 participants × 4 sensors are indexed with timestamp and source lineage. |
| 3 | ECG synchronization | Corrected ECG is monotonic with `0.98 <= alpha <= 1.02`, or has an explicit exclusion reason. |
| 4 | Segmentation and SQA | Only relax/task windows wholly inside buffered anchors are emitted; training tiers are excellent/good. |
| 5 | Gold v3 | Per-sensor hierarchical Parquet, feature table, IBI sequence table, and content-hashed manifest are produced. |
| 6 | Leakage-safe training | 28-participant nested LOSO and four-participant untouched holdout are disjoint; all transforms fit within folds. |
| 7 | Model selection | Candidate meets BA 0.80, F1 0.78, ROC-AUC 0.83 or release is blocked. |
| 8 | ONNX | FP32 opset 17 matches native probability on at least 100 rows with max absolute difference below 1e-5. |
| 9 | Android | Schema/hash/quality failures return no label and no probability; acquisition-supported sensor only is deployed. |
| 10 | Release evidence | Python vectors pass 100/100; physical Android vectors pass 100/100 before `android_ready=true`. |

## Public CLI

`clean`, `validate-inputs`, `ingest`, `sync-raw`, `segment`, `preprocess`,
`extract-features`, `train-stress`, `export-onnx`,
`generate-test-vectors`, `validate`, and `run-all`.

## Versions

- preprocessing: `v3.0.0`
- features: `v3.0.0_180s`
- model: `stress_v3.0.0`

## Out of scope

- Routine phase
- ECG as a model predictor
- Mandatory personality model
- Mandatory DagsHub/remote MLflow
- Claiming Android-ready without physical-device evidence
