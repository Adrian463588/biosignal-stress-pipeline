# Architecture Decision Records

Related release governance: [traceability matrix](traceability-matrix.md) and
[release freeze / pending evidence ledger](release-freeze-and-pending-evidence.md).

## ADR-001 — Local-only mandatory execution

Accepted. Pipeline, tracking, training, export, and validation run locally.
Remote tracking is optional and cannot be required for correctness.

## ADR-002 — Renamed raw source

Accepted. `DataRenamedPartisipan` is canonical for ECG and all PPG channels.
The trimmed tree is comparison-only to prevent duplicate slicing/buffering.

## ADR-003 — Two-stage ECG synchronization

Accepted. Apply bounded affine clock correction on full recordings, then
physiological R-peak/systolic or IBI validation. PPG remains fixed and PAT
100–400 ms is preserved. Invalid ECG is excluded without excluding good PPG.

## ADR-004 — Historical 28 plus 4 model lifecycle

Superseded by ADR-011. Historically, nested LOSO used a frozen 28-participant development pool. Four
participants are evaluated once. Production refit uses all 32 only after
features, threshold, and hyperparameters are frozen.

The stress decision threshold is pre-specified at `0.5`. Nested experiments
showed that per-fold threshold optimization was unstable across participants;
using the fixed calibrated-classifier boundary avoids outer-subject leakage and
is the value frozen before the untouched holdout is opened.

## ADR-005 — Android inference boundary

Accepted. A dedicated `:inference` module owns manifest validation,
preprocessing, ONNX execution, and fail-closed results. `:mobile` is a consumer.
Physical-device vector parity is required for an Android-ready release.

## ADR-011 — Release gate, holdout, and export correction

Accepted. ADR-007 and ADR-010 gate-lowering decisions are superseded. The
release gate is exactly `0.80`; ID-29..ID-32 are spent evidence; ID-33..ID-36
are the next one-time untouched cohort after a checksum-recorded freeze. Only
accepted, checksum- and parity-validated artifacts may enter the canonical
versioned `model_index.json` and Android asset tree.

## ADR-013 — Local validation and manual release governance

Accepted. Hosted CI/CD is replaced by fail-closed repository-owned PowerShell
runners and append-only local evidence. ADR-011 remains authoritative: the gate
is exactly `0.80`, ID-29..ID-32 are spent, ID-33..ID-36 are evaluated once only
after freeze, and manual sign-off cannot override a failed or pending gate.
