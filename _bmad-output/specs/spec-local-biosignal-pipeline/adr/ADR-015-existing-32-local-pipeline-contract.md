# ADR-015: Existing-32 Local Pipeline and Sensor-Specific Source Contract

## Status

Accepted — 2026-07-29.

## Context

Only ID-01..ID-32 exist. Earlier documents required a fabricated ID-33..ID-36
cohort and incorrectly classified all of `DataTrimmedPartisipan` as
comparison-only. The current filesystem is authoritative.

## Decision

1. Never create, require, or claim evidence from ID-33..ID-36.
2. Use ID-01..ID-28 for grouped development and ID-29..ID-32 as an isolated
   engineering parity pool. The parity pool influences no fitting,
   preprocessing, tuning, early stopping, threshold selection, or ranking.
3. Ingest complete ECG only from `DataRenamedPartisipan`; ingest PPG IR, Red,
   and Green only from `DataTrimmedPartisipan`; trimmed ECG is audit-only.
4. Lock release gates at balanced accuracy >= 0.80, F1 >= 0.78, and ROC-AUC
   >= 0.83. These are release gates, never tuning parameters.
5. Run the complete pipeline locally. `run_full_pipeline.ps1` remains
   validation-only; `run_local_training_pipeline.ps1` is mutating and resumable.
6. Refit an accepted frozen design on all 32 only after parity-pool evaluation
   passes every gate.

## Consequences

ADR-015 supersedes ADR-011 and conflicting portions of ADR-013, ADR-004,
AGENTS.md, and PRD.md concerning ID-33..ID-36, all-32 model selection, the
older gate values, and globally comparison-only trimmed data. Historical ADRs are
retained as incident evidence, not active data truth.

## Verification

- Input validation emits 256 production lineage rows with renamed ECG and
  trimmed PPG sources, plus a separate trimmed-ECG audit index.
- Gold contains exactly 28 `train` and four `parity_pool` participants.
- Local gate validation rejects fabricated participants and candidates missing
  any locked metric.
