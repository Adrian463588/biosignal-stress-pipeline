# ADR-016: Conditional Colab DL and Local Release Boundary

## Status

Accepted - 2026-08-09.

## Context

ADR-015 required the complete pipeline to run locally. The current PRD and
AGENTS contract instead require participant-safe classical ML to run locally
and permit deep learning only as a conditional remediation path on a Google
Colab T4. Raw biosignal files must remain local and immutable.

## Decision

1. Raw audit, Bronze/Silver/Gold wrangling, scientific validation, EDA,
   classical ML, ONNX export, parity verification, and Android builds run
   locally.
2. Deep learning runs only when a Green or IR classical model has development
   balanced accuracy below 0.80.
3. Conditional DL runs manually on a verified Google Colab T4. A missing or
   different accelerator is a hard failure, not a CPU fallback.
4. The handoff contains only channel-filtered Gold `ibi_values` and `ibi_mask`
   tensors plus the minimum lineage needed to enforce participant splits.
   ECG, Red PPG, PAT, motion features, and raw waveforms are forbidden.
5. Every handoff and checkpoint is bound to the Gold, configuration, and fold
   contract checksums. Resume rejects mismatched lineage.
6. Holdout ID-29..ID-32 remains closed until a development design is frozen.
   Colab does not alter the local release gates.
7. ONNX publication and Android asset promotion remain local and fail closed
   until development, holdout, ONNX parity, Kotlin parity, and required device
   evidence are all present.

## Consequences

This ADR supersedes only ADR-015 decision 5 where it requires deep learning to
run locally. ADR-015 remains authoritative for the 32-participant source and
split contract. Historical local-only DL statements are retained as audit
evidence but are not active execution policy.

## Verification

- Local ML reports state whether each channel requires conditional DL.
- The Colab handoff checksum is computed from its filtered Gold artifact.
- Handoff schema tests reject forbidden predictors and PPG Red.
- ONNX and Android stages refuse promotion when upstream gates are incomplete.
