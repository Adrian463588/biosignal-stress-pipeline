# ADR-011: Release Gate, Holdout, and Export Evidence Correction

## Status

Accepted — 2026-07-29. This ADR supersedes the gate-lowering decisions in
ADR-007 and ADR-010 and the original ID-29..ID-32 untouched-holdout lifecycle
in the local decision log. Historical results remain readable but are not
release evidence.

## Context

The fusion decision threshold was selected using labels from ID-29..ID-32,
and a fusion manifest recorded an unauthorized `0.70` performance gate.
Therefore ID-29..ID-32 are spent/contaminated evidence and neither their
threshold-dependent result nor artifacts authorized by a gate below `0.80`
can support deployment.

## Decision

1. The release gate is exactly `balanced_accuracy >= 0.80`. Values below or
   above `0.80` are invalid configuration; there is no runtime, per-sensor,
   fusion, manifest, or ADR exception mechanism.
2. All existing ID-29..ID-32 data may join the original 32-participant
   participant-grouped development pool. It must not be reported as untouched
   evidence again.
3. ID-33..ID-36 form the new untouched cohort. Before accessing their features,
   waveforms, or labels, freeze and checksum the architecture, hyperparameters,
   feature and preprocessing contracts, participant-grouped validation design,
   decision threshold, and executable/config revisions.
4. Train the frozen candidate once on the original 32, then evaluate ID-33..ID-36
   exactly once with the frozen threshold. No preprocessing fit, calibration,
   threshold choice, early stopping, architecture choice, or retry may use the
   new-cohort labels. Record access time, cohort IDs, checksums, and per-participant
   results in append-only evidence.
5. Passing requires new-cohort balanced accuracy `>= 0.80`. On failure, block
   release and treat ID-33..ID-36 as spent; do not tune against them. A future
   cohort requires a new corrective ADR. On pass, the frozen design may be
   refit on all 36 for production, while preserving pre-refit validation
   evidence and recording post-refit provenance.
6. Existing ONNX files, manifests, parity reports, and Android assets derived
   from a lower gate or contaminated holdout remain historical/quarantined and
   must not appear in the deployable index.
7. Canonical accepted export layout is append-only:

   ```text
   data/artifacts/export/{model_version}/stress_{sensor}_fp32.onnx
   data/artifacts/export/{model_version}/model_manifest_{sensor}.json
   data/artifacts/export/{model_version}/onnx_parity_{sensor}.json
   data/artifacts/export/{model_version}/test_vectors_{sensor}.json
   data/artifacts/export/{model_version}/model_index.json
   android/*/src/main/assets/models/{model_version}/{sensor}/
   ```

   `model_index.json` lists accepted artifacts only. Publication is atomic and
   requires the exact gate, static batch-1 named inputs, SHA-256, ONNX checker
   and real inference evidence, native/ONNX parity `< 1e-5`, frozen threshold,
   cohort IDs, versions, model family, refit provenance, and release status.
   Hybrid/fusion manifests enumerate every waveform/tabular input explicitly.
8. Quantized artifacts remain ineligible unless balanced accuracy is `>= 0.80`
   and degradation from FP32 is at most two percentage points.

## Consequences

- Release and Android packaging remain frozen until the new-cohort, artifact,
  AI/clinical-asset, and physical-device evidence gates all pass.
- Previously claimed fusion performance may guide engineering investigation
  only; it cannot establish product efficacy or be promoted.
- Historical files are preserved. Rollback changes only the canonical accepted
  index to the last fully validated version; it never overwrites raw data or
  evidence.

## Verification

- Configuration and publication tests reject any gate value other than `0.80`.
- A threshold-independence test changes new-cohort labels and proves the frozen
  threshold/checksum do not change.
- Artifact-index tests reject contaminated, failed-gate, missing-checksum,
  dynamic-shape, failed-checker, or failed-parity entries.
- Release evidence identifies missing ID-33..ID-36 data and physical-device
  results as blocking, never as a pass or silent skip.
