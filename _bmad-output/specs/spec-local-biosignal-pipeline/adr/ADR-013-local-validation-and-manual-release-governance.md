# ADR-013: Local Validation and Manual Release Governance

## Status

Accepted — 2026-07-29.

## Context

This project is required to execute entirely on local, user-controlled
hardware. Cloud CI runners, remote model validation, and hosted release gates
are not part of the production path. Existing GitHub Actions language duplicated
the real release controls and could imply that a remote job, rather than local
evidence, authorized deployment.

ADR-011 remains authoritative for the exact `0.80` balanced-accuracy gate,
the spent ID-29..ID-32 evidence, the one-time ID-33..ID-36 cohort protocol,
canonical export layout, and fail-closed publication.

## Decision

1. All required lint, type, test, leakage, ONNX, Android, checksum, asset, and
   release checks run through repository-owned local scripts on the designated
   workstation and physical devices.
2. `PipelinePython/configs/frozen_gates.yaml` remains locked at exactly `0.80`. The local gate
   checker terminates with a non-zero exit code for any other value. A documented
   accepted ADR is required for a deliberate gate-file edit, but an ADR can never
   authorize a value other than `0.80` while ADR-011 is active.
3. Local runners record UTC timestamps, exact commands, exit codes, and logs in
   an append-only directory under `build/release-evidence-local/runs/`.
4. Host checks may pass while the release remains blocked. Missing cohort,
   accepted model, AI/clinical asset, phone/wearable, or physical-performance
   evidence is never converted to a pass or silent skip.
5. Manual sign-off is allowed only after the generated release snapshot reports
   `candidate` and every required evidence record has status `passed`. Manual
   sign-off cannot override a failed or pending automated gate.
6. The local validation entry points are:

   ```text
   scripts/run_ml_validation.ps1
   scripts/run_android_validation.ps1
   scripts/verify_release_gate.ps1
   scripts/run_full_pipeline.ps1
   ```

7. `run_full_pipeline.ps1` is validation-only. It does not clean data, train,
   access the one-time cohort, export models, or publish assets. Mutating pipeline
   stages remain explicit, separately reviewed commands with versioned outputs.
8. The former hosted workflow is retained only as disabled historical evidence
   outside `.github/workflows/`; it is not an executable release mechanism.

## Consequences

- Local validation can be reproduced without network or hosted-runner access.
- Release authorization depends on evidence, not on merge state or a remote job.
- Physical-device checks remain mandatory and cannot be inferred from host tests.
- Historical logs and artifacts remain append-only; rollback never touches raw
  data and only repoints the canonical accepted model index.

## Verification

- Deliberately setting the frozen gate below or above `0.80` makes the local gate
  script fail.
- A blocked empty model index validates structurally but the release verifier
  exits non-zero because the overall release is not a candidate.
- Python and active Android host suites run from the local orchestrators and
  write command-level evidence.
- No active file remains under `.github/workflows/` after local equivalence is
  verified.

## Rollback

Restore the disabled workflow only as historical text. Do not use rollback to
weaken ADR-011, accept contaminated evidence, alter raw data, or publish an
unvalidated artifact.
